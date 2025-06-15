"""Orchestrator agent for managing all meta agents."""

import asyncio
import json
import logging
import redis
import websockets
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Type, Set
from dataclasses import dataclass
import schedule
import time
import threading
from fastapi import FastAPI, WebSocket, HTTPException
from pydantic import BaseModel, ValidationError

from .base_agent import BaseMetaAgent
from .code_review_agent import CodeReviewAgent
from .test_repair_agent import TestRepairAgent
from .performance_monitor_agent import PerformanceMonitorAgent
from .data_quality_agent import DataQualityAgent
from .code_generator import CodeGeneratorAgent

@dataclass
class Task:
    """Represents a task to be executed by an agent."""
    id: str
    type: str
    agent_id: Optional[str]
    dependencies: List[str]
    params: Dict[str, Any]
    status: str = "pending"
    created_at: str = datetime.utcnow().isoformat()
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

class TaskConfig(BaseModel):
    """Configuration for task execution."""
    max_retries: int = 3
    timeout_seconds: int = 300
    default_agent: str = "code_review"
    enable_websockets: bool = True
    redis_url: str = "redis://localhost:6379"
    log_level: str = "INFO"

class WebSocketManager:
    """Manages WebSocket connections and broadcasts."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.logger = logging.getLogger("WebSocketManager")
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Connect a new client."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.logger.info(f"Client {client_id} connected")
    
    async def disconnect(self, client_id: str):
        """Disconnect a client."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            self.logger.info(f"Client {client_id} disconnected")
    
    async def broadcast_task_update(self, task: Task):
        """Broadcast task status update to all clients."""
        message = {
            "type": "task_update",
            "task_id": task.id,
            "status": task.status,
            "timestamp": datetime.utcnow().isoformat(),
            "data": task.__dict__
        }
        await self.broadcast(message)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        disconnected = []
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_json(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.append(client_id)
        
        for client_id in disconnected:
            await self.disconnect(client_id)

class DevelopmentOrchestrator:
    """Orchestrates system-wide task execution."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the orchestrator.
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger("DevelopmentOrchestrator")
        self.config = self._load_config(config_path)
        self._validate_config()
        
        # Initialize components
        self.redis_client = redis.Redis.from_url(self.config.redis_url)
        self.websocket_manager = WebSocketManager()
        self.progress: Dict[str, Task] = {}
        
        # Initialize agents
        self.agents: Dict[str, BaseMetaAgent] = {
            "code_review": CodeReviewAgent(self.config.dict()),
            "test_repair": TestRepairAgent(self.config.dict()),
            "performance": PerformanceMonitorAgent(self.config.dict()),
            "data_quality": DataQualityAgent(self.config.dict()),
            "code_generator": CodeGeneratorAgent(self.config.dict())
        }
        
        # Setup FastAPI
        self.app = FastAPI(title="Development Orchestrator")
        self._setup_routes()
    
    def _load_config(self, config_path: Optional[str]) -> TaskConfig:
        """Load configuration from file or use defaults."""
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                config_data = json.load(f)
            return TaskConfig(**config_data)
        return TaskConfig()
    
    def _validate_config(self):
        """Validate configuration values."""
        try:
            self.config.validate()
        except ValidationError as e:
            self.logger.error(f"Invalid configuration: {str(e)}")
            raise
    
    def reload_config(self, config_path: str):
        """Reload configuration from file."""
        self.config = self._load_config(config_path)
        self._validate_config()
        self.logger.info("Configuration reloaded")
    
    async def schedule_task(self, task: Task) -> str:
        """Schedule a new task for execution.
        
        Args:
            task: Task to schedule
            
        Returns:
            Task ID
        """
        # Validate dependencies
        for dep_id in task.dependencies:
            if dep_id not in self.progress:
                raise ValueError(f"Unknown dependency: {dep_id}")
            if self.progress[dep_id].status != "completed":
                raise ValueError(f"Dependency {dep_id} not completed")
        
        # Store task
        self.progress[task.id] = task
        self.redis_client.hset("tasks", task.id, json.dumps(task.__dict__))
        
        # Broadcast update
        if self.config.enable_websockets:
            await self.websocket_manager.broadcast_task_update(task)
        
        return task.id
    
    async def run_task(self, task_id: str) -> Dict[str, Any]:
        """Run a scheduled task.
        
        Args:
            task_id: ID of task to run
            
        Returns:
            Task results
        """
        if task_id not in self.progress:
            raise ValueError(f"Unknown task: {task_id}")
        
        task = self.progress[task_id]
        task.status = "running"
        task.started_at = datetime.utcnow().isoformat()
        
        # Update Redis
        self.redis_client.hset("tasks", task_id, json.dumps(task.__dict__))
        
        # Broadcast update
        if self.config.enable_websockets:
            await self.websocket_manager.broadcast_task_update(task)
        
        try:
            # Get agent
            agent_id = task.agent_id or self.config.default_agent
            if agent_id not in self.agents:
                raise ValueError(f"Unknown agent: {agent_id}")
            
            # Run task
            result = await asyncio.wait_for(
                self.agents[agent_id].run_task(task),
                timeout=self.config.timeout_seconds
            )
            
            await self.complete_task(task_id, result)
            return result
            
        except Exception as e:
            await self.fail_task(task_id, str(e))
            raise
    
    async def complete_task(self, task_id: str, result: Dict[str, Any]):
        """Mark a task as completed.
        
        Args:
            task_id: ID of completed task
            result: Task results
        """
        task = self.progress[task_id]
        task.status = "completed"
        task.completed_at = datetime.utcnow().isoformat()
        task.result = result
        
        # Update Redis
        self.redis_client.hset("tasks", task_id, json.dumps(task.__dict__))
        
        # Broadcast update
        if self.config.enable_websockets:
            await self.websocket_manager.broadcast_task_update(task)
    
    async def fail_task(self, task_id: str, error: str):
        """Mark a task as failed.
        
        Args:
            task_id: ID of failed task
            error: Error message
        """
        task = self.progress[task_id]
        task.status = "failed"
        task.completed_at = datetime.utcnow().isoformat()
        task.error = error
        
        # Update Redis
        self.redis_client.hset("tasks", task_id, json.dumps(task.__dict__))
        
        # Broadcast update
        if self.config.enable_websockets:
            await self.websocket_manager.broadcast_task_update(task)
    
    async def simulate_task_run(self, task: Task) -> Dict[str, Any]:
        """Simulate running a task without actual execution.
        
        Args:
            task: Task to simulate
            
        Returns:
            Simulated results
        """
        self.logger.info(f"Simulating task: {task.id}")
        
        # Validate task
        if task.agent_id and task.agent_id not in self.agents:
            raise ValueError(f"Unknown agent: {task.agent_id}")
        
        # Simulate execution
        await asyncio.sleep(1)  # Simulate work
        
        return {
            "simulated": True,
            "task_id": task.id,
            "agent_id": task.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.post("/tasks")
        async def create_task(task: Task) -> Dict[str, str]:
            """Create a new task."""
            task_id = await self.schedule_task(task)
            return {"task_id": task_id}
        
        @self.app.post("/tasks/{task_id}/run")
        async def run_task_endpoint(task_id: str) -> Dict[str, Any]:
            """Run a scheduled task."""
            return await self.run_task(task_id)
        
        @self.app.get("/tasks/{task_id}")
        async def get_task(task_id: str) -> Dict[str, Any]:
            """Get task status."""
            if task_id not in self.progress:
                raise HTTPException(status_code=404, detail="Task not found")
            return self.progress[task_id].__dict__
        
        @self.app.post("/simulate/{task_id}")
        async def simulate_task(task_id: str) -> Dict[str, Any]:
            """Simulate running a task."""
            if task_id not in self.progress:
                raise HTTPException(status_code=404, detail="Task not found")
            return await self.simulate_task_run(self.progress[task_id])
        
        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            """WebSocket endpoint for real-time updates."""
            await self.websocket_manager.connect(websocket, client_id)
            try:
                while True:
                    data = await websocket.receive_json()
                    if data.get("type") == "pause":
                        # Handle pause command
                        pass
                    elif data.get("type") == "resume":
                        # Handle resume command
                        pass
            except websockets.exceptions.ConnectionClosed:
                await self.websocket_manager.disconnect(client_id)
    
    def cleanup(self):
        """Cleanup resources."""
        self.redis_client.close()
        for agent in self.agents.values():
            agent.cleanup() 