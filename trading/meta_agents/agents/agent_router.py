"""
Agent Router

This orchestrator routes tasks across the system's core agents:
- ModelBuilder: Handles model creation and updates
- PerformanceChecker: Monitors model performance and drift
- SelfRepair: Maintains system health and fixes issues

The router processes natural language prompts and internal events to determine
which agents to activate and how to coordinate their actions.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime
import yaml
from pathlib import Path
import re
from enum import Enum

from .model_builder import ModelBuilder
from .performance_checker import PerformanceChecker
from .self_repair import SelfRepairAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Priority levels for tasks"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class Task:
    """Container for task information"""
    task_type: str
    payload: Dict
    priority: TaskPriority
    timestamp: str
    source: str
    status: str = "pending"
    result: Optional[Dict] = None

class AgentRouter:
    def __init__(self, config_path: str = "config/agent_router_config.json"):
        """
        Initialize the AgentRouter.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        
        # Initialize agents
        self.model_agent = ModelBuilder()
        self.eval_agent = PerformanceChecker()
        self.repair_agent = SelfRepairAgent()
        
        # Initialize task queue
        self.task_queue: List[Task] = []
        self.running = False
        
        # Load task patterns
        self.task_patterns = self._load_task_patterns()
        
        logger.info("Initialized AgentRouter")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {}

    def _load_task_patterns(self) -> Dict[str, Dict]:
        """Load patterns for task detection"""
        return {
            "new_data": {
                "patterns": [
                    r"new data available",
                    r"data update",
                    r"new market data"
                ],
                "priority": TaskPriority.MEDIUM
            },
            "check_performance": {
                "patterns": [
                    r"check performance",
                    r"monitor drift",
                    r"evaluate model"
                ],
                "priority": TaskPriority.HIGH
            },
            "repair": {
                "patterns": [
                    r"fix issue",
                    r"repair system",
                    r"maintenance needed"
                ],
                "priority": TaskPriority.CRITICAL
            }
        }

    async def process_prompt(self, prompt: str) -> Task:
        """
        Process a natural language prompt and create a task.
        
        Args:
            prompt: The natural language prompt to process
            
        Returns:
            Task object representing the detected task
        """
        try:
            # Match prompt against patterns
            for task_type, pattern_info in self.task_patterns.items():
                for pattern in pattern_info["patterns"]:
                    if re.search(pattern, prompt, re.IGNORECASE):
                        return Task(
                            task_type=task_type,
                            payload={"prompt": prompt},
                            priority=pattern_info["priority"],
                            timestamp=datetime.now().isoformat(),
                            source="prompt"
                        )
            
            # If no pattern matches, create a generic task
            return Task(
                task_type="unknown",
                payload={"prompt": prompt},
                priority=TaskPriority.LOW,
                timestamp=datetime.now().isoformat(),
                source="prompt"
            )
            
        except Exception as e:
            logger.error(f"Error processing prompt: {e}")
            raise

    async def route(self, task: Task) -> Dict:
        """
        Route a task to the appropriate agent.
        
        Args:
            task: The task to route
            
        Returns:
            Dictionary containing the task result
        """
        try:
            if task.task_type == "new_data":
                result = await self._handle_new_data(task)
            elif task.task_type == "check_performance":
                result = await self._handle_performance_check(task)
            elif task.task_type == "repair":
                result = await self._handle_repair(task)
            else:
                result = {"status": "error", "message": f"Unknown task type: {task.task_type}"}
            
            task.result = result
            task.status = "completed"
            return result
            
        except Exception as e:
            logger.error(f"Error routing task: {e}")
            task.status = "failed"
            task.result = {"status": "error", "message": str(e)}
            return task.result

    async def _handle_new_data(self, task: Task) -> Dict:
        """Handle new data tasks"""
        try:
            # Detect data changes
            changes = await self.model_agent.detect_data_change()
            
            if changes:
                # Update models if needed
                update_result = await self.model_agent.update_models(changes)
                return {
                    "status": "success",
                    "changes_detected": changes,
                    "models_updated": update_result
                }
            else:
                return {
                    "status": "success",
                    "changes_detected": False
                }
                
        except Exception as e:
            logger.error(f"Error handling new data: {e}")
            raise

    async def _handle_performance_check(self, task: Task) -> Dict:
        """Handle performance check tasks"""
        try:
            # Check for drift
            drift_result = await self.eval_agent.detect_drift(task.payload)
            
            if drift_result.get("drift_detected"):
                # Trigger model update if needed
                await self.model_agent.update_models(drift_result)
            
            return {
                "status": "success",
                "drift_detected": drift_result.get("drift_detected", False),
                "drift_details": drift_result
            }
            
        except Exception as e:
            logger.error(f"Error handling performance check: {e}")
            raise

    async def _handle_repair(self, task: Task) -> Dict:
        """Handle repair tasks"""
        try:
            # Scan for issues
            issues = await self.repair_agent.scan_for_issues()
            
            if issues:
                # Apply fixes
                fixed_issues = await self.repair_agent.apply_patches(issues)
                
                # Log repairs
                for issue in fixed_issues:
                    self.repair_agent.log_repair(issue)
                
                return {
                    "status": "success",
                    "issues_found": len(issues),
                    "issues_fixed": len(fixed_issues)
                }
            else:
                return {
                    "status": "success",
                    "issues_found": 0
                }
                
        except Exception as e:
            logger.error(f"Error handling repair: {e}")
            raise

    async def add_task(self, task: Task) -> None:
        """
        Add a task to the queue.
        
        Args:
            task: The task to add
        """
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda x: x.priority.value, reverse=True)
        logger.info(f"Added task to queue: {task.task_type}")

    async def process_queue(self) -> None:
        """Process the task queue"""
        while self.running and self.task_queue:
            task = self.task_queue.pop(0)
            try:
                await self.route(task)
            except Exception as e:
                logger.error(f"Error processing task {task.task_type}: {e}")
                task.status = "failed"
                task.result = {"status": "error", "message": str(e)}

    async def start(self) -> None:
        """Start the router"""
        self.running = True
        logger.info("Starting AgentRouter")
        
        while self.running:
            await self.process_queue()
            await asyncio.sleep(1)  # Prevent busy waiting

    async def stop(self) -> None:
        """Stop the router"""
        self.running = False
        logger.info("Stopping AgentRouter")

    async def get_task_status(self, task_id: str) -> Dict:
        """
        Get the status of a task.
        
        Args:
            task_id: The ID of the task to check
            
        Returns:
            Dictionary containing task status
        """
        for task in self.task_queue:
            if task.task_type == task_id:
                return {
                    "status": task.status,
                    "result": task.result
                }
        return {"status": "not_found"}

    async def get_queue_status(self) -> Dict:
        """
        Get the status of the task queue.
        
        Returns:
            Dictionary containing queue status
        """
        return {
            "queue_length": len(self.task_queue),
            "tasks": [
                {
                    "type": task.task_type,
                    "priority": task.priority.name,
                    "status": task.status
                }
                for task in self.task_queue
            ]
        }

if __name__ == "__main__":
    router = AgentRouter()
    asyncio.run(router.start()) 