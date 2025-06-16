"""
Orchestrator

This module implements the orchestrator for managing and coordinating automation tasks.

Note: This module was adapted from the legacy automation/core/orchestrator.py file.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import json
import yaml
from .models import Task, Workflow, TaskStatus
from .task_manager import TaskManager
from .workflow_engine import WorkflowEngine
from .notification_service import NotificationService

class Orchestrator:
    """Orchestrator for managing and coordinating automation tasks."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize orchestrator."""
        self.config = config
        self.task_manager = TaskManager(config)
        self.workflow_engine = WorkflowEngine(config)
        self.notification_service = NotificationService(config)
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging for orchestrator."""
        log_path = Path("logs/orchestrator")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "orchestrator.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize orchestrator components."""
        try:
            # Initialize task manager
            await self.task_manager.initialize()
            
            # Initialize workflow engine
            await self.workflow_engine.initialize()
            
            # Initialize notification service
            await self.notification_service.initialize()
            
            self.logger.info("Orchestrator initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing orchestrator: {str(e)}")
            raise
    
    async def create_workflow(
        self,
        name: str,
        description: str,
        steps: List[Dict[str, Any]]
    ) -> str:
        """Create a new workflow."""
        try:
            workflow_id = await self.workflow_engine.create_workflow(
                name=name,
                description=description,
                steps=steps
            )
            
            self.logger.info(f"Created workflow: {workflow_id}")
            return workflow_id
        except Exception as e:
            self.logger.error(f"Error creating workflow: {str(e)}")
            raise
    
    async def execute_workflow(self, workflow_id: str) -> None:
        """Execute a workflow."""
        try:
            await self.workflow_engine.execute_workflow(workflow_id)
            self.logger.info(f"Executed workflow: {workflow_id}")
        except Exception as e:
            self.logger.error(f"Error executing workflow: {str(e)}")
            raise
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow status."""
        try:
            status = await self.workflow_engine.get_workflow_status(workflow_id)
            return status
        except Exception as e:
            self.logger.error(f"Error getting workflow status: {str(e)}")
            raise
    
    async def create_task(
        self,
        name: str,
        handler: str,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new task."""
        try:
            task_id = await self.task_manager.register_task(
                task_id=f"task_{int(datetime.now().timestamp()*1e6)}",
                name=name,
                handler=handler,
                args=args or [],
                kwargs=kwargs or {}
            )
            
            self.logger.info(f"Created task: {task_id}")
            return task_id
        except Exception as e:
            self.logger.error(f"Error creating task: {str(e)}")
            raise
    
    async def execute_task(self, task_id: str) -> None:
        """Execute a task."""
        try:
            await self.task_manager.execute_task(task_id)
            self.logger.info(f"Executed task: {task_id}")
        except Exception as e:
            self.logger.error(f"Error executing task: {str(e)}")
            raise
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status."""
        try:
            status = await self.task_manager.get_task_status(task_id)
            return status
        except Exception as e:
            self.logger.error(f"Error getting task status: {str(e)}")
            raise
    
    async def notify_status(
        self,
        message: str,
        channels: Optional[List[str]] = None
    ) -> None:
        """Send status notification."""
        try:
            await self.notification_service.send_notification(
                {"text": message},
                channels=channels
            )
            self.logger.info(f"Sent status notification: {message}")
        except Exception as e:
            self.logger.error(f"Error sending status notification: {str(e)}")
            raise
    
    async def start(self) -> None:
        """Start orchestrator."""
        try:
            await self.initialize()
            self.logger.info("Orchestrator started")
        except Exception as e:
            self.logger.error(f"Error starting orchestrator: {str(e)}")
            raise
    
    async def stop(self) -> None:
        """Stop orchestrator."""
        try:
            # TODO: Implement cleanup logic
            self.logger.info("Orchestrator stopped")
        except Exception as e:
            self.logger.error(f"Error stopping orchestrator: {str(e)}")
            raise 