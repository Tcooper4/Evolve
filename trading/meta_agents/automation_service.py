"""
Automation Service

This module implements a service for managing automation tasks and workflows.

Note: This module was adapted from the legacy automation/services/automation_service.py file.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from trading.models import Task, Workflow, TaskStatus
from trading.task_manager import TaskManager
from trading.workflow_engine import WorkflowEngine
from trading.service_manager import ServiceManager

class AutomationService:
    """Service for managing automation tasks and workflows."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the automation service."""
        self.config = config
        self.task_manager = TaskManager(config)
        self.workflow_engine = WorkflowEngine(config)
        self.service_manager = ServiceManager(config)
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging for automation."""
        log_path = Path("logs/automation")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "automation_service.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the automation service."""
        try:
            # Initialize components
            await self.task_manager.start()
            await self.workflow_engine.start()
            await self.service_manager.start()
            
            self.logger.info("Automation service initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize automation service: {str(e)}")
            raise
    
    async def create_task(
        self,
        name: str,
        task_type: str,
        description: str = "",
        priority: int = 1,
        scheduled_for: Optional[datetime] = None,
        dependencies: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new task."""
        try:
            task_id = await self.task_manager.register_task(
                task_id=f"task_{int(datetime.now().timestamp()*1e6)}",
                name=name,
                handler=self._get_task_handler(task_type),
                schedule=None,  # TODO: Implement scheduling
                args=[parameters or {}],
                kwargs=metadata or {}
            )
            
            self.logger.info(f"Created task: {task_id}")
            return task_id
            
        except Exception as e:
            self.logger.error(f"Error creating task: {str(e)}")
            raise
    
    async def create_workflow(
        self,
        name: str,
        description: str = "",
        tasks: Optional[List[Task]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new workflow."""
        try:
            workflow = Workflow(
                id=f"workflow_{int(datetime.now().timestamp()*1e6)}",
                name=name,
                description=description,
                tasks=tasks or [],
                metadata=metadata or {}
            )
            
            await self.workflow_engine.register_workflow(workflow)
            self.logger.info(f"Created workflow: {workflow.id}")
            return workflow.id
            
        except Exception as e:
            self.logger.error(f"Error creating workflow: {str(e)}")
            raise
    
    async def execute_task(self, task_id: str) -> None:
        """Execute a task."""
        try:
            await self.task_manager.execute_task(task_id)
            self.logger.info(f"Executed task: {task_id}")
        except Exception as e:
            self.logger.error(f"Error executing task {task_id}: {str(e)}")
            raise
    
    async def execute_workflow(self, workflow_id: str) -> None:
        """Execute a workflow."""
        try:
            await self.workflow_engine.execute_workflow(workflow_id)
            self.logger.info(f"Executed workflow: {workflow_id}")
        except Exception as e:
            self.logger.error(f"Error executing workflow {workflow_id}: {str(e)}")
            raise
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a task."""
        try:
            return await self.task_manager.get_task_status(task_id)
        except Exception as e:
            self.logger.error(f"Error getting task status: {str(e)}")
            raise
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get the status of a workflow."""
        try:
            return await self.workflow_engine.get_workflow_status(workflow_id)
        except Exception as e:
            self.logger.error(f"Error getting workflow status: {str(e)}")
            raise
    
    def _get_task_handler(self, task_type: str) -> Any:
        """Get a task handler by type."""
        from trading.task_handlers import TaskHandlerFactory
        factory = TaskHandlerFactory(self.config)
        handler = factory.get_handler(task_type)
        if not handler:
            raise ValueError(f"Unknown task type: {task_type}")
        return handler.handle
    
    async def start(self) -> None:
        """Start the automation service."""
        try:
            await self.initialize()
            self.logger.info("Automation service started")
        except Exception as e:
            self.logger.error(f"Error starting automation service: {str(e)}")
            raise
    
    async def stop(self) -> None:
        """Stop the automation service."""
        try:
            await self.task_manager.stop()
            await self.workflow_engine.stop()
            await self.service_manager.stop()
            self.logger.info("Automation service stopped")
        except Exception as e:
            self.logger.error(f"Error stopping automation service: {str(e)}")
            raise

    def schedule(self):
        raise NotImplementedError('Pending feature') 