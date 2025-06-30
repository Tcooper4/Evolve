"""
Task Agent

This module implements a specialized agent for managing and executing tasks
in the trading system. It handles task scheduling, execution, monitoring,
and error recovery.

Note: This module was adapted from the legacy automation/core/task_manager.py file.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from trading.base_agent import BaseAgent
from trading.task_manager import TaskManager
from trading.task_handlers import TaskHandlerFactory

class TaskAgent(BaseAgent):
    """Agent responsible for managing and executing tasks."""
    
    def __init__(self, config: Dict):
        """Initialize the task agent."""
        super().__init__(config)
        self.task_manager = TaskManager(config)
        self.setup_logging()
        
            return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def setup_logging(self):
        """Configure logging for task management."""
        log_path = Path("logs/tasks")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "task_agent.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    async def initialize(self) -> None:
        """Initialize the task agent."""
        try:
            await self.task_manager.initialize()
            self.logger.info("Task agent initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize task agent: {str(e)}")
            raise
    
    async def create_task(self, task_type: str, parameters: Dict[str, Any]) -> str:
        """Create a new task."""
        try:
            task_id = await self.task_manager.create_task(task_type, parameters)
            self.logger.info(f"Created task: {task_id}")
            return task_id
        except Exception as e:
            self.logger.error(f"Error creating task: {str(e)}")
            raise
    
    async def execute_task(self, task_id: str) -> Dict[str, Any]:
        """Execute a task."""
        try:
            result = await self.task_manager.execute_task(task_id)
            self.logger.info(f"Executed task: {task_id}")
            return result
        except Exception as e:
            self.logger.error(f"Error executing task: {str(e)}")
            raise
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a task."""
        try:
            status = await self.task_manager.get_task_status(task_id)
            return status
        except Exception as e:
            self.logger.error(f"Error getting task status: {str(e)}")
            raise
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        try:
            success = await self.task_manager.cancel_task(task_id)
            if success:
                self.logger.info(f"Cancelled task: {task_id}")
            return success
        except Exception as e:
            self.logger.error(f"Error cancelling task: {str(e)}")
            raise
    
    async def retry_task(self, task_id: str) -> bool:
        """Retry a failed task."""
        try:
            success = await self.task_manager.retry_task(task_id)
            if success:
                self.logger.info(f"Retrying task: {task_id}")
            return success
        except Exception as e:
            self.logger.error(f"Error retrying task: {str(e)}")
            raise
    
    async def get_task_history(
        self,
        task_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get task execution history."""
        try:
            history = await self.task_manager.get_task_history(
                task_type=task_type,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            return history
        except Exception as e:
            self.logger.error(f"Error getting task history: {str(e)}")
            raise
    
    async def monitor_tasks(self):
        """Monitor task execution and handle failures."""
        try:
            while True:
                # Get all active tasks
                active_tasks = await self.task_manager.get_active_tasks()
                
                for task in active_tasks:
                    # Check for stuck tasks
                    if task.get("status") == "running" and (
                        datetime.now() - datetime.fromisoformat(task["start_time"])
                    ).total_seconds() > self.config.get("task_timeout", 3600):
                        self.logger.warning(f"Task {task['id']} appears to be stuck")
                        await self.cancel_task(task["id"])
                        await self.retry_task(task["id"])
                    
                    # Check for failed tasks
                    if task.get("status") == "failed":
                        self.logger.warning(f"Task {task['id']} failed")
                        if task.get("retry_count", 0) < self.config.get("max_retries", 3):
                            await self.retry_task(task["id"])
                
                await asyncio.sleep(self.config.get("monitor_interval", 60))
                
        except Exception as e:
            self.logger.error(f"Error monitoring tasks: {str(e)}")
            raise
    
    async def start(self) -> None:
        """Start the task agent."""
        try:
            await self.initialize()
            self.logger.info("Task agent started")
            
            # Start task monitoring
            asyncio.create_task(self.monitor_tasks())
            
        except Exception as e:
            self.logger.error(f"Error starting task agent: {str(e)}")
            raise
    
    async def stop(self) -> None:
        """Stop the task agent."""
        try:
            await self.task_manager.cleanup()
            self.logger.info("Task agent stopped")
        except Exception as e:
            self.logger.error(f"Error stopping task agent: {str(e)}")
            raise 