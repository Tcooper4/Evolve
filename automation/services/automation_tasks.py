import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
from pathlib import Path
import json
from pydantic import BaseModel, Field
from cachetools import TTLCache
from ratelimit import limits, sleep_and_retry

from ..core.models.task import Task, TaskStatus, TaskPriority, TaskType
from .automation_core import AutomationCore

logger = logging.getLogger(__name__)

class TaskConfig(BaseModel):
    """Configuration for task management."""
    max_concurrent_tasks: int = Field(default=10)
    task_timeout: int = Field(default=300)
    retry_delay: int = Field(default=5)
    max_retries: int = Field(default=3)
    cleanup_interval: int = Field(default=3600)

class AutomationTasks:
    """Task management functionality."""
    
    def __init__(self, core: AutomationCore, config_path: str = "automation/config/tasks.json"):
        """Initialize task management."""
        self.core = core
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.setup_cache()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_results: Dict[str, Any] = {}
        self.task_errors: Dict[str, str] = {}
        self.lock = asyncio.Lock()
        
    def _load_config(self, config_path: str) -> TaskConfig:
        """Load task configuration."""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            return TaskConfig(**config_data)
        except Exception as e:
            logger.error(f"Failed to load task config: {str(e)}")
            raise
            
    def setup_logging(self):
        """Configure logging."""
        log_path = Path("automation/logs")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "tasks.log"),
                logging.StreamHandler()
            ]
        )
        
    def setup_cache(self):
        """Setup task result caching."""
        self.cache = TTLCache(
            maxsize=1000,
            ttl=3600
        )
        
    @sleep_and_retry
    @limits(calls=100, period=60)
    async def schedule_task(
        self,
        name: str,
        description: str,
        task_type: TaskType,
        priority: TaskPriority,
        scheduled_for: Optional[datetime] = None,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Schedule a task for execution."""
        try:
            async with self.lock:
                # Check concurrent task limit
                if len(self.running_tasks) >= self.config.max_concurrent_tasks:
                    raise ValueError("Maximum concurrent tasks reached")
                    
                # Create task
                task_id = await self.core.create_task(
                    name=name,
                    description=description,
                    task_type=task_type,
                    priority=priority,
                    parameters=parameters,
                    metadata=metadata
                )
                
                # Schedule execution
                if scheduled_for:
                    delay = (scheduled_for - datetime.now()).total_seconds()
                    if delay > 0:
                        await asyncio.sleep(delay)
                        
                # Start execution
                self.running_tasks[task_id] = asyncio.create_task(
                    self._execute_task(task_id)
                )
                
                logger.info(f"Scheduled task {task_id}: {name}")
                return task_id
                
        except Exception as e:
            logger.error(f"Failed to schedule task: {str(e)}")
            raise
            
    async def _execute_task(self, task_id: str) -> None:
        """Execute a task with retry logic."""
        try:
            for attempt in range(self.config.max_retries):
                try:
                    success = await self.core.execute_task(task_id)
                    if success:
                        return
                except Exception as e:
                    if attempt == self.config.max_retries - 1:
                        raise
                    logger.warning(f"Task {task_id} attempt {attempt + 1} failed: {str(e)}")
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                    
        except Exception as e:
            logger.error(f"Task {task_id} failed: {str(e)}")
            self.task_errors[task_id] = str(e)
        finally:
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
                
    async def get_task_result(self, task_id: str) -> Optional[Any]:
        """Get task result with caching."""
        try:
            # Try cache first
            if task_id in self.cache:
                return self.cache[task_id]
                
            # Get from task manager
            task = await self.core.get_task(task_id)
            if task and task.result:
                self.cache[task_id] = task.result
                return task.result
            return None
            
        except Exception as e:
            logger.error(f"Failed to get task result {task_id}: {str(e)}")
            return None
            
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        try:
            async with self.lock:
                if task_id in self.running_tasks:
                    self.running_tasks[task_id].cancel()
                    del self.running_tasks[task_id]
                    await self.core.update_task(task_id, status=TaskStatus.CANCELLED)
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {str(e)}")
            return False
            
    async def cleanup_completed_tasks(self):
        """Cleanup completed tasks."""
        try:
            async with self.lock:
                for task_id, task in self.running_tasks.items():
                    if task.done():
                        del self.running_tasks[task_id]
                        if task_id in self.cache:
                            del self.cache[task_id]
                            
        except Exception as e:
            logger.error(f"Failed to cleanup tasks: {str(e)}")
            
    async def get_running_tasks(self) -> List[str]:
        """Get list of running task IDs."""
        return list(self.running_tasks.keys())
        
    async def get_task_error(self, task_id: str) -> Optional[str]:
        """Get task error message."""
        return self.task_errors.get(task_id)
        
    async def cleanup(self):
        """Cleanup resources."""
        try:
            # Cancel all running tasks
            for task_id in list(self.running_tasks.keys()):
                await self.cancel_task(task_id)
                
            # Clear caches
            self.cache.clear()
            self.task_results.clear()
            self.task_errors.clear()
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            raise 