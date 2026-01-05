import asyncio
import logging
from typing import Any, Dict, Optional

import redis
from pydantic import BaseModel, Field
from ratelimit import limits, sleep_and_retry

# Try to import from system.infra.agents.core, fallback if not available
try:
    from system.infra.agents.core.models.task import (
        Task,
        TaskPriority,
        TaskStatus,
        TaskType,
    )
    from system.infra.agents.core.orchestrator import Orchestrator
    from system.infra.agents.core.task_manager import TaskManager
    CORE_AVAILABLE = True
except ImportError:
    # Create dummy classes if core modules not available
    class Task:
        pass
    class TaskPriority:
        pass
    class TaskStatus:
        pass
    class TaskType:
        pass
    class Orchestrator:
        pass
    class TaskManager:
        pass
    CORE_AVAILABLE = False
from utils.launch_utils import setup_logging

logger = logging.getLogger(__name__)


class AutomationConfig(BaseModel):
    """Configuration for automation service."""

    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    cache_size: int = Field(default=1000)
    cache_ttl: int = Field(default=3600)
    rate_limit_calls: int = Field(default=100)
    rate_limit_period: int = Field(default=60)
    max_retries: int = Field(default=3)
    timeout: int = Field(default=300)


class AutomationCoreService:
    def __init__(self, config: Optional[AutomationConfig] = None):
        """
        Initialize AutomationCoreService.
        
        Args:
            config: Optional AutomationConfig instance. If None, uses default config.
        """
        self.config = config or AutomationConfig()
        self.setup_logging()
        self.logger = logging.getLogger("automation")
        
        # Initialize components
        self.redis = None
        self.task_manager = None
        self.orchestrator = None
        self.cache = {}
        self.lock = asyncio.Lock()

    def setup_logging(self):
        return setup_logging(service_name="service")

    def setup_cache(self):
        """Set up cache for automation core service."""
        # Cache setup logic here

    def setup_redis(self):
        """Setup Redis connection."""
        self.redis = redis.Redis(
            host=self.config.redis_host,
            port=self.config.redis_port,
            db=self.config.redis_db,
            decode_responses=True,
        )

    def setup_task_manager(self):
        """Setup task manager."""
        self.task_manager = TaskManager(self.redis)

    def setup_orchestrator(self):
        """Setup orchestrator."""
        self.orchestrator = Orchestrator()

    @sleep_and_retry
    @limits(calls=100, period=60)
    async def create_task(
        self,
        name: str,
        description: str,
        task_type: TaskType,
        priority: TaskPriority,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new task with rate limiting."""
        try:
            async with self.lock:
                task_id = await self.task_manager.create_task(
                    name=name,
                    description=description,
                    task_type=task_type,
                    priority=priority,
                    parameters=parameters,
                    metadata=metadata,
                )
                logger.info(f"Created task {task_id}: {name}")
                return task_id
        except Exception as e:
            logger.error(f"Failed to create task: {str(e)}")
            raise

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID with caching."""
        try:
            # Try cache first
            if task_id in self.cache:
                return self.cache[task_id]

            # Get from task manager
            task = await self.task_manager.get_task(task_id)
            if task:
                self.cache[task_id] = task
            return task
        except Exception as e:
            logger.error(f"Failed to get task {task_id}: {str(e)}")
            return None

    async def update_task(
        self,
        task_id: str,
        status: Optional[TaskStatus] = None,
        priority: Optional[TaskPriority] = None,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Task]:
        """Update task with validation."""
        try:
            async with self.lock:
                task = await self.task_manager.update_task(
                    task_id=task_id,
                    status=status,
                    priority=priority,
                    parameters=parameters,
                    metadata=metadata,
                )
                if task:
                    # Update cache
                    self.cache[task_id] = task
                return task
        except Exception as e:
            logger.error(f"Failed to update task {task_id}: {str(e)}")
            return None

    async def execute_task(self, task_id: str) -> bool:
        """Execute task with retry logic."""
        try:
            task = await self.get_task(task_id)
            if not task:
                return False

            for attempt in range(self.config.max_retries):
                try:
                    success = await self.task_manager.execute_task(task_id)
                    if success:
                        return True
                except Exception as e:
                    if attempt == self.config.max_retries - 1:
                        raise
                    logger.warning(
                        f"Task {task_id} attempt {attempt + 1} failed: {str(e)}"
                    )
                    await asyncio.sleep(2**attempt)  # Exponential backoff

            return False
        except Exception as e:
            logger.error(f"Failed to execute task {task_id}: {str(e)}")
            return False

    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get task status with caching."""
        try:
            task = await self.get_task(task_id)
            return task.status if task else None
        except Exception as e:
            logger.error(f"Failed to get task status {task_id}: {str(e)}")
            return None

    async def cleanup(self):
        """Cleanup resources."""
        try:
            await self.task_manager.cleanup()
            await self.orchestrator.stop()
            self.cache.clear()
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            raise
