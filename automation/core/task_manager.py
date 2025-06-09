import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import redis
import json
import heapq
from .models.task import Task, TaskStatus, TaskPriority, TaskType

logger = logging.getLogger(__name__)

class TaskManager:
    def __init__(self, redis_client: redis.Redis):
        """Initialize the task manager."""
        self.redis = redis_client
        self.tasks: Dict[str, Task] = {}
        self.task_queue = []  # Priority queue for tasks
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_results: Dict[str, Any] = {}
        self.task_errors: Dict[str, str] = {}
        self.logger = logging.getLogger(__name__)

    async def create_task(
        self,
        name: str,
        description: str,
        task_type: TaskType,
        priority: TaskPriority,
        handler: Optional[callable] = None,
        scheduled_for: Optional[datetime] = None,
        dependencies: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs
    ) -> str:
        """Create a new task."""
        try:
            # Validate dependencies
            if dependencies:
                for dep_id in dependencies:
                    if dep_id not in self.tasks:
                        raise ValueError(f"Dependency task {dep_id} not found")

            # Create task
            task = Task(
                name=name,
                description=description,
                type=task_type,
                priority=priority,
                scheduled_for=scheduled_for,
                dependencies=dependencies or [],
                parameters=parameters or {},
                metadata=metadata or {},
                handler=handler,
                handler_args=args,
                handler_kwargs=kwargs
            )

            # Store task
            self.tasks[task.id] = task
            await self._store_task(task)

            # Add to priority queue if not scheduled for future
            if not scheduled_for or scheduled_for <= datetime.now():
                self._add_to_queue(task)

            self.logger.info(f"Created task {task.id}: {name}")
            return task.id

        except Exception as e:
            self.logger.error(f"Error creating task: {str(e)}")
            raise

    async def _store_task(self, task: Task) -> None:
        """Store task in Redis."""
        try:
            # Store task data
            await self.redis.hset(
                "tasks",
                task.id,
                task.json()
            )

            # Add to type-specific list
            type_key = f"tasks:{task.type.value}"
            await self.redis.lpush(type_key, task.id)

            # Add to user's tasks if user_id is in metadata
            if task.metadata.get("user_id"):
                user_key = f"user_tasks:{task.metadata['user_id']}"
                await self.redis.lpush(user_key, task.id)

        except Exception as e:
            self.logger.error(f"Error storing task: {str(e)}")
            raise

    def _add_to_queue(self, task: Task) -> None:
        """Add task to priority queue."""
        # Priority is negative because heapq is a min-heap
        heapq.heappush(
            self.task_queue,
            (-task.priority.value, task.scheduled_for or datetime.now(), task.id)
        )

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        try:
            # Try to get from memory first
            if task_id in self.tasks:
                return self.tasks[task_id]

            # Get from Redis
            task_data = await self.redis.hget("tasks", task_id)
            if task_data:
                task = Task.parse_raw(task_data)
                self.tasks[task_id] = task
                return task

            return None

        except Exception as e:
            self.logger.error(f"Error getting task: {str(e)}")
            return None

    async def get_tasks(
        self,
        task_type: Optional[TaskType] = None,
        status: Optional[TaskStatus] = None,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Task]:
        """Get tasks with optional filtering."""
        try:
            # Determine which list to query
            if task_type:
                key = f"tasks:{task_type.value}"
            elif user_id:
                key = f"user_tasks:{user_id}"
            else:
                key = "tasks"

            # Get task IDs
            task_ids = await self.redis.lrange(key, offset, offset + limit - 1)
            
            # Get tasks
            tasks = []
            for task_id in task_ids:
                task = await self.get_task(task_id.decode())
                if task and (not status or task.status == status):
                    tasks.append(task)

            return tasks

        except Exception as e:
            self.logger.error(f"Error getting tasks: {str(e)}")
            return []

    async def update_task(
        self,
        task_id: str,
        status: Optional[TaskStatus] = None,
        priority: Optional[TaskPriority] = None,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Task]:
        """Update a task."""
        try:
            task = await self.get_task(task_id)
            if not task:
                return None

            # Update fields
            if status:
                task.update_status(status)
            if priority:
                task.priority = priority
            if parameters:
                task.parameters.update(parameters)
            if metadata:
                task.metadata.update(metadata)

            # Store updated task
            await self._store_task(task)

            # Update queue if priority changed
            if priority:
                self._add_to_queue(task)

            return task

        except Exception as e:
            self.logger.error(f"Error updating task: {str(e)}")
            return None

    async def delete_task(self, task_id: str) -> bool:
        """Delete a task."""
        try:
            task = await self.get_task(task_id)
            if not task:
                return False

            # Remove from Redis
            await self.redis.hdel("tasks", task_id)
            await self.redis.lrem(f"tasks:{task.type.value}", 0, task_id)

            # Remove from user's tasks if user_id is in metadata
            if task.metadata.get("user_id"):
                await self.redis.lrem(f"user_tasks:{task.metadata['user_id']}", 0, task_id)

            # Remove from memory
            self.tasks.pop(task_id, None)
            self.task_results.pop(task_id, None)
            self.task_errors.pop(task_id, None)

            return True

        except Exception as e:
            self.logger.error(f"Error deleting task: {str(e)}")
            return False

    async def execute_task(self, task_id: str) -> bool:
        """Execute a task."""
        try:
            task = await self.get_task(task_id)
            if not task or not task.handler:
                return False

            # Update status
            task.update_status(TaskStatus.RUNNING)
            await self._store_task(task)

            # Create task coroutine
            task_coro = asyncio.create_task(
                self._run_task(task),
                name=f"task_{task_id}"
            )

            # Store running task
            self.running_tasks[task_id] = task_coro

            return True

        except Exception as e:
            self.logger.error(f"Error executing task: {str(e)}")
            return False

    async def _run_task(self, task: Task) -> None:
        """Run a task with timeout."""
        try:
            # Run the task with timeout
            result = await asyncio.wait_for(
                task.handler(*task.handler_args, **task.handler_kwargs),
                timeout=task.timeout
            )

            # Store result
            task.result = result
            task.update_status(TaskStatus.COMPLETED)
            self.task_results[task.id] = result

            # Store updated task
            await self._store_task(task)

            self.logger.info(f"Task completed: {task.id}")

        except asyncio.TimeoutError:
            task.error = f"Task timed out after {task.timeout} seconds"
            task.update_status(TaskStatus.FAILED)
            await self._store_task(task)
            self.logger.error(f"Task timed out: {task.id}")

        except Exception as e:
            task.error = str(e)
            task.update_status(TaskStatus.FAILED)
            await self._store_task(task)
            self.logger.error(f"Task failed: {task.id} - {str(e)}")

            # Handle retries
            if task.retries < task.max_retries:
                task.retries += 1
                task.update_status(TaskStatus.PENDING)
                await self._store_task(task)
                self._add_to_queue(task)
                self.logger.info(f"Rescheduled task {task.id} (retry {task.retries})")

        finally:
            self.running_tasks.pop(task.id, None)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        try:
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                task.cancel()
                await self.update_task(task_id, status=TaskStatus.CANCELLED)
                self.logger.info(f"Cancelled task: {task_id}")
                return True
            return False

        except Exception as e:
            self.logger.error(f"Error cancelling task: {str(e)}")
            return False 