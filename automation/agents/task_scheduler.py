import logging
from typing import Dict, List, Optional, Callable
import json
from pathlib import Path
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
import uuid
import heapq
from enum import Enum

class TaskPriority(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Task:
    id: str
    name: str
    description: str
    priority: TaskPriority
    status: TaskStatus
    created_at: str
    scheduled_for: str
    dependencies: List[str]
    handler: Callable
    args: tuple
    kwargs: dict
    result: Optional[Dict] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    timeout: int = 300  # 5 minutes

class TaskScheduler:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.tasks: Dict[str, Task] = {}
        self.task_queue: List[tuple] = []  # Priority queue
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_handlers: Dict[str, Callable] = {}
        self.task_results: Dict[str, Dict] = {}
        self.task_errors: Dict[str, str] = {}

    def setup_logging(self):
        """Configure logging for the task scheduler."""
        log_path = Path("automation/logs/task_scheduler")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "task_scheduler.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def register_handler(self, task_type: str, handler: Callable):
        """Register a handler for a task type."""
        self.task_handlers[task_type] = handler
        self.logger.info(f"Registered handler for task type: {task_type}")

    async def schedule_task(
        self,
        name: str,
        description: str,
        priority: TaskPriority,
        handler: Callable,
        scheduled_for: Optional[datetime] = None,
        dependencies: Optional[List[str]] = None,
        *args,
        **kwargs
    ) -> str:
        """
        Schedule a new task.
        
        Args:
            name: Task name
            description: Task description
            priority: Task priority
            handler: Task handler function
            scheduled_for: When to run the task
            dependencies: List of task IDs this task depends on
            *args: Arguments for the handler
            **kwargs: Keyword arguments for the handler
        
        Returns:
            str: Task ID
        """
        task_id = str(uuid.uuid4())
        
        if scheduled_for is None:
            scheduled_for = datetime.now()
        
        task = Task(
            id=task_id,
            name=name,
            description=description,
            priority=priority,
            status=TaskStatus.PENDING,
            created_at=datetime.now().isoformat(),
            scheduled_for=scheduled_for.isoformat(),
            dependencies=dependencies or [],
            handler=handler,
            args=args,
            kwargs=kwargs
        )
        
        self.tasks[task_id] = task
        self._add_to_queue(task)
        
        self.logger.info(f"Scheduled task: {task_id} ({name})")
        return task_id

    def _add_to_queue(self, task: Task):
        """Add a task to the priority queue."""
        # Priority is negative because heapq is a min-heap
        heapq.heappush(
            self.task_queue,
            (-task.priority.value, task.scheduled_for, task.id)
        )

    async def start(self):
        """Start the task scheduler."""
        self.logger.info("Starting task scheduler")
        
        while True:
            try:
                await self._process_queue()
                await asyncio.sleep(1)  # Check queue every second
            except Exception as e:
                self.logger.error(f"Error in task scheduler: {str(e)}")
                await asyncio.sleep(5)  # Wait longer on error

    async def _process_queue(self):
        """Process the task queue."""
        now = datetime.now()
        
        while self.task_queue:
            # Peek at the next task
            priority, scheduled_for, task_id = self.task_queue[0]
            scheduled_time = datetime.fromisoformat(scheduled_for)
            
            # Check if it's time to run the task
            if scheduled_time > now:
                break
            
            # Remove task from queue
            heapq.heappop(self.task_queue)
            
            # Get task details
            task = self.tasks[task_id]
            
            # Check dependencies
            if not await self._check_dependencies(task):
                # Re-add to queue if dependencies aren't met
                self._add_to_queue(task)
                continue
            
            # Run the task
            await self._run_task(task)

    async def _check_dependencies(self, task: Task) -> bool:
        """Check if all dependencies are met."""
        for dep_id in task.dependencies:
            if dep_id not in self.tasks:
                self.logger.error(f"Dependency not found: {dep_id}")
                return False
            
            dep_task = self.tasks[dep_id]
            if dep_task.status != TaskStatus.COMPLETED:
                return False
        
        return True

    async def _run_task(self, task: Task):
        """Run a task."""
        task.status = TaskStatus.RUNNING
        self.logger.info(f"Running task: {task.id} ({task.name})")
        
        try:
            # Create task coroutine
            task_coro = asyncio.create_task(
                self._execute_task(task),
                name=f"task_{task.id}"
            )
            
            # Store running task
            self.running_tasks[task.id] = task_coro
            
            # Wait for task to complete
            await task_coro
            
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            self.logger.info(f"Task cancelled: {task.id}")
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            self.logger.error(f"Task failed: {task.id} - {str(e)}")
            
            # Handle retries
            if task.retries < task.max_retries:
                task.retries += 1
                task.status = TaskStatus.PENDING
                self._add_to_queue(task)
                self.logger.info(f"Rescheduled task {task.id} (retry {task.retries})")
        finally:
            self.running_tasks.pop(task.id, None)

    async def _execute_task(self, task: Task):
        """Execute a task with timeout."""
        try:
            # Run the task with timeout
            result = await asyncio.wait_for(
                task.handler(*task.args, **task.kwargs),
                timeout=task.timeout
            )
            
            # Store result
            task.result = result
            task.status = TaskStatus.COMPLETED
            self.task_results[task.id] = result
            
            self.logger.info(f"Task completed: {task.id}")
            
        except asyncio.TimeoutError:
            raise Exception(f"Task timed out after {task.timeout} seconds")
        except Exception as e:
            self.task_errors[task.id] = str(e)
            raise

    async def cancel_task(self, task_id: str):
        """Cancel a running task."""
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            task.cancel()
            self.logger.info(f"Cancelled task: {task_id}")

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get status of a task."""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        return {
            'id': task.id,
            'name': task.name,
            'status': task.status.value,
            'created_at': task.created_at,
            'scheduled_for': task.scheduled_for,
            'result': task.result,
            'error': task.error,
            'retries': task.retries
        }

    def get_all_tasks(self) -> List[Dict]:
        """Get status of all tasks."""
        return [self.get_task_status(task_id) for task_id in self.tasks]

    def clear_completed_tasks(self):
        """Clear completed tasks from memory."""
        completed_ids = [
            task_id for task_id, task in self.tasks.items()
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
        ]
        
        for task_id in completed_ids:
            self.tasks.pop(task_id)
            self.task_results.pop(task_id, None)
            self.task_errors.pop(task_id, None)
        
        self.logger.info(f"Cleared {len(completed_ids)} completed tasks") 