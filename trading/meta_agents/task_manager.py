"""
# Adapted from automation/core/task_manager.py — legacy task management logic

TaskManager for managing, queuing, and executing agentic tasks in the trading system.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import heapq
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# --- Task Models ---
class TaskStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority:
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    URGENT = 5

class TaskType:
    DATA_COLLECTION = "data_collection"
    MODEL_TRAINING = "model_training"
    BACKTESTING = "backtesting"
    PORTFOLIO_UPDATE = "portfolio_update"
    RISK_ASSESSMENT = "risk_assessment"
    OPTIMIZATION = "optimization"
    EVALUATION = "evaluation"
    DEPLOYMENT = "deployment"

class Task(BaseModel):
    id: str = Field(default_factory=lambda: f"task_{int(datetime.now().timestamp()*1e6)}")
    name: str
    description: str = ""
    type: str
    status: str = TaskStatus.PENDING
    priority: int = TaskPriority.MEDIUM
    scheduled_for: Optional[datetime] = None
    dependencies: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    handler: Optional[Callable] = None
    handler_args: tuple = ()
    handler_kwargs: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = None

    def update_status(self, status: str):
        self.status = status
        self.updated_at = datetime.now()

class TaskManager:
    """Manages agentic tasks, their queue, and execution."""
    def __init__(self):
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
        task_type: str,
        priority: int = TaskPriority.MEDIUM,
        handler: Optional[Callable] = None,
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
            # Add to priority queue if not scheduled for future
            if not scheduled_for or scheduled_for <= datetime.now():
                self._add_to_queue(task)
            self.logger.info(f"Created task {task.id}: {name}")
            return task.id
        except Exception as e:
            self.logger.error(f"Error creating task: {str(e)}")
            raise

    def _add_to_queue(self, task: Task) -> None:
        """Add task to priority queue."""
        heapq.heappush(
            self.task_queue,
            (-task.priority, task.scheduled_for or datetime.now(), task.id)
        )

    async def get_task(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)

    async def get_tasks(
        self,
        task_type: Optional[str] = None,
        status: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Task]:
        tasks = list(self.tasks.values())
        if task_type:
            tasks = [t for t in tasks if t.type == task_type]
        if status:
            tasks = [t for t in tasks if t.status == status]
        if user_id:
            tasks = [t for t in tasks if t.metadata.get("user_id") == user_id]
        return tasks[offset:offset+limit]

    async def update_task(
        self,
        task_id: str,
        status: Optional[str] = None,
        priority: Optional[int] = None,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Task]:
        task = self.tasks.get(task_id)
        if not task:
            return None
        if status:
            task.update_status(status)
        if priority:
            task.priority = priority
        if parameters:
            task.parameters.update(parameters)
        if metadata:
            task.metadata.update(metadata)
        return task

    async def delete_task(self, task_id: str) -> bool:
        if task_id in self.tasks:
            self.tasks.pop(task_id)
            self.task_results.pop(task_id, None)
            self.task_errors.pop(task_id, None)
            return True
        return False

    async def execute_task(self, task_id: str) -> bool:
        task = self.tasks.get(task_id)
        if not task or not task.handler:
            return False
        task.update_status(TaskStatus.RUNNING)
        # Create task coroutine
        task_coro = asyncio.create_task(
            self._run_task(task),
            name=f"task_{task_id}"
        )
        self.running_tasks[task_id] = task_coro
        return True

    async def _run_task(self, task: Task) -> None:
        try:
            result = await task.handler(*task.handler_args, **task.handler_kwargs)
            task.update_status(TaskStatus.COMPLETED)
            self.task_results[task.id] = result
        except Exception as e:
            task.update_status(TaskStatus.FAILED)
            self.task_errors[task.id] = str(e)
            self.logger.error(f"Task {task.id} failed: {str(e)}")

    async def cancel_task(self, task_id: str) -> bool:
        coro = self.running_tasks.get(task_id)
        if coro:
            coro.cancel()
            self.tasks[task_id].update_status(TaskStatus.CANCELLED)
            return True
        return False 