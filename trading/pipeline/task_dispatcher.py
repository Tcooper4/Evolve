"""
Enhanced Task Dispatcher with Redis Failover

This module provides robust task dispatching with:
- Task ID registry to prevent duplicate enqueuing
- Comprehensive try/except error handling
- Redis failover to local in-memory task queue
- Task prioritization and retry logic
- Performance monitoring and metrics
"""

import asyncio
import logging
import time
import uuid
import threading
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
from collections import defaultdict, deque
import weakref

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class TaskStatus(Enum):
    """Task status states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

@dataclass
class Task:
    """Task definition with metadata."""
    id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[float] = None
    result: Any = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class TaskRegistry:
    """Thread-safe task registry to prevent duplicates."""
    
    def __init__(self):
        self._tasks: Dict[str, Task] = {}
        self._lock = threading.RLock()
        self._task_signatures: Dict[str, List[str]] = defaultdict(list)
    
    def register_task(self, task: Task) -> bool:
        """Register a task, return False if duplicate."""
        with self._lock:
            # Check for exact duplicate
            if task.id in self._tasks:
                logger.warning(f"⚠️ Task {task.id} already registered")
                return False
            
            # Check for functional duplicates (same func + args + kwargs)
            signature = self._create_signature(task)
            if signature in self._task_signatures:
                existing_ids = self._task_signatures[signature]
                # Check if any existing task is still pending
                for existing_id in existing_ids:
                    if existing_id in self._tasks:
                        existing_task = self._tasks[existing_id]
                        if existing_task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                            logger.warning(f"⚠️ Duplicate task detected: {task.id} matches {existing_id}")
                            return False
            
            # Register the task
            self._tasks[task.id] = task
            self._task_signatures[signature].append(task.id)
            logger.info(f"✅ Registered task {task.id}")
            return True
    
    def unregister_task(self, task_id: str) -> bool:
        """Unregister a task."""
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                signature = self._create_signature(task)
                
                # Remove from both registries
                del self._tasks[task_id]
                if signature in self._task_signatures:
                    self._task_signatures[signature].remove(task_id)
                    if not self._task_signatures[signature]:
                        del self._task_signatures[signature]
                
                logger.info(f"✅ Unregistered task {task_id}")
                return True
            return False
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        with self._lock:
            return self._tasks.get(task_id)
    
    def get_pending_tasks(self) -> List[Task]:
        """Get all pending tasks."""
        with self._lock:
            return [task for task in self._tasks.values() if task.status == TaskStatus.PENDING]
    
    def cleanup_completed_tasks(self, max_age: float = 3600) -> int:
        """Clean up completed tasks older than max_age seconds."""
        with self._lock:
            current_time = time.time()
            to_remove = []
            
            for task_id, task in self._tasks.items():
                if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                    task.completed_at and (current_time - task.completed_at) > max_age):
                    to_remove.append(task_id)
            
            for task_id in to_remove:
                self.unregister_task(task_id)
            
            logger.info(f"🧹 Cleaned up {len(to_remove)} completed tasks")
            return len(to_remove)
    
    def _create_signature(self, task: Task) -> str:
        """Create a signature for duplicate detection."""
        # Create a hash of function name and arguments
        func_name = getattr(task.func, '__name__', str(task.func))
        args_str = str(task.args)
        kwargs_str = str(sorted(task.kwargs.items()))
        return f"{func_name}:{args_str}:{kwargs_str}"

class LocalTaskQueue:
    """Local in-memory task queue with priority support."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._queues: Dict[TaskPriority, deque] = {
            priority: deque() for priority in TaskPriority
        }
        self._lock = threading.RLock()
        self._size = 0
    
    def put(self, task: Task) -> bool:
        """Add task to queue, return False if full."""
        with self._lock:
            if self._size >= self.max_size:
                logger.warning(f"⚠️ Task queue full ({self._size}/{self.max_size})")
                return False
            
            self._queues[task.priority].append(task)
            self._size += 1
            logger.debug(f"📥 Added task {task.id} to queue (priority: {task.priority.name})")
            return True
    
    def get(self) -> Optional[Task]:
        """Get next task by priority."""
        with self._lock:
            # Get highest priority task
            for priority in reversed(list(TaskPriority)):
                if self._queues[priority]:
                    task = self._queues[priority].popleft()
                    self._size -= 1
                    logger.debug(f"📤 Retrieved task {task.id} from queue (priority: {priority.name})")
                    return task
            return None
    
    def size(self) -> int:
        """Get current queue size."""
        with self._lock:
            return self._size
    
    def clear(self) -> int:
        """Clear all tasks, return number of cleared tasks."""
        with self._lock:
            total_cleared = self._size
            for queue in self._queues.values():
                queue.clear()
            self._size = 0
            logger.info(f"🧹 Cleared {total_cleared} tasks from queue")
            return total_cleared

class RedisTaskQueue:
    """Redis-based task queue with failover support."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", queue_name: str = "task_queue"):
        self.redis_url = redis_url
        self.queue_name = queue_name
        self.redis_client = None
        self._connected = False
        self._connect()
    
    def _connect(self) -> bool:
        """Connect to Redis."""
        if not REDIS_AVAILABLE:
            logger.warning("⚠️ Redis not available, using local queue only")
            return False
        
        try:
            self.redis_client = redis.from_url(self.redis_url)
            # Test connection
            self.redis_client.ping()
            self._connected = True
            logger.info("✅ Connected to Redis")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            self._connected = False
            return False
    
    def put(self, task: Task) -> bool:
        """Add task to Redis queue."""
        if not self._connected and not self._connect():
            return False
        
        try:
            task_data = pickle.dumps(task)
            self.redis_client.lpush(self.queue_name, task_data)
            logger.debug(f"📥 Added task {task.id} to Redis queue")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to add task to Redis: {e}")
            self._connected = False
            return False
    
    def get(self) -> Optional[Task]:
        """Get task from Redis queue."""
        if not self._connected and not self._connect():
            return None
        
        try:
            task_data = self.redis_client.rpop(self.queue_name)
            if task_data:
                task = pickle.loads(task_data)
                logger.debug(f"📤 Retrieved task {task.id} from Redis queue")
                return task
            return None
        except Exception as e:
            logger.error(f"❌ Failed to get task from Redis: {e}")
            self._connected = False
            return None
    
    def size(self) -> int:
        """Get Redis queue size."""
        if not self._connected and not self._connect():
            return 0
        
        try:
            return self.redis_client.llen(self.queue_name)
        except Exception as e:
            logger.error(f"❌ Failed to get Redis queue size: {e}")
            self._connected = False
            return 0

class TaskDispatcher:
    """Enhanced task dispatcher with Redis failover."""
    
    def __init__(
        self,
        max_workers: int = 10,
        redis_url: Optional[str] = None,
        enable_redis: bool = True
    ):
        self.max_workers = max_workers
        self.registry = TaskRegistry()
        self.local_queue = LocalTaskQueue()
        self.redis_queue = RedisTaskQueue(redis_url) if enable_redis and redis_url else None
        self.workers: List[asyncio.Task] = []
        self.running = False
        self._lock = asyncio.Lock()
        self._metrics = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'tasks_retried': 0,
            'redis_failovers': 0
        }
    
    async def start(self) -> None:
        """Start the task dispatcher."""
        async with self._lock:
            if self.running:
                logger.warning("⚠️ Task dispatcher already running")
                return
            
            self.running = True
            # Start worker tasks
            for i in range(self.max_workers):
                worker = asyncio.create_task(self._worker(f"worker-{i}"))
                self.workers.append(worker)
            
            logger.info(f"✅ Started task dispatcher with {self.max_workers} workers")
    
    async def stop(self) -> None:
        """Stop the task dispatcher."""
        async with self._lock:
            if not self.running:
                return
            
            self.running = False
            
            # Cancel all workers
            for worker in self.workers:
                worker.cancel()
            
            # Wait for workers to finish
            if self.workers:
                await asyncio.gather(*self.workers, return_exceptions=True)
            
            self.workers.clear()
            logger.info("✅ Stopped task dispatcher")
    
    async def submit_task(
        self,
        func: Callable,
        *args,
        task_id: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 3,
        timeout: Optional[float] = None,
        **kwargs
    ) -> str:
        """Submit a task for execution."""
        task_id = task_id or str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries,
            timeout=timeout
        )
        
        # Register task
        if not self.registry.register_task(task):
            raise ValueError(f"Task {task_id} is a duplicate")
        
        # Try to enqueue in Redis first, fallback to local
        enqueued = False
        if self.redis_queue:
            enqueued = self.redis_queue.put(task)
            if not enqueued:
                self._metrics['redis_failovers'] += 1
                logger.warning(f"⚠️ Redis failed, falling back to local queue for task {task_id}")
        
        if not enqueued:
            enqueued = self.local_queue.put(task)
            if not enqueued:
                self.registry.unregister_task(task_id)
                raise RuntimeError("Task queue is full")
        
        self._metrics['tasks_submitted'] += 1
        logger.info(f"📋 Submitted task {task_id} (priority: {priority.name})")
        return task_id
    
    async def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get the result of a completed task."""
        start_time = time.time()
        
        while True:
            task = self.registry.get_task(task_id)
            if not task:
                raise ValueError(f"Task {task_id} not found")
            
            if task.status == TaskStatus.COMPLETED:
                return task.result
            
            if task.status == TaskStatus.FAILED:
                raise task.error or RuntimeError(f"Task {task_id} failed")
            
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Timeout waiting for task {task_id}")
            
            await asyncio.sleep(0.1)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        task = self.registry.get_task(task_id)
        if not task:
            return False
        
        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            self.registry.unregister_task(task_id)
            logger.info(f"❌ Cancelled task {task_id}")
            return True
        
        return False
    
    async def _worker(self, worker_name: str) -> None:
        """Worker task that processes queued tasks."""
        logger.info(f"🔄 Started worker {worker_name}")
        
        while self.running:
            try:
                # Try to get task from Redis first, then local queue
                task = None
                if self.redis_queue:
                    task = self.redis_queue.get()
                
                if not task:
                    task = self.local_queue.get()
                
                if not task:
                    await asyncio.sleep(0.1)
                    continue
                
                # Execute task
                await self._execute_task(task, worker_name)
                
            except asyncio.CancelledError:
                logger.info(f"🛑 Worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Worker {worker_name} error: {e}")
                await asyncio.sleep(1)
        
        logger.info(f"🛑 Worker {worker_name} stopped")
    
    async def _execute_task(self, task: Task, worker_name: str) -> None:
        """Execute a single task with comprehensive error handling."""
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        
        logger.info(f"🚀 Worker {worker_name} executing task {task.id}")
        
        try:
            # Execute the task
            if asyncio.iscoroutinefunction(task.func):
                result = await asyncio.wait_for(
                    task.func(*task.args, **task.kwargs),
                    timeout=task.timeout
                )
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, task.func, *task.args, **task.kwargs),
                    timeout=task.timeout
                )
            
            # Task completed successfully
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = time.time()
            self._metrics['tasks_completed'] += 1
            
            logger.info(f"✅ Task {task.id} completed successfully")
            
        except asyncio.TimeoutError:
            task.error = TimeoutError(f"Task {task.id} timed out")
            await self._handle_task_failure(task, worker_name)
            
        except Exception as e:
            task.error = e
            await self._handle_task_failure(task, worker_name)
        
        finally:
            # Cleanup
            self.registry.unregister_task(task.id)
    
    async def _handle_task_failure(self, task: Task, worker_name: str) -> None:
        """Handle task failure with retry logic."""
        task.retry_count += 1
        
        if task.retry_count <= task.max_retries:
            task.status = TaskStatus.RETRYING
            self._metrics['tasks_retried'] += 1
            
            logger.warning(f"🔄 Retrying task {task.id} (attempt {task.retry_count}/{task.max_retries})")
            
            # Re-queue for retry with exponential backoff
            await asyncio.sleep(2 ** task.retry_count)
            
            # Reset task for retry
            task.status = TaskStatus.PENDING
            task.started_at = None
            task.completed_at = None
            task.error = None
            
            # Re-queue
            if self.redis_queue and self.redis_queue.put(task):
                pass
            else:
                self.local_queue.put(task)
        else:
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
            self._metrics['tasks_failed'] += 1
            
            logger.error(f"❌ Task {task.id} failed permanently after {task.max_retries} retries")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return {
            **self._metrics,
            'queue_size': self.local_queue.size() + (self.redis_queue.size() if self.redis_queue else 0),
            'active_workers': len(self.workers),
            'pending_tasks': len(self.registry.get_pending_tasks())
        }
    
    async def cleanup(self) -> None:
        """Clean up old completed tasks."""
        cleaned = self.registry.cleanup_completed_tasks()
        logger.info(f"🧹 Cleaned up {cleaned} old tasks")

# Global dispatcher instance
_dispatcher: Optional[TaskDispatcher] = None

def get_dispatcher() -> TaskDispatcher:
    """Get the global task dispatcher instance."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = TaskDispatcher()
    return _dispatcher

async def submit_task(func: Callable, *args, **kwargs) -> str:
    """Submit a task using the global dispatcher."""
    dispatcher = get_dispatcher()
    return await dispatcher.submit_task(func, *args, **kwargs)

async def get_task_result(task_id: str, timeout: Optional[float] = None) -> Any:
    """Get task result using the global dispatcher."""
    dispatcher = get_dispatcher()
    return await dispatcher.get_task_result(task_id, timeout) 