"""
Scheduler module for managing periodic tasks and updates.

This module provides a comprehensive task scheduling system for the trading platform,
supporting various scheduling patterns, task prioritization, and error handling.
"""

import time
import threading
import logging
import asyncio
import json
from typing import Callable, Optional, Dict, Any, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import schedule
import queue
import signal
import sys

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ScheduleType(Enum):
    """Types of scheduling patterns."""
    INTERVAL = "interval"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CRON = "cron"

@dataclass
class ScheduledTask:
    """Represents a scheduled task."""
    id: str
    name: str
    func: Callable
    schedule_type: ScheduleType
    schedule_config: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    enabled: bool = True
    max_retries: int = 3
    retry_delay: int = 60  # seconds
    timeout: Optional[int] = None  # seconds
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    error_count: int = 0
    success_count: int = 0
    total_runtime: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['func'] = self.func.__name__ if hasattr(self.func, '__name__') else str(self.func)
        result['last_run'] = self.last_run.isoformat() if self.last_run else None
        result['next_run'] = self.next_run.isoformat() if self.next_run else None
        return result

class TaskScheduler:
    """Advanced task scheduler with comprehensive features."""
    
    def __init__(self, max_workers: int = 4, task_queue_size: int = 100):
        """
        Initialize the task scheduler.
        
        Args:
            max_workers: Maximum number of worker threads
            task_queue_size: Size of the task queue
        """
        self.max_workers = max_workers
        self.task_queue = queue.PriorityQueue(maxsize=task_queue_size)
        self.tasks: Dict[str, ScheduledTask] = {}
        self.running = False
        self.workers: List[threading.Thread] = []
        self.scheduler_thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.stats = {
            'total_tasks_executed': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'total_runtime': 0.0,
            'start_time': None
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info(f"TaskScheduler initialized with {max_workers} workers")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down gracefully")
        self.stop()
        sys.exit(0)
    
    def add_task(self, task_id: str, name: str, func: Callable, 
                 schedule_type: ScheduleType, schedule_config: Dict[str, Any],
                 priority: TaskPriority = TaskPriority.NORMAL,
                 enabled: bool = True, max_retries: int = 3,
                 retry_delay: int = 60, timeout: Optional[int] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a new scheduled task.
        
        Args:
            task_id: Unique identifier for the task
            name: Human-readable task name
            func: Function to execute
            schedule_type: Type of scheduling
            schedule_config: Configuration for the schedule
            priority: Task priority
            enabled: Whether the task is enabled
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds
            timeout: Task timeout in seconds
            metadata: Additional task metadata
            
        Returns:
            True if task was added successfully
        """
        try:
            if task_id in self.tasks:
                self.logger.warning(f"Task {task_id} already exists, updating")
            
            task = ScheduledTask(
                id=task_id,
                name=name,
                func=func,
                schedule_type=schedule_type,
                schedule_config=schedule_config,
                priority=priority,
                enabled=enabled,
                max_retries=max_retries,
                retry_delay=retry_delay,
                timeout=timeout,
                metadata=metadata or {}
            )
            
            # Calculate next run time
            task.next_run = self._calculate_next_run(task)
            
            self.tasks[task_id] = task
            self.logger.info(f"Task '{name}' (ID: {task_id}) added successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding task {task_id}: {e}")
            return False
    
    def remove_task(self, task_id: str) -> bool:
        """
        Remove a scheduled task.
        
        Args:
            task_id: ID of the task to remove
            
        Returns:
            True if task was removed successfully
        """
        try:
            if task_id in self.tasks:
                del self.tasks[task_id]
                self.logger.info(f"Task {task_id} removed successfully")
                return True
            else:
                self.logger.warning(f"Task {task_id} not found")
                return False
                
        except Exception as e:
            self.logger.error(f"Error removing task {task_id}: {e}")
            return False
    
    def enable_task(self, task_id: str) -> bool:
        """Enable a task."""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = True
            self.logger.info(f"Task {task_id} enabled")
            return True
        return False
    
    def disable_task(self, task_id: str) -> bool:
        """Disable a task."""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = False
            self.logger.info(f"Task {task_id} disabled")
            return True
        return False
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task status dictionary or None if not found
        """
        if task_id in self.tasks:
            task = self.tasks[task_id]
            return {
                'id': task.id,
                'name': task.name,
                'status': task.status.value,
                'enabled': task.enabled,
                'last_run': task.last_run.isoformat() if task.last_run else None,
                'next_run': task.next_run.isoformat() if task.next_run else None,
                'error_count': task.error_count,
                'success_count': task.success_count,
                'total_runtime': task.total_runtime
            }
        return None
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get status of all tasks."""
        return [self.get_task_status(task_id) for task_id in self.tasks.keys()]
    
    def start(self) -> None:
        """Start the scheduler."""
        if self.running:
            self.logger.warning("Scheduler is already running")
            return
        
        self.running = True
        self.stats['start_time'] = datetime.now()
        
        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        self.logger.info(f"Scheduler started with {self.max_workers} workers")
    
    def stop(self) -> None:
        """Stop the scheduler."""
        if not self.running:
            return
        
        self.logger.info("Stopping scheduler...")
        self.running = False
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        self.logger.info("Scheduler stopped")
    
    def _scheduler_loop(self) -> None:
        """Main scheduler loop that checks for tasks to run."""
        while self.running:
            try:
                current_time = datetime.now()
                
                for task in self.tasks.values():
                    if not task.enabled or task.status == TaskStatus.RUNNING:
                        continue
                    
                    if task.next_run and current_time >= task.next_run:
                        # Add task to queue with priority
                        priority = (task.priority.value, current_time.timestamp())
                        self.task_queue.put((priority, task))
                        
                        # Calculate next run time
                        task.next_run = self._calculate_next_run(task)
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                time.sleep(5)
    
    def _worker_loop(self, worker_id: int) -> None:
        """Worker thread loop that executes tasks."""
        self.logger.info(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # Get task from queue with timeout
                try:
                    priority, task = self.task_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                self._execute_task(task, worker_id)
                self.task_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in worker {worker_id}: {e}")
                time.sleep(1)
        
        self.logger.info(f"Worker {worker_id} stopped")
    
    def _execute_task(self, task: ScheduledTask, worker_id: int) -> None:
        """Execute a single task."""
        start_time = time.time()
        task.status = TaskStatus.RUNNING
        
        self.logger.info(f"Worker {worker_id} executing task: {task.name}")
        
        try:
            # Execute task with timeout if specified
            if task.timeout:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(task.func)
                    result = future.result(timeout=task.timeout)
            else:
                result = task.func()
            
            # Task completed successfully
            task.status = TaskStatus.COMPLETED
            task.success_count += 1
            task.error_count = 0
            task.last_run = datetime.now()
            task.total_runtime += time.time() - start_time
            
            self.stats['successful_tasks'] += 1
            self.logger.info(f"Task {task.name} completed successfully in {time.time() - start_time:.2f}s")
            
        except Exception as e:
            # Task failed
            task.error_count += 1
            task.last_run = datetime.now()
            task.total_runtime += time.time() - start_time
            
            self.stats['failed_tasks'] += 1
            self.logger.error(f"Task {task.name} failed: {e}")
            
            # Retry logic
            if task.error_count < task.max_retries:
                task.status = TaskStatus.PENDING
                # Reschedule for retry
                retry_time = datetime.now() + timedelta(seconds=task.retry_delay)
                task.next_run = retry_time
                self.logger.info(f"Task {task.name} will be retried at {retry_time}")
            else:
                task.status = TaskStatus.FAILED
                self.logger.error(f"Task {task.name} failed permanently after {task.max_retries} retries")
        
        finally:
            self.stats['total_tasks_executed'] += 1
            self.stats['total_runtime'] += time.time() - start_time
    
    def _calculate_next_run(self, task: ScheduledTask) -> Optional[datetime]:
        """Calculate the next run time for a task."""
        try:
            current_time = datetime.now()
            
            if task.schedule_type == ScheduleType.INTERVAL:
                interval_seconds = task.schedule_config.get('seconds', 3600)
                return current_time + timedelta(seconds=interval_seconds)
            
            elif task.schedule_type == ScheduleType.DAILY:
                time_str = task.schedule_config.get('time', '00:00')
                hour, minute = map(int, time_str.split(':'))
                next_run = current_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if next_run <= current_time:
                    next_run += timedelta(days=1)
                return next_run
            
            elif task.schedule_type == ScheduleType.WEEKLY:
                weekday = task.schedule_config.get('weekday', 0)  # Monday = 0
                time_str = task.schedule_config.get('time', '00:00')
                hour, minute = map(int, time_str.split(':'))
                
                days_ahead = weekday - current_time.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                
                next_run = current_time + timedelta(days=days_ahead)
                next_run = next_run.replace(hour=hour, minute=minute, second=0, microsecond=0)
                return next_run
            
            elif task.schedule_type == ScheduleType.MONTHLY:
                day = task.schedule_config.get('day', 1)
                time_str = task.schedule_config.get('time', '00:00')
                hour, minute = map(int, time_str.split(':'))
                
                next_run = current_time.replace(day=day, hour=hour, minute=minute, second=0, microsecond=0)
                if next_run <= current_time:
                    # Move to next month
                    if current_time.month == 12:
                        next_run = next_run.replace(year=current_time.year + 1, month=1)
                    else:
                        next_run = next_run.replace(month=current_time.month + 1)
                return next_run
            
            elif task.schedule_type == ScheduleType.CRON:
                # Simple cron-like scheduling
                cron_expr = task.schedule_config.get('expression', '0 0 * * *')
                return self._parse_cron_expression(cron_expr, current_time)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error calculating next run time for task {task.name}: {e}")
            return None
    
    def _parse_cron_expression(self, cron_expr: str, current_time: datetime) -> datetime:
        """Parse a simple cron expression and return next run time."""
        try:
            parts = cron_expr.split()
            if len(parts) != 5:
                raise ValueError("Cron expression must have 5 parts")
            
            minute, hour, day, month, weekday = parts
            
            # Simple implementation - just return current time + 1 hour
            # In a production system, you'd want a proper cron parser
            return current_time + timedelta(hours=1)
            
        except Exception as e:
            self.logger.error(f"Error parsing cron expression {cron_expr}: {e}")
            return current_time + timedelta(hours=1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        uptime = None
        if self.stats['start_time']:
            uptime = (datetime.now() - self.stats['start_time']).total_seconds()
        
        return {
            'running': self.running,
            'total_tasks': len(self.tasks),
            'enabled_tasks': sum(1 for task in self.tasks.values() if task.enabled),
            'total_tasks_executed': self.stats['total_tasks_executed'],
            'successful_tasks': self.stats['successful_tasks'],
            'failed_tasks': self.stats['failed_tasks'],
            'total_runtime': self.stats['total_runtime'],
            'uptime_seconds': uptime,
            'queue_size': self.task_queue.qsize(),
            'active_workers': len([w for w in self.workers if w.is_alive()])
        }
    
    def save_state(self, filepath: str = "scheduler_state.json") -> bool:
        """Save scheduler state to file."""
        try:
            state = {
                'tasks': {task_id: task.to_dict() for task_id, task in self.tasks.items()},
                'stats': self.stats,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.info(f"Scheduler state saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving scheduler state: {e}")
            return False
    
    def load_state(self, filepath: str = "scheduler_state.json") -> bool:
        """Load scheduler state from file."""
        try:
            if not Path(filepath).exists():
                self.logger.warning(f"State file {filepath} not found")
                return False
            
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Note: Functions cannot be restored from JSON, so tasks will need to be re-registered
            self.stats = state.get('stats', self.stats)
            
            self.logger.info(f"Scheduler state loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading scheduler state: {e}")
            return False

# Global scheduler instance
_scheduler: Optional[TaskScheduler] = None

def get_scheduler() -> TaskScheduler:
    """Get the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = TaskScheduler()
    return _scheduler

# Convenience functions
def schedule_task(task_id: str, name: str, func: Callable, 
                 schedule_type: ScheduleType, schedule_config: Dict[str, Any],
                 **kwargs) -> bool:
    """Schedule a task using the global scheduler."""
    return get_scheduler().add_task(task_id, name, func, schedule_type, schedule_config, **kwargs)

def start_scheduler() -> None:
    """Start the global scheduler."""
    get_scheduler().start()

def stop_scheduler() -> None:
    """Stop the global scheduler."""
    if _scheduler:
        _scheduler.stop()

def get_scheduler_stats() -> Dict[str, Any]:
    """Get statistics from the global scheduler."""
    return get_scheduler().get_stats() 