"""
Strategy Executor

Enhanced strategy execution with:
- Task queue length guards to prevent async overload
- Comprehensive logging for dropped/failed strategy results
- Execution timeouts per task using asyncio.wait_for()
- Performance monitoring and metrics
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import weakref
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    DROPPED = "dropped"


@dataclass
class TaskResult:
    """Result of task execution."""
    task_id: str
    strategy_name: str
    status: TaskStatus
    result: Any = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    timeout: float = 30.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        if self.start_time:
            result['start_time'] = self.start_time.isoformat()
        if self.end_time:
            result['end_time'] = self.end_time.isoformat()
        return result


@dataclass
class StrategyTask:
    """Strategy execution task."""
    task_id: str
    strategy_name: str
    strategy_func: Callable
    args: tuple = ()
    kwargs: dict = None
    priority: int = 1
    timeout: float = 30.0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
        if self.created_at is None:
            self.created_at = datetime.now()


class StrategyExecutor:
    """
    Enhanced strategy executor with queue management and timeout controls.
    
    Features:
    - Task queue length guards to prevent async overload
    - Comprehensive logging for dropped/failed strategy results
    - Execution timeouts per task using asyncio.wait_for()
    - Performance monitoring and metrics
    - Priority-based task scheduling
    """
    
    def __init__(self, 
                 max_queue_size: int = 100,
                 max_concurrent_tasks: int = 10,
                 default_timeout: float = 30.0,
                 enable_metrics: bool = True):
        """Initialize the strategy executor."""
        self.max_queue_size = max_queue_size
        self.max_concurrent_tasks = max_concurrent_tasks
        self.default_timeout = default_timeout
        self.enable_metrics = enable_metrics
        
        # Task management
        self.task_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: List[TaskResult] = []
        self.failed_tasks: List[TaskResult] = []
        self.dropped_tasks: List[TaskResult] = []
        
        # Performance metrics
        self.metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'dropped_tasks': 0,
            'timeout_tasks': 0,
            'average_execution_time': 0.0,
            'queue_overflow_count': 0,
            'start_time': datetime.now()
        }
        
        # Control flags
        self.running = False
        self._task_counter = 0
        self._lock = asyncio.Lock()
        
        logger.info(f"StrategyExecutor initialized: max_queue={max_queue_size}, max_concurrent={max_concurrent_tasks}")
    
    async def start(self):
        """Start the strategy executor."""
        if self.running:
            logger.warning("StrategyExecutor already running")
            return
        
        self.running = True
        logger.info("🚀 Starting StrategyExecutor...")
        
        # Start worker tasks
        workers = []
        for i in range(self.max_concurrent_tasks):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            workers.append(worker)
        
        # Start metrics collection
        if self.enable_metrics:
            metrics_task = asyncio.create_task(self._metrics_collector())
            workers.append(metrics_task)
        
        # Wait for all workers
        try:
            await asyncio.gather(*workers)
        except Exception as e:
            logger.error(f"Worker error: {e}")
        finally:
            self.running = False
            logger.info("🛑 StrategyExecutor stopped")
    
    async def stop(self):
        """Stop the strategy executor gracefully."""
        logger.info("🛑 Stopping StrategyExecutor...")
        self.running = False
        
        # Cancel running tasks
        for task_id, task in self.running_tasks.items():
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled running task: {task_id}")
        
        # Wait for tasks to complete
        if self.running_tasks:
            await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)
        
        logger.info("✅ StrategyExecutor stopped gracefully")
    
    async def submit_task(self, 
                         strategy_name: str,
                         strategy_func: Callable,
                         *args,
                         priority: int = 1,
                         timeout: Optional[float] = None,
                         **kwargs) -> Optional[str]:
        """Submit a strategy task for execution."""
        if not self.running:
            logger.error("StrategyExecutor not running")
            return None
        
        task_id = f"task_{self._task_counter}_{int(time.time())}"
        self._task_counter += 1
        
        # Check queue size
        if self.task_queue.qsize() >= self.max_queue_size:
            await self._handle_queue_overflow(task_id, strategy_name)
            return None
        
        # Create task
        task = StrategyTask(
            task_id=task_id,
            strategy_name=strategy_name,
            strategy_func=strategy_func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout or self.default_timeout
        )
        
        try:
            # Add to queue with timeout
            await asyncio.wait_for(
                self.task_queue.put(task),
                timeout=5.0
            )
            
            self.metrics['total_tasks'] += 1
            logger.debug(f"Task submitted: {task_id} ({strategy_name})")
            return task_id
            
        except asyncio.TimeoutError:
            await self._handle_queue_overflow(task_id, strategy_name)
            return None
        except Exception as e:
            logger.error(f"Failed to submit task {task_id}: {e}")
            return None
    
    async def _handle_queue_overflow(self, task_id: str, strategy_name: str):
        """Handle queue overflow by dropping task and logging."""
        dropped_result = TaskResult(
            task_id=task_id,
            strategy_name=strategy_name,
            status=TaskStatus.DROPPED,
            error_message="Queue overflow - task dropped",
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        self.dropped_tasks.append(dropped_result)
        self.metrics['dropped_tasks'] += 1
        self.metrics['queue_overflow_count'] += 1
        
        logger.warning(f"Queue overflow - dropped task {task_id} ({strategy_name})")
        logger.warning(f"Queue size: {self.task_queue.qsize()}/{self.max_queue_size}")
    
    async def _worker(self, worker_name: str):
        """Worker task that processes strategy tasks."""
        logger.info(f"Worker {worker_name} started")
        
        while self.running:
            try:
                # Get task from queue
                task = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )
                
                # Execute task
                await self._execute_task(task, worker_name)
                
                # Mark task as done
                self.task_queue.task_done()
                
            except asyncio.TimeoutError:
                # No tasks available, continue
                continue
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
        
        logger.info(f"Worker {worker_name} stopped")
    
    async def _execute_task(self, task: StrategyTask, worker_name: str):
        """Execute a single strategy task."""
        start_time = datetime.now()
        task_result = TaskResult(
            task_id=task.task_id,
            strategy_name=task.strategy_name,
            status=TaskStatus.RUNNING,
            start_time=start_time,
            timeout=task.timeout
        )
        
        # Add to running tasks
        self.running_tasks[task.task_id] = asyncio.current_task()
        
        try:
            logger.debug(f"Executing task {task.task_id} ({task.strategy_name}) on {worker_name}")
            
            # Execute strategy with timeout
            result = await asyncio.wait_for(
                self._call_strategy(task),
                timeout=task.timeout
            )
            
            # Task completed successfully
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            task_result.status = TaskStatus.COMPLETED
            task_result.result = result
            task_result.end_time = end_time
            task_result.execution_time = execution_time
            
            self.completed_tasks.append(task_result)
            self.metrics['completed_tasks'] += 1
            
            logger.info(f"✅ Task {task.task_id} completed in {execution_time:.2f}s")
            
        except asyncio.TimeoutError:
            # Task timed out
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            task_result.status = TaskStatus.TIMEOUT
            task_result.error_message = f"Task timed out after {task.timeout}s"
            task_result.end_time = end_time
            task_result.execution_time = execution_time
            
            self.failed_tasks.append(task_result)
            self.metrics['timeout_tasks'] += 1
            
            logger.error(f"⏰ Task {task.task_id} timed out after {execution_time:.2f}s")
            
        except Exception as e:
            # Task failed
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            task_result.status = TaskStatus.FAILED
            task_result.error_message = str(e)
            task_result.end_time = end_time
            task_result.execution_time = execution_time
            
            self.failed_tasks.append(task_result)
            self.metrics['failed_tasks'] += 1
            
            logger.error(f"❌ Task {task.task_id} failed after {execution_time:.2f}s: {e}")
            
        finally:
            # Remove from running tasks
            self.running_tasks.pop(task.task_id, None)
    
    async def _call_strategy(self, task: StrategyTask) -> Any:
        """Call the strategy function with proper error handling."""
        try:
            if asyncio.iscoroutinefunction(task.strategy_func):
                # Async function
                result = await task.strategy_func(*task.args, **task.kwargs)
            else:
                # Sync function - run in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, 
                    task.strategy_func, 
                    *task.args, 
                    **task.kwargs
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Strategy execution error in {task.strategy_name}: {e}")
            raise
    
    async def _metrics_collector(self):
        """Collect and log performance metrics."""
        logger.info("📊 Metrics collector started")
        
        while self.running:
            try:
                await asyncio.sleep(60)  # Collect metrics every minute
                
                # Calculate average execution time
                if self.completed_tasks:
                    total_time = sum(task.execution_time for task in self.completed_tasks)
                    self.metrics['average_execution_time'] = total_time / len(self.completed_tasks)
                
                # Log metrics
                logger.info(f"📊 Metrics: {self.metrics}")
                
                # Log failed/dropped tasks summary
                if self.failed_tasks or self.dropped_tasks:
                    logger.warning(f"⚠️ Failed tasks: {len(self.failed_tasks)}, Dropped tasks: {len(self.dropped_tasks)}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collector error: {e}")
        
        logger.info("📊 Metrics collector stopped")
    
    def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """Get status of a specific task."""
        # Check running tasks
        if task_id in self.running_tasks:
            return TaskResult(
                task_id=task_id,
                strategy_name="unknown",
                status=TaskStatus.RUNNING
            )
        
        # Check completed tasks
        for task in self.completed_tasks:
            if task.task_id == task_id:
                return task
        
        # Check failed tasks
        for task in self.failed_tasks:
            if task.task_id == task_id:
                return task
        
        # Check dropped tasks
        for task in self.dropped_tasks:
            if task.task_id == task_id:
                return task
        
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        metrics = self.metrics.copy()
        metrics.update({
            'queue_size': self.task_queue.qsize(),
            'running_tasks': len(self.running_tasks),
            'uptime_seconds': (datetime.now() - metrics['start_time']).total_seconds()
        })
        return metrics
    
    def get_failed_tasks_summary(self) -> Dict[str, Any]:
        """Get summary of failed and dropped tasks."""
        failed_summary = {}
        dropped_summary = {}
        
        # Group failed tasks by strategy
        for task in self.failed_tasks:
            strategy = task.strategy_name
            if strategy not in failed_summary:
                failed_summary[strategy] = []
            failed_summary[strategy].append(task.to_dict())
        
        # Group dropped tasks by strategy
        for task in self.dropped_tasks:
            strategy = task.strategy_name
            if strategy not in dropped_summary:
                dropped_summary[strategy] = []
            dropped_summary[strategy].append(task.to_dict())
        
        return {
            'failed_tasks': failed_summary,
            'dropped_tasks': dropped_summary,
            'total_failed': len(self.failed_tasks),
            'total_dropped': len(self.dropped_tasks)
        }
    
    def clear_completed_tasks(self, max_age_hours: int = 24):
        """Clear old completed tasks to free memory."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        # Clear old completed tasks
        original_count = len(self.completed_tasks)
        self.completed_tasks = [
            task for task in self.completed_tasks
            if task.end_time and task.end_time > cutoff_time
        ]
        cleared_count = original_count - len(self.completed_tasks)
        
        # Clear old failed tasks
        original_failed_count = len(self.failed_tasks)
        self.failed_tasks = [
            task for task in self.failed_tasks
            if task.end_time and task.end_time > cutoff_time
        ]
        cleared_failed_count = original_failed_count - len(self.failed_tasks)
        
        logger.info(f"Cleared {cleared_count} completed tasks and {cleared_failed_count} failed tasks")


def process_strategy_signals(strategy_results):
    """
    Process signals from multiple strategies:
    - Skip None or empty results with warning
    - Deduplicate buy/sell signals with identical timestamps
    - Apply confidence-based weighting if multiple strategies overlap
    """
    all_signals = []
    for strat_name, result in strategy_results.items():
        if result is None or (hasattr(result, 'empty') and result.empty):
            logger.warning(f"Strategy {strat_name} returned None or empty result. Skipping.")
            continue
        if isinstance(result, pd.DataFrame):
            all_signals.append((strat_name, result))
        else:
            logger.warning(f"Strategy {strat_name} returned non-DataFrame result. Skipping.")

    if not all_signals:
        logger.warning("No valid strategy signals to process.")
        return pd.DataFrame()

    # Concatenate all signals
    combined = pd.concat([df.assign(strategy=strat) for strat, df in all_signals], axis=0, ignore_index=True)

    # Deduplicate by timestamp and signal type (buy/sell), keep highest confidence
    deduped = (
        combined.sort_values("confidence", ascending=False)
        .drop_duplicates(subset=["timestamp", "signal_type"], keep="first")
        .reset_index(drop=True)
    )

    # If multiple strategies overlap on the same timestamp, compute weighted signal
    grouped = deduped.groupby(["timestamp", "signal_type"])
    weighted_signals = []
    for (timestamp, signal_type), group in grouped:
        if len(group) == 1:
            weighted_signals.append(group.iloc[0])
        else:
            # Confidence-based weighting
            total_conf = group["confidence"].sum()
            if total_conf == 0:
                weights = [1.0 / len(group)] * len(group)
            else:
                weights = group["confidence"] / total_conf
            weighted_signal = group.iloc[0].copy()
            weighted_signal["confidence"] = group["confidence"].max()
            weighted_signal["weighted_value"] = (group["signal_value"] * weights).sum()
            weighted_signal["strategies"] = ",".join(group["strategy"])
            weighted_signals.append(weighted_signal)
    result_df = pd.DataFrame(weighted_signals)
    return result_df


# Global executor instance
_global_executor: Optional[StrategyExecutor] = None


def get_strategy_executor() -> StrategyExecutor:
    """Get global strategy executor instance."""
    global _global_executor
    if _global_executor is None:
        _global_executor = StrategyExecutor()
    return _global_executor


async def submit_strategy_task(strategy_name: str, 
                              strategy_func: Callable,
                              *args,
                              **kwargs) -> Optional[str]:
    """Submit a task to the global strategy executor."""
    executor = get_strategy_executor()
    return await executor.submit_task(strategy_name, strategy_func, *args, **kwargs)


if __name__ == "__main__":
    # Demo usage
    async def demo():
        print("🎯 Strategy Executor Demo")
        print("=" * 50)
        
        # Create executor
        executor = StrategyExecutor(max_queue_size=10, max_concurrent_tasks=2)
        
        # Define sample strategies
        async def sample_strategy_1():
            await asyncio.sleep(2)
            return "Strategy 1 completed"
        
        async def sample_strategy_2():
            await asyncio.sleep(1)
            return "Strategy 2 completed"
        
        async def failing_strategy():
            await asyncio.sleep(0.5)
            raise ValueError("Strategy failed intentionally")
        
        async def timeout_strategy():
            await asyncio.sleep(35)  # Longer than default timeout
            return "This should timeout"
        
        # Start executor
        executor_task = asyncio.create_task(executor.start())
        
        # Submit tasks
        print("\n📝 Submitting tasks...")
        tasks = [
            ("strategy_1", sample_strategy_1),
            ("strategy_2", sample_strategy_2),
            ("failing_strategy", failing_strategy),
            ("timeout_strategy", timeout_strategy)
        ]
        
        task_ids = []
        for name, func in tasks:
            task_id = await executor.submit_task(name, func)
            if task_id:
                task_ids.append(task_id)
                print(f"Submitted: {name} -> {task_id}")
        
        # Wait a bit for execution
        await asyncio.sleep(5)
        
        # Get metrics
        metrics = executor.get_metrics()
        print(f"\n📊 Metrics: {metrics}")
        
        # Get failed tasks summary
        failed_summary = executor.get_failed_tasks_summary()
        print(f"\n❌ Failed tasks: {failed_summary}")
        
        # Stop executor
        await executor.stop()
        executor_task.cancel()
        
        print("\n✅ Demo completed!")
    
    asyncio.run(demo()) 