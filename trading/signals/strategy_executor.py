"""
Strategy Executor

This module provides an enhanced strategy executor with queue management,
timeout controls, and comprehensive monitoring capabilities.
"""

import asyncio
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

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
            result["start_time"] = self.start_time.isoformat()
        if self.end_time:
            result["end_time"] = self.end_time.isoformat()
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

    def __init__(
        self,
        max_queue_size: int = 100,
        max_concurrent_tasks: int = 10,
        default_timeout: float = 30.0,
        enable_metrics: bool = True,
    ):
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
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "dropped_tasks": 0,
            "timeout_tasks": 0,
            "average_execution_time": 0.0,
            "queue_overflow_count": 0,
            "start_time": datetime.now(),
        }

        # Control flags
        self.running = False
        self._task_counter = 0
        self._lock = asyncio.Lock()

        logger.info(
            f"StrategyExecutor initialized: max_queue={max_queue_size}, max_concurrent={max_concurrent_tasks}"
        )

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

    async def submit_task(
        self,
        strategy_name: str,
        strategy_func: Callable,
        *args,
        priority: int = 1,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Optional[str]:
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
            timeout=timeout or self.default_timeout,
        )

        try:
            # Add to queue with timeout
            await asyncio.wait_for(self.task_queue.put(task), timeout=5.0)

            self.metrics["total_tasks"] += 1
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
            end_time=datetime.now(),
        )

        self.dropped_tasks.append(dropped_result)
        self.metrics["dropped_tasks"] += 1
        self.metrics["queue_overflow_count"] += 1

        logger.warning(f"Queue overflow - dropped task {task_id} ({strategy_name})")
        logger.warning(f"Queue size: {self.task_queue.qsize()}/{self.max_queue_size}")

    async def _worker(self, worker_name: str):
        """Worker task that processes strategy tasks."""
        logger.info(f"Worker {worker_name} started")

        while self.running:
            try:
                # Get task from queue
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)

                # Execute task
                await self._execute_task(task, worker_name)

                # Mark task as done
                self.task_queue.task_done()

            except asyncio.TimeoutError:
                # No tasks in queue, continue
                continue
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                continue

        logger.info(f"Worker {worker_name} stopped")

    async def _execute_task(self, task: StrategyTask, worker_name: str):
        """Execute a single strategy task."""
        start_time = datetime.now()
        task_result = TaskResult(
            task_id=task.task_id,
            strategy_name=task.strategy_name,
            status=TaskStatus.RUNNING,
            start_time=start_time,
            timeout=task.timeout,
        )

        # Add to running tasks
        self.running_tasks[task.task_id] = asyncio.current_task()

        try:
            logger.debug(
                f"Executing task {task.task_id} ({task.strategy_name}) on {worker_name}"
            )

            # Execute strategy with timeout
            result = await asyncio.wait_for(
                self._call_strategy(task), timeout=task.timeout
            )

            # Task completed successfully
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            task_result.status = TaskStatus.COMPLETED
            task_result.result = result
            task_result.end_time = end_time
            task_result.execution_time = execution_time

            self.completed_tasks.append(task_result)
            self.metrics["completed_tasks"] += 1

            logger.debug(f"Task {task.task_id} completed in {execution_time:.2f}s")

        except asyncio.TimeoutError:
            # Task timed out
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            task_result.status = TaskStatus.TIMEOUT
            task_result.error_message = f"Task timed out after {task.timeout}s"
            task_result.end_time = end_time
            task_result.execution_time = execution_time

            self.failed_tasks.append(task_result)
            self.metrics["timeout_tasks"] += 1

            logger.warning(f"Task {task.task_id} timed out after {execution_time:.2f}s")

        except Exception as e:
            # Task failed
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            task_result.status = TaskStatus.FAILED
            task_result.error_message = str(e)
            task_result.end_time = end_time
            task_result.execution_time = execution_time

            self.failed_tasks.append(task_result)
            self.metrics["failed_tasks"] += 1

            logger.error(f"Task {task.task_id} failed: {e}")

        finally:
            # Remove from running tasks
            self.running_tasks.pop(task.task_id, None)

    async def _call_strategy(self, task: StrategyTask) -> Any:
        """Call the strategy function."""
        if asyncio.iscoroutinefunction(task.strategy_func):
            # Async function
            return await task.strategy_func(*task.args, **task.kwargs)
        else:
            # Sync function - run in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, task.strategy_func, *task.args, **task.kwargs
            )

    async def _metrics_collector(self):
        """Collect and update metrics periodically."""
        logger.info("Metrics collector started")

        while self.running:
            try:
                await asyncio.sleep(60)  # Update every minute

                # Calculate average execution time
                if self.completed_tasks:
                    total_time = sum(t.execution_time for t in self.completed_tasks)
                    self.metrics["average_execution_time"] = total_time / len(
                        self.completed_tasks
                    )

                logger.debug(f"Metrics updated: {self.metrics}")

            except Exception as e:
                logger.error(f"Metrics collector error: {e}")

        logger.info("Metrics collector stopped")

    def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """Get status of a specific task."""
        # Check running tasks
        if task_id in self.running_tasks:
            return TaskResult(
                task_id=task_id, strategy_name="unknown", status=TaskStatus.RUNNING
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
        """Get current metrics."""
        metrics = self.metrics.copy()
        metrics["queue_size"] = self.task_queue.qsize()
        metrics["running_tasks"] = len(self.running_tasks)
        metrics["uptime"] = (datetime.now() - metrics["start_time"]).total_seconds()
        return metrics

    def get_failed_tasks_summary(self) -> Dict[str, Any]:
        """Get summary of failed tasks."""
        if not self.failed_tasks:
            return {"error": "No failed tasks"}

        error_counts = {}
        for task in self.failed_tasks:
            error_type = (
                task.error_message.split(":")[0] if task.error_message else "Unknown"
            )
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

        return {
            "total_failed": len(self.failed_tasks),
            "error_counts": error_counts,
            "recent_failures": [
                {
                    "task_id": task.task_id,
                    "strategy": task.strategy_name,
                    "error": task.error_message,
                    "time": task.end_time.isoformat() if task.end_time else None,
                }
                for task in self.failed_tasks[-10:]  # Last 10 failures
            ],
        }

    def clear_completed_tasks(self, max_age_hours: int = 24):
        """Clear old completed tasks to free memory."""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)

        # Filter out old tasks
        self.completed_tasks = [
            task
            for task in self.completed_tasks
            if task.end_time and task.end_time.timestamp() > cutoff_time
        ]

        self.failed_tasks = [
            task
            for task in self.failed_tasks
            if task.end_time and task.end_time.timestamp() > cutoff_time
        ]

        self.dropped_tasks = [
            task
            for task in self.dropped_tasks
            if task.end_time and task.end_time.timestamp() > cutoff_time
        ]

        logger.info(f"Cleared tasks older than {max_age_hours} hours")


def process_strategy_signals(strategy_results):
    """
    Process strategy execution results and generate signals.

    Args:
        strategy_results: List of strategy execution results

    Returns:
        Processed signals dictionary
    """
    signals = {
        "buy_signals": [],
        "sell_signals": [],
        "hold_signals": [],
        "confidence_scores": [],
        "risk_levels": [],
    }

    for result in strategy_results:
        if result.status == TaskStatus.COMPLETED:
            # Process successful strategy result
            if hasattr(result.result, "signal_type"):
                signal_type = result.result.signal_type
                if signal_type in ["buy", "strong_buy"]:
                    signals["buy_signals"].append(result.result)
                elif signal_type in ["sell", "strong_sell"]:
                    signals["sell_signals"].append(result.result)
                else:
                    signals["hold_signals"].append(result.result)

            # Extract confidence and risk information
            if hasattr(result.result, "confidence"):
                signals["confidence_scores"].append(result.result.confidence)

            if hasattr(result.result, "risk_level"):
                signals["risk_levels"].append(result.result.risk_level)

    return signals


# Global executor instance
_executor: Optional[StrategyExecutor] = None


def get_strategy_executor() -> StrategyExecutor:
    """Get the global strategy executor instance."""
    global _executor
    if _executor is None:
        _executor = StrategyExecutor()
    return _executor


async def submit_strategy_task(
    strategy_name: str, strategy_func: Callable, *args, **kwargs
) -> Optional[str]:
    """Submit a strategy task using the global executor."""
    executor = get_strategy_executor()
    return await executor.submit_task(strategy_name, strategy_func, *args, **kwargs)


if __name__ == "__main__":

    async def demo():
        """Demo the strategy executor."""
        executor = StrategyExecutor(max_concurrent_tasks=2)

        async def sample_strategy_1():
            await asyncio.sleep(2)
            return {"signal": "buy", "confidence": 0.8}

        async def sample_strategy_2():
            await asyncio.sleep(1)
            return {"signal": "sell", "confidence": 0.6}

        async def failing_strategy():
            raise ValueError("Strategy failed")

        async def timeout_strategy():
            await asyncio.sleep(35)  # Longer than default timeout
            return {"signal": "hold"}

        # Start executor
        asyncio.create_task(executor.start())

        # Wait a moment for executor to start
        await asyncio.sleep(1)

        # Submit tasks
        await submit_strategy_task("strategy1", sample_strategy_1)
        await submit_strategy_task("strategy2", sample_strategy_2)
        await submit_strategy_task("failing", failing_strategy)
        await submit_strategy_task("timeout", timeout_strategy, timeout=5.0)

        # Wait for tasks to complete
        await asyncio.sleep(10)

        # Print results
        print("Metrics:", executor.get_metrics())
        print("Failed tasks:", executor.get_failed_tasks_summary())

        # Stop executor
        await executor.stop()

    # Run demo
    asyncio.run(demo())
