"""
Task Executor Module

This module contains task execution functionality for the task orchestrator.
Extracted from the original task_orchestrator.py for modularity.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .task_models import TaskConfig, TaskExecution, TaskStatus, TaskType


class TaskExecutor:
    """Handles task execution and management."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Execution management
        self.executor = ThreadPoolExecutor(max_workers=config.get("max_workers", 10))
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_queue = asyncio.Queue()

        # Task providers
        self.task_providers: Dict[str, Callable] = {}

        # Execution tracking
        self.execution_history: List[TaskExecution] = []
        self.current_executions: Dict[str, TaskExecution] = {}

    def register_task_provider(self, task_type: str, provider: Callable) -> None:
        """Register a task provider for a specific task type."""
        self.task_providers[task_type] = provider
        self.logger.info(f"Registered task provider for: {task_type}")

    async def execute_task(
        self,
        task_name: str,
        task_config: TaskConfig,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> TaskExecution:
        """Execute a task."""
        execution = TaskExecution(
            task_id=f"{task_name}_{int(time.time())}",
            task_name=task_name,
            task_type=task_config.task_type,
            start_time=datetime.utcnow().isoformat(),
            status=TaskStatus.RUNNING,
        )

        self.current_executions[task_name] = execution

        try:
            self.logger.info(f"Starting task execution: {task_name}")

            # Check if task provider exists
            if task_config.task_type.value not in self.task_providers:
                raise RuntimeError(
                    f"No provider registered for task type: {task_config.task_type.value}"
                )

            # Execute task
            start_time = time.time()
            result = await self._run_task_method(task_name, parameters or {})
            end_time = time.time()

            # Update execution record
            execution.end_time = datetime.utcnow().isoformat()
            execution.status = TaskStatus.COMPLETED
            execution.result = result
            execution.duration_seconds = end_time - start_time
            execution.performance_score = self._calculate_performance_score(result)

            self.logger.info(
                f"Task completed successfully: {task_name} (duration: {execution.duration_seconds:.2f}s)"
            )

        except Exception as e:
            self.logger.error(f"Task execution failed: {task_name} - {e}")

            execution.end_time = datetime.utcnow().isoformat()
            execution.status = TaskStatus.FAILED
            execution.error_message = str(e)
            execution.duration_seconds = (
                time.time() - start_time if "start_time" in locals() else 0
            )

        finally:
            # Remove from current executions
            if task_name in self.current_executions:
                del self.current_executions[task_name]

            # Add to history
            self.execution_history.append(execution)

            # Keep history size manageable
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-500:]

        return execution

    async def _run_task_method(
        self, task_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run the actual task method."""
        # Get task provider
        task_type = self._get_task_type(task_name)
        provider = self.task_providers.get(task_type)

        if not provider:
            raise RuntimeError(f"No provider found for task type: {task_type}")

        # Execute in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor, provider, task_name, parameters
        )

        return result or {}

    def _get_task_type(self, task_name: str) -> str:
        """Get task type from task name."""
        # This is a simplified mapping - in practice, you'd have a proper task registry
        task_type_mapping = {
            "model_innovation": "model_innovation",
            "strategy_research": "strategy_research",
            "sentiment_fetch": "sentiment_fetch",
            "meta_control": "meta_control",
            "risk_management": "risk_management",
            "execution": "execution",
            "explanation": "explanation",
            "system_health": "system_health",
            "data_sync": "data_sync",
            "performance_analysis": "performance_analysis",
        }

        for key, value in task_type_mapping.items():
            if key in task_name.lower():
                return value

        return "unknown"

    def _calculate_performance_score(self, result: Dict[str, Any]) -> float:
        """Calculate performance score for a task execution."""
        if not result:
            return 0.0

        # Simple scoring based on result success
        if result.get("success", False):
            return 1.0
        elif "error" in result:
            return 0.0
        else:
            return 0.5

    async def execute_task_now(
        self, task_name: str, parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute a task immediately."""
        # Get task config (this would come from the scheduler)
        task_config = TaskConfig(
            name=task_name,
            task_type=TaskType.SYSTEM_HEALTH,  # Default type
            enabled=True,
        )

        # Execute task
        execution = await self.execute_task(task_name, task_config, parameters)

        return execution.task_id

    def get_task_status(self, task_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a task."""
        # Check current executions
        if task_name in self.current_executions:
            execution = self.current_executions[task_name]
            return {
                "task_name": execution.task_name,
                "status": execution.status.value,
                "start_time": execution.start_time,
                "duration": execution.duration_seconds or 0,
            }

        # Check recent history
        for execution in reversed(self.execution_history):
            if execution.task_name == task_name:
                return {
                    "task_name": execution.task_name,
                    "status": execution.status.value,
                    "start_time": execution.start_time,
                    "end_time": execution.end_time,
                    "duration": execution.duration_seconds or 0,
                    "performance_score": execution.performance_score,
                    "error": execution.error_message,
                }

        return None

    def get_execution_history(
        self, task_name: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get execution history."""
        history = self.execution_history

        if task_name:
            history = [
                execution for execution in history if execution.task_name == task_name
            ]

        # Return recent history
        recent_history = history[-limit:] if limit > 0 else history

        return [execution.__dict__ for execution in recent_history]

    def get_current_executions(self) -> Dict[str, Dict[str, Any]]:
        """Get currently running tasks."""
        return {
            task_name: {
                "task_name": execution.task_name,
                "status": execution.status.value,
                "start_time": execution.start_time,
                "duration": execution.duration_seconds or 0,
            }
            for task_name, execution in self.current_executions.items()
        }

    def cancel_task(self, task_name: str) -> bool:
        """Cancel a running task."""
        if task_name in self.current_executions:
            execution = self.current_executions[task_name]
            execution.status = TaskStatus.CANCELLED
            execution.end_time = datetime.utcnow().isoformat()

            # Move to history
            self.execution_history.append(execution)
            del self.current_executions[task_name]

            self.logger.info(f"Cancelled task: {task_name}")
            return True

        return False

    def clear_history(self) -> None:
        """Clear execution history."""
        self.execution_history.clear()
        self.logger.info("Cleared execution history")
