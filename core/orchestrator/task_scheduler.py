"""
Task Scheduler Module

This module contains task scheduling functionality for the task orchestrator.
Extracted from the original task_orchestrator.py for modularity.
"""

import logging
from typing import Any, Dict, List, Optional

import schedule

from .task_models import TaskConfig


class TaskScheduler:
    """Handles task scheduling and timing."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Task registry
        self.tasks: Dict[str, TaskConfig] = {}
        self.scheduled_tasks: Dict[str, Any] = {}

        # Scheduling state
        self.is_running = False
        self.scheduler_thread = None

    def add_task(self, task_config: TaskConfig) -> None:
        """Add a task to the scheduler."""
        self.tasks[task_config.name] = task_config
        self._schedule_task(task_config)

    def remove_task(self, task_name: str) -> None:
        """Remove a task from the scheduler."""
        if task_name in self.tasks:
            del self.tasks[task_name]

        if task_name in self.scheduled_tasks:
            schedule.clear(task_name)
            del self.scheduled_tasks[task_name]

    def update_task(self, task_name: str, updates: Dict[str, Any]) -> None:
        """Update task configuration."""
        if task_name in self.tasks:
            task_config = self.tasks[task_name]

            # Update task config
            for key, value in updates.items():
                if hasattr(task_config, key):
                    setattr(task_config, key, value)

            # Reschedule task
            self._schedule_task(task_config)

    def _schedule_task(self, task_config: TaskConfig) -> None:
        """Schedule a task based on its configuration."""
        if not task_config.enabled:
            return

        # Clear existing schedule for this task
        if task_config.name in self.scheduled_tasks:
            schedule.clear(task_config.name)

        # Create schedule based on interval
        if task_config.interval_minutes <= 0:
            # One-time task
            job = (
                schedule.every()
                .day.at("00:00")
                .do(self._task_wrapper, task_config.name)
                .tag(task_config.name)
            )
        else:
            # Recurring task
            job = (
                schedule.every(task_config.interval_minutes)
                .minutes.do(self._task_wrapper, task_config.name)
                .tag(task_config.name)
            )

        self.scheduled_tasks[task_config.name] = job
        self.logger.info(
            f"Scheduled task: {task_config.name} (every {task_config.interval_minutes} minutes)"
        )

    def _task_wrapper(self, task_name: str) -> None:
        """Wrapper for scheduled tasks."""
        try:
            self.logger.info(f"Executing scheduled task: {task_name}")
            # This would typically trigger task execution
            # For now, just log the execution
        except Exception as e:
            self.logger.error(f"Failed to execute scheduled task {task_name}: {e}")

    def start(self) -> None:
        """Start the scheduler."""
        self.is_running = True
        self.logger.info("Task scheduler started")

    def stop(self) -> None:
        """Stop the scheduler."""
        self.is_running = False
        self.logger.info("Task scheduler stopped")

    def run_pending(self) -> None:
        """Run pending scheduled tasks."""
        if self.is_running:
            schedule.run_pending()

    def get_scheduled_tasks(self) -> Dict[str, Any]:
        """Get information about scheduled tasks."""
        scheduled_info = {}

        for task_name, task_config in self.tasks.items():
            if task_name in self.scheduled_tasks:
                job = self.scheduled_tasks[task_name]
                scheduled_info[task_name] = {
                    "enabled": task_config.enabled,
                    "interval_minutes": task_config.interval_minutes,
                    "next_run": (
                        str(job.next_run) if hasattr(job, "next_run") else "unknown"
                    ),
                    "priority": task_config.priority.value,
                }

        return scheduled_info

    def get_task_config(self, task_name: str) -> Optional[TaskConfig]:
        """Get task configuration."""
        return self.tasks.get(task_name)

    def get_all_tasks(self) -> List[TaskConfig]:
        """Get all task configurations."""
        return list(self.tasks.values())

    def enable_task(self, task_name: str) -> None:
        """Enable a task."""
        if task_name in self.tasks:
            self.tasks[task_name].enabled = True
            self._schedule_task(self.tasks[task_name])

    def disable_task(self, task_name: str) -> None:
        """Disable a task."""
        if task_name in self.tasks:
            self.tasks[task_name].enabled = False
            if task_name in self.scheduled_tasks:
                schedule.clear(task_name)
                del self.scheduled_tasks[task_name]
