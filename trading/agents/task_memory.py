# -*- coding: utf-8 -*-
"""Persistent memory for agent tasks with automatic persistence."""

# Standard library imports
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Third-party imports
from pydantic import Field

# Local imports
from trading.config.settings import MEMORY_BACKEND, MEMORY_DIR


class TaskStatus(str, Enum):
    """Possible task statuses."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Task data structure."""

    task_id: str
    task_type: str
    status: TaskStatus
    agent: str
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status.value,
            "agent": self.agent,
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": json.dumps(self.metadata) if self.metadata else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create task from dictionary."""
        metadata = data.get("metadata")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = None
        return cls(
            task_id=data["task_id"],
            task_type=data["task_type"],
            status=TaskStatus(data["status"]),
            agent=data["agent"],
            notes=data.get("notes"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=metadata,
        )


class TaskMemory:
    """Persistent memory for agent tasks."""

    def __init__(
        self, memory_file: Optional[Path] = None, backend: str = MEMORY_BACKEND
    ):
        """Initialize task memory.

        Args:
            memory_file: Path to memory file (defaults to MEMORY_DIR/task_memory)
            backend: Storage backend ('json' or 'sqlite')
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.backend = backend
        self.memory_file = memory_file or MEMORY_DIR / "task_memory"
        self.tasks: Dict[str, Task] = {}
        # Initialize storage
        if self.backend == "sqlite":
            self._init_sqlite()
        else:
            self._init_json()
        self._load_memory()
        self.init_status = {
            "success": True,
            "message": "TaskMemory initialized successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def _init_sqlite(self) -> dict:
        """Initialize SQLite database."""
        try:
            db_path = self.memory_file.with_suffix(".db")
            self.conn = sqlite3.connect(str(db_path))
            self.cursor = self.conn.cursor()
            # Create tasks table
            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    agent TEXT NOT NULL,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT
                )
            """
            )
            self.conn.commit()
            return {
                "success": True,
                "message": "SQLite database initialized",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _init_json(self) -> dict:
        """Initialize JSON storage."""
        try:
            self.memory_file = self.memory_file.with_suffix(".json")
            if not self.memory_file.exists():
                self._save_memory()
            return {
                "success": True,
                "message": "JSON storage initialized",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _load_memory(self) -> dict:
        """Load tasks from storage."""
        try:
            if self.backend == "sqlite":
                self._load_from_sqlite()
            else:
                self._load_from_json()
            self.logger.info(
                f"Loaded {len(self.tasks)} tasks from {self.backend} storage"
            )
            return {
                "success": True,
                "message": f"Loaded {len(self.tasks)} tasks",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error loading memory: {e}")
            self.tasks = {}
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _load_from_sqlite(self) -> dict:
        """Load tasks from SQLite database."""
        try:
            self.cursor.execute("SELECT * FROM tasks")
            rows = self.cursor.fetchall()
            for row in rows:
                task_data = {
                    "task_id": row[0],
                    "task_type": row[1],
                    "status": row[2],
                    "agent": row[3],
                    "notes": row[4],
                    "created_at": row[5],
                    "updated_at": row[6],
                    "metadata": row[7],
                }
                task = Task.from_dict(task_data)
                self.tasks[task.task_id] = task
            return {
                "success": True,
                "message": f"Loaded {len(rows)} tasks from SQLite",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _load_from_json(self) -> dict:
        """Load tasks from JSON file."""
        try:
            if not self.memory_file.exists():
                return {
                    "success": True,
                    "message": "No JSON file exists, starting fresh",
                    "timestamp": datetime.now().isoformat(),
                }
            with open(self.memory_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.tasks = {
                task_id: Task.from_dict(task_data)
                for task_id, task_data in data.items()
            }
            return {
                "success": True,
                "message": f"Loaded {len(self.tasks)} tasks from JSON",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _save_memory(self) -> dict:
        """Save tasks to storage."""
        try:
            if self.backend == "sqlite":
                self._save_to_sqlite()
            else:
                self._save_to_json()
            self.logger.debug(
                f"Saved {len(self.tasks)} tasks to {self.backend} storage"
            )
            return {
                "success": True,
                "message": f"Saved {len(self.tasks)} tasks",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error saving memory: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def add_task(self, task: Task) -> dict:
        """Add a new task to memory."""
        try:
            self.tasks[task.task_id] = task
            self._save_memory()
            return {
                "success": True,
                "message": f"Task {task.task_id} added",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def update_task(self, task_id: str, **kwargs) -> dict:
        """Update an existing task."""
        try:
            if task_id not in self.tasks:
                return {
                    "success": False,
                    "error": f"Task {task_id} not found",
                    "timestamp": datetime.now().isoformat(),
                }
            task = self.tasks[task_id]
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            task.updated_at = datetime.now()
            self._save_memory()
            return {
                "success": True,
                "message": f"Task {task_id} updated",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def get_task(self, task_id: str) -> dict:
        """Get a task by ID."""
        try:
            task = self.tasks.get(task_id)
            if task:
                return {
                    "success": True,
                    "result": task,
                    "message": f"Task {task_id} retrieved",
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                return {
                    "success": False,
                    "error": f"Task {task_id} not found",
                    "timestamp": datetime.now().isoformat(),
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def get_tasks_by_status(self, status: Union[TaskStatus, List[TaskStatus]]) -> dict:
        """Get tasks by status."""
        try:
            if isinstance(status, TaskStatus):
                status_list = [status]
            else:
                status_list = status
            filtered_tasks = [
                task for task in self.tasks.values() if task.status in status_list
            ]
            return {
                "success": True,
                "result": filtered_tasks,
                "message": f"Found {len(filtered_tasks)} tasks with status {status}",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def get_tasks_by_agent(self, agent: str) -> dict:
        """Get tasks by agent."""
        try:
            filtered_tasks = [
                task for task in self.tasks.values() if task.agent == agent
            ]
            return {
                "success": True,
                "result": filtered_tasks,
                "message": f"Found {len(filtered_tasks)} tasks for agent {agent}",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def get_tasks_by_type(self, task_type: str) -> dict:
        """Get tasks by type."""
        try:
            filtered_tasks = [
                task for task in self.tasks.values() if task.task_type == task_type
            ]
            return {
                "success": True,
                "result": filtered_tasks,
                "message": f"Found {len(filtered_tasks)} tasks of type {task_type}",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def delete_task(self, task_id: str) -> dict:
        """Delete a task."""
        try:
            if task_id not in self.tasks:
                return {
                    "success": False,
                    "error": f"Task {task_id} not found",
                    "timestamp": datetime.now().isoformat(),
                }
            del self.tasks[task_id]
            self._save_memory()
            return {
                "success": True,
                "message": f"Task {task_id} deleted",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def clear_completed_tasks(self, older_than_days: int = 30) -> dict:
        """Clear completed tasks older than specified days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=older_than_days)
            tasks_to_remove = []
            for task_id, task in self.tasks.items():
                if (
                    task.status == TaskStatus.COMPLETED
                    and task.updated_at < cutoff_date
                ):
                    tasks_to_remove.append(task_id)
            for task_id in tasks_to_remove:
                del self.tasks[task_id]
            self._save_memory()
            return {
                "success": True,
                "message": f"Cleared {len(tasks_to_remove)} completed tasks",
                "timestamp": datetime.now().isoformat(),
                "cleared_count": len(tasks_to_remove),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def __del__(self):
        """Cleanup method."""
        try:
            if hasattr(self, "conn") and self.conn:
                self.conn.close()
            return {
                "success": True,
                "message": "TaskMemory cleanup completed",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Test task memory
    memory = TaskMemory()

    # Create test task
    task = Task(
        task_id="test_001",
        task_type="forecast",
        status=TaskStatus.PENDING,
        agent="forecaster",
        notes="Test task",
    )

    # Add task
    result = memory.add_task(task)
    memory.logger.info(f"Add task result: {result}")

    # Update task
    result = memory.update_task("test_001", status=TaskStatus.COMPLETED)
    memory.logger.info(f"Update task result: {result}")

    # Get tasks
    pending_result = memory.get_tasks_by_status(TaskStatus.PENDING)
    completed_result = memory.get_tasks_by_status(TaskStatus.COMPLETED)

    pending_tasks = (
        pending_result.get("result", []) if pending_result.get("success") else []
    )
    completed_tasks = (
        completed_result.get("result", []) if completed_result.get("success") else []
    )

    memory.logger.info(f"Pending tasks: {len(pending_tasks)}")
    memory.logger.info(f"Completed tasks: {len(completed_tasks)}")
