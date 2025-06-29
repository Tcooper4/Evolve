# -*- coding: utf-8 -*-
"""Persistent memory for agent tasks with automatic persistence."""

# Standard library imports
import json
import logging
import sqlite3
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import os

# Third-party imports
from pydantic import BaseModel, Field

# Local imports
from trading.config.settings import MEMORY_DIR, MEMORY_BACKEND
from trading.utils.error_handling import handle_file_errors

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
            "metadata": json.dumps(self.metadata) if self.metadata else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
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
            metadata=metadata
        )

class TaskMemory:
    """Persistent memory for agent tasks."""
    
    def __init__(self, memory_file: Optional[Path] = None, backend: str = MEMORY_BACKEND):
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
    
    def _init_sqlite(self):
        """Initialize SQLite database."""
        db_path = self.memory_file.with_suffix('.db')
        self.conn = sqlite3.connect(str(db_path))
        self.cursor = self.conn.cursor()
        
        # Create tasks table
        self.cursor.execute("""
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
        """)
        self.conn.commit()
    
    def _init_json(self):
        """Initialize JSON storage."""
        self.memory_file = self.memory_file.with_suffix('.json')
        if not self.memory_file.exists():
            self._save_memory()
    
    def _load_memory(self) -> None:
        """Load tasks from storage."""
        try:
            if self.backend == "sqlite":
                self._load_from_sqlite()
            else:
                self._load_from_json()
                
            self.logger.info(f"Loaded {len(self.tasks)} tasks from {self.backend} storage")
            
        except Exception as e:
            self.logger.error(f"Error loading memory: {e}")
            self.tasks = {}
    
    def _load_from_sqlite(self) -> None:
        """Load tasks from SQLite database."""
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
                "metadata": row[7]
            }
            task = Task.from_dict(task_data)
            self.tasks[task.task_id] = task
    
    def _load_from_json(self) -> None:
        """Load tasks from JSON file."""
        if not self.memory_file.exists():
            return
            
        with open(self.memory_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.tasks = {
            task_id: Task.from_dict(task_data)
            for task_id, task_data in data.items()
        }
    
    def _save_memory(self) -> None:
        """Save tasks to storage."""
        try:
            if self.backend == "sqlite":
                self._save_to_sqlite()
            else:
                self._save_to_json()
                
            self.logger.debug(f"Saved {len(self.tasks)} tasks to {self.backend} storage")
            
        except Exception as e:
            self.logger.error(f"Error saving memory: {e}")
            raise
    
    def _save_to_sqlite(self) -> None:
        """Save tasks to SQLite database."""
        # Clear existing data
        self.cursor.execute("DELETE FROM tasks")
        
        # Insert new data
        for task in self.tasks.values():
            task_dict = task.to_dict()
            self.cursor.execute("""
                INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task_dict["task_id"],
                task_dict["task_type"],
                task_dict["status"],
                task_dict["agent"],
                task_dict["notes"],
                task_dict["created_at"],
                task_dict["updated_at"],
                task_dict["metadata"]
            ))
            
        self.conn.commit()
    
    def _save_to_json(self) -> None:
        """Save tasks to JSON file."""
        # Ensure directory exists
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert tasks to dictionary
        data = {
            task_id: task.to_dict()
            for task_id, task in self.tasks.items()
        }
        
        # Save to file
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def add_task(self, task: Task) -> None:
        """Add a new task to memory.
        
        Args:
            task: Task to add
        """
        self.tasks[task.task_id] = task
        self._save_memory()
        self.logger.info(f"Added task {task.task_id} ({task.task_type})")
    
    def update_task(self, task_id: str, **kwargs) -> Optional[Task]:
        """Update an existing task.
        
        Args:
            task_id: ID of task to update
            **kwargs: Fields to update
            
        Returns:
            Updated task or None if not found
        """
        if task_id not in self.tasks:
            self.logger.warning(f"Task {task_id} not found")
            return None
            
        task = self.tasks[task_id]
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(task, key):
                setattr(task, key, value)
        
        # Update timestamp
        task.updated_at = datetime.now()
        
        self._save_memory()
        self.logger.info(f"Updated task {task_id}")
        return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID.
        
        Args:
            task_id: ID of task to get
            
        Returns:
            Task or None if not found
        """
        return self.tasks.get(task_id)
    
    def get_tasks_by_status(self, status: Union[TaskStatus, List[TaskStatus]]) -> List[Task]:
        """Get tasks by status.
        
        Args:
            status: Status or list of statuses to filter by
            
        Returns:
            List of matching tasks
        """
        if isinstance(status, TaskStatus):
            status = [status]
            
        return [
            task for task in self.tasks.values()
            if task.status in status
        ]
    
    def get_tasks_by_agent(self, agent: str) -> List[Task]:
        """Get tasks by agent.
        
        Args:
            agent: Agent name to filter by
            
        Returns:
            List of matching tasks
        """
        return [
            task for task in self.tasks.values()
            if task.agent == agent
        ]
    
    def get_tasks_by_type(self, task_type: str) -> List[Task]:
        """Get tasks by type.
        
        Args:
            task_type: Task type to filter by
            
        Returns:
            List of matching tasks
        """
        return [
            task for task in self.tasks.values()
            if task.task_type == task_type
        ]
    
    def delete_task(self, task_id: str) -> bool:
        """Delete a task.
        
        Args:
            task_id: ID of task to delete
            
        Returns:
            True if task was deleted
        """
        if task_id not in self.tasks:
            return False
            
        del self.tasks[task_id]
        self._save_memory()
        self.logger.info(f"Deleted task {task_id}")
        return True
    
    def clear_completed_tasks(self, older_than_days: int = 30) -> int:
        """Clear completed tasks older than specified days.
        
        Args:
            older_than_days: Age threshold in days
            
        Returns:
            Number of tasks cleared
        """
        cutoff = datetime.now() - timedelta(days=older_than_days)
        to_delete = [
            task_id for task_id, task in self.tasks.items()
            if (task.status == TaskStatus.COMPLETED and 
                task.updated_at < cutoff)
        ]
        
        for task_id in to_delete:
            del self.tasks[task_id]
            
        if to_delete:
            self._save_memory()
            self.logger.info(f"Cleared {len(to_delete)} completed tasks")
            
        return len(to_delete)
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'conn'):
            self.conn.close()

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
        notes="Test task"
    )
    
    # Add task
    memory.add_task(task)
    
    # Update task
    memory.update_task("test_001", status=TaskStatus.COMPLETED)
    
    # Get tasks
    pending_tasks = memory.get_tasks_by_status(TaskStatus.PENDING)
    completed_tasks = memory.get_tasks_by_status(TaskStatus.COMPLETED)
    
    print(f"Pending tasks: {len(pending_tasks)}")
    print(f"Completed tasks: {len(completed_tasks)}") 