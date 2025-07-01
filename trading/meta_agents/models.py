"""
Automation Models

This module implements models for automation-related data structures.

Note: This module was adapted from the legacy automation/models/automation.py file.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    """Task priority enumeration."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

class AlertSeverity(Enum):
    """Alert severity enumeration."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Task:
    """Task model."""
    id: str
    name: str
    description: str = ""
    type: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    scheduled_for: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None
    
    def update_status(self, status: TaskStatus) -> None:
        """Update task status."""
        self.status = status
        self.updated_at = datetime.now()

@dataclass
class Workflow:
    """Workflow model."""
    id: str
    name: str
    description: str = ""
    tasks: List[Task] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_task(self, task: Task) -> None:
        """Add a task to the workflow."""
        self.tasks.append(task)
        self.updated_at = datetime.now()

    def remove_task(self, task_id: str) -> None:
        """Remove a task from the workflow."""
        self.tasks = [t for t in self.tasks if t.id != task_id]
        self.updated_at = datetime.now()

    def update_status(self, status: TaskStatus) -> None:
        """Update workflow status."""
        self.status = status
        self.updated_at = datetime.now()

@dataclass
class Alert:
    """Alert model."""
    id: str
    title: str
    message: str
    severity: AlertSeverity = AlertSeverity.INFO
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Alert":
        """Create alert from dictionary."""
        return cls(
            id=data["id"],
            title=data["title"],
            message=data["message"],
            severity=AlertSeverity(data["severity"]),
            source=data.get("source", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            status=data.get("status", "active"),
            metadata=data.get("metadata", {})
        )

@dataclass
class Service:
    """Service model."""
    id: str
    name: str
    description: str = ""
    version: str = "1.0.0"
    status: str = "stopped"
    endpoints: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def update_status(self, status: str) -> None:
        """Update service status."""
        self.status = status
        self.updated_at = datetime.now()

    def add_endpoint(self, name: str, url: str) -> None:
        """Add an endpoint to the service."""
        self.endpoints[name] = url
        self.updated_at = datetime.now()

    def remove_endpoint(self, name: str) -> None:
        """Remove an endpoint from the service."""
        if name in self.endpoints:
            del self.endpoints[name]
            self.updated_at = datetime.now()

    def add_dependency(self, service_id: str) -> None:
        """Add a service dependency."""
        if service_id not in self.dependencies:
            self.dependencies.append(service_id)
            self.updated_at = datetime.now()

    def remove_dependency(self, service_id: str) -> None:
        """Remove a service dependency."""
        if service_id in self.dependencies:
            self.dependencies.remove(service_id)
            self.updated_at = datetime.now() 
