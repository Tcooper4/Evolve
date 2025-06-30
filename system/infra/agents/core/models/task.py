from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from pydantic import BaseModel, Field
import uuid

class TaskPriority(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

class TaskStatus(Enum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskType(Enum):
    SYSTEM = "system"
    DATA_PROCESSING = "data_processing"
    MODEL_TRAINING = "model_training"
    DEPLOYMENT = "deployment"
    MAINTENANCE = "maintenance"
    CUSTOM = "custom"

class Task(BaseModel):
    """Centralized task model for the automation system."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    type: TaskType
    priority: TaskPriority
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    scheduled_for: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    dependencies: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    timeout: int = 300  # 5 minutes
    metadata: Dict[str, Any] = Field(default_factory=dict)
    handler: Optional[Callable] = None
    handler_args: tuple = Field(default_factory=tuple)
    handler_kwargs: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def update_status(self, status: TaskStatus) -> None:
        """Update task status and timestamps."""
        self.status = status
        self.updated_at = datetime.now()
        if status == TaskStatus.COMPLETED:
            self.completed_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary format."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.type.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "scheduled_for": self.scheduled_for.isoformat() if self.scheduled_for else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "dependencies": self.dependencies,
            "parameters": self.parameters,
            "result": self.result,
            "error": self.error,
            "retries": self.retries,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary format."""
        # Convert string values to enums
        if isinstance(data.get("type"), str):
            data["type"] = TaskType(data["type"])
        if isinstance(data.get("priority"), (int, str)):
            data["priority"] = TaskPriority(int(data["priority"]))
        if isinstance(data.get("status"), str):
            data["status"] = TaskStatus(data["status"])
        
        # Convert string timestamps to datetime
        for field in ["created_at", "updated_at", "scheduled_for", "completed_at"]:
            if isinstance(data.get(field), str):
                data[field] = datetime.fromisoformat(data[field])
        
        return cls(**data)