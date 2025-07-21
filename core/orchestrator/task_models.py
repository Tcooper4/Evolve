"""
Task Models Module

This module contains task-related data classes and models for the task orchestrator.
Extracted from the original task_orchestrator.py for modularity.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskStatus(Enum):
    """Task execution status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskType(Enum):
    """Task types"""

    MODEL_INNOVATION = "model_innovation"
    STRATEGY_RESEARCH = "strategy_research"
    SENTIMENT_FETCH = "sentiment_fetch"
    META_CONTROL = "meta_control"
    RISK_MANAGEMENT = "risk_management"
    EXECUTION = "execution"
    EXPLANATION = "explanation"
    SYSTEM_HEALTH = "system_health"
    DATA_SYNC = "data_sync"
    PERFORMANCE_ANALYSIS = "performance_analysis"


@dataclass
class TaskConfig:
    """Task configuration"""

    name: str
    task_type: TaskType
    enabled: bool = True
    interval_minutes: int = 60
    priority: TaskPriority = TaskPriority.MEDIUM
    max_duration_minutes: int = 30
    retry_count: int = 3
    retry_delay_minutes: int = 5
    dependencies: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_minutes: int = 15
    concurrent_execution: bool = False
    error_threshold: int = 5
    performance_threshold: float = 0.8
    retry_attempts: int = 3
    retry_delay: int = 60
    skip_on_failure: bool = False


@dataclass
class TaskExecution:
    """Task execution record"""

    task_id: str
    task_name: str
    task_type: TaskType
    start_time: str
    end_time: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None
    performance_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentStatus:
    """Agent status information"""

    agent_name: str
    is_running: bool = False
    last_execution: Optional[str] = None
    next_scheduled: Optional[str] = None
    success_count: int = 0
    failure_count: int = 0
    average_duration: float = 0.0
    health_score: float = 1.0
    error_history: List[str] = field(default_factory=list)
