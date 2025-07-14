"""
Base Agent Interface

This module defines the base interface for all pluggable agents in the system.
All agents must implement this interface to be compatible with the agent manager.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class AgentPriority(Enum):
    """Agent priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class AgentState(Enum):
    """Agent execution states."""

    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    DISABLED = "disabled"
    ERROR = "error"


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    name: str
    enabled: bool = True
    priority: Union[int, AgentPriority] = AgentPriority.NORMAL
    max_concurrent_runs: int = 1
    timeout_seconds: int = 300
    retry_attempts: int = 3
    retry_delay_seconds: int = 60
    custom_config: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)
    description: str = ""
    version: str = "1.0.0"
    author: str = "Unknown"

    def __post_init__(self):
        """Convert priority to enum if needed."""
        if isinstance(self.priority, int):
            self.priority = AgentPriority(self.priority)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "priority": self.priority.value,
            "max_concurrent_runs": self.max_concurrent_runs,
            "timeout_seconds": self.timeout_seconds,
            "retry_attempts": self.retry_attempts,
            "retry_delay_seconds": self.retry_delay_seconds,
            "custom_config": self.custom_config,
            "tags": self.tags,
            "description": self.description,
            "version": self.version,
            "author": self.author,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class AgentStatus:
    """Status information for an agent."""

    name: str
    enabled: bool
    state: AgentState = AgentState.IDLE
    is_running: bool = False
    last_run: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    current_error: Optional[str] = None
    current_run_start: Optional[datetime] = None
    average_execution_time: float = 0.0
    total_execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "state": self.state.value,
            "is_running": self.is_running,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "last_success": self.last_success.isoformat()
            if self.last_success
            else None,
            "last_failure": self.last_failure.isoformat()
            if self.last_failure
            else None,
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "current_error": self.current_error,
            "current_run_start": self.current_run_start.isoformat()
            if self.current_run_start
            else None,
            "average_execution_time": self.average_execution_time,
            "total_execution_time": self.total_execution_time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentStatus":
        """Create from dictionary."""
        # Convert string enums back to enum objects
        if isinstance(data["state"], str):
            data["state"] = AgentState(data["state"])

        # Convert datetime strings back to datetime objects
        for field in ["last_run", "last_success", "last_failure", "current_run_start"]:
            if data.get(field) and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])

        return cls(**data)


@dataclass
class AgentResult:
    """Result from agent execution."""

    success: bool
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "data": self.data,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentResult":
        """Create from dictionary."""
        if isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class BaseAgent(ABC):
    """Base interface for all pluggable agents."""

    # Class-level metadata for registration
    __version__ = "1.0.0"
    __author__ = "Unknown"
    __description__ = "Base agent interface"
    __tags__ = []
    __capabilities__ = []
    __dependencies__ = []
    __config_schema__ = None

    def __init__(self, config: AgentConfig):
        """
        Initialize the agent with configuration.

        Args:
            config: Agent configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
        self.status = AgentStatus(
            name=config.name, enabled=config.enabled, state=AgentState.IDLE
        )

        # Performance tracking
        self.execution_history: List[AgentResult] = []
        self.max_history_size = 100

        # Setup the agent
        self._setup()

        self.logger.info(f"Agent {config.name} initialized successfully")

    @abstractmethod
    async def execute(self, **kwargs) -> AgentResult:
        """
        Execute the agent's main logic.

        This is the primary method that all agents must implement.
        It should contain the core business logic of the agent.

        Args:
            **kwargs: Arguments specific to the agent's functionality

        Returns:
            AgentResult: Result of the execution
        """

    @abstractmethod
    def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters before execution.

        This method should validate all input parameters to ensure
        they meet the agent's requirements before execution begins.

        Args:
            **kwargs: Input parameters to validate

        Returns:
            bool: True if input is valid, False otherwise
        """

    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate the agent's configuration.

        This method should validate the agent's configuration to ensure
        all required settings are present and valid.

        Returns:
            bool: True if configuration is valid, False otherwise
        """

    @abstractmethod
    def handle_error(self, error: Exception) -> AgentResult:
        """
        Handle errors during execution.

        This method should provide consistent error handling for all agents.
        It should log the error and return an appropriate AgentResult.

        Args:
            error: Exception that occurred during execution

        Returns:
            AgentResult: Error result with appropriate error information
        """

    @abstractmethod
    def _setup(self) -> None:
        """
        Setup method called during initialization.

        This method should perform any necessary setup operations
        such as loading models, connecting to services, etc.

        Subclasses must implement this method to ensure proper initialization.
        """

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Get the agent's capabilities.

        This method should return a list of capabilities that the agent provides.
        These capabilities are used for agent discovery and routing.

        Returns:
            List[str]: List of capability names
        """

    @abstractmethod
    def get_requirements(self) -> Dict[str, Any]:
        """
        Get the agent's requirements.

        This method should return a dictionary describing the agent's requirements
        such as dependencies, system requirements, etc.

        Returns:
            Dict[str, Any]: Dictionary of requirements
        """

    def enable(self) -> None:
        """Enable the agent."""
        self.config.enabled = True
        self.status.enabled = True
        self.logger.info(f"Agent {self.config.name} enabled")

    def disable(self) -> None:
        """Disable the agent."""
        self.config.enabled = False
        self.status.enabled = False
        self.logger.info(f"Agent {self.config.name} disabled")

    def is_enabled(self) -> bool:
        """Check if the agent is enabled."""
        return self.config.enabled and self.status.enabled

    def is_running(self) -> bool:
        """Check if the agent is currently running."""
        return self.status.is_running

    def get_status(self) -> AgentStatus:
        """Get the current status of the agent."""
        return self.status

    def get_config(self) -> AgentConfig:
        """Get the agent's configuration."""
        return self.config

    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """
        Update the agent's configuration.

        Args:
            new_config: New configuration values

        Returns:
            bool: True if update was successful
        """
        try:
            # Validate new configuration
            if not self._validate_new_config(new_config):
                return False

            # Update configuration
            for key, value in new_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                elif key == "custom_config":
                    if self.config.custom_config is None:
                        self.config.custom_config = {}
                    self.config.custom_config.update(value)

            # Re-validate configuration
            if not self.validate_config():
                self.logger.error("Configuration validation failed after update")
                return False

            self.logger.info(f"Updated configuration for agent {self.config.name}")
            return True

        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            return False

    def _validate_new_config(self, new_config: Dict[str, Any]) -> bool:
        """
        Validate new configuration values.

        Args:
            new_config: New configuration to validate

        Returns:
            bool: True if configuration is valid
        """
        try:
            # Check for invalid keys
            valid_keys = {
                "enabled",
                "priority",
                "max_concurrent_runs",
                "timeout_seconds",
                "retry_attempts",
                "retry_delay_seconds",
                "custom_config",
                "tags",
                "description",
                "version",
                "author",
            }

            for key in new_config.keys():
                if key not in valid_keys:
                    self.logger.warning(f"Invalid configuration key: {key}")
                    return False

            # Validate specific fields
            if "max_concurrent_runs" in new_config:
                if (
                    not isinstance(new_config["max_concurrent_runs"], int)
                    or new_config["max_concurrent_runs"] < 1
                ):
                    self.logger.error("max_concurrent_runs must be a positive integer")
                    return False

            if "timeout_seconds" in new_config:
                if (
                    not isinstance(new_config["timeout_seconds"], int)
                    or new_config["timeout_seconds"] < 1
                ):
                    self.logger.error("timeout_seconds must be a positive integer")
                    return False

            if "retry_attempts" in new_config:
                if (
                    not isinstance(new_config["retry_attempts"], int)
                    or new_config["retry_attempts"] < 0
                ):
                    self.logger.error("retry_attempts must be a non-negative integer")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating new configuration: {e}")
            return False
