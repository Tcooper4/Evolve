"""
Base agent interface for the trading system.

This module provides the base classes and interfaces that all agents must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class Task:
    """Task data structure for agents."""

    task_id: str
    task_type: str
    status: str
    agent: str
    notes: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Result of an agent's execution."""

    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[Exception] = None


class BaseAgent(ABC):
    """Base class for all agents in the system."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the agent.

        Args:
            name: Unique identifier for the agent
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self._setup()

    def __str__(self) -> str:
        """
        String representation for easier logging/debugging of agent identity.

        Returns:
            str: Human-readable representation of the agent
        """
        config_summary = ""
        if self.config:
            # Include key config items for identification
            key_configs = ["model_type", "strategy", "risk_level", "enabled"]
            config_items = [f"{k}={v}" for k, v in self.config.items() if k in key_configs and v is not None]
            if config_items:
                config_summary = f" ({', '.join(config_items)})"

        return f"{self.__class__.__name__}[{self.name}]{config_summary}"

    def __repr__(self) -> str:
        """
        Detailed string representation for debugging.

        Returns:
            str: Detailed representation of the agent
        """
        return f"{self.__class__.__name__}(name='{self.name}', config={self.config})"

    def _setup(self):
        """Setup method to be overridden by subclasses."""

    @abstractmethod
    def run(self, prompt: str, **kwargs) -> AgentResult:
        """
        Execute the agent's main logic.

        Args:
            prompt: Input prompt or task description
            **kwargs: Additional arguments specific to the agent

        Returns:
            AgentResult: Result of the execution
        """

    def validate_input(self, prompt: str, **kwargs) -> bool:
        """
        Validate input parameters.

        Args:
            prompt: Input prompt to validate
            **kwargs: Additional parameters to validate

        Returns:
            bool: True if input is valid
        """
        return bool(prompt and isinstance(prompt, str))

    def handle_error(self, error: Exception) -> AgentResult:
        """
        Handle errors during execution.

        Args:
            error: Exception that occurred

        Returns:
            AgentResult: Error result
        """
        return AgentResult(success=False, message=f"Error in {self.name}: {str(error)}", error=error)

    def log_execution(self, result: AgentResult):
        """
        Log execution results.

        Args:
            result: Result to log
        """
        # Implementation can be overridden by subclasses


# For backward compatibility
BaseMetaAgent = BaseAgent

__all__ = ["BaseAgent", "BaseMetaAgent", "AgentResult", "Task"]
