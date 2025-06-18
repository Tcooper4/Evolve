"""
Base agent interface for the financial forecasting system.

This module defines the core interface that all agents must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
from datetime import datetime

@dataclass
class AgentResult:
    """Result of an agent's execution."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = datetime.now()
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
        
    def _setup(self):
        """Setup method to be overridden by subclasses."""
        pass
        
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
        pass
        
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
        return AgentResult(
            success=False,
            message=f"Error in {self.name}: {str(error)}",
            error=error
        )
        
    def log_execution(self, result: AgentResult):
        """
        Log execution results.
        
        Args:
            result: Result to log
        """
        # Implementation can be overridden by subclasses
        pass 