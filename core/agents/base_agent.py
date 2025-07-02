"""
Base agent interface for the financial forecasting system.

This module defines the core interface that all agents must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Union
from datetime import datetime
import logging
import uuid

logger = logging.getLogger(__name__)

@dataclass
class AgentResult:
    """Result of an agent's execution."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[Exception] = None
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Optional[Dict[str, Any]] = None

class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the agent.
        
        Args:
            name: Unique identifier for the agent
            config: Optional configuration dictionary
        """
        self.name: str = name
        self.config: Dict[str, Any] = config or {}
        self.execution_count: int = 0
        self.last_execution: Optional[datetime] = None
        self.execution_history: List[AgentResult] = []
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        self._setup()
    
    def _setup(self) -> None:
        """Setup method to be overridden by subclasses."""
        self.logger.info(f"Initializing agent: {self.name}")
    
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
        if not prompt or not isinstance(prompt, str):
            self.logger.warning("Invalid prompt: must be non-empty string")
            return False
        
        if len(prompt.strip()) == 0:
            self.logger.warning("Invalid prompt: cannot be empty or whitespace")
            return False
            
        return True
        
    def handle_error(self, error: Exception) -> AgentResult:
        """
        Handle errors during execution.
        
        Args:
            error: Exception that occurred
            
        Returns:
            AgentResult: Error result
        """
        self.logger.error(f"Error in {self.name}: {error}")
        return AgentResult(
            success=False,
            message=f"Error in {self.name}: {str(error)}",
            error=error,
            metadata={"error_type": type(error).__name__}
        )
        
    def log_execution(self, result: AgentResult) -> None:
        """
        Log execution results.
        
        Args:
            result: Result to log
        """
        self.execution_count += 1
        self.last_execution = datetime.now()
        self.execution_history.append(result)
        
        # Keep only last 100 executions
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
        
        if result.success:
            self.logger.info(f"Execution successful: {result.message}")
        else:
            self.logger.error(f"Execution failed: {result.message}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get agent status information.
        
        Returns:
            Dictionary containing agent status
        """
        return {
            "name": self.name,
            "execution_count": self.execution_count,
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
            "success_rate": self._calculate_success_rate(),
            "config": self.config,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_success_rate(self) -> float:
        """
        Calculate success rate from execution history.
        
        Returns:
            Success rate as a float between 0 and 1
        """
        if not self.execution_history:
            return 0.0
        
        successful_executions = sum(1 for result in self.execution_history if result.success)
        return successful_executions / len(self.execution_history)
    
    def reset_history(self) -> None:
        """Reset execution history."""
        self.execution_history.clear()
        self.execution_count = 0
        self.last_execution = None
        self.logger.info(f"Reset execution history for agent: {self.name}")
    
    def get_recent_executions(self, limit: int = 10) -> List[AgentResult]:
        """
        Get recent execution results.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of recent AgentResult objects
        """
        return self.execution_history[-limit:]
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update agent configuration.
        
        Args:
            new_config: New configuration dictionary
        """
        self.config.update(new_config)
        self.logger.info(f"Updated configuration for agent: {self.name}")
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(name='{self.name}', executions={self.execution_count})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the agent."""
        return f"{self.__class__.__name__}(name='{self.name}', config={self.config}, executions={self.execution_count})"