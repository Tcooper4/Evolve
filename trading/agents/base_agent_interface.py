"""
Base Agent Interface

This module defines the base interface for all pluggable agents in the system.
All agents must implement this interface to be compatible with the agent manager.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import logging

@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    enabled: bool = True
    priority: int = 1
    max_concurrent_runs: int = 1
    timeout_seconds: int = 300
    retry_attempts: int = 3
    custom_config: Optional[Dict[str, Any]] = None

@dataclass
class AgentStatus:
    """Status information for an agent."""
    name: str
    enabled: bool
    is_running: bool
    last_run: Optional[datetime] = None
    last_success: Optional[datetime] = None
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    current_error: Optional[str] = None

@dataclass
class AgentResult:
    """Result from agent execution."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class BaseAgent(ABC):
    """Base interface for all pluggable agents."""
    
    def __init__(self, config: AgentConfig):
        """Initialize the agent with configuration.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
        self.status = AgentStatus(
            name=config.name,
            enabled=config.enabled,
            is_running=False
        )
        self._setup()

    def _setup(self) -> None:
        """Setup method called during initialization.
        
        Override this method to perform any agent-specific setup.
        """
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> AgentResult:
        """Execute the agent's main logic.
        
        Args:
            **kwargs: Agent-specific parameters
            
        Returns:
            AgentResult: Result of the execution
        """
        pass
    
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
    
    def get_status(self) -> AgentStatus:
        """Get the current status of the agent."""
        return self.status
    
    def get_config(self) -> AgentConfig:
        """Get the agent configuration."""
        return self.config
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update agent configuration.
        
        Args:
            new_config: New configuration values
        """
        for key, value in new_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            elif self.config.custom_config is not None:
                self.config.custom_config[key] = value
        
        self.logger.info(f"Updated configuration for agent {self.config.name}")

    def validate_input(self, **kwargs) -> bool:
        """Validate input parameters.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            bool: True if input is valid
        """
        return True
    
    def handle_error(self, error: Exception) -> AgentResult:
        """Handle errors during execution.
        
        Args:
            error: Exception that occurred
            
        Returns:
            AgentResult: Error result
        """
        self.status.failed_runs += 1
        self.status.current_error = str(error)
        self.status.is_running = False
        
        return AgentResult(
            success=False,
            error_message=str(error),
            timestamp=datetime.now()
        )
    
    def _update_status_on_start(self) -> None:
        """Update status when execution starts."""
        self.status.is_running = True
        self.status.last_run = datetime.now()
        self.status.total_runs += 1
        self.status.current_error = None

    def _update_status_on_success(self, execution_time: float) -> None:
        """Update status when execution succeeds."""
        self.status.is_running = False
        self.status.last_success = datetime.now()
        self.status.successful_runs += 1
        self.status.current_error = None

    def _update_status_on_failure(self, error: str) -> None:
        """Update status when execution fails."""
        self.status.is_running = False
        self.status.failed_runs += 1
        self.status.current_error = error

    async def run(self, **kwargs) -> AgentResult:
        """Run the agent with error handling and status updates.
        
        Args:
            **kwargs: Agent-specific parameters
            
        Returns:
            AgentResult: Result of the execution
        """
        if not self.is_enabled():
            return AgentResult(
                success=False,
                error_message=f"Agent {self.config.name} is disabled",
                timestamp=datetime.now()
            )
        
        if not self.validate_input(**kwargs):
            return AgentResult(
                success=False,
                error_message=f"Invalid input for agent {self.config.name}",
                timestamp=datetime.now()
            )
        
        self._update_status_on_start()
        start_time = datetime.now()
        
        try:
            result = await self.execute(**kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            
            if result.success:
                self._update_status_on_success(execution_time)
            else:
                self._update_status_on_failure(result.error_message or "Unknown error")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_status_on_failure(str(e))
            return self.handle_error(e)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get agent metadata for registration.
        
        Returns:
            Dict containing agent metadata
        """
        return {
            "name": self.config.name,
            "version": getattr(self, 'version', '1.0.0'),
            "description": getattr(self, 'description', ''),
            "author": getattr(self, 'author', ''),
            "tags": getattr(self, 'tags', []),
            "capabilities": getattr(self, 'capabilities', []),
            "dependencies": getattr(self, 'dependencies', [])
        } 