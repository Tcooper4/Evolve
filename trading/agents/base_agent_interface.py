"""
Base Agent Interface

This module defines the base interface for all pluggable agents in the system.
All agents must implement this interface to be compatible with the agent manager.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import asyncio
import json
import traceback
from enum import Enum
from pathlib import Path

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
            'name': self.name,
            'enabled': self.enabled,
            'priority': self.priority.value,
            'max_concurrent_runs': self.max_concurrent_runs,
            'timeout_seconds': self.timeout_seconds,
            'retry_attempts': self.retry_attempts,
            'retry_delay_seconds': self.retry_delay_seconds,
            'custom_config': self.custom_config,
            'tags': self.tags,
            'description': self.description,
            'version': self.version,
            'author': self.author
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfig':
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
            'name': self.name,
            'enabled': self.enabled,
            'state': self.state.value,
            'is_running': self.is_running,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'last_success': self.last_success.isoformat() if self.last_success else None,
            'last_failure': self.last_failure.isoformat() if self.last_failure else None,
            'total_runs': self.total_runs,
            'successful_runs': self.successful_runs,
            'failed_runs': self.failed_runs,
            'current_error': self.current_error,
            'current_run_start': self.current_run_start.isoformat() if self.current_run_start else None,
            'average_execution_time': self.average_execution_time,
            'total_execution_time': self.total_execution_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentStatus':
        """Create from dictionary."""
        # Convert string enums back to enum objects
        if isinstance(data['state'], str):
            data['state'] = AgentState(data['state'])
        
        # Convert datetime strings back to datetime objects
        for field in ['last_run', 'last_success', 'last_failure', 'current_run_start']:
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
            'success': self.success,
            'data': self.data,
            'error_message': self.error_message,
            'error_type': self.error_type,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentResult':
        """Create from dictionary."""
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
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
            name=config.name,
            enabled=config.enabled,
            state=AgentState.IDLE
        )
        
        # Performance tracking
        self.execution_history: List[AgentResult] = []
        self.max_history_size = 100
        
        # Setup the agent
        self._setup()
        
        self.logger.info(f"Agent {config.name} initialized successfully")

    def _setup(self) -> None:
        """
        Setup method called during initialization.
        
        Override this method to perform any agent-specific setup.
        """
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> AgentResult:
        """
        Execute the agent's main logic.
        
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
        self.status.state = AgentState.IDLE
        self.logger.info(f"Agent {self.config.name} enabled")

    def disable(self) -> None:
        """Disable the agent."""
        self.config.enabled = False
        self.status.enabled = False
        self.status.state = AgentState.DISABLED
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
        """Get the agent configuration."""
        return self.config
    
    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """
        Update agent configuration.
        
        Args:
            new_config: New configuration values
            
        Returns:
            bool: True if update was successful
        """
        try:
            for key, value in new_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                elif self.config.custom_config is not None:
                    self.config.custom_config[key] = value
                else:
                    self.config.custom_config = {key: value}
            
            self.logger.info(f"Updated configuration for agent {self.config.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            return False

    def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            bool: True if input is valid
        """
        return True
    
    def validate_config(self) -> bool:
        """
        Validate agent configuration.
        
        Returns:
            bool: True if configuration is valid
        """
        try:
            if not self.config.name:
                return False
            
            if self.config.timeout_seconds <= 0:
                return False
            
            if self.config.max_concurrent_runs <= 0:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            return False
    
    def handle_error(self, error: Exception) -> AgentResult:
        """
        Handle errors during execution.
        
        Args:
            error: Exception that occurred
            
        Returns:
            AgentResult: Error result
        """
        error_type = type(error).__name__
        error_message = str(error)
        error_traceback = traceback.format_exc()
        
        self.logger.error(f"Agent {self.config.name} error: {error_message}")
        self.logger.debug(f"Error traceback: {error_traceback}")
        
        self.status.failed_runs += 1
        self.status.current_error = error_message
        self.status.is_running = False
        self.status.state = AgentState.ERROR
        self.status.last_failure = datetime.now()
        
        return AgentResult(
            success=False,
            error_message=error_message,
            error_type=error_type,
            metadata={'traceback': error_traceback}
        )
    
    def _update_status_on_start(self) -> None:
        """Update status when execution starts."""
        self.status.is_running = True
        self.status.state = AgentState.RUNNING
        self.status.last_run = datetime.now()
        self.status.current_run_start = datetime.now()
        self.status.total_runs += 1
        self.status.current_error = None

    def _update_status_on_success(self, execution_time: float) -> None:
        """Update status when execution succeeds."""
        self.status.is_running = False
        self.status.state = AgentState.SUCCESS
        self.status.last_success = datetime.now()
        self.status.successful_runs += 1
        self.status.current_error = None
        self.status.current_run_start = None
        
        # Update execution time statistics
        self.status.total_execution_time += execution_time
        if self.status.total_runs > 0:
            self.status.average_execution_time = self.status.total_execution_time / self.status.total_runs

    def _update_status_on_failure(self, error: str, execution_time: float = 0.0) -> None:
        """Update status when execution fails."""
        self.status.is_running = False
        self.status.state = AgentState.FAILED
        self.status.failed_runs += 1
        self.status.current_error = error
        self.status.current_run_start = None
        self.status.last_failure = datetime.now()
        
        # Update execution time statistics
        if execution_time > 0:
            self.status.total_execution_time += execution_time
            if self.status.total_runs > 0:
                self.status.average_execution_time = self.status.total_execution_time / self.status.total_runs

    async def run(self, **kwargs) -> AgentResult:
        """
        Run the agent with error handling and status updates.
        
        Args:
            **kwargs: Agent-specific parameters
            
        Returns:
            AgentResult: Result of the execution
        """
        # Check if agent is enabled
        if not self.is_enabled():
            return AgentResult(
                success=False,
                error_message=f"Agent {self.config.name} is disabled",
                error_type="AgentDisabled"
            )
        
        # Check if agent is already running
        if self.is_running():
            return AgentResult(
                success=False,
                error_message=f"Agent {self.config.name} is already running",
                error_type="AgentAlreadyRunning"
            )
        
        # Validate configuration
        if not self.validate_config():
            return AgentResult(
                success=False,
                error_message=f"Invalid configuration for agent {self.config.name}",
                error_type="InvalidConfiguration"
            )
        
        # Validate input
        if not self.validate_input(**kwargs):
            return AgentResult(
                success=False,
                error_message=f"Invalid input for agent {self.config.name}",
                error_type="InvalidInput"
            )
        
        # Start execution
        self._update_status_on_start()
        start_time = datetime.now()
        
        try:
            # Execute with timeout
            if self.config.timeout_seconds > 0:
                result = await asyncio.wait_for(
                    self.execute(**kwargs),
                    timeout=self.config.timeout_seconds
                )
            else:
                result = await self.execute(**kwargs)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            
            if result.success:
                self._update_status_on_success(execution_time)
            else:
                self._update_status_on_failure(result.error_message or "Unknown error", execution_time)
            
            # Add to history
            self._add_to_history(result)
            
            return result
            
        except asyncio.TimeoutError:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Agent {self.config.name} execution timed out after {self.config.timeout_seconds} seconds"
            self._update_status_on_failure(error_msg, execution_time)
            
            result = AgentResult(
                success=False,
                error_message=error_msg,
                error_type="TimeoutError",
                execution_time=execution_time
            )
            self._add_to_history(result)
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            result = self.handle_error(e)
            result.execution_time = execution_time
            self._add_to_history(result)
            return result
    
    def _add_to_history(self, result: AgentResult) -> None:
        """Add result to execution history."""
        self.execution_history.append(result)
        
        # Keep only the last N results
        if len(self.execution_history) > self.max_history_size:
            self.execution_history = self.execution_history[-self.max_history_size:]
    
    def get_execution_history(self, limit: Optional[int] = None) -> List[AgentResult]:
        """
        Get execution history.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of execution results
        """
        history = self.execution_history.copy()
        if limit:
            history = history[-limit:]
        return history
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        if not self.execution_history:
            return {
                'total_executions': 0,
                'success_rate': 0.0,
                'average_execution_time': 0.0,
                'min_execution_time': 0.0,
                'max_execution_time': 0.0,
                'total_execution_time': 0.0
            }
        
        successful_runs = [r for r in self.execution_history if r.success]
        execution_times = [r.execution_time for r in self.execution_history if r.execution_time > 0]
        
        return {
            'total_executions': len(self.execution_history),
            'successful_executions': len(successful_runs),
            'failed_executions': len(self.execution_history) - len(successful_runs),
            'success_rate': len(successful_runs) / len(self.execution_history) if self.execution_history else 0.0,
            'average_execution_time': sum(execution_times) / len(execution_times) if execution_times else 0.0,
            'min_execution_time': min(execution_times) if execution_times else 0.0,
            'max_execution_time': max(execution_times) if execution_times else 0.0,
            'total_execution_time': sum(execution_times),
            'last_execution': self.execution_history[-1].timestamp.isoformat() if self.execution_history else None
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get agent metadata for registration.
        
        Returns:
            Dict containing agent metadata
        """
        return {
            "name": self.config.name,
            "version": getattr(self, '__version__', self.config.version),
            "description": getattr(self, '__description__', self.config.description),
            "author": getattr(self, '__author__', self.config.author),
            "tags": getattr(self, '__tags__', self.config.tags),
            "capabilities": getattr(self, '__capabilities__', []),
            "dependencies": getattr(self, '__dependencies__', []),
            "config_schema": getattr(self, '__config_schema__', None),
            "priority": self.config.priority.value,
            "enabled": self.config.enabled
        }
    
    def save_state(self, filepath: str) -> bool:
        """
        Save agent state to file.
        
        Args:
            filepath: Path to save state
            
        Returns:
            bool: True if save was successful
        """
        try:
            state = {
                'config': self.config.to_dict(),
                'status': self.status.to_dict(),
                'execution_history': [r.to_dict() for r in self.execution_history],
                'performance_stats': self.get_performance_stats(),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.info(f"Agent state saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving agent state: {e}")
            return False
    
    def load_state(self, filepath: str) -> bool:
        """
        Load agent state from file.
        
        Args:
            filepath: Path to load state from
            
        Returns:
            bool: True if load was successful
        """
        try:
            if not Path(filepath).exists():
                self.logger.warning(f"State file {filepath} not found")
                return False
            
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Load configuration
            if 'config' in state:
                self.config = AgentConfig.from_dict(state['config'])
            
            # Load status
            if 'status' in state:
                self.status = AgentStatus.from_dict(state['status'])
            
            # Load execution history
            if 'execution_history' in state:
                self.execution_history = [
                    AgentResult.from_dict(r) for r in state['execution_history']
                ]
            
            self.logger.info(f"Agent state loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading agent state: {e}")
            return False
    
    def reset(self) -> None:
        """Reset agent state and statistics."""
        self.status = AgentStatus(
            name=self.config.name,
            enabled=self.config.enabled,
            state=AgentState.IDLE
        )
        self.execution_history.clear()
        self.logger.info(f"Agent {self.config.name} state reset")
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"Agent(name={self.config.name}, state={self.status.state.value}, enabled={self.config.enabled})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the agent."""
        return (f"Agent(name={self.config.name}, state={self.status.state.value}, "
                f"enabled={self.config.enabled}, total_runs={self.status.total_runs}, "
                f"success_rate={self.get_performance_stats()['success_rate']:.2%})") 