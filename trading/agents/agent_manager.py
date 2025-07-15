"""
Enhanced Agent Manager with Batch 12 Features

This module manages all pluggable agents in the system, providing dynamic
enable/disable functionality, agent registration, and execution coordination.

Enhanced with Batch 12 features:
- Comprehensive try/except with retry/backoff logic for each agent task
- Detailed logging of all failures with full traceback
- Exponential backoff with jitter for retry attempts
- Graceful degradation and fallback mechanisms
- Enhanced error tracking and recovery
"""

import asyncio
import json
import logging
import random
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional, Type, Union
import time

from trading.agents.agent_leaderboard import AgentLeaderboard

# Local imports
from trading.agents.base_agent_interface import (
    AgentConfig,
    AgentResult,
    AgentStatus,
    BaseAgent,
)
from trading.agents.execution.execution_agent import ExecutionAgent
from trading.agents.model_builder_agent import ModelBuilderAgent
from trading.agents.performance_critic_agent import PerformanceCriticAgent
from trading.agents.updater_agent import UpdaterAgent


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    retry_on_exceptions: tuple = (Exception,)
    exponential_backoff: bool = True


@dataclass
class AgentTask:
    """Agent task with retry configuration."""
    
    agent_name: str
    task_id: str
    kwargs: Dict[str, Any]
    retry_config: RetryConfig
    created_at: datetime
    attempts: int = 0
    last_attempt: Optional[datetime] = None
    next_retry: Optional[datetime] = None
    error_history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.error_history is None:
            self.error_history = []


@dataclass
class AgentManagementRequest:
    """Agent management request."""

    action: str  # 'register', 'unregister', 'enable', 'disable', 'execute', 'get_status', 'update_config'
    agent_name: Optional[str] = None
    agent_class: Optional[Type[BaseAgent]] = None
    config: Optional[AgentConfig] = None
    kwargs: Optional[Dict[str, Any]] = None
    retry_config: Optional[RetryConfig] = None


@dataclass
class AgentManagementResult:
    """Agent management result."""

    success: bool
    data: Dict[str, Any]
    error_message: Optional[str] = None
    retry_attempts: int = 0
    total_duration: float = 0.0


@dataclass
class AgentRegistryEntry:
    """Entry in the agent registry."""

    agent_class: Type[BaseAgent]
    config: AgentConfig
    instance: Optional[BaseAgent] = None
    metadata: Optional[Dict[str, Any]] = None
    retry_config: RetryConfig = None
    
    def __post_init__(self):
        if self.retry_config is None:
            self.retry_config = RetryConfig()


@dataclass
class AgentManagerConfig:
    """Configuration for the agent manager."""

    config_file: str = "trading/agents/agent_config.json"
    auto_start: bool = True
    max_concurrent_agents: int = 5
    execution_timeout: int = 300
    enable_logging: bool = True
    enable_metrics: bool = True
    default_retry_config: RetryConfig = None
    
    def __post_init__(self):
        if self.default_retry_config is None:
            self.default_retry_config = RetryConfig()


class EnhancedAgentManager:
    """Enhanced agent manager with comprehensive retry/backoff logic."""

    def __init__(self, config: Optional[AgentManagerConfig] = None):
        """Initialize the Enhanced Agent Manager.

        Args:
            config: Agent manager configuration
        """
        self.config = config or AgentManagerConfig()
        self.logger = logging.getLogger(__name__)

        # Agent registry
        self.agent_registry: Dict[str, AgentRegistryEntry] = {}

        # Execution queue and thread pool
        self.execution_queue = Queue()
        self.execution_thread = None
        self.running = False

        # Enhanced retry and error tracking
        self.pending_tasks: Dict[str, AgentTask] = {}
        self.completed_tasks: Dict[str, AgentTask] = {}
        self.failed_tasks: Dict[str, AgentTask] = {}
        self.retry_queue: Queue = Queue()
        
        # Error tracking
        self.error_log: List[Dict[str, Any]] = []
        self.agent_error_counts: Dict[str, int] = {}
        self.global_error_count = 0

        # Metrics and monitoring
        self.execution_history: List[Dict[str, Any]] = []
        self.agent_metrics: Dict[str, Dict[str, Any]] = {}

        # Agent health monitoring and restart logic
        self.agent_health: Dict[str, Dict[str, Any]] = {}
        self.agent_restart_count: Dict[str, int] = {}
        self.max_restart_attempts = self.config.max_concurrent_agents
        self.restart_delay = 30  # seconds
        self.health_check_interval = 60  # seconds
        self.agent_timeout_threshold = 300  # seconds
        
        # Callback system
        self.callbacks: Dict[str, List[callable]] = {
            'agent_started': [],
            'agent_completed': [],
            'agent_failed': [],
            'agent_restarted': [],
            'agent_timeout': [],
            'agent_retry': [],
            'agent_backoff': []
        }

        # Restart monitoring
        self._restart_monitor_running = False
        self._restart_monitor_thread = None
        self._retry_monitor_running = False
        self._retry_monitor_thread = None
        self._agent_locks: Dict[str, asyncio.Lock] = {}

        # Leaderboard
        self.leaderboard = AgentLeaderboard()

        # Load configuration and register default agents
        self._load_config()
        self._register_default_agents()

        self.logger.info("Enhanced AgentManager initialized")
        self.status = 'idle'

    def _calculate_backoff_delay(self, attempt: int, retry_config: RetryConfig) -> float:
        """Calculate backoff delay with jitter.
        
        Args:
            attempt: Current attempt number
            retry_config: Retry configuration
            
        Returns:
            Delay in seconds
        """
        if retry_config.exponential_backoff:
            delay = min(
                retry_config.base_delay * (retry_config.backoff_factor ** (attempt - 1)),
                retry_config.max_delay
            )
        else:
            delay = retry_config.base_delay
        
        # Add jitter if enabled
        if retry_config.jitter:
            jitter = random.uniform(0, delay * 0.1)
            delay += jitter
        
        return delay

    def _log_error_with_traceback(
        self, 
        agent_name: str, 
        error: Exception, 
        task_id: str,
        attempt: int,
        context: Dict[str, Any] = None
    ):
        """Log error with full traceback and context.
        
        Args:
            agent_name: Name of the agent
            error: Exception that occurred
            task_id: Task ID
            attempt: Attempt number
            context: Additional context
        """
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "task_id": task_id,
            "attempt": attempt,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {}
        }
        
        # Log to error log
        self.error_log.append(error_info)
        
        # Update error counts
        self.agent_error_counts[agent_name] = self.agent_error_counts.get(agent_name, 0) + 1
        self.global_error_count += 1
        
        # Log with appropriate level
        if attempt == 1:
            self.logger.error(
                f"Agent {agent_name} failed on first attempt: {error}",
                extra={"error_info": error_info}
            )
        else:
            self.logger.warning(
                f"Agent {agent_name} failed on attempt {attempt}: {error}",
                extra={"error_info": error_info}
            )
        
        # Trigger callbacks
        self._trigger_callbacks('agent_failed', agent_name=agent_name, error=error, task_id=task_id)

    async def execute_agent_with_retry(
        self, 
        name: str, 
        retry_config: Optional[RetryConfig] = None,
        **kwargs
    ) -> AgentResult:
        """Execute agent with comprehensive retry logic.
        
        Args:
            name: Agent name
            retry_config: Retry configuration
            **kwargs: Agent execution parameters
            
        Returns:
            AgentResult: Result of agent execution
        """
        if name not in self.agent_registry:
            return AgentResult(
                success=False,
                error_message=f"Agent '{name}' not found in registry",
                error_type="AgentNotFound"
            )
        
        # Get retry configuration
        if retry_config is None:
            retry_config = self.agent_registry[name].retry_config or self.config.default_retry_config
        
        # Create task
        task_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        task = AgentTask(
            agent_name=name,
            task_id=task_id,
            kwargs=kwargs,
            retry_config=retry_config,
            created_at=datetime.now()
        )
        
        self.pending_tasks[task_id] = task
        
        try:
            return await self._execute_with_retry(task)
        finally:
            # Clean up task
            if task_id in self.pending_tasks:
                del self.pending_tasks[task_id]

    async def _execute_with_retry(self, task: AgentTask) -> AgentResult:
        """Execute task with retry logic.
        
        Args:
            task: Agent task
            
        Returns:
            AgentResult: Final result
        """
        agent_name = task.agent_name
        retry_config = task.retry_config
        start_time = time.time()
        
        for attempt in range(1, retry_config.max_retries + 1):
            task.attempts = attempt
            task.last_attempt = datetime.now()
            
            try:
                # Execute agent
                result = await self._execute_single_attempt(agent_name, task.kwargs)
                
                # Check if successful
                if result.success:
                    # Record successful execution
                    task.total_duration = time.time() - start_time
                    self.completed_tasks[task.task_id] = task
                    self._record_execution(agent_name, result)
                    
                    self.logger.info(
                        f"Agent {agent_name} completed successfully on attempt {attempt}",
                        extra={"task_id": task.task_id, "duration": task.total_duration}
                    )
                    
                    # Trigger callbacks
                    self._trigger_callbacks('agent_completed', agent_name=agent_name, result=result, task_id=task.task_id)
                    
                    return result
                else:
                    # Agent returned failure but didn't raise exception
                    raise Exception(f"Agent execution failed: {result.error_message}")
                    
            except Exception as e:
                # Log error with full traceback
                self._log_error_with_traceback(
                    agent_name, e, task.task_id, attempt, 
                    {"kwargs": task.kwargs, "retry_config": retry_config.__dict__}
                )
                
                # Check if we should retry
                if not isinstance(e, retry_config.retry_on_exceptions):
                    self.logger.info(f"Not retrying {agent_name} - exception type not in retry list")
                    break
                
                if attempt < retry_config.max_retries:
                    # Calculate backoff delay
                    delay = self._calculate_backoff_delay(attempt, retry_config)
                    task.next_retry = datetime.now() + timedelta(seconds=delay)
                    
                    self.logger.info(
                        f"Retrying {agent_name} in {delay:.2f}s (attempt {attempt}/{retry_config.max_retries})",
                        extra={"task_id": task.task_id, "delay": delay}
                    )
                    
                    # Trigger retry callback
                    self._trigger_callbacks('agent_retry', agent_name=agent_name, attempt=attempt, delay=delay, task_id=task.task_id)
                    
                    # Wait before retry
                    await asyncio.sleep(delay)
                else:
                    # Max retries reached
                    self.logger.error(
                        f"Agent {agent_name} failed after {retry_config.max_retries} attempts",
                        extra={"task_id": task.task_id, "total_duration": time.time() - start_time}
                    )
                    break
        
        # All attempts failed
        task.total_duration = time.time() - start_time
        self.failed_tasks[task.task_id] = task
        
        # Trigger failure callback
        self._trigger_callbacks('agent_failed', agent_name=agent_name, task_id=task.task_id, final_failure=True)
        
        return AgentResult(
            success=False,
            error_message=f"Agent '{agent_name}' failed after {retry_config.max_retries} attempts",
            error_type="MaxRetriesExceeded",
            metadata={
                "attempts": task.attempts,
                "total_duration": task.total_duration,
                "error_history": task.error_history
            }
        )

    async def _execute_single_attempt(self, agent_name: str, kwargs: Dict[str, Any]) -> AgentResult:
        """Execute a single attempt of agent execution.
        
        Args:
            agent_name: Name of the agent
            kwargs: Execution parameters
            
        Returns:
            AgentResult: Result of execution
        """
        agent_entry = self.agent_registry[agent_name]
        
        # Check if agent is available
        if not agent_entry.instance:
            return AgentResult(
                success=False,
                error_message=f"Agent '{agent_name}' instance not available",
                error_type="AgentNotAvailable"
            )
        
        # Check if agent is enabled
        if not agent_entry.instance.config.enabled:
            return AgentResult(
                success=False,
                error_message=f"Agent '{agent_name}' is disabled",
                error_type="AgentDisabled"
            )
        
        # Execute with timeout
        try:
            if asyncio.iscoroutinefunction(agent_entry.instance.execute):
                result = await asyncio.wait_for(
                    agent_entry.instance.execute(**kwargs),
                    timeout=self.config.execution_timeout
                )
            else:
                # Handle synchronous agents
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: agent_entry.instance.execute(**kwargs)
                )
            
            return result
            
        except asyncio.TimeoutError:
            return AgentResult(
                success=False,
                error_message=f"Agent '{agent_name}' execution timed out after {self.config.execution_timeout}s",
                error_type="ExecutionTimeout"
            )

    def start_restart_monitor(self) -> None:
        """Start the agent restart monitoring system."""
        if self._restart_monitor_running:
            self.logger.warning("Restart monitor is already running")
            return

        self._restart_monitor_running = True
        self._restart_monitor_thread = threading.Thread(
            target=self._restart_monitor_loop, daemon=True
        )
        self._restart_monitor_thread.start()
        self.logger.info("Agent restart monitor started")

    def stop_restart_monitor(self) -> None:
        """Stop the agent restart monitoring system."""
        self._restart_monitor_running = False
        if self._restart_monitor_thread:
            self._restart_monitor_thread.join(timeout=5)
        self.logger.info("Agent restart monitor stopped")

    def _restart_monitor_loop(self) -> None:
        """Main loop for monitoring agent health and restarting failed agents."""
        import time

        while self._restart_monitor_running:
            try:
                for agent_name in list(self.agent_registry.keys()):
                    if not self._restart_monitor_running:
                        break

                    self._check_agent_health(agent_name)

                time.sleep(self.health_check_interval)

            except Exception as e:
                self.logger.error(f"Error in restart monitor loop: {e}")
                time.sleep(10)  # Brief pause on error

    def _check_agent_health(self, agent_name: str) -> None:
        """Check the health of a specific agent and restart if needed."""
        try:
            if agent_name not in self.agent_registry:
                return

            agent_entry = self.agent_registry[agent_name]
            current_time = datetime.now()

            # Initialize health tracking if not exists
            if agent_name not in self.agent_health:
                self.agent_health[agent_name] = {
                    "last_heartbeat": current_time,
                    "last_execution": None,
                    "error_count": 0,
                    "status": "unknown",
                    "restart_count": 0,
                }

            health = self.agent_health[agent_name]

            # Check if agent is responsive
            if agent_entry.instance:
                is_responsive = self._is_agent_responsive(agent_entry.instance)
                if not is_responsive:
                    health["error_count"] += 1
                    self.logger.warning(
                        f"Agent {agent_name} is not responsive (error count: {health['error_count']})"
                    )
                else:
                    health["last_heartbeat"] = current_time
                    health["error_count"] = 0

            # Check for timeout conditions
            if health["last_execution"]:
                time_since_execution = (
                    current_time - health["last_execution"]
                ).total_seconds()
                if time_since_execution > self.agent_timeout_threshold:
                    health["error_count"] += 1
                    self.logger.warning(
                        f"Agent {agent_name} has exceeded timeout threshold"
                    )

            # Determine if restart is needed
            if health["error_count"] >= 3:  # 3 consecutive errors
                if health["restart_count"] < self.max_restart_attempts:
                    self._restart_agent(agent_name)
                else:
                    self.logger.error(
                        f"Agent {agent_name} exceeded maximum restart attempts"
                    )
                    health["status"] = "failed"

        except Exception as e:
            self.logger.error(f"Error checking health for agent {agent_name}: {e}")

    def _is_agent_responsive(self, agent_instance: BaseAgent) -> bool:
        """Check if an agent instance is responsive."""
        try:
            # Try to call a simple method to check responsiveness
            if hasattr(agent_instance, "is_alive"):
                return agent_instance.is_alive()

            if hasattr(agent_instance, "get_status"):
                status = agent_instance.get_status()
                return status.get("running", False)

            if hasattr(agent_instance, "health_check"):
                return agent_instance.health_check()

            # Default: assume responsive if we can access the instance
            return True

        except Exception as e:
            self.logger.debug(f"Agent responsiveness check failed: {e}")
            return False

    def _restart_agent(self, agent_name: str) -> bool:
        """Restart a specific agent."""
        try:
            if agent_name not in self.agent_registry:
                return False

            agent_entry = self.agent_registry[agent_name]
            health = self.agent_health[agent_name]

            self.logger.info(
                f"Restarting agent {agent_name} (attempt {health['restart_count'] + 1})"
            )

            # Stop current instance if running
            if agent_entry.instance:
                try:
                    if hasattr(agent_entry.instance, "stop"):
                        agent_entry.instance.stop()
                except Exception as e:
                    self.logger.debug(f"Error stopping agent {agent_name}: {e}")

            # Create new instance
            try:
                new_instance = agent_entry.agent_class(agent_entry.config)
                agent_entry.instance = new_instance

                # Update health tracking
                health["restart_count"] += 1
                health["error_count"] = 0
                health["last_heartbeat"] = datetime.now()
                health["status"] = "restarted"

                self.logger.info(f"Successfully restarted agent {agent_name}")
                return True

            except Exception as e:
                self.logger.error(f"Failed to restart agent {agent_name}: {e}")
                health["status"] = "restart_failed"
                return False

        except Exception as e:
            self.logger.error(f"Error in restart process for agent {agent_name}: {e}")
            return False

    def force_restart_agent(self, agent_name: str) -> bool:
        """Force restart an agent regardless of health status."""
        try:
            if agent_name not in self.agent_registry:
                self.logger.warning(f"Agent {agent_name} not found in registry")
                return False

            # Reset health tracking
            if agent_name in self.agent_health:
                self.agent_health[agent_name]["error_count"] = 0
                self.agent_health[agent_name]["status"] = "force_restart"

            return self._restart_agent(agent_name)

        except Exception as e:
            self.logger.error(f"Error force restarting agent {agent_name}: {e}")
            return False

    def get_agent_health_status(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get the health status of a specific agent."""
        if agent_name not in self.agent_health:
            return None

        health = self.agent_health[agent_name].copy()
        health["agent_name"] = agent_name
        health["last_heartbeat"] = (
            health["last_heartbeat"].isoformat() if health["last_heartbeat"] else None
        )
        health["last_execution"] = (
            health["last_execution"].isoformat() if health["last_execution"] else None
        )

        return health

    def get_all_agent_health_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all agents."""
        return {
            agent_name: self.get_agent_health_status(agent_name)
            for agent_name in self.agent_registry.keys()
        }

    def reset_agent_health_tracking(self, agent_name: str) -> bool:
        """Reset health tracking for a specific agent."""
        try:
            if agent_name in self.agent_health:
                self.agent_health[agent_name] = {
                    "last_heartbeat": datetime.now(),
                    "last_execution": None,
                    "error_count": 0,
                    "status": "reset",
                    "restart_count": 0,
                }
                self.logger.info(f"Reset health tracking for agent {agent_name}")
                return True
            return False
        except Exception as e:
            self.logger.error(
                f"Error resetting health tracking for agent {agent_name}: {e}"
            )
            return False

    def _load_config(self) -> None:
        """Load agent configuration from file."""
        config_path = Path(self.config.config_file)
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config_data = json.load(f)
                self.logger.info(f"Loaded agent configuration from {config_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
                config_data = {}
        else:
            config_data = {}
            self._create_default_config(config_path)

        self.agent_configs = config_data

    def _create_default_config(self, config_path: Path) -> None:
        """Create default configuration file."""
        default_config = {
            "agents": {
                "model_builder": {
                    "enabled": True,
                    "priority": 1,
                    "max_concurrent_runs": 1,
                    "timeout_seconds": 300,
                    "retry_attempts": 3,
                    "custom_config": {},
                },
                "performance_critic": {
                    "enabled": True,
                    "priority": 2,
                    "max_concurrent_runs": 1,
                    "timeout_seconds": 300,
                    "retry_attempts": 3,
                    "custom_config": {},
                },
                "updater": {
                    "enabled": True,
                    "priority": 3,
                    "max_concurrent_runs": 1,
                    "timeout_seconds": 300,
                    "retry_attempts": 3,
                    "custom_config": {},
                },
                "execution_agent": {
                    "enabled": True,
                    "priority": 4,
                    "max_concurrent_runs": 1,
                    "timeout_seconds": 300,
                    "retry_attempts": 3,
                    "custom_config": {
                        "execution_mode": "simulation",
                        "max_positions": 10,
                        "min_confidence": 0.7,
                        "max_slippage": 0.001,
                        "execution_delay": 1.0,
                        "risk_per_trade": 0.02,
                        "max_position_size": 0.2,
                        "base_fee": 0.001,
                        "min_fee": 1.0,
                    },
                },
            },
            "manager": {
                "auto_start": True,
                "max_concurrent_agents": 5,
                "execution_timeout": 300,
                "enable_logging": True,
                "enable_metrics": True,
            },
        }

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=2)

        self.logger.info(f"Created default configuration at {config_path}")

    def _register_default_agents(self) -> None:
        """Register the default agents."""
        default_agents = {
            "model_builder": ModelBuilderAgent,
            "performance_critic": PerformanceCriticAgent,
            "updater": UpdaterAgent,
            "execution_agent": ExecutionAgent,
        }

        for agent_name, agent_class in default_agents.items():
            self.register_agent(agent_name, agent_class)

    def register_agent(
        self,
        name: str,
        agent_class: Type[BaseAgent],
        config: Optional[AgentConfig] = None,
    ) -> None:
        """Register an agent with the manager.

        Args:
            name: Name of the agent
            agent_class: Agent class to register
            config: Agent configuration (optional)
        """
        if name in self.agent_registry:
            self.logger.warning(f"Agent {name} already registered, overwriting")

        # Get configuration from file or use default
        agent_config_data = self.agent_configs.get("agents", {}).get(name, {})

        if config is None:
            config = AgentConfig(
                name=name,
                enabled=agent_config_data.get("enabled", True),
                priority=agent_config_data.get("priority", 1),
                max_concurrent_runs=agent_config_data.get("max_concurrent_runs", 1),
                timeout_seconds=agent_config_data.get("timeout_seconds", 300),
                retry_attempts=agent_config_data.get("retry_attempts", 3),
                custom_config=agent_config_data.get("custom_config", {}),
            )

        # Always instantiate the agent with AgentConfig
        instance = agent_class(config)

        # Create registry entry
        entry = AgentRegistryEntry(
            agent_class=agent_class,
            config=config,
            instance=instance,
            metadata=agent_class.get_metadata()
            if hasattr(agent_class, "get_metadata")
            else None,
        )
        self.agent_registry[name] = entry
        self.logger.info(f"Registered agent: {name}")

    def unregister_agent(self, name: str) -> None:
        """Unregister an agent.

        Args:
            name: Name of the agent to unregister
        """
        if name in self.agent_registry:
            # Stop the agent if it's running
            if self.agent_registry[name].instance:
                self.agent_registry[name].instance.disable()

            del self.agent_registry[name]
            self.logger.info(f"Unregistered agent: {name}")
        else:
            self.logger.warning(f"Agent {name} not found in registry")

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get an agent instance by name."""
        entry = self.agent_registry.get(name)
        if entry:
            if entry.instance is None:
                # Instantiate if not already done
                entry.instance = entry.agent_class(entry.config)
            return entry.instance
        return None

    def enable_agent(self, name: str) -> bool:
        """Enable an agent.

        Args:
            name: Name of the agent to enable

        Returns:
            True if successful, False otherwise
        """
        agent = self.get_agent(name)
        if agent:
            agent.enable()
            self.logger.info(f"Enabled agent: {name}")
            return True
        else:
            self.logger.warning(f"Agent {name} not found or cannot be enabled")
            return False

    def disable_agent(self, name: str) -> bool:
        """Disable an agent.

        Args:
            name: Name of the agent to disable

        Returns:
            True if successful, False otherwise
        """
        agent = self.get_agent(name)
        if agent:
            agent.disable()
            self.logger.info(f"Disabled agent: {name}")
            return True
        else:
            self.logger.warning(f"Agent {name} not found")
            return False

    def start_retry_monitor(self) -> None:
        """Start the retry monitoring system."""
        if self._retry_monitor_running:
            self.logger.warning("Retry monitor is already running")
            return

        self._retry_monitor_running = True
        self._retry_monitor_thread = threading.Thread(
            target=self._retry_monitor_loop, daemon=True
        )
        self._retry_monitor_thread.start()
        self.logger.info("Retry monitor started")

    def stop_retry_monitor(self) -> None:
        """Stop the retry monitoring system."""
        self._retry_monitor_running = False
        if self._retry_monitor_thread:
            self._retry_monitor_thread.join(timeout=5)
        self.logger.info("Retry monitor stopped")

    def _retry_monitor_loop(self) -> None:
        """Monitor loop for handling retries and backoff."""
        while self._retry_monitor_running:
            try:
                # Process retry queue
                while not self.retry_queue.empty():
                    task = self.retry_queue.get_nowait()
                    
                    # Check if it's time to retry
                    if task.next_retry and datetime.now() >= task.next_retry:
                        # Schedule retry
                        asyncio.create_task(self._execute_with_retry(task))
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error in retry monitor loop: {e}")
                time.sleep(5)

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics.
        
        Returns:
            Dictionary with error statistics
        """
        stats = {
            "total_errors": self.global_error_count,
            "agent_error_counts": self.agent_error_counts.copy(),
            "recent_errors": self.error_log[-100:] if self.error_log else [],
            "error_types": {},
            "task_statistics": {
                "pending": len(self.pending_tasks),
                "completed": len(self.completed_tasks),
                "failed": len(self.failed_tasks)
            }
        }
        
        # Analyze error types
        for error in self.error_log:
            error_type = error.get("error_type", "Unknown")
            stats["error_types"][error_type] = stats["error_types"].get(error_type, 0) + 1
        
        return stats

    def clear_error_log(self, older_than_days: int = 7) -> int:
        """Clear old error logs.
        
        Args:
            older_than_days: Clear errors older than this many days
            
        Returns:
            Number of errors cleared
        """
        cutoff_time = datetime.now() - timedelta(days=older_than_days)
        original_count = len(self.error_log)
        
        self.error_log = [
            error for error in self.error_log
            if datetime.fromisoformat(error["timestamp"]) > cutoff_time
        ]
        
        cleared_count = original_count - len(self.error_log)
        self.logger.info(f"Cleared {cleared_count} old error logs")
        
        return cleared_count

    def get_agent_retry_config(self, agent_name: str) -> Optional[RetryConfig]:
        """Get retry configuration for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            RetryConfig or None if agent not found
        """
        if agent_name in self.agent_registry:
            return self.agent_registry[agent_name].retry_config
        return None

    def update_agent_retry_config(self, agent_name: str, retry_config: RetryConfig) -> bool:
        """Update retry configuration for an agent.
        
        Args:
            agent_name: Name of the agent
            retry_config: New retry configuration
            
        Returns:
            True if updated successfully
        """
        if agent_name in self.agent_registry:
            self.agent_registry[agent_name].retry_config = retry_config
            self.logger.info(f"Updated retry config for agent {agent_name}")
            return True
        return False

    def _record_execution(self, agent_name: str, result: AgentResult) -> None:
        """Record agent execution for metrics.

        Args:
            agent_name: Name of the agent
            result: Execution result
        """
        execution_record = {
            "agent_name": agent_name,
            "timestamp": result.timestamp,
            "success": result.success,
            "execution_time": result.execution_time,
            "error_message": result.error_message,
        }

        self.execution_history.append(execution_record)

        # Update agent metrics
        if agent_name not in self.agent_metrics:
            self.agent_metrics[agent_name] = {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "total_execution_time": 0.0,
                "avg_execution_time": 0.0,
            }

        metrics = self.agent_metrics[agent_name]
        metrics["total_executions"] += 1
        metrics["total_execution_time"] += result.execution_time

        if result.success:
            metrics["successful_executions"] += 1
        else:
            metrics["failed_executions"] += 1

        metrics["avg_execution_time"] = (
            metrics["total_execution_time"] / metrics["total_executions"]
        )

        # Update leaderboard
        self.leaderboard.update_performance(
            agent_name=agent_name,
            sharpe_ratio=result.sharpe_ratio,
            max_drawdown=result.max_drawdown,
            win_rate=result.win_rate,
            total_return=result.total_return,
            extra_metrics=result.extra_metrics,
        )

    def get_agent_status(self, name: str) -> Optional[AgentStatus]:
        """Get the status of an agent.

        Args:
            name: Name of the agent

        Returns:
            Agent status if found, None otherwise
        """
        agent = self.get_agent(name)
        if agent:
            return agent.get_status()
        return None

    def get_all_agent_statuses(self) -> Dict[str, AgentStatus]:
        """Get status of all agents.

        Returns:
            Dictionary mapping agent names to their statuses
        """
        statuses = {}
        for name in self.agent_registry.keys():
            status = self.get_agent_status(name)
            if status:
                statuses[name] = status
        return statuses

    def get_agent_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for an agent.

        Args:
            name: Name of the agent

        Returns:
            Agent metadata if found, None otherwise
        """
        if name in self.agent_registry:
            return self.agent_registry[name].metadata
        return None

    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents.

        Returns:
            List of agent information dictionaries
        """
        agents = []
        for name, entry in self.agent_registry.items():
            agent_info = {
                "name": name,
                "enabled": entry.config.enabled,
                "priority": entry.config.priority,
                "metadata": entry.metadata,
                "status": self.get_agent_status(name),
            }
            agents.append(agent_info)

        return sorted(agents, key=lambda x: x["priority"])

    def update_agent_config(self, name: str, new_config: Dict[str, Any]) -> bool:
        """Update agent configuration.

        Args:
            name: Name of the agent
            new_config: New configuration values

        Returns:
            True if successful, False otherwise
        """
        agent = self.get_agent(name)
        if agent:
            agent.update_config(new_config)
            self.logger.info(f"Updated configuration for agent: {name}")
            return True
        else:
            self.logger.warning(f"Agent {name} not found")
            return False

    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution metrics for all agents.

        Returns:
            Dictionary containing execution metrics
        """
        return {
            "success": True,
            "result": {
                "agent_metrics": self.agent_metrics,
                "total_executions": len(self.execution_history),
                "recent_executions": self.execution_history[-10:]
                if self.execution_history
                else [],
            },
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def save_config(self) -> None:
        """Save current agent configuration to file."""
        config_data = {
            "agents": {},
            "manager": {
                "auto_start": self.config.auto_start,
                "max_concurrent_agents": self.config.max_concurrent_agents,
                "execution_timeout": self.config.execution_timeout,
                "enable_logging": self.config.enable_logging,
                "enable_metrics": self.config.enable_metrics,
            },
        }

        for name, entry in self.agent_registry.items():
            config_data["agents"][name] = {
                "enabled": entry.config.enabled,
                "priority": entry.config.priority,
                "max_concurrent_runs": entry.config.max_concurrent_runs,
                "timeout_seconds": entry.config.timeout_seconds,
                "retry_attempts": entry.config.retry_attempts,
                "custom_config": entry.config.custom_config or {},
            }

        config_path = Path(self.config.config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

        self.logger.info(f"Saved agent configuration to {config_path}")

    def start(self) -> None:
        self.logger.info("AgentManager starting...")
        self.status = 'running'
        self._trigger_callbacks('agent_started', manager=self)
        try:
            if self.config.auto_start:
                # Enable all agents that should be auto-started
                for name, entry in self.agent_registry.items():
                    if entry.config.enabled:
                        self.enable_agent(name)
            self.logger.info("AgentManager started.")
        except Exception as e:
            self.status = 'failed'
            self.logger.error(f"AgentManager failed to start: {e}")
            self._trigger_callbacks('agent_failed', manager=self, exception=e)
            raise

    def stop(self) -> None:
        self.logger.info("AgentManager stopping...")
        try:
            # Disable all agents
            for name in self.agent_registry.keys():
                self.disable_agent(name)
            self.logger.info("AgentManager stopped.")
            self._trigger_callbacks('agent_completed', manager=self)
        except Exception as e:
            self.status = 'failed'
            self.logger.error(f"AgentManager failed to stop: {e}")
            self._trigger_callbacks('agent_failed', manager=self, exception=e)
            raise

    def log_agent_performance(
        self,
        agent_name: str,
        sharpe_ratio: float,
        max_drawdown: float,
        win_rate: float,
        total_return: float,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ):
        """Log agent performance to the leaderboard."""
        self.leaderboard.update_performance(
            agent_name=agent_name,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_return=total_return,
            extra_metrics=extra_metrics,
        )

    def get_leaderboard(
        self, top_n: int = 10, sort_by: str = "sharpe_ratio"
    ) -> List[Dict[str, Any]]:
        """Expose leaderboard data for dashboard/reporting."""
        return self.leaderboard.get_leaderboard(top_n=top_n, sort_by=sort_by)

    def get_deprecated_agents(self) -> List[str]:
        return self.leaderboard.get_deprecated_agents()

    def get_active_agents(self) -> List[str]:
        return self.leaderboard.get_active_agents()

    def get_leaderboard_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self.leaderboard.get_history(limit=limit)

    def get_leaderboard_dataframe(self):
        return self.leaderboard.as_dataframe()

    def register_callback(self, event_name: str, handler_fn):
        """Register a callback handler for a specific event."""
        if event_name not in self.callbacks:
            self.callbacks[event_name] = []
        self.callbacks[event_name].append(handler_fn)
        self.logger.info(f"Callback registered for event: {event_name}")

    def _trigger_callbacks(self, event_name: str, **kwargs):
        """Trigger all callbacks for a given event."""
        for handler in self.callbacks.get(event_name, []):
            try:
                handler(**kwargs)
            except Exception as e:
                self.logger.error(f"Callback for {event_name} failed: {e}")


# Global agent manager instance
_agent_manager: Optional[EnhancedAgentManager] = None


def get_agent_manager() -> EnhancedAgentManager:
    """Get the global agent manager instance.

    Returns:
        Agent manager instance
    """
    global _agent_manager
    if _agent_manager is None:
        _agent_manager = EnhancedAgentManager()
    return _agent_manager


def register_agent(
    name: str, agent_class: Type[BaseAgent], config: Optional[AgentConfig] = None
) -> None:
    """Register an agent with the global agent manager.

    Args:
        name: Name of the agent
        agent_class: Agent class to register
        config: Agent configuration (optional)
    """
    manager = get_agent_manager()
    manager.register_agent(name, agent_class, config)


async def execute_agent(name: str, **kwargs) -> AgentResult:
    """Execute an agent using the global agent manager.

    Args:
        name: Name of the agent to execute
        **kwargs: Parameters to pass to the agent

    Returns:
        AgentResult: Result of the execution
    """
    manager = get_agent_manager()
    return await manager.execute_agent_with_retry(name, **kwargs)

# Backward compatibility aliases
AgentManager = EnhancedAgentManager
