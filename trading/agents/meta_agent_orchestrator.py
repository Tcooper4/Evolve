"""
Meta Agent Orchestrator

Orchestrates multiple agents with error handling and fallback mechanisms.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class AgentCall:
    """Agent call configuration."""
    agent_name: str
    method: str
    args: tuple
    kwargs: dict
    timeout: float = 30.0
    retry_attempts: int = 3
    fallback_agent: Optional[str] = None


@dataclass
class OrchestrationResult:
    """Result of agent orchestration."""
    success: bool
    primary_result: Optional[Any] = None
    fallback_result: Optional[Any] = None
    errors: List[str] = None
    execution_time: float = 0.0
    agent_used: str = ""


class MetaAgentOrchestrator:
    """Orchestrates multiple agents with error handling and fallbacks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the meta agent orchestrator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.default_agent = self.config.get("default_agent", "GeneralAgent")
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 1.0)
        self.agent_registry = {}
        self.error_history = []
        
    def register_agent(self, agent_name: str, agent_instance: Any):
        """Register an agent for orchestration.
        
        Args:
            agent_name: Name of the agent
            agent_instance: Agent instance
        """
        self.agent_registry[agent_name] = agent_instance
        logger.info(f"Registered agent: {agent_name}")
        
    def orchestrate_agents(
        self, 
        agent_calls: List[AgentCall],
        fallback_strategy: str = "default"
    ) -> OrchestrationResult:
        """
        Orchestrate multiple agent calls with error handling.
        
        Args:
            agent_calls: List of agent calls to execute
            fallback_strategy: Fallback strategy ('default', 'next', 'best')
            
        Returns:
            OrchestrationResult with results and error information
        """
        start_time = datetime.now()
        errors = []
        primary_result = None
        fallback_result = None
        agent_used = ""
        
        # Try primary agents
        for agent_call in agent_calls:
            try:
                logger.info(f"Attempting to call agent: {agent_call.agent_name}")
                
                if agent_call.agent_name not in self.agent_registry:
                    raise ValueError(f"Agent {agent_call.agent_name} not registered")
                    
                agent = self.agent_registry[agent_call.agent_name]
                method = getattr(agent, agent_call.method, None)
                
                if method is None:
                    raise AttributeError(f"Method {agent_call.method} not found on {agent_call.agent_name}")
                    
                # Execute agent call with retries
                result = self._execute_with_retries(
                    method, 
                    agent_call.args, 
                    agent_call.kwargs, 
                    agent_call.timeout,
                    agent_call.retry_attempts
                )
                
                primary_result = result
                agent_used = agent_call.agent_name
                logger.info(f"Successfully executed {agent_call.agent_name}")
                break
                
            except Exception as e:
                error_msg = f"Agent {agent_call.agent_name} failed: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                self.error_history.append({
                    "agent": agent_call.agent_name,
                    "error": str(e),
                    "timestamp": datetime.now()
                })
                
                # Try fallback agent if specified
                if agent_call.fallback_agent:
                    try:
                        fallback_result = self._execute_fallback(agent_call.fallback_agent)
                        if fallback_result:
                            agent_used = agent_call.fallback_agent
                            break
                    except Exception as fallback_error:
                        errors.append(f"Fallback agent {agent_call.fallback_agent} failed: {str(fallback_error)}")
                        
        # If no agents succeeded, use default fallback
        if not primary_result and not fallback_result:
            try:
                fallback_result = self._execute_fallback(self.default_agent)
                agent_used = self.default_agent
                logger.info(f"Using default fallback agent: {self.default_agent}")
            except Exception as e:
                error_msg = f"Default fallback agent failed: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return OrchestrationResult(
            success=primary_result is not None or fallback_result is not None,
            primary_result=primary_result,
            fallback_result=fallback_result,
            errors=errors,
            execution_time=execution_time,
            agent_used=agent_used
        )
        
    def _execute_with_retries(
        self, 
        method: Callable, 
        args: tuple, 
        kwargs: dict, 
        timeout: float,
        max_retries: int
    ) -> Any:
        """Execute a method with retries and timeout.
        
        Args:
            method: Method to execute
            args: Method arguments
            kwargs: Method keyword arguments
            timeout: Timeout in seconds
            max_retries: Maximum number of retry attempts
            
        Returns:
            Method result
            
        Raises:
            Exception: If all retries fail
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Execute with timeout
                if asyncio.iscoroutinefunction(method):
                    # Handle async methods
                    loop = asyncio.get_event_loop()
                    return loop.run_until_complete(
                        asyncio.wait_for(method(*args, **kwargs), timeout=timeout)
                    )
                else:
                    # Handle sync methods
                    return method(*args, **kwargs)
                    
            except asyncio.TimeoutError:
                last_error = TimeoutError(f"Method execution timed out after {timeout}s")
                logger.warning(f"Attempt {attempt + 1} timed out for {method.__name__}")
                
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed for {method.__name__}: {str(e)}")
                
            if attempt < max_retries - 1:
                import time
                time.sleep(self.retry_delay)
                
        raise last_error
        
    def _execute_fallback(self, agent_name: str) -> Optional[Any]:
        """Execute fallback agent.
        
        Args:
            agent_name: Name of the fallback agent
            
        Returns:
            Fallback result or None if failed
        """
        try:
            if agent_name not in self.agent_registry:
                logger.error(f"Fallback agent {agent_name} not registered")
                return None
                
            agent = self.agent_registry[agent_name]
            
            # Try common fallback methods
            fallback_methods = ["handle_fallback", "default_response", "help"]
            
            for method_name in fallback_methods:
                if hasattr(agent, method_name):
                    method = getattr(agent, method_name)
                    try:
                        result = method()
                        logger.info(f"Fallback agent {agent_name} executed {method_name}")
                        return result
                    except Exception as e:
                        logger.warning(f"Fallback method {method_name} failed: {str(e)}")
                        continue
                        
            logger.warning(f"No suitable fallback method found for {agent_name}")
            return None
            
        except Exception as e:
            logger.error(f"Fallback execution failed for {agent_name}: {str(e)}")
            return None
            
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors.
        
        Returns:
            Error summary dictionary
        """
        if not self.error_history:
            return {"total_errors": 0, "recent_errors": []}
            
        # Get recent errors (last 24 hours)
        recent_cutoff = datetime.now().timestamp() - 86400
        recent_errors = [
            error for error in self.error_history 
            if error["timestamp"].timestamp() > recent_cutoff
        ]
        
        # Count errors by agent
        agent_error_counts = {}
        for error in recent_errors:
            agent = error["agent"]
            agent_error_counts[agent] = agent_error_counts.get(agent, 0) + 1
            
        return {
            "total_errors": len(self.error_history),
            "recent_errors": len(recent_errors),
            "agent_error_counts": agent_error_counts,
            "most_recent_errors": recent_errors[-5:] if recent_errors else []
        }
        
    def clear_error_history(self):
        """Clear error history."""
        self.error_history.clear()
        logger.info("Error history cleared")
        
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all registered agents.
        
        Returns:
            Dictionary with agent status information
        """
        status = {}
        for agent_name, agent in self.agent_registry.items():
            try:
                # Check if agent has status method
                if hasattr(agent, "get_status"):
                    agent_status = agent.get_status()
                elif hasattr(agent, "is_available"):
                    agent_status = {"available": agent.is_available}
                else:
                    agent_status = {"available": True}
                    
                status[agent_name] = agent_status
                
            except Exception as e:
                status[agent_name] = {"available": False, "error": str(e)}
                
        return status
