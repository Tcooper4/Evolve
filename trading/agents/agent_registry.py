# -*- coding: utf-8 -*-
"""
Agent registry for the trading system.

This module provides a registry of available agent types and their capabilities,
enabling dynamic agent discovery and loading.
"""

import logging
import importlib
import inspect
from typing import Dict, List, Optional, Any, Type, Set
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from trading.base_agent import BaseAgent

logger = logging.getLogger(__name__)

@dataclass
class AgentCapability:
    """Agent capability definition."""
    name: str
    description: str
    required_params: List[str]
    optional_params: List[str]
    return_type: str

@dataclass
class AgentInfo:
    """Agent information and metadata."""
    name: str
    class_name: str
    module_path: str
    description: str
    capabilities: List[AgentCapability]
    dependencies: List[str]
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()

class AgentRegistry:
    """Registry for managing available agent types."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the agent registry.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.agents: Dict[str, AgentInfo] = {}
        self.capabilities: Dict[str, Set[str]] = {}  # capability -> agent names
        self._discover_agents()

    def _discover_agents(self):
        """Discover available agents in the system."""
        # Get agent directories from config
        agent_dirs = self.config.get("agent_dirs", ["trading.agents"])
        
        for dir_path in agent_dirs:
            try:
                # Import the module
                module = importlib.import_module(dir_path)
                
                # Find all agent classes
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseAgent) and 
                        obj != BaseAgent):
                        
                        # Get agent info
                        agent_info = self._get_agent_info(obj)
                        self.register_agent(agent_info)
                        
            except ImportError as e:
                logger.error(f"Failed to import agent directory {dir_path}: {e}")

    def _get_agent_info(self, agent_class: Type[BaseAgent]) -> AgentInfo:
        """Get information about an agent class.
        
        Args:
            agent_class: Agent class to analyze
            
        Returns:
            AgentInfo object containing agent metadata
        """
        # Get class docstring
        doc = inspect.getdoc(agent_class) or ""
        
        # Get capabilities from run method
        capabilities = []
        run_method = getattr(agent_class, "run", None)
        if run_method:
            sig = inspect.signature(run_method)
            required_params = []
            optional_params = []
            
            for name, param in sig.parameters.items():
                if param.default == inspect.Parameter.empty:
                    required_params.append(name)
                else:
                    optional_params.append(name)
                    
            capabilities.append(AgentCapability(
                name="run",
                description=inspect.getdoc(run_method) or "Run the agent",
                required_params=required_params,
                optional_params=optional_params,
                return_type=str(sig.return_annotation)
            ))
            
        # Get dependencies from class attributes
        dependencies = []
        for name, value in inspect.getmembers(agent_class):
            if name.startswith("_") and isinstance(value, (str, list)):
                dependencies.extend(value if isinstance(value, list) else [value])
                
        return AgentInfo(
            name=agent_class.__name__.lower(),
            class_name=agent_class.__name__,
            module_path=f"{agent_class.__module__}.{agent_class.__name__}",
            description=doc,
            capabilities=capabilities,
            dependencies=dependencies
        )
        
    def register_agent(self, agent_info: AgentInfo):
        """Register an agent in the registry.
        
        Args:
            agent_info: Information about the agent to register
        """
        self.agents[agent_info.name] = agent_info
        
        # Update capability index
        for capability in agent_info.capabilities:
            if capability.name not in self.capabilities:
                self.capabilities[capability.name] = set()
            self.capabilities[capability.name].add(agent_info.name)
            
        logger.info(f"Registered agent: {agent_info.name}")

    def get_agent(self, name: str) -> Optional[AgentInfo]:
        """Get information about an agent.
        
        Args:
            name: Name of the agent
            
        Returns:
            AgentInfo object or None if not found
        """
        return self.agents.get(name)
        
    def get_agents_by_capability(self, capability: str) -> List[AgentInfo]:
        """Get agents that provide a specific capability.
        
        Args:
            capability: Name of the capability
            
        Returns:
            List of agents with the capability
        """
        agent_names = self.capabilities.get(capability, set())
        return [self.agents[name] for name in agent_names]
        
    def list_agents(self) -> List[str]:
        """Get list of all registered agent names.
        
        Returns:
            List of agent names
        """
        return list(self.agents.keys())
        
    def list_capabilities(self) -> List[str]:
        """Get list of all available capabilities.
        
        Returns:
            List of capability names
        """
        return list(self.capabilities.keys())
        
    def get_agent_class(self, name: str) -> Optional[Type[BaseAgent]]:
        """Get the class for an agent.
        
        Args:
            name: Name of the agent
            
        Returns:
            Agent class or None if not found
        """
        agent_info = self.get_agent(name)
        if not agent_info:
            return None
            
        try:
            module = importlib.import_module(agent_info.module_path.rsplit(".", 1)[0])
            return getattr(module, agent_info.class_name)
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load agent class {name}: {e}")
            return None