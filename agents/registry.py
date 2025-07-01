"""
Centralized Agent Registry

This module provides a unified registry for all agents across the codebase.
It dynamically discovers and registers agent classes from multiple directories
and provides a clean interface for agent instantiation and management.
"""

import logging
import importlib
import inspect
from typing import Dict, List, Optional, Any, Type, Union
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import sys

logger = logging.getLogger(__name__)

@dataclass
class AgentMetadata:
    """Metadata for registered agents."""
    name: str
    class_name: str
    module_path: str
    description: str
    capabilities: List[str]
    dependencies: List[str]
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()

class AgentRegistry:
    """
    Centralized registry for all agents in the system.
    
    Features:
    - Dynamic discovery of agents from multiple directories
    - Automatic registration of agent classes
    - Clean interface for agent instantiation
    - Fallback handling for missing agents
    - Support for both agents/ and trading/agents/ directories
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the agent registry.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.agents: Dict[str, Type] = {}
        self.metadata: Dict[str, AgentMetadata] = {}
        self.agent_directories = [
            "agents",
            "trading.agents"
        ]
        
        # Discover and register all agents
        self._discover_agents()
        
    def _discover_agents(self):
        """Discover and register all available agents."""
        logger.info("Starting agent discovery...")
        
        for directory in self.agent_directories:
            try:
                self._discover_agents_in_directory(directory)
            except Exception as e:
                logger.warning(f"Failed to discover agents in {directory}: {e}")
        
        logger.info(f"Agent discovery complete. Registered {len(self.agents)} agents.")
    
    def _discover_agents_in_directory(self, directory: str):
        """Discover agents in a specific directory."""
        try:
            # Import the directory module
            module = importlib.import_module(directory)
            
            # Get all members of the module
            for name, obj in inspect.getmembers(module):
                if self._is_agent_class(obj):
                    self._register_agent(name, obj, directory)
                    
        except ImportError as e:
            logger.warning(f"Could not import {directory}: {e}")
    
    def _is_agent_class(self, obj) -> bool:
        """Check if an object is an agent class."""
        if not inspect.isclass(obj):
            return False
            
        # Check if it has an execute method (common agent interface)
        if not hasattr(obj, 'execute'):
            return False
            
        # Check if it's not a built-in or standard library class
        if obj.__module__.startswith('builtins') or obj.__module__.startswith('__'):
            return False
            
        # Check if it's not an abstract base class
        if inspect.isabstract(obj):
            return False
            
        return True
    
    def _register_agent(self, name: str, agent_class: Type, directory: str):
        """Register an agent class."""
        # Normalize agent name
        agent_name = name.lower()
        
        # Create metadata
        metadata = AgentMetadata(
            name=agent_name,
            class_name=name,
            module_path=f"{directory}.{name}",
            description=inspect.getdoc(agent_class) or "",
            capabilities=self._extract_capabilities(agent_class),
            dependencies=self._extract_dependencies(agent_class)
        )
        
        # Register the agent
        self.agents[agent_name] = agent_class
        self.metadata[agent_name] = metadata
        
        logger.debug(f"Registered agent: {agent_name} from {directory}")
    
    def _extract_capabilities(self, agent_class: Type) -> List[str]:
        """Extract capabilities from an agent class."""
        capabilities = []
        
        # Check for common agent methods
        if hasattr(agent_class, 'execute'):
            capabilities.append('execute')
        if hasattr(agent_class, 'process'):
            capabilities.append('process')
        if hasattr(agent_class, 'analyze'):
            capabilities.append('analyze')
        if hasattr(agent_class, 'forecast'):
            capabilities.append('forecast')
        if hasattr(agent_class, 'optimize'):
            capabilities.append('optimize')
        if hasattr(agent_class, 'train'):
            capabilities.append('train')
        
        return capabilities
    
    def _extract_dependencies(self, agent_class: Type) -> List[str]:
        """Extract dependencies from an agent class."""
        dependencies = []
        
        # Check for common dependency patterns
        for name, value in inspect.getmembers(agent_class):
            if name.startswith('_') and isinstance(value, (str, list)):
                if isinstance(value, list):
                    dependencies.extend(value)
                else:
                    dependencies.append(value)
        
        return dependencies
    
    def get_agent(self, name: str, **kwargs) -> Optional[Any]:
        """
        Get an agent instance by name.
        
        Args:
            name: Name of the agent (case-insensitive)
            **kwargs: Arguments to pass to the agent constructor
            
        Returns:
            Agent instance or None if not found
        """
        agent_name = name.lower()
        
        if agent_name not in self.agents:
            logger.warning(f"Agent '{name}' not found in registry")
            return self._get_fallback_agent(name, **kwargs)
        
        try:
            agent_class = self.agents[agent_name]
            return agent_class(**kwargs)
        except Exception as e:
            logger.error(f"Failed to instantiate agent '{name}': {e}")
            return self._get_fallback_agent(name, **kwargs)
    
    def _get_fallback_agent(self, name: str, **kwargs) -> Optional[Any]:
        """Get a fallback agent when the requested agent is not available."""
        logger.info(f"Using fallback for agent '{name}'")
        
        # Try to create a basic agent with the same name
        try:
            # Import base agent if available
            from trading.agents.base_agent_interface import BaseAgent
            return BaseAgent(name=name, **kwargs)
        except ImportError:
            logger.error(f"No fallback available for agent '{name}'")
            return None
    
    def list_agents(self) -> List[str]:
        """Get list of all registered agent names."""
        return list(self.agents.keys())
    
    def get_agent_metadata(self, name: str) -> Optional[AgentMetadata]:
        """Get metadata for a specific agent."""
        return self.metadata.get(name.lower())
    
    def get_agents_by_capability(self, capability: str) -> List[str]:
        """Get agents that provide a specific capability."""
        matching_agents = []
        
        for name, metadata in self.metadata.items():
            if capability in metadata.capabilities:
                matching_agents.append(name)
        
        return matching_agents
    
    def search_agents(self, query: str) -> List[str]:
        """Search for agents by name or description."""
        query = query.lower()
        matching_agents = []
        
        for name, metadata in self.metadata.items():
            if (query in name or 
                query in metadata.description.lower() or
                query in metadata.class_name.lower()):
                matching_agents.append(name)
        
        return matching_agents
    
    def reload_agents(self):
        """Reload all agents from their modules."""
        logger.info("Reloading agents...")
        
        # Clear existing registrations
        self.agents.clear()
        self.metadata.clear()
        
        # Rediscover agents
        self._discover_agents()
        
        logger.info("Agent reload complete")
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get status information about the registry."""
        return {
            'total_agents': len(self.agents),
            'agent_names': list(self.agents.keys()),
            'capabilities': list(set(
                cap for metadata in self.metadata.values() 
                for cap in metadata.capabilities
            )),
            'last_updated': datetime.now().isoformat(),
            'directories_searched': self.agent_directories
        }

# Global registry instance
_registry = None

def get_registry() -> AgentRegistry:
    """Get the global agent registry instance."""
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry

def get_agent(name: str, **kwargs) -> Optional[Any]:
    """
    Convenience function to get an agent by name.
    
    Args:
        name: Name of the agent
        **kwargs: Arguments to pass to the agent constructor
        
    Returns:
        Agent instance or None if not found
    """
    return get_registry().get_agent(name, **kwargs)

def list_agents() -> List[str]:
    """Convenience function to list all agents."""
    return get_registry().list_agents()

def search_agents(query: str) -> List[str]:
    """Convenience function to search for agents."""
    return get_registry().search_agents(query)

# Export commonly used agents for easy access
def get_prompt_router_agent(**kwargs):
    """Get the prompt router agent."""
    return get_agent('promptrouteragent', **kwargs)

def get_model_builder_agent(**kwargs):
    """Get the model builder agent."""
    return get_agent('modelbuilderagent', **kwargs)

def get_performance_checker_agent(**kwargs):
    """Get the performance checker agent."""
    return get_agent('performancecheckeragent', **kwargs)

def get_voice_prompt_agent(**kwargs):
    """Get the voice prompt agent."""
    return get_agent('voicepromptagent', **kwargs)

# Dictionary for easy access to all agents
ALL_AGENTS = {
    'prompt_router': get_prompt_router_agent,
    'model_builder': get_model_builder_agent,
    'performance_checker': get_performance_checker_agent,
    'voice_prompt': get_voice_prompt_agent,
}

# Add all discovered agents to ALL_AGENTS
def _populate_all_agents():
    """Populate ALL_AGENTS with discovered agents."""
    registry = get_registry()
    for agent_name in registry.list_agents():
        if agent_name not in ALL_AGENTS:
            ALL_AGENTS[agent_name] = lambda name=agent_name, **kwargs: get_agent(name, **kwargs)

# Initialize ALL_AGENTS when module is imported
_populate_all_agents() 