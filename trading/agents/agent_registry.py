# -*- coding: utf-8 -*-
"""
Agent registry for the trading system.

This module provides a registry of available agent types and their capabilities,
enabling dynamic agent discovery and loading.
"""

import logging
import importlib
import inspect
import json
import yaml
from typing import Dict, List, Optional, Any, Type, Set, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from trading.agents.base_agent_interface import BaseAgent

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Agent status enumeration."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    DISABLED = "disabled"
    TESTING = "testing"

class AgentCategory(Enum):
    """Agent category enumeration."""
    TRADING = "trading"
    ANALYSIS = "analysis"
    EXECUTION = "execution"
    MONITORING = "monitoring"
    UTILITY = "utility"

@dataclass
class AgentCapability:
    """Agent capability definition."""
    name: str
    description: str
    required_params: List[str]
    optional_params: List[str]
    return_type: str
    version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentCapability':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class AgentInfo:
    """Agent information and metadata."""
    name: str
    class_name: str
    module_path: str
    description: str
    capabilities: List[AgentCapability]
    dependencies: List[str]
    category: AgentCategory = AgentCategory.UTILITY
    status: AgentStatus = AgentStatus.ACTIVE
    version: str = "1.0.0"
    author: str = "Unknown"
    tags: List[str] = None
    config_schema: Optional[Dict[str, Any]] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['created_at'] = self.created_at.isoformat()
        result['updated_at'] = self.updated_at.isoformat()
        result['category'] = self.category.value
        result['status'] = self.status.value
        result['capabilities'] = [cap.to_dict() for cap in self.capabilities]
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentInfo':
        """Create from dictionary."""
        # Convert string enums back to enum objects
        if isinstance(data['category'], str):
            data['category'] = AgentCategory(data['category'])
        if isinstance(data['status'], str):
            data['status'] = AgentStatus(data['status'])
        
        # Convert datetime strings back to datetime objects
        if isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        # Convert capabilities
        if 'capabilities' in data and isinstance(data['capabilities'], list):
            data['capabilities'] = [AgentCapability.from_dict(cap) for cap in data['capabilities']]
        
        return cls(**data)

class AgentRegistry:
    """Registry for managing available agent types."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                 registry_file: Optional[str] = None):
        """
        Initialize the agent registry.
        
        Args:
            config: Optional configuration dictionary
            registry_file: Path to registry file for persistence
        """
        self.config = config or {}
        self.registry_file = Path(registry_file) if registry_file else Path("data/agent_registry.json")
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Registry storage
        self.agents: Dict[str, AgentInfo] = {}
        self.capabilities: Dict[str, Set[str]] = {}  # capability -> agent names
        self.categories: Dict[AgentCategory, Set[str]] = {}  # category -> agent names
        
        # Performance tracking
        self.stats = {
            'total_agents': 0,
            'active_agents': 0,
            'total_capabilities': 0,
            'last_discovery': None
        }
        
        # Load existing registry
        self._load_registry()
        
        # Discover agents
        self._discover_agents()
        
        logger.info(f"AgentRegistry initialized with {len(self.agents)} agents")

    def _load_registry(self) -> None:
        """Load registry from file."""
        try:
            if self.registry_file.exists():
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                    
                # Load agents
                if 'agents' in data:
                    for name, agent_data in data['agents'].items():
                        self.agents[name] = AgentInfo.from_dict(agent_data)
                
                # Load capabilities index
                if 'capabilities' in data:
                    self.capabilities = {
                        cap: set(agents) for cap, agents in data['capabilities'].items()
                    }
                
                # Load categories index
                if 'categories' in data:
                    self.categories = {
                        AgentCategory(cat): set(agents) for cat, agents in data['categories'].items()
                    }
                
                logger.info(f"Loaded registry from {self.registry_file}")
                
        except Exception as e:
            logger.error(f"Error loading registry: {e}")

    def _save_registry(self) -> None:
        """Save registry to file."""
        try:
            data = {
                'agents': {name: agent.to_dict() for name, agent in self.agents.items()},
                'capabilities': {cap: list(agents) for cap, agents in self.capabilities.items()},
                'categories': {cat.value: list(agents) for cat, agents in self.categories.items()},
                'stats': self.stats,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Registry saved to {self.registry_file}")
            
        except Exception as e:
            logger.error(f"Error saving registry: {e}")

    def _discover_agents(self) -> None:
        """Discover available agents in the system."""
        try:
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
                            if agent_info:
                                self.register_agent(agent_info)
                            
                except ImportError as e:
                    logger.error(f"Failed to import agent directory {dir_path}: {e}")
                except Exception as e:
                    logger.error(f"Error discovering agents in {dir_path}: {e}")
            
            self.stats['last_discovery'] = datetime.now().isoformat()
            self._update_stats()
            self._save_registry()
            
        except Exception as e:
            logger.error(f"Error during agent discovery: {e}")

    def _get_agent_info(self, agent_class: Type[BaseAgent]) -> Optional[AgentInfo]:
        """
        Get information about an agent class.
        
        Args:
            agent_class: Agent class to analyze
            
        Returns:
            AgentInfo object containing agent metadata or None if error
        """
        try:
            # Get class docstring
            doc = inspect.getdoc(agent_class) or ""
            
            # Get capabilities from execute method
            capabilities = []
            execute_method = getattr(agent_class, "execute", None)
            if execute_method:
                sig = inspect.signature(execute_method)
                required_params = []
                optional_params = []
                
                for name, param in sig.parameters.items():
                    if name == 'self':
                        continue
                    if param.default == inspect.Parameter.empty:
                        required_params.append(name)
                    else:
                        optional_params.append(name)
                
                capabilities.append(AgentCapability(
                    name="execute",
                    description=inspect.getdoc(execute_method) or "Execute the agent",
                    required_params=required_params,
                    optional_params=optional_params,
                    return_type=str(sig.return_annotation),
                    version="1.0.0"
                ))
            
            # Get additional capabilities from other methods
            for name, method in inspect.getmembers(agent_class, inspect.isfunction):
                if name.startswith('_') or name == 'execute':
                    continue
                
                sig = inspect.signature(method)
                required_params = []
                optional_params = []
                
                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue
                    if param.default == inspect.Parameter.empty:
                        required_params.append(param_name)
                    else:
                        optional_params.append(param_name)
                
                capabilities.append(AgentCapability(
                    name=name,
                    description=inspect.getdoc(method) or f"{name} method",
                    required_params=required_params,
                    optional_params=optional_params,
                    return_type=str(sig.return_annotation),
                    version="1.0.0"
                ))
            
            # Get dependencies from class attributes
            dependencies = []
            for name, value in inspect.getmembers(agent_class):
                if name.startswith("_") and isinstance(value, (str, list)):
                    dependencies.extend(value if isinstance(value, list) else [value])
            
            # Determine category from class name or module
            category = self._determine_category(agent_class)
            
            # Get version and author from class attributes
            version = getattr(agent_class, '__version__', '1.0.0')
            author = getattr(agent_class, '__author__', 'Unknown')
            
            # Get tags from class attributes
            tags = getattr(agent_class, '__tags__', [])
            if isinstance(tags, str):
                tags = [tags]
            
            # Get config schema if available
            config_schema = getattr(agent_class, '__config_schema__', None)
            
            return AgentInfo(
                name=agent_class.__name__.lower(),
                class_name=agent_class.__name__,
                module_path=f"{agent_class.__module__}.{agent_class.__name__}",
                description=doc,
                capabilities=capabilities,
                dependencies=dependencies,
                category=category,
                version=version,
                author=author,
                tags=tags,
                config_schema=config_schema
            )
            
        except Exception as e:
            logger.error(f"Error getting agent info for {agent_class.__name__}: {e}")
            return None
    
    def _determine_category(self, agent_class: Type[BaseAgent]) -> AgentCategory:
        """Determine agent category based on class name and module."""
        class_name = agent_class.__name__.lower()
        module_name = agent_class.__module__.lower()
        
        # Check for category indicators in class name
        if any(word in class_name for word in ['trade', 'execution', 'order']):
            return AgentCategory.EXECUTION
        elif any(word in class_name for word in ['analysis', 'analyzer', 'research']):
            return AgentCategory.ANALYSIS
        elif any(word in class_name for word in ['monitor', 'watch', 'alert']):
            return AgentCategory.MONITORING
        elif any(word in class_name for word in ['trading', 'strategy']):
            return AgentCategory.TRADING
        
        # Check module path
        if 'execution' in module_name:
            return AgentCategory.EXECUTION
        elif 'analysis' in module_name:
            return AgentCategory.ANALYSIS
        elif 'monitoring' in module_name:
            return AgentCategory.MONITORING
        elif 'trading' in module_name:
            return AgentCategory.TRADING
        
        return AgentCategory.UTILITY
        
    def register_agent(self, agent_info: AgentInfo) -> bool:
        """
        Register an agent in the registry.
        
        Args:
            agent_info: Information about the agent to register
            
        Returns:
            True if registration was successful
        """
        try:
            self.agents[agent_info.name] = agent_info
            
            # Update capability index
            for capability in agent_info.capabilities:
                if capability.name not in self.capabilities:
                    self.capabilities[capability.name] = set()
                self.capabilities[capability.name].add(agent_info.name)
            
            # Update category index
            if agent_info.category not in self.categories:
                self.categories[agent_info.category] = set()
            self.categories[agent_info.category].add(agent_info.name)
            
            # Update timestamp
            agent_info.updated_at = datetime.now()
            
            logger.info(f"Registered agent: {agent_info.name} ({agent_info.category.value})")
            return True
            
        except Exception as e:
            logger.error(f"Error registering agent {agent_info.name}: {e}")
            return False

    def unregister_agent(self, name: str) -> bool:
        """
        Unregister an agent from the registry.
        
        Args:
            name: Name of the agent to unregister
            
        Returns:
            True if unregistration was successful
        """
        try:
            if name not in self.agents:
                logger.warning(f"Agent {name} not found in registry")
                return False
            
            agent_info = self.agents[name]
            
            # Remove from capability index
            for capability in agent_info.capabilities:
                if capability.name in self.capabilities:
                    self.capabilities[capability.name].discard(name)
                    if not self.capabilities[capability.name]:
                        del self.capabilities[capability.name]
            
            # Remove from category index
            if agent_info.category in self.categories:
                self.categories[agent_info.category].discard(name)
                if not self.categories[agent_info.category]:
                    del self.categories[agent_info.category]
            
            # Remove from agents
            del self.agents[name]
            
            logger.info(f"Unregistered agent: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error unregistering agent {name}: {e}")
            return False

    def get_agent(self, name: str) -> Optional[AgentInfo]:
        """
        Get information about an agent.
        
        Args:
            name: Name of the agent
            
        Returns:
            AgentInfo object or None if not found
        """
        return self.agents.get(name)
        
    def get_agents_by_capability(self, capability: str) -> List[AgentInfo]:
        """
        Get agents that provide a specific capability.
        
        Args:
            capability: Name of the capability
            
        Returns:
            List of agents with the capability
        """
        agent_names = self.capabilities.get(capability, set())
        return [self.agents[name] for name in agent_names if name in self.agents]
    
    def get_agents_by_category(self, category: Union[str, AgentCategory]) -> List[AgentInfo]:
        """
        Get agents in a specific category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of agents in the category
        """
        if isinstance(category, str):
            try:
                category = AgentCategory(category)
            except ValueError:
                logger.error(f"Invalid category: {category}")
                return []
        
        agent_names = self.categories.get(category, set())
        return [self.agents[name] for name in agent_names if name in self.agents]
    
    def get_agents_by_status(self, status: Union[str, AgentStatus]) -> List[AgentInfo]:
        """
        Get agents with a specific status.
        
        Args:
            status: Status to filter by
            
        Returns:
            List of agents with the status
        """
        if isinstance(status, str):
            try:
                status = AgentStatus(status)
            except ValueError:
                logger.error(f"Invalid status: {status}")
                return []
        
        return [agent for agent in self.agents.values() if agent.status == status]
        
    def list_agents(self, category: Optional[Union[str, AgentCategory]] = None,
                   status: Optional[Union[str, AgentStatus]] = None) -> List[str]:
        """
        Get list of registered agent names with optional filtering.
        
        Args:
            category: Filter by category
            status: Filter by status
            
        Returns:
            List of agent names
        """
        agents = list(self.agents.keys())
        
        if category:
            category_agents = set(self.get_agents_by_category(category))
            agents = [name for name in agents if self.agents[name] in category_agents]
        
        if status:
            status_agents = set(self.get_agents_by_status(status))
            agents = [name for name in agents if self.agents[name] in status_agents]
        
        return agents
        
    def list_capabilities(self) -> List[str]:
        """
        Get list of all available capabilities.
        
        Returns:
            List of capability names
        """
        return list(self.capabilities.keys())
    
    def list_categories(self) -> List[str]:
        """
        Get list of all available categories.
        
        Returns:
            List of category names
        """
        return [cat.value for cat in self.categories.keys()]
        
    def get_agent_class(self, name: str) -> Optional[Type[BaseAgent]]:
        """
        Get the class for an agent.
        
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
    
    def create_agent_instance(self, name: str, **kwargs) -> Optional[BaseAgent]:
        """
        Create an instance of an agent.
        
        Args:
            name: Name of the agent
            **kwargs: Arguments to pass to agent constructor
            
        Returns:
            Agent instance or None if creation failed
        """
        try:
            agent_class = self.get_agent_class(name)
            if not agent_class:
                return None
            
            return agent_class(**kwargs)
            
        except Exception as e:
            logger.error(f"Failed to create agent instance {name}: {e}")
            return None
    
    def update_agent_status(self, name: str, status: Union[str, AgentStatus]) -> bool:
        """
        Update the status of an agent.
        
        Args:
            name: Name of the agent
            status: New status
            
        Returns:
            True if update was successful
        """
        try:
            if name not in self.agents:
                logger.warning(f"Agent {name} not found")
                return False
            
            if isinstance(status, str):
                status = AgentStatus(status)
            
            self.agents[name].status = status
            self.agents[name].updated_at = datetime.now()
            
            logger.info(f"Updated agent {name} status to {status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating agent status: {e}")
            return False
    
    def search_agents(self, query: str) -> List[AgentInfo]:
        """
        Search agents by name, description, or tags.
        
        Args:
            query: Search query
            
        Returns:
            List of matching agents
        """
        query = query.lower()
        results = []
        
        for agent in self.agents.values():
            # Search in name
            if query in agent.name.lower():
                results.append(agent)
                continue
            
            # Search in description
            if query in agent.description.lower():
                results.append(agent)
                continue
            
            # Search in tags
            if any(query in tag.lower() for tag in agent.tags):
                results.append(agent)
                continue
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            **self.stats,
            'total_agents': len(self.agents),
            'active_agents': len(self.get_agents_by_status(AgentStatus.ACTIVE)),
            'total_capabilities': len(self.capabilities),
            'total_categories': len(self.categories),
            'registry_file': str(self.registry_file)
        }
    
    def export_registry(self, filepath: str = "agent_registry_export.json") -> bool:
        """
        Export registry to file.
        
        Args:
            filepath: Path to export file
            
        Returns:
            True if export was successful
        """
        try:
            data = {
                'agents': {name: agent.to_dict() for name, agent in self.agents.items()},
                'capabilities': {cap: list(agents) for cap, agents in self.capabilities.items()},
                'categories': {cat.value: list(agents) for cat, agents in self.categories.items()},
                'stats': self.get_stats(),
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Registry exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting registry: {e}")
            return False
    
    def _update_stats(self) -> None:
        """Update internal statistics."""
        self.stats['total_agents'] = len(self.agents)
        self.stats['active_agents'] = len(self.get_agents_by_status(AgentStatus.ACTIVE))
        self.stats['total_capabilities'] = len(self.capabilities)

# Global registry instance
_registry: Optional[AgentRegistry] = None

def get_registry() -> AgentRegistry:
    """Get the global registry instance."""
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry

# Convenience functions
def register_agent(agent_info: AgentInfo) -> bool:
    """Register an agent using the global registry."""
    return get_registry().register_agent(agent_info)

def get_agent(name: str) -> Optional[AgentInfo]:
    """Get agent information using the global registry."""
    return get_registry().get_agent(name)

def list_agents(**kwargs) -> List[str]:
    """List agents using the global registry."""
    return get_registry().list_agents(**kwargs)

def get_agent_class(name: str) -> Optional[Type[BaseAgent]]:
    """Get agent class using the global registry."""
    return get_registry().get_agent_class(name)

def create_agent_instance(name: str, **kwargs) -> Optional[BaseAgent]:
    """Create agent instance using the global registry."""
    return get_registry().create_agent_instance(name, **kwargs)