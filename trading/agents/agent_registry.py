# -*- coding: utf-8 -*-
"""
Agent registry for the trading system.

This module provides a registry of available agent types and their capabilities,
enabling dynamic agent discovery and loading.
"""

import importlib
import inspect
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union

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
class AgentRegistrationRequest:
    """Agent registration request."""

    action: str  # 'register', 'unregister', 'get', 'list', 'search', 'update_status'
    agent_info: Optional["AgentInfo"] = None
    agent_name: Optional[str] = None
    status: Optional[str] = None
    query: Optional[str] = None
    category: Optional[str] = None


@dataclass
class AgentRegistrationResult:
    """Agent registration result."""

    success: bool
    data: Dict[str, Any]
    error_message: Optional[str] = None


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
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCapability":
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
        result["created_at"] = self.created_at.isoformat()
        result["updated_at"] = self.updated_at.isoformat()
        result["category"] = self.category.value
        result["status"] = self.status.value
        result["capabilities"] = [cap.to_dict() for cap in self.capabilities]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentInfo":
        """Create from dictionary."""
        # Convert string enums back to enum objects
        if isinstance(data["category"], str):
            data["category"] = AgentCategory(data["category"])
        if isinstance(data["status"], str):
            data["status"] = AgentStatus(data["status"])

        # Convert datetime strings back to datetime objects
        if isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])

        # Convert capabilities
        if "capabilities" in data and isinstance(data["capabilities"], list):
            data["capabilities"] = [
                AgentCapability.from_dict(cap) for cap in data["capabilities"]
            ]

        return cls(**data)


class AgentRegistry:
    """Registry for managing available agent types."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        registry_file: Optional[str] = None,
    ):
        """
        Initialize the agent registry.

        Args:
            config: Optional configuration dictionary
            registry_file: Path to registry file for persistence
        """
        self.config = config or {}
        self.registry_file = (
            Path(registry_file) if registry_file else Path("data/agent_registry.json")
        )
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)

        # Registry storage
        self.agents: Dict[str, AgentInfo] = {}
        self.capabilities: Dict[str, Set[str]] = {}  # capability -> agent names
        self.categories: Dict[AgentCategory, Set[str]] = {}  # category -> agent names

        # Performance tracking
        self.stats = {
            "total_agents": 0,
            "active_agents": 0,
            "total_capabilities": 0,
            "last_discovery": None,
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
                with open(self.registry_file, "r") as f:
                    data = json.load(f)

                # Load agents
                if "agents" in data:
                    for name, agent_data in data["agents"].items():
                        self.agents[name] = AgentInfo.from_dict(agent_data)

                # Load capabilities index
                if "capabilities" in data:
                    self.capabilities = {
                        cap: set(agents) for cap, agents in data["capabilities"].items()
                    }

                # Load categories index
                if "categories" in data:
                    self.categories = {
                        AgentCategory(cat): set(agents)
                        for cat, agents in data["categories"].items()
                    }

                logger.info(f"Loaded registry from {self.registry_file}")

        except Exception as e:
            logger.error(f"Error loading registry: {e}")

    def _save_registry(self) -> None:
        """Save registry to file."""
        try:
            data = {
                "agents": {
                    name: agent.to_dict() for name, agent in self.agents.items()
                },
                "capabilities": {
                    cap: list(agents) for cap, agents in self.capabilities.items()
                },
                "categories": {
                    cat.value: list(agents) for cat, agents in self.categories.items()
                },
                "stats": self.stats,
                "last_updated": datetime.now().isoformat(),
            }

            with open(self.registry_file, "w") as f:
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
                        if (
                            inspect.isclass(obj)
                            and issubclass(obj, BaseAgent)
                            and obj != BaseAgent
                        ):
                            # Get agent info
                            agent_info = self._get_agent_info(obj)
                            if agent_info:
                                self.register_agent(agent_info)

                except ImportError as e:
                    logger.error(f"Failed to import agent directory {dir_path}: {e}")
                except Exception as e:
                    logger.error(f"Error discovering agents in {dir_path}: {e}")

            self.stats["last_discovery"] = datetime.now().isoformat()
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
                    if name == "self":
                        continue
                    if param.default == inspect.Parameter.empty:
                        required_params.append(name)
                    else:
                        optional_params.append(name)

                capabilities.append(
                    AgentCapability(
                        name="execute",
                        description=inspect.getdoc(execute_method)
                        or "Execute the agent",
                        required_params=required_params,
                        optional_params=optional_params,
                        return_type=str(sig.return_annotation),
                        version="1.0.0",
                    )
                )

            # Get additional capabilities from other methods
            for name, method in inspect.getmembers(agent_class, inspect.isfunction):
                if name.startswith("_") or name == "execute":
                    continue

                sig = inspect.signature(method)
                required_params = []
                optional_params = []

                for param_name, param in sig.parameters.items():
                    if param_name == "self":
                        continue
                    if param.default == inspect.Parameter.empty:
                        required_params.append(param_name)
                    else:
                        optional_params.append(param_name)

                capabilities.append(
                    AgentCapability(
                        name=name,
                        description=inspect.getdoc(method) or f"{name} method",
                        required_params=required_params,
                        optional_params=optional_params,
                        return_type=str(sig.return_annotation),
                        version="1.0.0",
                    )
                )

            # Get dependencies from class attributes
            dependencies = []
            for name, value in inspect.getmembers(agent_class):
                if name.startswith("_") and isinstance(value, (str, list)):
                    dependencies.extend(value if isinstance(value, list) else [value])

            # Determine category from class name or module
            category = self._determine_category(agent_class)

            # Get version and author from class attributes
            version = getattr(agent_class, "__version__", "1.0.0")
            author = getattr(agent_class, "__author__", "Unknown")

            # Get tags from class attributes
            tags = getattr(agent_class, "__tags__", [])
            if isinstance(tags, str):
                tags = [tags]

            # Get config schema if available
            config_schema = getattr(agent_class, "__config_schema__", None)

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
                config_schema=config_schema,
            )

        except Exception as e:
            logger.error(f"Error getting agent info for {agent_class.__name__}: {e}")
            return None

    def _determine_category(self, agent_class: Type[BaseAgent]) -> AgentCategory:
        """Determine agent category based on class name and module."""
        class_name = agent_class.__name__.lower()
        module_name = agent_class.__module__.lower()

        # Check for category indicators in class name
        if any(word in class_name for word in ["trade", "execution", "order"]):
            return AgentCategory.EXECUTION
        elif any(word in class_name for word in ["analysis", "analyzer", "research"]):
            return AgentCategory.ANALYSIS
        elif any(word in class_name for word in ["monitor", "watch", "alert"]):
            return AgentCategory.MONITORING
        elif any(word in class_name for word in ["trading", "strategy"]):
            return AgentCategory.TRADING

        # Check module path
        if "execution" in module_name:
            return AgentCategory.EXECUTION
        elif "analysis" in module_name:
            return AgentCategory.ANALYSIS
        elif "monitoring" in module_name:
            return AgentCategory.MONITORING
        elif "trading" in module_name:
            return AgentCategory.TRADING

        return AgentCategory.UTILITY

    def register_agent(self, agent_info: AgentInfo) -> bool:
        """Register an agent in the registry.

        Args:
            agent_info: Agent information to register

        Returns:
            True if registration was successful
        """
        try:
            # Validate agent info
            validation_result = self._validate_agent_info(agent_info)
            if not validation_result["valid"]:
                logger.error(f"Agent validation failed: {validation_result['errors']}")
                return False

            # Validate agent class
            agent_class = self._get_agent_class_from_info(agent_info)
            if agent_class:
                class_validation = self._validate_agent_class(agent_class)
                if not class_validation["valid"]:
                    logger.error(
                        f"Agent class validation failed: {class_validation['errors']}"
                    )
                    return False

            # Check for conflicts
            if agent_info.name in self.agents:
                existing_agent = self.agents[agent_info.name]
                if existing_agent.version == agent_info.version:
                    logger.warning(
                        f"Agent {agent_info.name} version {agent_info.version} already registered"
                    )
                    return False

            # Register the agent
            self.agents[agent_info.name] = agent_info

            # Update indexes
            self._update_capability_index(agent_info)
            self._update_category_index(agent_info)

            # Update stats
            self._update_stats()

            # Save registry
            self._save_registry()

            logger.info(
                f"Registered agent: {agent_info.name} (version {agent_info.version})"
            )
            return True

        except Exception as e:
            logger.error(f"Error registering agent {agent_info.name}: {e}")
            return False

    def _validate_agent_info(self, agent_info: AgentInfo) -> Dict[str, Any]:
        """Validate agent information before registration.

        Args:
            agent_info: Agent information to validate

        Returns:
            Validation result with 'valid' boolean and 'errors' list
        """
        errors = []

        # Required fields validation
        if not agent_info.name or not agent_info.name.strip():
            errors.append("Agent name is required and cannot be empty")

        if not agent_info.class_name or not agent_info.class_name.strip():
            errors.append("Class name is required and cannot be empty")

        if not agent_info.module_path or not agent_info.module_path.strip():
            errors.append("Module path is required and cannot be empty")

        if not agent_info.description or not agent_info.description.strip():
            errors.append("Description is required and cannot be empty")

        # Name format validation
        if agent_info.name and not self._is_valid_agent_name(agent_info.name):
            errors.append("Agent name must be alphanumeric with underscores only")

        # Version validation
        if agent_info.version and not self._is_valid_version(agent_info.version):
            errors.append("Version must be in semantic versioning format (e.g., 1.0.0)")

        # Capabilities validation
        if not agent_info.capabilities:
            errors.append("At least one capability must be defined")
        else:
            for i, capability in enumerate(agent_info.capabilities):
                cap_errors = self._validate_capability(capability)
                for error in cap_errors:
                    errors.append(f"Capability {i + 1}: {error}")

        # Dependencies validation
        if agent_info.dependencies:
            for dep in agent_info.dependencies:
                if not self._is_valid_dependency(dep):
                    errors.append(f"Invalid dependency format: {dep}")

        # Config schema validation
        if agent_info.config_schema:
            schema_errors = self._validate_config_schema(agent_info.config_schema)
            errors.extend(schema_errors)

        return {"valid": len(errors) == 0, "errors": errors}

    def _validate_agent_class(self, agent_class: Type[BaseAgent]) -> Dict[str, Any]:
        """Validate agent class structure and implementation.

        Args:
            agent_class: Agent class to validate

        Returns:
            Validation result with 'valid' boolean and 'errors' list
        """
        errors = []

        try:
            # Check if class inherits from BaseAgent
            if not issubclass(agent_class, BaseAgent):
                errors.append("Agent class must inherit from BaseAgent")

            # Check for required methods
            required_methods = ["run", "validate_input", "handle_error"]
            for method_name in required_methods:
                if not hasattr(agent_class, method_name):
                    errors.append(f"Required method '{method_name}' not found")

            # Check for proper method signatures
            if hasattr(agent_class, "run"):
                run_method = getattr(agent_class, "run")
                if not callable(run_method):
                    errors.append("'run' method must be callable")
                else:
                    # Check if it's an abstract method
                    if (
                        hasattr(run_method, "__isabstractmethod__")
                        and run_method.__isabstractmethod__
                    ):
                        errors.append("'run' method must be implemented (not abstract)")

            # Check for proper constructor
            if hasattr(agent_class, "__init__"):
                init_method = getattr(agent_class, "__init__")
                if not callable(init_method):
                    errors.append("Constructor must be callable")

            # Check for proper string representation
            if not hasattr(agent_class, "__str__") and not hasattr(
                agent_class, "__repr__"
            ):
                errors.append(
                    "Agent class should have __str__ or __repr__ method for debugging"
                )

        except Exception as e:
            errors.append(f"Error validating agent class: {str(e)}")

        return {"valid": len(errors) == 0, "errors": errors}

    def _validate_capability(self, capability: AgentCapability) -> List[str]:
        """Validate a single capability definition.

        Args:
            capability: Capability to validate

        Returns:
            List of validation errors
        """
        errors = []

        if not capability.name or not capability.name.strip():
            errors.append("Capability name is required")

        if not capability.description or not capability.description.strip():
            errors.append("Capability description is required")

        if not isinstance(capability.required_params, list):
            errors.append("Required parameters must be a list")

        if not isinstance(capability.optional_params, list):
            errors.append("Optional parameters must be a list")

        if not capability.return_type or not capability.return_type.strip():
            errors.append("Return type is required")

        if capability.version and not self._is_valid_version(capability.version):
            errors.append("Capability version must be in semantic versioning format")

        return errors

    def _validate_config_schema(self, schema: Dict[str, Any]) -> List[str]:
        """Validate configuration schema structure.

        Args:
            schema: Configuration schema to validate

        Returns:
            List of validation errors
        """
        errors = []

        if not isinstance(schema, dict):
            errors.append("Config schema must be a dictionary")
            return errors

        # Check for required schema fields
        required_fields = ["type", "properties"]
        for field in required_fields:
            if field not in schema:
                errors.append(f"Config schema missing required field: {field}")

        # Validate schema type
        if "type" in schema and schema["type"] not in [
            "object",
            "array",
            "string",
            "number",
            "boolean",
        ]:
            errors.append("Invalid schema type")

        # Validate properties if present
        if "properties" in schema and isinstance(schema["properties"], dict):
            for prop_name, prop_schema in schema["properties"].items():
                if not isinstance(prop_schema, dict):
                    errors.append(f"Property '{prop_name}' schema must be a dictionary")
                elif "type" not in prop_schema:
                    errors.append(f"Property '{prop_name}' missing type definition")

        return errors

    def _is_valid_agent_name(self, name: str) -> bool:
        """Check if agent name follows naming conventions.

        Args:
            name: Agent name to validate

        Returns:
            True if name is valid
        """
        import re

        # Allow alphanumeric characters and underscores, must start with letter
        pattern = r"^[a-zA-Z][a-zA-Z0-9_]*$"
        return bool(re.match(pattern, name))

    def _is_valid_version(self, version: str) -> bool:
        """Check if version follows semantic versioning.

        Args:
            version: Version string to validate

        Returns:
            True if version is valid
        """
        import re

        # Basic semantic versioning pattern (major.minor.patch)
        pattern = r"^\d+\.\d+\.\d+$"
        return bool(re.match(pattern, version))

    def _is_valid_dependency(self, dependency: str) -> bool:
        """Check if dependency string is valid.

        Args:
            dependency: Dependency string to validate

        Returns:
            True if dependency is valid
        """
        import re

        pattern = r"^[a-zA-Z][a-zA-Z0-9_-]*(\s*[<>=!]+\s*\d+\.\d+\.\d+)?$"
        return bool(re.match(pattern, dependency))

    def _get_agent_class_from_info(
        self, agent_info: AgentInfo
    ) -> Optional[Type[BaseAgent]]:
        """Get agent class from agent info.

        Args:
            agent_info: Agent information

        Returns:
            Agent class or None if not found
        """
        try:
            module = importlib.import_module(agent_info.module_path)
            agent_class = getattr(module, agent_info.class_name, None)
            return agent_class if agent_class else None
        except Exception as e:
            logger.debug(
                f"Could not load agent class {agent_info.class_name} from {agent_info.module_path}: {e}"
            )
            return None

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

    def get_agents_by_category(
        self, category: Union[str, AgentCategory]
    ) -> List[AgentInfo]:
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

    def list_agents(
        self,
        category: Optional[Union[str, AgentCategory]] = None,
        status: Optional[Union[str, AgentStatus]] = None,
    ) -> List[str]:
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
            "total_agents": len(self.agents),
            "active_agents": len(self.get_agents_by_status(AgentStatus.ACTIVE)),
            "total_capabilities": len(self.capabilities),
            "total_categories": len(self.categories),
            "registry_file": str(self.registry_file),
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
                "agents": {
                    name: agent.to_dict() for name, agent in self.agents.items()
                },
                "capabilities": {
                    cap: list(agents) for cap, agents in self.capabilities.items()
                },
                "categories": {
                    cat.value: list(agents) for cat, agents in self.categories.items()
                },
                "stats": self.get_stats(),
                "export_timestamp": datetime.now().isoformat(),
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Registry exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error exporting registry: {e}")
            return False

    def _update_stats(self) -> None:
        """Update internal statistics."""
        self.stats["total_agents"] = len(self.agents)
        self.stats["active_agents"] = len(self.get_agents_by_status(AgentStatus.ACTIVE))
        self.stats["total_capabilities"] = len(self.capabilities)

    def _update_capability_index(self, agent_info: AgentInfo) -> None:
        """Update the capability index for an agent."""
        for capability in agent_info.capabilities:
            if capability.name not in self.capabilities:
                self.capabilities[capability.name] = set()
            self.capabilities[capability.name].add(agent_info.name)

    def _update_category_index(self, agent_info: AgentInfo) -> None:
        """Update the category index for an agent."""
        if agent_info.category not in self.categories:
            self.categories[agent_info.category] = set()
        self.categories[agent_info.category].add(agent_info.name)


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
