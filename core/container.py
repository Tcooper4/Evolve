"""
Dependency Injection Container

This module provides a service container for dependency injection,
enabling loose coupling between system components.
"""

import logging
from typing import Any, Dict, Type, Optional, Callable, Union
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

from .interfaces import (
    IDataProvider, IModel, IStrategy, IAgent, IExecutionEngine,
    IRiskManager, IPortfolioManager, IEventBus, IConfigManager,
    ILogger, IPlugin
)

logger = logging.getLogger(__name__)

class ServiceLifetime(Enum):
    """Service lifetime options."""
    TRANSIENT = "transient"  # New instance each time
    SINGLETON = "singleton"  # Single instance
    SCOPED = "scoped"       # Single instance per scope

@dataclass
class ServiceRegistration:
    """Registration information for a service."""
    interface: Type
    implementation: Type
    factory: Optional[Callable] = None
    lifetime: ServiceLifetime = ServiceLifetime.SINGLETON
    instance: Optional[Any] = None
    dependencies: Optional[Dict[str, Type]] = None

class ServiceContainer:
    """
    Dependency injection container for managing service dependencies.
    
    This container provides:
    - Service registration and resolution
    - Automatic dependency injection
    - Singleton and transient service management
    - Factory pattern support
    - Circular dependency detection
    """
    
    def __init__(self):
        """Initialize the service container."""
        self._services: Dict[Type, ServiceRegistration] = {}
        self._singletons: Dict[Type, Any] = {}
        self._scopes: Dict[str, Dict[Type, Any]] = {}
        self._current_scope: Optional[str] = None
        self._resolving: set = set()
        
        # Register core services
        self._register_core_services()
    
    def _register_core_services(self) -> None:
        """Register core system services."""
        # Register the container itself
        self.register_singleton(ServiceContainer, self)
        
        # Register logging
        self.register_singleton(ILogger, self._create_logger)
    
    def _create_logger(self) -> ILogger:
        """Create a logger instance."""
        from .logging import SystemLogger
        return SystemLogger()
    
    def register(
        self,
        interface: Type,
        implementation: Type,
        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
        factory: Optional[Callable] = None,
        dependencies: Optional[Dict[str, Type]] = None
    ) -> None:
        """
        Register a service with the container.
        
        Args:
            interface: The interface type
            implementation: The implementation type
            lifetime: Service lifetime
            factory: Optional factory function
            dependencies: Optional dependency overrides
        """
        registration = ServiceRegistration(
            interface=interface,
            implementation=implementation,
            factory=factory,
            lifetime=lifetime,
            dependencies=dependencies
        )
        
        self._services[interface] = registration
        logger.debug(f"Registered service: {interface.__name__} -> {implementation.__name__}")
    
    def register_singleton(
        self,
        interface: Type,
        implementation: Union[Type, Any],
        factory: Optional[Callable] = None,
        dependencies: Optional[Dict[str, Type]] = None
    ) -> None:
        """Register a singleton service."""
        if not isinstance(implementation, type):
            # Implementation is already an instance
            self._singletons[interface] = implementation
            self.register(interface, type(implementation), ServiceLifetime.SINGLETON, factory, dependencies)
        else:
            self.register(interface, implementation, ServiceLifetime.SINGLETON, factory, dependencies)
    
    def register_transient(
        self,
        interface: Type,
        implementation: Type,
        factory: Optional[Callable] = None,
        dependencies: Optional[Dict[str, Type]] = None
    ) -> None:
        """Register a transient service."""
        self.register(interface, implementation, ServiceLifetime.TRANSIENT, factory, dependencies)
    
    def register_scoped(
        self,
        interface: Type,
        implementation: Type,
        factory: Optional[Callable] = None,
        dependencies: Optional[Dict[str, Type]] = None
    ) -> None:
        """Register a scoped service."""
        self.register(interface, implementation, ServiceLifetime.SCOPED, factory, dependencies)
    
    def resolve(self, interface: Type) -> Any:
        """
        Resolve a service from the container.
        
        Args:
            interface: The interface type to resolve
            
        Returns:
            The resolved service instance
            
        Raises:
            ValueError: If service is not registered
            RuntimeError: If circular dependency detected
        """
        if interface not in self._services:
            raise ValueError(f"Service {interface.__name__} is not registered")
        
        registration = self._services[interface]
        
        # Check for circular dependencies
        if interface in self._resolving:
            raise RuntimeError(f"Circular dependency detected for {interface.__name__}")
        
        # Handle different lifetimes
        if registration.lifetime == ServiceLifetime.SINGLETON:
            return self._resolve_singleton(registration)
        elif registration.lifetime == ServiceLifetime.SCOPED:
            return self._resolve_scoped(registration)
        else:  # TRANSIENT
            return self._resolve_transient(registration)
    
    def _resolve_singleton(self, registration: ServiceRegistration) -> Any:
        """Resolve a singleton service."""
        if registration.interface in self._singletons:
            return self._singletons[registration.interface]
        
        instance = self._create_instance(registration)
        self._singletons[registration.interface] = instance
        return instance
    
    def _resolve_scoped(self, registration: ServiceRegistration) -> Any:
        """Resolve a scoped service."""
        if self._current_scope is None:
            raise RuntimeError("No active scope for scoped service")
        
        if self._current_scope not in self._scopes:
            self._scopes[self._current_scope] = {}
        
        if registration.interface in self._scopes[self._current_scope]:
            return self._scopes[self._current_scope][registration.interface]
        
        instance = self._create_instance(registration)
        self._scopes[self._current_scope][registration.interface] = instance
        return instance
    
    def _resolve_transient(self, registration: ServiceRegistration) -> Any:
        """Resolve a transient service."""
        return self._create_instance(registration)
    
    def _create_instance(self, registration: ServiceRegistration) -> Any:
        """Create a new instance of a service."""
        self._resolving.add(registration.interface)
        
        try:
            if registration.factory:
                # Use factory function
                return registration.factory(self)
            else:
                # Use constructor injection
                return self._create_instance_with_injection(registration)
        finally:
            self._resolving.remove(registration.interface)
    
    def _create_instance_with_injection(self, registration: ServiceRegistration) -> Any:
        """Create instance with constructor dependency injection."""
        implementation = registration.implementation
        
        # Get constructor parameters
        import inspect
        sig = inspect.signature(implementation.__init__)
        params = sig.parameters
        
        # Skip 'self' parameter
        param_names = list(params.keys())[1:]
        
        # Resolve dependencies
        args = []
        for param_name in param_names:
            param = params[param_name]
            
            # Check for dependency override
            if registration.dependencies and param_name in registration.dependencies:
                dependency_type = registration.dependencies[param_name]
            else:
                # Try to infer type from annotation
                dependency_type = param.annotation
                
                if dependency_type == inspect.Parameter.empty:
                    # No type annotation, try to find by name
                    dependency_type = self._find_service_by_name(param_name)
            
            if dependency_type is None:
                raise ValueError(f"Cannot resolve dependency '{param_name}' for {implementation.__name__}")
            
            args.append(self.resolve(dependency_type))
        
        return implementation(*args)
    
    def _find_service_by_name(self, name: str) -> Optional[Type]:
        """Find a service by parameter name."""
        # This is a simple heuristic - in practice, you'd want more sophisticated logic
        for interface in self._services.keys():
            if interface.__name__.lower() == name.lower():
                return interface
        return None
    
    def create_scope(self, scope_name: str) -> 'ServiceScope':
        """Create a new service scope."""
        return ServiceScope(self, scope_name)
    
    def get_registered_services(self) -> Dict[str, str]:
        """Get list of registered services."""
        return {
            interface.__name__: registration.implementation.__name__
            for interface, registration in self._services.items()
        }
    
    def is_registered(self, interface: Type) -> bool:
        """Check if a service is registered."""
        return interface in self._services
    
    def clear(self) -> None:
        """Clear all registered services."""
        self._services.clear()
        self._singletons.clear()
        self._scopes.clear()
        self._resolving.clear()

class ServiceScope:
    """Context manager for service scopes."""
    
    def __init__(self, container: ServiceContainer, scope_name: str):
        self.container = container
        self.scope_name = scope_name
        self._previous_scope = None
    
    def __enter__(self):
        """Enter the scope."""
        self._previous_scope = self.container._current_scope
        self.container._current_scope = self.scope_name
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the scope."""
        self.container._current_scope = self._previous_scope
        
        # Clean up scoped instances
        if self.scope_name in self.container._scopes:
            del self.container._scopes[self.scope_name]

# Global container instance
_container: Optional[ServiceContainer] = None

def get_container() -> ServiceContainer:
    """Get the global service container instance."""
    global _container
    if _container is None:
        _container = ServiceContainer()
    return _container

def register_service(
    interface: Type,
    implementation: Type,
    lifetime: ServiceLifetime = ServiceLifetime.SINGLETON
) -> None:
    """Register a service with the global container."""
    container = get_container()
    container.register(interface, implementation, lifetime)

def resolve_service(interface: Type) -> Any:
    """Resolve a service from the global container."""
    container = get_container()
    return container.resolve(interface)

def create_scope(scope_name: str) -> ServiceScope:
    """Create a scope with the global container."""
    container = get_container()
    return container.create_scope(scope_name)

# Convenience functions for common services
def get_data_provider() -> IDataProvider:
    """Get the data provider service."""
    return resolve_service(IDataProvider)

def get_model_service() -> IModel:
    """Get the model service."""
    return resolve_service(IModel)

def get_strategy_service() -> IStrategy:
    """Get the strategy service."""
    return resolve_service(IStrategy)

def get_agent_service() -> IAgent:
    """Get the agent service."""
    return resolve_service(IAgent)

def get_execution_engine() -> IExecutionEngine:
    """Get the execution engine service."""
    return resolve_service(IExecutionEngine)

def get_risk_manager() -> IRiskManager:
    """Get the risk manager service."""
    return resolve_service(IRiskManager)

def get_portfolio_manager() -> IPortfolioManager:
    """Get the portfolio manager service."""
    return resolve_service(IPortfolioManager)

def get_event_bus() -> IEventBus:
    """Get the event bus service."""
    return resolve_service(IEventBus)

def get_config_manager() -> IConfigManager:
    """Get the configuration manager service."""
    return resolve_service(IConfigManager)

def get_logger() -> ILogger:
    """Get the logger service."""
    return resolve_service(ILogger)

__all__ = [
    'ServiceContainer', 'ServiceScope', 'ServiceLifetime', 'ServiceRegistration',
    'get_container', 'register_service', 'resolve_service', 'create_scope',
    'get_data_provider', 'get_model_service', 'get_strategy_service',
    'get_agent_service', 'get_execution_engine', 'get_risk_manager',
    'get_portfolio_manager', 'get_event_bus', 'get_config_manager', 'get_logger'
] 