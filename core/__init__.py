"""
Core Module for Evolve Trading Platform

This module provides the foundational architecture components including:
- Dependency injection container
- Event-driven architecture
- Plugin system
- Interface definitions
- Logging system
"""

from typing import Dict, Any

# Import interfaces
from .interfaces import (
    # Data types
    DataRequest, DataResponse, ModelConfig, TrainingResult,
    PredictionResult, SignalData, TradeSignal, EventType, SystemEvent,
    
    # Core interfaces
    IDataProvider, IModel, IStrategy, IAgent, IExecutionEngine,
    IRiskManager, IPortfolioManager, IEventBus, IConfigManager,
    ILogger, IPlugin
)

# Import dependency injection
from .container import (
    ServiceContainer, ServiceScope, ServiceLifetime, ServiceRegistration,
    get_container, register_service, resolve_service, create_scope,
    get_data_provider, get_model_service, get_strategy_service,
    get_agent_service, get_execution_engine, get_risk_manager,
    get_portfolio_manager, get_event_bus, get_config_manager, get_logger
)

# Import event system
from .events import (
    EventBus, EventPriority, EventHandler,
    publish_data_loaded, publish_model_trained, publish_signal_generated,
    publish_trade_executed, publish_risk_alert, publish_system_error,
    get_event_bus, set_event_bus
)

# Import plugin system
from .plugins import (
    PluginManager, PluginInfo, PluginStatus,
    get_plugin_manager, set_plugin_manager
)

# Import logging
from .logging import (
    SystemLogger, get_logger,
    log_info, log_warning, log_error, log_debug, log_event
)

# Import existing components
from .agent_hub import AgentHub
from .capability_router import CapabilityRouter
from .error_handler import ErrorHandler
from .session_utils import SessionUtils
from .sanity_checks import SanityChecks

# Version information
__version__ = "2.1.0"
__author__ = "Evolve Team"
__description__ = "Core architecture components for Evolve trading platform"

# Export all public components
__all__ = [
    # Interfaces and data types
    'DataRequest', 'DataResponse', 'ModelConfig', 'TrainingResult',
    'PredictionResult', 'SignalData', 'TradeSignal', 'EventType', 'SystemEvent',
    'IDataProvider', 'IModel', 'IStrategy', 'IAgent', 'IExecutionEngine',
    'IRiskManager', 'IPortfolioManager', 'IEventBus', 'IConfigManager',
    'ILogger', 'IPlugin',
    
    # Dependency injection
    'ServiceContainer', 'ServiceScope', 'ServiceLifetime', 'ServiceRegistration',
    'get_container', 'register_service', 'resolve_service', 'create_scope',
    'get_data_provider', 'get_model_service', 'get_strategy_service',
    'get_agent_service', 'get_execution_engine', 'get_risk_manager',
    'get_portfolio_manager', 'get_event_bus', 'get_config_manager', 'get_logger',
    
    # Event system
    'EventBus', 'EventPriority', 'EventHandler',
    'publish_data_loaded', 'publish_model_trained', 'publish_signal_generated',
    'publish_trade_executed', 'publish_risk_alert', 'publish_system_error',
    'get_event_bus', 'set_event_bus',
    
    # Plugin system
    'PluginManager', 'PluginInfo', 'PluginStatus',
    'get_plugin_manager', 'set_plugin_manager',
    
    # Logging
    'SystemLogger', 'get_logger',
    'log_info', 'log_warning', 'log_error', 'log_debug', 'log_event',
    
    # Existing components
    'AgentHub', 'CapabilityRouter', 'ErrorHandler', 'SessionUtils', 'SanityChecks'
]

def initialize_core_system() -> Dict[str, Any]:
    """
    Initialize the core system components.
    
    Returns:
        Dictionary containing initialized components
    """
    from .logging import log_info
    
    log_info("Initializing core system components")
    
    # Initialize service container
    container = get_container()
    
    # Initialize event bus
    event_bus = get_event_bus()
    
    # Initialize plugin manager
    plugin_manager = get_plugin_manager()
    
    # Initialize logger
    logger = get_logger()
    
    # Register core services
    container.register_singleton(IEventBus, event_bus)
    container.register_singleton(ILogger, logger)
    
    log_info("Core system initialization complete")
    
    return {
        'container': container,
        'event_bus': event_bus,
        'plugin_manager': plugin_manager,
        'logger': logger
    }

def get_system_status() -> Dict[str, Any]:
    """
    Get the status of all core system components.
    
    Returns:
        Dictionary containing system status information
    """
    try:
        container = get_container()
        event_bus = get_event_bus()
        plugin_manager = get_plugin_manager()
        logger = get_logger()
        
        return {
            'status': 'healthy',
            'components': {
                'service_container': {
                    'status': 'active',
                    'registered_services': len(container.get_registered_services())
                },
                'event_bus': {
                    'status': 'active',
                    'stats': event_bus.get_stats()
                },
                'plugin_manager': {
                    'status': 'active',
                    'stats': plugin_manager.get_plugin_stats()
                },
                'logger': {
                    'status': 'active',
                    'stats': logger.get_stats()
                }
            }
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

def shutdown_core_system() -> None:
    """Shutdown the core system components."""
    from .logging import log_info
    
    log_info("Shutting down core system components")
    
    try:
        # Shutdown event bus
        event_bus = get_event_bus()
        event_bus.shutdown()
        
        # Cleanup logger
        logger = get_logger()
        logger.cleanup()
        
        log_info("Core system shutdown complete")
    except Exception as e:
        log_info(f"Error during core system shutdown: {e}") 