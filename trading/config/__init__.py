"""
Configuration package for the trading system.

This module provides centralized configuration management for all trading components
including models, data sources, web services, monitoring, and agent settings.
"""

from datetime import datetime
from typing import Any, Dict

from .configuration import (
    ConfigManager,
    DataConfig,
    ModelConfig,
    MonitoringConfig,
    TrainingConfig,
    WebConfig,
)
from .enhanced_settings import (
    AgentConfig,
    EnhancedSettings,
    PerformanceConfig,
    RiskConfig,
    TradingConfig,
)

# Import commonly used settings explicitly
from .settings import (
    AGENT_LOG_LEVEL,
    AGENT_MEMORY_SIZE,
    AGENT_TIMEOUT,
    ALPHA_VANTAGE_API_KEY,
    BACKTEST_DAYS,
    BATCH_SIZE,
    CACHE_DIR,
    DATA_DIR,
    DATA_LOG_LEVEL,
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_PORT,
    DB_USER,
    DEBUG,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_MODEL,
    DEFAULT_STRATEGY,
    DEFAULT_TICKERS,
    ENV,
    GRAFANA_PORT,
    HUGGINGFACE_API_KEY,
    HUGGINGFACE_MODEL,
    JWT_SECRET_KEY,
    LEARNING_RATE,
    LOG_LEVEL,
    MAX_CONCURRENT_AGENTS,
    MEMORY_BACKEND,
    MEMORY_DIR,
    METRIC_LOGGING_ENABLED,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_LOG_LEVEL,
    OPENAI_API_KEY,
    POLYGON_API_KEY,
    PROMETHEUS_PORT,
    REDIS_DB,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    STRATEGY_DIR,
    STRATEGY_REGISTRY_PATH,
    STRATEGY_SWITCH_API_ENDPOINT,
    STRATEGY_SWITCH_BACKEND,
    STRATEGY_SWITCH_LOCK_TIMEOUT,
    STRATEGY_SWITCH_LOG_PATH,
    WEB_DEBUG,
    WEB_HOST,
    WEB_PORT,
    WEB_SECRET_KEY,
    get_config_dict,
    get_config_value,
    validate_config,
)


class Config:
    """Configuration object that provides .get() method for accessing settings."""

    def __init__(self):
        """Initialize configuration object."""
        self._settings = settings
        self._config_manager = ConfigManager()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        try:
            # Try to get from settings first
            if hasattr(self._settings, key):
                return getattr(self._settings, key)

            # Try to get from config manager
            return self._config_manager.get(key, default)

        except Exception as e:
            return default

    def set(self, key: str, value: Any) -> bool:
        """Set a configuration value.

        Args:
            key: Configuration key
            value: Configuration value

        Returns:
            True if successful, False otherwise
        """
        try:
            if hasattr(self._settings, key):
                setattr(self._settings, key, value)
                return True
            return False
        except Exception:
            return False

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values.

        Returns:
            Dictionary of all configuration values
        """
        try:
            return get_config_dict()
        except Exception:
            return {}

    def validate(self) -> Dict[str, Any]:
        """Validate all configuration values.

        Returns:
            Validation result dictionary
        """
        try:
            return validate_config()
        except Exception as e:
            return {
                "success": False,
                "message": f"Configuration validation failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
            }


# Create a config instance
config = Config()

__all__ = [
    # Configuration classes
    "ConfigManager",
    "ModelConfig",
    "DataConfig",
    "TrainingConfig",
    "WebConfig",
    "MonitoringConfig",
    "EnhancedSettings",
    "TradingConfig",
    "AgentConfig",
    "RiskConfig",
    "PerformanceConfig",
    # Configuration instance
    "config",
    # Utility functions
    "get_config_value",
    "get_config_dict",
    "validate_config",
    # Environment variables
    "ENV",
    "DEBUG",
    "LOG_LEVEL",
    "AGENT_LOG_LEVEL",
    "MODEL_LOG_LEVEL",
    "DATA_LOG_LEVEL",
    "ALPHA_VANTAGE_API_KEY",
    "POLYGON_API_KEY",
    "OPENAI_API_KEY",
    "JWT_SECRET_KEY",
    "WEB_SECRET_KEY",
    "DB_HOST",
    "DB_PORT",
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD",
    "REDIS_HOST",
    "REDIS_PORT",
    "REDIS_DB",
    "REDIS_PASSWORD",
    "MODEL_DIR",
    "DEFAULT_MODEL",
    "BATCH_SIZE",
    "LEARNING_RATE",
    "DATA_DIR",
    "CACHE_DIR",
    "DEFAULT_TICKERS",
    "WEB_HOST",
    "WEB_PORT",
    "WEB_DEBUG",
    "PROMETHEUS_PORT",
    "GRAFANA_PORT",
    "AGENT_TIMEOUT",
    "MAX_CONCURRENT_AGENTS",
    "AGENT_MEMORY_SIZE",
    "STRATEGY_DIR",
    "DEFAULT_STRATEGY",
    "BACKTEST_DAYS",
    "DEFAULT_LLM_PROVIDER",
    "HUGGINGFACE_API_KEY",
    "HUGGINGFACE_MODEL",
    "MEMORY_DIR",
    "MEMORY_BACKEND",
    "METRIC_LOGGING_ENABLED",
    "METRICS_PATH",
    "STRATEGY_SWITCH_LOG_PATH",
    "STRATEGY_REGISTRY_PATH",
    "STRATEGY_SWITCH_LOCK_TIMEOUT",
    "STRATEGY_SWITCH_BACKEND",
    "STRATEGY_SWITCH_API_ENDPOINT",
]

__version__ = "1.0.0"
__author__ = "Evolve Trading System"
__description__ = "Trading System Configuration Management"
