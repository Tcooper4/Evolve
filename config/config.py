"""
Configuration utilities and simplified interface.

This module provides utilities for configuration management and a simplified
interface for accessing configuration settings.
"""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .app_config import AppConfig, get_config

logger = logging.getLogger(__name__)


class Config:
    """
    Simplified configuration interface.

    This class provides a simplified interface for accessing configuration
    settings with fallback values and type conversion.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize configuration interface.

        Args:
            config: AppConfig instance, uses global config if None
        """
        self._config = config or get_config()

    # Server settings
    @property
    def host(self) -> str:
        """Get server host."""
        return self._config.server.host

    @property
    def port(self) -> int:
        """Get server port."""
        return self._config.server.port

    @property
    def debug(self) -> bool:
        """Get debug mode."""
        return self._config.server.debug

    @property
    def workers(self) -> int:
        """Get number of workers."""
        return self._config.server.workers

    @property
    def timeout(self) -> int:
        """Get server timeout."""
        return self._config.server.timeout

    # Logging settings
    @property
    def log_level(self) -> str:
        """Get log level."""
        return self._config.logging.level

    @property
    def log_file(self) -> str:
        """Get log file path."""
        return self._config.logging.file

    @property
    def log_format(self) -> str:
        """Get log format."""
        return self._config.logging.format

    # Database settings
    @property
    def redis_host(self) -> str:
        """Get Redis host."""
        return self._config.database.redis_host

    @property
    def redis_port(self) -> int:
        """Get Redis port."""
        return self._config.database.redis_port

    @property
    def redis_db(self) -> int:
        """Get Redis database number."""
        return self._config.database.redis_db

    @property
    def redis_password(self) -> Optional[str]:
        """Get Redis password."""
        return self._config.database.redis_password

    @property
    def sqlite_path(self) -> str:
        """Get SQLite database path."""
        return self._config.database.sqlite_path

    # Market data settings
    @property
    def default_timeframe(self) -> str:
        """Get default timeframe."""
        return self._config.market_data.default_timeframe

    @property
    def default_assets(self) -> List[str]:
        """Get default assets."""
        return self._config.market_data.default_assets

    @property
    def cache_ttl(self) -> int:
        """Get cache TTL."""
        return self._config.market_data.cache_ttl

    # Model settings
    @property
    def forecast_horizon(self) -> int:
        """Get forecast horizon."""
        return self._config.models.forecast_horizon

    @property
    def confidence_interval(self) -> float:
        """Get confidence interval."""
        return self._config.models.confidence_interval

    @property
    def min_training_samples(self) -> int:
        """Get minimum training samples."""
        return self._config.models.min_training_samples

    @property
    def ensemble_size(self) -> int:
        """Get ensemble size."""
        return self._config.models.ensemble_size

    # Strategy settings
    @property
    def position_size(self) -> float:
        """Get default position size."""
        return self._config.strategies.position_size

    @property
    def stop_loss(self) -> float:
        """Get default stop loss."""
        return self._config.strategies.stop_loss

    @property
    def take_profit(self) -> float:
        """Get default take profit."""
        return self._config.strategies.take_profit

    @property
    def max_positions(self) -> int:
        """Get maximum positions."""
        return self._config.strategies.max_positions

    # Risk settings
    @property
    def max_drawdown(self) -> float:
        """Get maximum drawdown."""
        return self._config.risk.max_drawdown

    @property
    def max_leverage(self) -> float:
        """Get maximum leverage."""
        return self._config.risk.max_leverage

    @property
    def position_limits(self) -> Dict[str, float]:
        """Get position limits."""
        return self._config.risk.position_limits

    # Agent settings
    @property
    def goal_planner_enabled(self) -> bool:
        """Get goal planner enabled status."""
        return self._config.agents.goal_planner_enabled

    @property
    def router_enabled(self) -> bool:
        """Get router enabled status."""
        return self._config.agents.router_enabled

    @property
    def router_confidence(self) -> float:
        """Get router confidence threshold."""
        return self._config.agents.router_confidence

    @property
    def self_improving_enabled(self) -> bool:
        """Get self-improving agent enabled status."""
        return self._config.agents.self_improving_enabled

    # NLP settings
    @property
    def nlp_confidence_threshold(self) -> float:
        """Get NLP confidence threshold."""
        return self._config.nlp.confidence_threshold

    @property
    def nlp_max_tokens(self) -> int:
        """Get NLP max tokens."""
        return self._config.nlp.max_tokens

    @property
    def nlp_temperature(self) -> float:
        """Get NLP temperature."""
        return self._config.nlp.temperature

    # API settings
    @property
    def api_rate_limit(self) -> int:
        """Get API rate limit."""
        return self._config.api.rate_limit

    @property
    def api_timeout(self) -> int:
        """Get API timeout."""
        return self._config.api.timeout

    @property
    def api_version(self) -> str:
        """Get API version."""
        return self._config.api.version

    # Monitoring settings
    @property
    def monitoring_enabled(self) -> bool:
        """Get monitoring enabled status."""
        return self._config.monitoring.enabled

    @property
    def dashboard_enabled(self) -> bool:
        """Get dashboard enabled status."""
        return self._config.monitoring.dashboard_enabled

    @property
    def dashboard_port(self) -> int:
        """Get dashboard port."""
        return self._config.monitoring.dashboard_port

    # Security settings
    @property
    def ssl_enabled(self) -> bool:
        """Get SSL enabled status."""
        return self._config.security.ssl_enabled

    @property
    def auth_enabled(self) -> bool:
        """Get authentication enabled status."""
        return self._config.security.auth_enabled

    @property
    def rate_limiting_enabled(self) -> bool:
        """Get rate limiting enabled status."""
        return self._config.security.rate_limiting_enabled

    # Development settings
    @property
    def dev_debug(self) -> bool:
        """Get development debug status."""
        return self._config.development.debug

    @property
    def test_mode(self) -> bool:
        """Get test mode status."""
        return self._config.development.test_mode

    @property
    def mock_data(self) -> bool:
        """Get mock data status."""
        return self._config.development.mock_data

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (dot notation supported)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        try:
            keys = key.split(".")
            value = self._config

            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                elif isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default

            return value
        except Exception as e:
            logger.warning(f"Error getting config key '{key}': {e}")
            return default

    def set(self, key: str, value: Any) -> bool:
        """
        Set configuration value by key.

        Args:
            key: Configuration key (dot notation supported)
            value: Value to set

        Returns:
            True if set successfully
        """
        try:
            keys = key.split(".")
            obj = self._config

            # Navigate to parent object
            for k in keys[:-1]:
                if hasattr(obj, k):
                    obj = getattr(obj, k)
                elif isinstance(obj, dict):
                    if k not in obj:
                        obj[k] = {}
                    obj = obj[k]
                else:
                    return False

            # Set value on final object
            final_key = keys[-1]
            if hasattr(obj, final_key):
                setattr(obj, final_key, value)
            elif isinstance(obj, dict):
                obj[final_key] = value
            else:
                return False

            return True
        except Exception as e:
            logger.error(f"Error setting config key '{key}': {e}")
            return False

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation
        """
        return self._config.to_dict()

    def validate(self) -> Dict[str, Any]:
        """
        Validate configuration.

        Returns:
            Validation results
        """
        return self._config.validate()

    def reload(self) -> bool:
        """
        Reload configuration from file.

        Returns:
            True if reloaded successfully
        """
        try:
            new_config = AppConfig.from_yaml("config/app_config.yaml")
            self._config = new_config
            logger.info("Configuration reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error reloading configuration: {e}")
            return False


@lru_cache(maxsize=1)
def get_simple_config() -> Config:
    """
    Get cached simple configuration instance.

    Returns:
        Config instance
    """
    return Config()


def load_env_config() -> Dict[str, str]:
    """
    Load configuration from environment variables.

    Returns:
        Dictionary of environment variables
    """
    config_vars = {}

    # Common environment variables
    env_vars = [
        "HOST",
        "PORT",
        "DEBUG_MODE",
        "LOG_LEVEL",
        "LOG_FILE",
        "REDIS_HOST",
        "REDIS_PORT",
        "REDIS_PASSWORD",
        "DEFAULT_TIMEFRAME",
        "FORECAST_HORIZON",
        "CONFIDENCE_INTERVAL",
        "POSITION_SIZE",
        "STOP_LOSS",
        "TAKE_PROFIT",
        "MAX_POSITIONS",
        "MAX_DRAWDOWN",
        "MAX_LEVERAGE",
        "NLP_CONFIDENCE_THRESHOLD",
        "API_RATE_LIMIT",
        "API_TIMEOUT",
        "MONITORING_ENABLED",
        "SSL_ENABLED",
        "AUTH_ENABLED",
        "DEV_DEBUG",
        "TEST_MODE",
    ]

    for var in env_vars:
        value = os.getenv(var)
        if value is not None:
            config_vars[var] = value

    return config_vars


def save_env_config(config_vars: Dict[str, str], file_path: Union[str, Path]) -> bool:
    """
    Save environment variables to file.

    Args:
        config_vars: Dictionary of environment variables
        file_path: Path to save file

    Returns:
        True if saved successfully
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            for key, value in config_vars.items():
                f.write(f"{key}={value}\n")

        logger.info(f"Environment configuration saved to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving environment configuration: {e}")
        return False


def create_default_config(config_path: Union[str, Path] = "config/app_config.yaml") -> bool:
    """
    Create default configuration file.

    Args:
        config_path: Path to save configuration

    Returns:
        True if created successfully
    """
    try:
        config = AppConfig()
        return config.save_to_yaml(config_path)
    except Exception as e:
        logger.error(f"Error creating default configuration: {e}")
        return False


def validate_config_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate configuration file.

    Args:
        config_path: Path to configuration file

    Returns:
        Validation results
    """
    try:
        config = AppConfig.from_yaml(config_path)
        return config.validate()
    except Exception as e:
        logger.error(f"Error validating configuration file: {e}")
        return {"valid": False, "errors": [f"Configuration file error: {e}"], "warnings": []}


# Convenience functions


def get_server_config() -> Dict[str, Any]:
    """Get server configuration."""
    config = get_simple_config()
    return {
        "host": config.host,
        "port": config.port,
        "debug": config.debug,
        "workers": config.workers,
        "timeout": config.timeout,
    }


def get_database_config() -> Dict[str, Any]:
    """Get database configuration."""
    config = get_simple_config()
    return {
        "redis": {
            "host": config.redis_host,
            "port": config.redis_port,
            "db": config.redis_db,
            "password": config.redis_password,
        },
        "sqlite": {"path": config.sqlite_path},
    }


def get_model_config() -> Dict[str, Any]:
    """Get model configuration."""
    config = get_simple_config()
    return {
        "forecast_horizon": config.forecast_horizon,
        "confidence_interval": config.confidence_interval,
        "min_training_samples": config.min_training_samples,
        "ensemble_size": config.ensemble_size,
    }


def get_strategy_config() -> Dict[str, Any]:
    """Get strategy configuration."""
    config = get_simple_config()
    return {
        "position_size": config.position_size,
        "stop_loss": config.stop_loss,
        "take_profit": config.take_profit,
        "max_positions": config.max_positions,
    }


def get_risk_config() -> Dict[str, Any]:
    """Get risk configuration."""
    config = get_simple_config()
    return {
        "max_drawdown": config.max_drawdown,
        "max_leverage": config.max_leverage,
        "position_limits": config.position_limits,
    }


def get_agent_config() -> Dict[str, Any]:
    """Get agent configuration."""
    config = get_simple_config()
    return {
        "goal_planner_enabled": config.goal_planner_enabled,
        "router_enabled": config.router_enabled,
        "router_confidence": config.router_confidence,
        "self_improving_enabled": config.self_improving_enabled,
    }


def get_nlp_config() -> Dict[str, Any]:
    """Get NLP configuration."""
    config = get_simple_config()
    return {
        "confidence_threshold": config.nlp_confidence_threshold,
        "max_tokens": config.nlp_max_tokens,
        "temperature": config.nlp_temperature,
    }


def get_api_config() -> Dict[str, Any]:
    """Get API configuration."""
    config = get_simple_config()
    return {"rate_limit": config.api_rate_limit, "timeout": config.api_timeout, "version": config.api_version}


def get_monitoring_config() -> Dict[str, Any]:
    """Get monitoring configuration."""
    config = get_simple_config()
    return {
        "enabled": config.monitoring_enabled,
        "dashboard_enabled": config.dashboard_enabled,
        "dashboard_port": config.dashboard_port,
    }


def get_security_config() -> Dict[str, Any]:
    """Get security configuration."""
    config = get_simple_config()
    return {
        "ssl_enabled": config.ssl_enabled,
        "auth_enabled": config.auth_enabled,
        "rate_limiting_enabled": config.rate_limiting_enabled,
    }


def get_development_config() -> Dict[str, Any]:
    """Get development configuration."""
    config = get_simple_config()
    return {"debug": config.dev_debug, "test_mode": config.test_mode, "mock_data": config.mock_data}
