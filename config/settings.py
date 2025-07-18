"""Configuration settings for the trading system."""

import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Environment
ENV = os.getenv("TRADING_ENV", "development")
DEBUG = ENV == "development"

# Logging
LOG_DIR = Path(os.getenv("LOG_DIR", PROJECT_ROOT / "logs"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
AGENT_LOG_LEVEL = os.getenv("AGENT_LOG_LEVEL", "INFO")
MODEL_LOG_LEVEL = os.getenv("MODEL_LOG_LEVEL", "INFO")
DATA_LOG_LEVEL = os.getenv("DATA_LOG_LEVEL", "INFO")
ROOT_LOG_LEVEL = os.getenv("ROOT_LOG_LEVEL", "WARNING")

# API Keys
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Database
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "trading")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# Redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

# Model Settings
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "transformer")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))

# Data Settings
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", PROJECT_ROOT / "cache"))
DEFAULT_TICKERS = os.getenv("DEFAULT_TICKERS", "AAPL,MSFT,GOOGL").split(",")

# Web Settings
WEB_HOST = os.getenv("WEB_HOST", "localhost")
WEB_PORT = int(os.getenv("WEB_PORT", "5000"))
WEB_DEBUG = DEBUG
WEB_SECRET_KEY = os.getenv("WEB_SECRET_KEY", "dev-secret-key")

# Monitoring
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "9090"))
GRAFANA_PORT = int(os.getenv("GRAFANA_PORT", "3000"))
ALERT_EMAIL = os.getenv("ALERT_EMAIL")
ALERT_WEBHOOK = os.getenv("ALERT_WEBHOOK")

# Agent Settings
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "300"))
MAX_CONCURRENT_AGENTS = int(os.getenv("MAX_CONCURRENT_AGENTS", "5"))
AGENT_MEMORY_SIZE = int(os.getenv("AGENT_MEMORY_SIZE", "1000"))

# Strategy Settings
STRATEGY_DIR = Path(os.getenv("STRATEGY_DIR", PROJECT_ROOT / "strategies"))
DEFAULT_STRATEGY = os.getenv("DEFAULT_STRATEGY", "momentum")
BACKTEST_DAYS = int(os.getenv("BACKTEST_DAYS", "365"))


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a configuration value with type conversion.

    Args:
        key: Configuration key
        default: Default value if key not found

    Returns:
        Configuration value with appropriate type
    """
    value = os.getenv(key, default)
    if value is None:
        return None

    # Try to convert to appropriate type
    if isinstance(default, bool):
        return value.lower() in ("true", "1", "yes")
    elif isinstance(default, int):
        return int(value)
    elif isinstance(default, float):
        return float(value)
    elif isinstance(default, list):
        return value.split(",")
    elif isinstance(default, Path):
        return Path(value)
    return value


def get_config_dict() -> Dict[str, Any]:
    """Get all configuration values as a dictionary.

    Returns:
        Dictionary of all configuration values
    """
    return {
        key: value
        for key, value in globals().items()
        if key.isupper() and not key.startswith("_")
    }


def validate_config() -> bool:
    """Validate the configuration.

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate required API keys
    if ENV == "production":
        if not ALPHA_VANTAGE_API_KEY:
            raise ValueError("ALPHA_VANTAGE_API_KEY is required in production")
        if not POLYGON_API_KEY:
            raise ValueError("POLYGON_API_KEY is required in production")

    # Validate directories
    for dir_path in [LOG_DIR, MODEL_DIR, DATA_DIR, CACHE_DIR, STRATEGY_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Validate ports
    if not 1024 <= WEB_PORT <= 65535:
        raise ValueError(f"Invalid WEB_PORT: {WEB_PORT}")
    if not 1024 <= PROMETHEUS_PORT <= 65535:
        raise ValueError(f"Invalid PROMETHEUS_PORT: {PROMETHEUS_PORT}")
    if not 1024 <= GRAFANA_PORT <= 65535:
        raise ValueError(f"Invalid GRAFANA_PORT: {GRAFANA_PORT}")

    return True


# Validate configuration on import
validate_config()

# Upgrader Agent Settings
UPGRADER_SETTINGS = {
    "upgrade_interval": 24,  # hours between automatic upgrade checks
    "max_retries": 3,  # maximum number of retry attempts for failed upgrades
    "retry_delay": 300,  # seconds to wait between retry attempts
    "log_retention_days": 30,  # number of days to keep log files
    "memory_retention_days": 7,  # number of days to keep task memory
    "model_drift_threshold": 0.1,  # threshold for detecting model drift
    "version_check_interval": 3600,  # seconds between version checks
    "safe_mode": True,  # enable safe fallback mode
    "backup_before_upgrade": True,  # create backups before upgrades
    "notify_on_failure": True,  # send notifications on upgrade failures
    "max_concurrent_upgrades": 3,  # maximum number of concurrent upgrades
    "upgrade_timeout": 3600,  # maximum time (seconds) for an upgrade to complete
}
