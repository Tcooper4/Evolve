"""Configuration settings for the trading system."""

import os
from pathlib import Path
from typing import Any, Optional, Dict, Union
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Environment
ENV = os.getenv('TRADING_ENV', 'development')
DEBUG = ENV == 'development'

# Logging
LOG_DIR = Path(os.getenv('LOG_DIR', PROJECT_ROOT / 'logs'))
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
AGENT_LOG_LEVEL = os.getenv('AGENT_LOG_LEVEL', 'INFO')
MODEL_LOG_LEVEL = os.getenv('MODEL_LOG_LEVEL', 'INFO')
DATA_LOG_LEVEL = os.getenv('DATA_LOG_LEVEL', 'INFO')
ROOT_LOG_LEVEL = os.getenv('ROOT_LOG_LEVEL', 'WARNING')

# API Keys
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo-alpha-vantage-key')
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY', 'demo-polygon-key')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'demo-openai-key')
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', 'demo-finnhub-key')
IEX_API_KEY = os.getenv('IEX_API_KEY', 'demo-iex-key')

# Security and Authentication
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'dev-jwt-secret-key-change-in-production')
WEB_SECRET_KEY = os.getenv('WEB_SECRET_KEY', 'dev-secret-key-change-in-production')

# Notification Settings
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', 'dev-email-password')
SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL', '')
EMAIL_HOST = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.getenv('EMAIL_PORT', '587'))
EMAIL_USER = os.getenv('EMAIL_USER', 'your-email@gmail.com')

# Database
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = int(os.getenv('DB_PORT', '5432'))
DB_NAME = os.getenv('DB_NAME', 'trading')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')

# Redis
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
REDIS_DB = int(os.getenv('REDIS_DB', '0'))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', '')

# Model Settings
MODEL_DIR = Path(os.getenv('MODEL_DIR', PROJECT_ROOT / 'models'))
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'transformer')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', '0.001'))

# Data Settings
DATA_DIR = Path(os.getenv('DATA_DIR', PROJECT_ROOT / 'data'))
CACHE_DIR = Path(os.getenv('CACHE_DIR', PROJECT_ROOT / 'cache'))
DEFAULT_TICKERS = os.getenv('DEFAULT_TICKERS', 'AAPL,MSFT,GOOGL').split(',')

# Web Settings
WEB_HOST = os.getenv('WEB_HOST', 'localhost')
WEB_PORT = int(os.getenv('WEB_PORT', '5000'))
WEB_DEBUG = DEBUG

# Monitoring
PROMETHEUS_PORT = int(os.getenv('PROMETHEUS_PORT', '9090'))
GRAFANA_PORT = int(os.getenv('GRAFANA_PORT', '3000'))
ALERT_EMAIL = os.getenv('ALERT_EMAIL')
ALERT_WEBHOOK = os.getenv('ALERT_WEBHOOK')

# Agent Settings
AGENT_TIMEOUT = int(os.getenv('AGENT_TIMEOUT', '300'))
MAX_CONCURRENT_AGENTS = int(os.getenv('MAX_CONCURRENT_AGENTS', '5'))
AGENT_MEMORY_SIZE = int(os.getenv('AGENT_MEMORY_SIZE', '1000'))

# Strategy Settings
STRATEGY_DIR = Path(os.getenv('STRATEGY_DIR', PROJECT_ROOT / 'strategies'))
DEFAULT_STRATEGY = os.getenv('DEFAULT_STRATEGY', 'momentum')
BACKTEST_DAYS = int(os.getenv('BACKTEST_DAYS', '365'))

# LLM Settings
DEFAULT_LLM_PROVIDER = os.getenv('DEFAULT_LLM_PROVIDER', 'openai')  # openai, huggingface, fallback
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
HUGGINGFACE_MODEL = os.getenv('HUGGINGFACE_MODEL', 'gpt2')

# Memory Settings
MEMORY_DIR = Path(os.getenv('MEMORY_DIR', PROJECT_ROOT / 'memory'))
MEMORY_BACKEND = os.getenv('MEMORY_BACKEND', 'json')

# Metric Logging Settings
METRIC_LOGGING_ENABLED = True
METRICS_PATH = LOG_DIR / "metrics.log"

# Strategy Switch Settings
STRATEGY_SWITCH_LOG_PATH = LOG_DIR / "strategy_switches.json"
STRATEGY_REGISTRY_PATH = STRATEGY_DIR / "strategy_registry.json"
STRATEGY_SWITCH_LOCK_TIMEOUT = 30
STRATEGY_SWITCH_BACKEND = "file"
STRATEGY_SWITCH_API_ENDPOINT = "http://localhost:8000/api/strategy-switches"

# Data Provider Configuration
DATA_PROVIDER = os.getenv('DATA_PROVIDER', 'yahoo')  # yahoo, alpha_vantage, polygon, finnhub, iex
DEFAULT_PROVIDER = os.getenv('DEFAULT_PROVIDER', 'yahoo')

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
        return value.lower() in ('true', '1', 'yes')
    elif isinstance(default, int):
        return int(value)
    elif isinstance(default, float):
        return float(value)
    elif isinstance(default, list):
        return value.split(',')
    elif isinstance(default, Path):
        return Path(value)
    return value

def get_config_dict() -> Dict[str, Any]:
    """Get all configuration values as a dictionary.
    
    Returns:
        Dictionary of all configuration values
    """
    return {
        key: value for key, value in globals().items()
        if key.isupper() and not key.startswith('_')
    }

def validate_config() -> bool:
    """Validate the configuration.
    
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate required API keys
    if ENV == 'production':
        if not ALPHA_VANTAGE_API_KEY:
            raise ValueError("ALPHA_VANTAGE_API_KEY is required in production")
        if not POLYGON_API_KEY:
            raise ValueError("POLYGON_API_KEY is required in production")
        if JWT_SECRET_KEY == 'dev-jwt-secret-key-change-in-production':
            raise ValueError("JWT_SECRET_KEY must be set in production")
        if WEB_SECRET_KEY == 'dev-secret-key-change-in-production':
            raise ValueError("WEB_SECRET_KEY must be set in production")
            
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