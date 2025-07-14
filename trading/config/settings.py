"""Configuration settings for the trading system."""

import logging
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

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
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
IEX_API_KEY = os.getenv("IEX_API_KEY", "")
QUANDL_API_KEY = os.getenv("QUANDL_API_KEY", "")
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "")

# Security and Authentication
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-jwt-secret-key-change-in-production")
WEB_SECRET_KEY = os.getenv("WEB_SECRET_KEY", "dev-secret-key-change-in-production")
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "dev-encryption-key-change-in-production")

# Notification Settings
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_USER = os.getenv("EMAIL_USER", "")
EMAIL_USE_TLS = os.getenv("EMAIL_USE_TLS", "true").lower() == "true"
EMAIL_USE_SSL = os.getenv("EMAIL_USE_SSL", "false").lower() == "true"

# Database
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "trading")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_SSL_MODE = os.getenv("DB_SSL_MODE", "prefer")
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))

# Redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_SSL = os.getenv("REDIS_SSL", "false").lower() == "true"
REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "10"))

# Model Settings
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "transformer")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
MODEL_SAVE_FREQUENCY = int(os.getenv("MODEL_SAVE_FREQUENCY", "10"))
MODEL_EVALUATION_FREQUENCY = int(os.getenv("MODEL_EVALUATION_FREQUENCY", "50"))
MODEL_BACKUP_COUNT = int(os.getenv("MODEL_BACKUP_COUNT", "5"))

# Data Settings
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", PROJECT_ROOT / "cache"))
DEFAULT_TICKERS = os.getenv("DEFAULT_TICKERS", "AAPL,MSFT,GOOGL").split(",")
DATA_CACHE_TTL = int(os.getenv("DATA_CACHE_TTL", "3600"))  # 1 hour
DATA_RETRY_ATTEMPTS = int(os.getenv("DATA_RETRY_ATTEMPTS", "3"))
DATA_RETRY_DELAY = float(os.getenv("DATA_RETRY_DELAY", "1.0"))
DATA_TIMEOUT = int(os.getenv("DATA_TIMEOUT", "30"))

# Web Settings
WEB_HOST = os.getenv("WEB_HOST", "localhost")
WEB_PORT = int(os.getenv("WEB_PORT", "5000"))
WEB_DEBUG = DEBUG
WEB_WORKERS = int(os.getenv("WEB_WORKERS", "4"))
WEB_MAX_REQUESTS = int(os.getenv("WEB_MAX_REQUESTS", "1000"))
WEB_MAX_REQUESTS_JITTER = int(os.getenv("WEB_MAX_REQUESTS_JITTER", "100"))
WEB_TIMEOUT = int(os.getenv("WEB_TIMEOUT", "30"))

# Monitoring
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "9090"))
GRAFANA_PORT = int(os.getenv("GRAFANA_PORT", "3000"))
ALERT_EMAIL = os.getenv("ALERT_EMAIL", "")
ALERT_WEBHOOK = os.getenv("ALERT_WEBHOOK", "")
METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() == "true"
HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "60"))

# Agent Settings
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "300"))
MAX_CONCURRENT_AGENTS = int(os.getenv("MAX_CONCURRENT_AGENTS", "5"))
AGENT_MEMORY_SIZE = int(os.getenv("AGENT_MEMORY_SIZE", "1000"))
AGENT_RETRY_ATTEMPTS = int(os.getenv("AGENT_RETRY_ATTEMPTS", "3"))
AGENT_HEARTBEAT_INTERVAL = int(os.getenv("AGENT_HEARTBEAT_INTERVAL", "30"))
AGENT_CLEANUP_INTERVAL = int(os.getenv("AGENT_CLEANUP_INTERVAL", "300"))

# Strategy Settings
STRATEGY_DIR = Path(os.getenv("STRATEGY_DIR", PROJECT_ROOT / "strategies"))
DEFAULT_STRATEGY = os.getenv("DEFAULT_STRATEGY", "momentum")
BACKTEST_DAYS = int(os.getenv("BACKTEST_DAYS", "365"))
STRATEGY_EVALUATION_FREQUENCY = int(os.getenv("STRATEGY_EVALUATION_FREQUENCY", "7"))
STRATEGY_SWITCH_THRESHOLD = float(os.getenv("STRATEGY_SWITCH_THRESHOLD", "0.1"))

# LLM Settings
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "gpt2")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))

# Memory Settings
MEMORY_DIR = Path(os.getenv("MEMORY_DIR", PROJECT_ROOT / "memory"))
MEMORY_BACKEND = os.getenv("MEMORY_BACKEND", "json")
MEMORY_MAX_SIZE = int(os.getenv("MEMORY_MAX_SIZE", "10000"))
MEMORY_CLEANUP_INTERVAL = int(os.getenv("MEMORY_CLEANUP_INTERVAL", "3600"))
MEMORY_TTL = int(os.getenv("MEMORY_TTL", "86400"))  # 24 hours

# Risk Management
MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", "0.20"))
MAX_CORRELATION = float(os.getenv("MAX_CORRELATION", "0.70"))
MAX_CONCENTRATION = float(os.getenv("MAX_CONCENTRATION", "0.30"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.02"))
STOP_LOSS = float(os.getenv("STOP_LOSS", "0.05"))
TAKE_PROFIT = float(os.getenv("TAKE_PROFIT", "0.10"))
MAX_LEVERAGE = float(os.getenv("MAX_LEVERAGE", "1.0"))

# Performance Settings
PERFORMANCE_THRESHOLD = float(os.getenv("PERFORMANCE_THRESHOLD", "0.05"))
IMPROVEMENT_THRESHOLD = float(os.getenv("IMPROVEMENT_THRESHOLD", "0.02"))
SHARPE_THRESHOLD = float(os.getenv("SHARPE_THRESHOLD", "1.0"))
SORTINO_THRESHOLD = float(os.getenv("SORTINO_THRESHOLD", "1.0"))
MAX_DRAWDOWN_THRESHOLD = float(os.getenv("MAX_DRAWDOWN_THRESHOLD", "0.15"))

# Execution Settings
SLIPPAGE = float(os.getenv("SLIPPAGE", "0.001"))
TRANSACTION_COST = float(os.getenv("TRANSACTION_COST", "0.001"))
MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", "0.25"))
MIN_POSITION_SIZE = float(os.getenv("MIN_POSITION_SIZE", "0.01"))
EXECUTION_TIMEOUT = int(os.getenv("EXECUTION_TIMEOUT", "30"))
EXECUTION_RETRY_ATTEMPTS = int(os.getenv("EXECUTION_RETRY_ATTEMPTS", "3"))

# Feature Engineering
FEATURE_WINDOW = int(os.getenv("FEATURE_WINDOW", "20"))
FEATURE_SCALING = os.getenv("FEATURE_SCALING", "standard")  # standard, minmax, robust
FEATURE_SELECTION_METHOD = os.getenv("FEATURE_SELECTION_METHOD", "correlation")
FEATURE_CORRELATION_THRESHOLD = float(
    os.getenv("FEATURE_CORRELATION_THRESHOLD", "0.95")
)

# Backtesting
BACKTEST_START_DATE = os.getenv("BACKTEST_START_DATE", "")
BACKTEST_END_DATE = os.getenv("BACKTEST_END_DATE", "")
BACKTEST_INITIAL_CAPITAL = float(os.getenv("BACKTEST_INITIAL_CAPITAL", "100000"))
BACKTEST_COMMISSION = float(os.getenv("BACKTEST_COMMISSION", "0.001"))
BACKTEST_SLIPPAGE = float(os.getenv("BACKTEST_SLIPPAGE", "0.001"))

# Reporting
REPORT_FREQUENCY = os.getenv("REPORT_FREQUENCY", "daily")
REPORT_DIR = Path(os.getenv("REPORT_DIR", PROJECT_ROOT / "reports"))
REPORT_RETENTION_DAYS = int(os.getenv("REPORT_RETENTION_DAYS", "30"))
REPORT_FORMAT = os.getenv("REPORT_FORMAT", "html")  # html, pdf, json

# Logging Configuration
LOG_RETENTION_DAYS = int(os.getenv("LOG_RETENTION_DAYS", "30"))
LOG_ROTATION_SIZE = int(os.getenv("LOG_ROTATION_SIZE", "10485760"))  # 10MB
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))
LOG_COMPRESSION = os.getenv("LOG_COMPRESSION", "gzip")

# Cache Settings
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))
CACHE_CLEANUP_INTERVAL = int(os.getenv("CACHE_CLEANUP_INTERVAL", "300"))

# Security Settings
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))
SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "3600"))

# Development Settings
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
PROFILING_ENABLED = os.getenv("PROFILING_ENABLED", "false").lower() == "true"
TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"
MOCK_EXTERNAL_APIS = os.getenv("MOCK_EXTERNAL_APIS", "false").lower() == "true"

# Metric Logging Settings
METRIC_LOGGING_ENABLED = os.getenv("METRIC_LOGGING_ENABLED", "true").lower() == "true"
METRICS_PATH = LOG_DIR / "metrics.log"
METRICS_FLUSH_INTERVAL = int(os.getenv("METRICS_FLUSH_INTERVAL", "60"))

# Strategy Switch Settings
STRATEGY_SWITCH_LOG_PATH = LOG_DIR / "strategy_switches.json"
STRATEGY_REGISTRY_PATH = STRATEGY_DIR / "strategy_registry.json"
STRATEGY_SWITCH_LOCK_TIMEOUT = int(os.getenv("STRATEGY_SWITCH_LOCK_TIMEOUT", "30"))
STRATEGY_SWITCH_BACKEND = os.getenv("STRATEGY_SWITCH_BACKEND", "file")
STRATEGY_SWITCH_API_ENDPOINT = os.getenv(
    "STRATEGY_SWITCH_API_ENDPOINT", "http://localhost:8000/api/strategy-switches"
)

# Data Provider Configuration
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "yahoo")
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "yahoo")
DATA_PROVIDER_FALLBACK = os.getenv("DATA_PROVIDER_FALLBACK", "true").lower() == "true"
DATA_PROVIDER_TIMEOUT = int(os.getenv("DATA_PROVIDER_TIMEOUT", "30"))

# === Compatibility Aliases and Defaults ===
PERFORMANCE_CONFIG_PATH = PROJECT_ROOT / "config" / "performance_config.json"
MODELS_DIR = MODEL_DIR  # Alias for compatibility
STRATEGIES_DIR = STRATEGY_DIR  # Alias for compatibility
DEFAULT_PERFORMANCE_THRESHOLDS = {
    "min_sharpe": SHARPE_THRESHOLD,
    "max_drawdown": MAX_DRAWDOWN_THRESHOLD,
    "min_accuracy": 0.6,
}


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
    errors = []

    # Validate required API keys for production
    if ENV == "production":
        if not ALPHA_VANTAGE_API_KEY:
            errors.append("ALPHA_VANTAGE_API_KEY is required in production")
        if not POLYGON_API_KEY:
            errors.append("POLYGON_API_KEY is required in production")
        if JWT_SECRET_KEY == "dev-jwt-secret-key-change-in-production":
            errors.append("JWT_SECRET_KEY must be set in production")
        if WEB_SECRET_KEY == "dev-secret-key-change-in-production":
            errors.append("WEB_SECRET_KEY must be set in production")

    # Validate directories
    for dir_path in [
        LOG_DIR,
        MODEL_DIR,
        DATA_DIR,
        CACHE_DIR,
        STRATEGY_DIR,
        MEMORY_DIR,
        REPORT_DIR,
    ]:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Failed to create directory {dir_path}: {e}")

    # Validate ports
    for port_name, port_value in [
        ("WEB_PORT", WEB_PORT),
        ("PROMETHEUS_PORT", PROMETHEUS_PORT),
        ("GRAFANA_PORT", GRAFANA_PORT),
    ]:
        if not 1024 <= port_value <= 65535:
            errors.append(f"Invalid {port_name}: {port_value}")

    # Validate numeric ranges
    if not 0 <= MAX_DRAWDOWN <= 1:
        errors.append(f"Invalid MAX_DRAWDOWN: {MAX_DRAWDOWN}")
    if not 0 <= MAX_CORRELATION <= 1:
        errors.append(f"Invalid MAX_CORRELATION: {MAX_CORRELATION}")
    if not 0 <= MAX_CONCENTRATION <= 1:
        errors.append(f"Invalid MAX_CONCENTRATION: {MAX_CONCENTRATION}")
    if not 0 <= RISK_PER_TRADE <= 1:
        errors.append(f"Invalid RISK_PER_TRADE: {RISK_PER_TRADE}")
    if not 0 <= STOP_LOSS <= 1:
        errors.append(f"Invalid STOP_LOSS: {STOP_LOSS}")
    if not 0 <= TAKE_PROFIT <= 1:
        errors.append(f"Invalid TAKE_PROFIT: {TAKE_PROFIT}")
    if MAX_LEVERAGE <= 0:
        errors.append(f"Invalid MAX_LEVERAGE: {MAX_LEVERAGE}")

    # Validate timeouts
    for timeout_name, timeout_value in [
        ("AGENT_TIMEOUT", AGENT_TIMEOUT),
        ("EXECUTION_TIMEOUT", EXECUTION_TIMEOUT),
        ("LLM_TIMEOUT", LLM_TIMEOUT),
    ]:
        if timeout_value <= 0:
            errors.append(f"Invalid {timeout_name}: {timeout_value}")

    # Validate intervals
    for interval_name, interval_value in [
        ("AGENT_HEARTBEAT_INTERVAL", AGENT_HEARTBEAT_INTERVAL),
        ("MEMORY_CLEANUP_INTERVAL", MEMORY_CLEANUP_INTERVAL),
    ]:
        if interval_value <= 0:
            errors.append(f"Invalid {interval_name}: {interval_value}")

    if errors:
        raise ValueError(
            f"Configuration validation failed:\n"
            + "\n".join(f"  - {error}" for error in errors)
        )

    return True


def create_env_template() -> str:
    """Create a template .env file with all configuration options.

    Returns:
        Template string for .env file
    """
    template = """# Trading System Environment Configuration
# Copy this file to .env and modify as needed

# Environment
TRADING_ENV=development

# Logging
LOG_LEVEL=INFO
AGENT_LOG_LEVEL=INFO
MODEL_LOG_LEVEL=INFO
DATA_LOG_LEVEL=INFO
ROOT_LOG_LEVEL=WARNING

# API Keys
ALPHA_VANTAGE_API_KEY=
POLYGON_API_KEY=
OPENAI_API_KEY=
FINNHUB_API_KEY=
IEX_API_KEY=
QUANDL_API_KEY=
TWELVE_DATA_API_KEY=

# Security
JWT_SECRET_KEY=dev-jwt-secret-key-change-in-production
WEB_SECRET_KEY=dev-secret-key-change-in-production
ENCRYPTION_KEY=dev-encryption-key-change-in-production

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading
DB_USER=postgres
DB_PASSWORD=
DB_SSL_MODE=prefer
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
REDIS_SSL=false
REDIS_MAX_CONNECTIONS=10

# Model Settings
DEFAULT_MODEL=transformer
BATCH_SIZE=32
LEARNING_RATE=0.001
MODEL_SAVE_FREQUENCY=10
MODEL_EVALUATION_FREQUENCY=50
MODEL_BACKUP_COUNT=5

# Data Settings
DEFAULT_TICKERS=AAPL,MSFT,GOOGL
DATA_CACHE_TTL=3600
DATA_RETRY_ATTEMPTS=3
DATA_RETRY_DELAY=1.0
DATA_TIMEOUT=30

# Web Settings
WEB_HOST=localhost
WEB_PORT=5000
WEB_WORKERS=4
WEB_MAX_REQUESTS=1000
WEB_MAX_REQUESTS_JITTER=100
WEB_TIMEOUT=30

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
ALERT_EMAIL=
ALERT_WEBHOOK=
METRICS_ENABLED=true
HEALTH_CHECK_INTERVAL=60

# Agent Settings
AGENT_TIMEOUT=300
MAX_CONCURRENT_AGENTS=5
AGENT_MEMORY_SIZE=1000
AGENT_RETRY_ATTEMPTS=3
AGENT_HEARTBEAT_INTERVAL=30
AGENT_CLEANUP_INTERVAL=300

# Strategy Settings
DEFAULT_STRATEGY=momentum
BACKTEST_DAYS=365
STRATEGY_EVALUATION_FREQUENCY=7
STRATEGY_SWITCH_THRESHOLD=0.1

# LLM Settings
DEFAULT_LLM_PROVIDER=openai
HUGGINGFACE_API_KEY=
HUGGINGFACE_MODEL=gpt2
ANTHROPIC_API_KEY=
LLM_MAX_TOKENS=1000
LLM_TEMPERATURE=0.7
LLM_TIMEOUT=60

# Memory Settings
MEMORY_BACKEND=json
MEMORY_MAX_SIZE=10000
MEMORY_CLEANUP_INTERVAL=3600
MEMORY_TTL=86400

# Risk Management
MAX_DRAWDOWN=0.20
MAX_CORRELATION=0.70
MAX_CONCENTRATION=0.30
RISK_PER_TRADE=0.02
STOP_LOSS=0.05
TAKE_PROFIT=0.10
MAX_LEVERAGE=1.0

# Performance Settings
PERFORMANCE_THRESHOLD=0.05
IMPROVEMENT_THRESHOLD=0.02
SHARPE_THRESHOLD=1.0
SORTINO_THRESHOLD=1.0
MAX_DRAWDOWN_THRESHOLD=0.15

# Execution Settings
SLIPPAGE=0.001
TRANSACTION_COST=0.001
MAX_POSITION_SIZE=0.25
MIN_POSITION_SIZE=0.01
EXECUTION_TIMEOUT=30
EXECUTION_RETRY_ATTEMPTS=3

# Feature Engineering
FEATURE_WINDOW=20
FEATURE_SCALING=standard
FEATURE_SELECTION_METHOD=correlation
FEATURE_CORRELATION_THRESHOLD=0.95

# Backtesting
BACKTEST_START_DATE=
BACKTEST_END_DATE=
BACKTEST_INITIAL_CAPITAL=100000
BACKTEST_COMMISSION=0.001
BACKTEST_SLIPPAGE=0.001

# Reporting
REPORT_FREQUENCY=daily
REPORT_RETENTION_DAYS=30
REPORT_FORMAT=html

# Logging Configuration
LOG_RETENTION_DAYS=30
LOG_ROTATION_SIZE=10485760
LOG_BACKUP_COUNT=5
LOG_COMPRESSION=gzip

# Cache Settings
CACHE_ENABLED=true
CACHE_TTL=3600
CACHE_MAX_SIZE=1000
CACHE_CLEANUP_INTERVAL=300

# Security Settings
CORS_ORIGINS=*
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600
SESSION_TIMEOUT=3600

# Development Settings
DEBUG_MODE=false
PROFILING_ENABLED=false
TEST_MODE=false
MOCK_EXTERNAL_APIS=false

# Data Provider
DATA_PROVIDER=yahoo
DEFAULT_PROVIDER=yahoo
DATA_PROVIDER_FALLBACK=true
DATA_PROVIDER_TIMEOUT=30
"""
    return template


# Validate configuration on import
try:
    validate_config()
    logger.info("Configuration validation completed successfully")
except Exception as e:
    logger.error(f"Configuration validation failed: {e}")
    if ENV == "production":
        raise
