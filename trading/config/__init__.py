"""Configuration package."""

from trading.config.configuration import (
    ConfigManager,
    ModelConfig,
    DataConfig,
    TrainingConfig,
    WebConfig,
    MonitoringConfig
)

# Import commonly used settings explicitly
from trading.config.settings import (
    ENV, DEBUG, LOG_LEVEL, AGENT_LOG_LEVEL, MODEL_LOG_LEVEL, DATA_LOG_LEVEL,
    ALPHA_VANTAGE_API_KEY, POLYGON_API_KEY, OPENAI_API_KEY,
    JWT_SECRET_KEY, WEB_SECRET_KEY,
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD,
    REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD,
    MODEL_DIR, DEFAULT_MODEL, BATCH_SIZE, LEARNING_RATE,
    DATA_DIR, CACHE_DIR, DEFAULT_TICKERS,
    WEB_HOST, WEB_PORT, WEB_DEBUG,
    PROMETHEUS_PORT, GRAFANA_PORT,
    AGENT_TIMEOUT, MAX_CONCURRENT_AGENTS, AGENT_MEMORY_SIZE,
    STRATEGY_DIR, DEFAULT_STRATEGY, BACKTEST_DAYS,
    DEFAULT_LLM_PROVIDER, HUGGINGFACE_API_KEY, HUGGINGFACE_MODEL,
    MEMORY_DIR, MEMORY_BACKEND,
    METRIC_LOGGING_ENABLED, METRICS_PATH,
    STRATEGY_SWITCH_LOG_PATH, STRATEGY_REGISTRY_PATH,
    STRATEGY_SWITCH_LOCK_TIMEOUT, STRATEGY_SWITCH_BACKEND,
    STRATEGY_SWITCH_API_ENDPOINT,
    get_config_value, get_config_dict, validate_config, Settings, settings
)

class Config:
    """Configuration object that provides .get() method for accessing settings."""
    
    def get(self, key: str, default=None):
        """Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return globals().get(key, default)

# Create a config instance
config = Config()

__all__ = [
    'ConfigManager',
    'ModelConfig',
    'DataConfig',
    'TrainingConfig',
    'WebConfig',
    'MonitoringConfig',
    'config',
    'Settings',
    'settings'
] 