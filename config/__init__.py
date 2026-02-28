"""
Configuration module for application settings and logging.

Single source of truth: use get_config() from config.app_config (or
config.primary_config.get_primary_config()). See config/CONFIG_README.md.
"""

from .app_config import AppConfig, get_config
from .config import Config
from .logging_config import LoggingConfig
from .market_analysis_config import MarketAnalysisConfig

try:
    from .primary_config import get_primary_config
except ImportError:
    get_primary_config = get_config  # fallback

__all__ = [
    "AppConfig",
    "Config",
    "get_config",
    "get_primary_config",
    "LoggingConfig",
    "MarketAnalysisConfig",
]
