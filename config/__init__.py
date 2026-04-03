"""
Configuration module for application settings and logging.

Single source of truth: use get_config() from config.app_config (or
config.primary_config.get_primary_config()). See config/CONFIG_README.md.
"""

from .app_config import AppConfig, get_config
from .config import Config
from .logging_config import LoggingConfig

# Archived modules (no import): config/market_analysis_config.py, config/primary_config.py
MarketAnalysisConfig = None  # type: ignore[misc, assignment]
get_primary_config = get_config

__all__ = [
    "AppConfig",
    "Config",
    "get_config",
    "get_primary_config",
    "LoggingConfig",
    "MarketAnalysisConfig",
]
