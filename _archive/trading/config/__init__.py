"""
Configuration package for the trading system.

Canonical trading-side settings live in ``configuration.py`` (dataclasses / YAML).
``enhanced_settings.py`` and ``settings.py`` were moved to ``_archive/``; use
``config.app_config`` / ``config.llm_config`` / ``config.user_store`` for app-wide config.
"""

from .configuration import (
    ConfigManager,
    DataConfig,
    ModelConfig,
    MonitoringConfig,
    TrainingConfig,
    WebConfig,
)

__all__ = [
    "ConfigManager",
    "ModelConfig",
    "DataConfig",
    "TrainingConfig",
    "WebConfig",
    "MonitoringConfig",
]
