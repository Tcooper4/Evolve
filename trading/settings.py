"""Global settings for the trading system."""

import os
import logging
from typing import Optional
from pathlib import Path

# Default settings
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_METRIC_LOGGING = True
DEFAULT_METRICS_PATH = "logs/metrics.jsonl"

# Global settings
LOG_LEVEL = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL)
METRIC_LOGGING_ENABLED = os.getenv("METRIC_LOGGING_ENABLED", str(DEFAULT_METRIC_LOGGING)).lower() == "true"
METRICS_PATH = os.getenv("METRICS_PATH", DEFAULT_METRICS_PATH)

def set_log_level(level: str) -> None:
    """Set the global log level.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    global LOG_LEVEL
    LOG_LEVEL = level.upper()
    
    # Update root logger
    logging.getLogger().setLevel(getattr(logging, LOG_LEVEL, "INFO"))
    
    # Update all existing loggers
    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, LOG_LEVEL, "INFO"))

def set_metric_logging(enabled: bool) -> None:
    """Enable or disable metric logging.
    
    Args:
        enabled: Whether to enable metric logging
    """
    global METRIC_LOGGING_ENABLED
    METRIC_LOGGING_ENABLED = enabled

def set_metrics_path(path: str) -> None:
    """Set the path for metrics logging.
    
    Args:
        path: Path to metrics file
    """
    global METRICS_PATH
    METRICS_PATH = path
    
    # Ensure directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def get_settings() -> dict:
    """Get current settings.
    
    Returns:
        Dictionary of current settings
    """
    return {
        "LOG_LEVEL": LOG_LEVEL,
        "METRIC_LOGGING_ENABLED": METRIC_LOGGING_ENABLED,
        "METRICS_PATH": METRICS_PATH
    } 