"""
Centralized Logging System

This module provides a comprehensive logging system with support for:
- Structured logging with JSON Lines format
- Rotating file handlers
- Metrics logging
- LLM-specific metrics
- Backtest performance metrics
- Agent decision metrics
"""

import json
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Configuration constants
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
DEFAULT_LOG_FILE = "logs/app.log"
DEFAULT_MAX_BYTES = 5 * 1024 * 1024  # 5MB
DEFAULT_BACKUP_COUNT = 3

# Metrics logging configuration
METRIC_LOGGING_ENABLED = os.getenv("METRIC_LOGGING_ENABLED", "true").lower() == "true"
METRICS_PATH = os.getenv("METRICS_PATH", "logs/metrics.jsonl")


def _load_config() -> dict:
    """Load logging configuration from YAML file if it exists."""
    config_path = Path("config/log_config.yaml")
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f).get("logging", {})
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error loading log config: {e}")
    return {}


def _get_log_level() -> int:
    """Get log level from environment or config."""
    config = _load_config()
    level = os.getenv("LOG_LEVEL") or config.get("level", DEFAULT_LOG_LEVEL)
    return getattr(logging, level.upper())


def _get_log_format() -> str:
    """Get log format from config."""
    config = _load_config()
    return config.get("format", DEFAULT_LOG_FORMAT)


def _get_log_file() -> str:
    """Get log file path from config."""
    config = _load_config()
    return config.get("file", DEFAULT_LOG_FILE)


def _get_max_bytes() -> int:
    """Get max bytes for rotating file handler from config."""
    config = _load_config()
    return config.get("max_bytes", DEFAULT_MAX_BYTES)


def _get_backup_count() -> int:
    """Get backup count for rotating file handler from config."""
    config = _load_config()
    return config.get("backup_count", DEFAULT_BACKUP_COUNT)


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Name of the logger (typically __name__)

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)

    # Return existing logger if already configured
    if logger.handlers:
        return logger

    # Set log level
    logger.setLevel(_get_log_level())

    # Create formatter
    formatter = logging.Formatter(_get_log_format())

    # Create and configure stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Create and configure file handler
    log_file = _get_log_file()
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(log_file, maxBytes=_get_max_bytes(), backupCount=_get_backup_count())
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def log_metrics(data: Dict[str, Any], path: Optional[str] = None) -> None:
    """Log metrics to JSON Lines file.

    Args:
        data: Dictionary of metrics to log
        path: Optional path to metrics file (defaults to METRICS_PATH)
    """
    if not METRIC_LOGGING_ENABLED:
        return

    path = path or METRICS_PATH

    # Ensure directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Add timestamp if not present
    if "timestamp" not in data:
        data["timestamp"] = datetime.utcnow().isoformat()

    try:
        with open(path, "a") as f:
            json.dump(data, f)
            f.write("\n")
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error writing metrics: {e}")


def log_llm_metrics(
    model: str,
    latency: float,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Log LLM-specific metrics.

    Args:
        model: Name of the model
        latency: Response time in seconds
        prompt_tokens: Number of tokens in prompt
        completion_tokens: Number of tokens in completion
        total_tokens: Total tokens used
        metadata: Optional additional metadata
    """
    data = {
        "type": "llm_metrics",
        "model": model,
        "latency": latency,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

    if metadata:
        data["metadata"] = metadata

    log_metrics(data)


def log_backtest_metrics(
    strategy: str, sharpe_ratio: float, win_rate: float, mse: float, metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Log backtest performance metrics.

    Args:
        strategy: Name of the strategy
        sharpe_ratio: Sharpe ratio
        win_rate: Win rate (0-1)
        mse: Mean squared error
        metadata: Optional additional metadata
    """
    data = {
        "type": "backtest_metrics",
        "strategy": strategy,
        "sharpe_ratio": sharpe_ratio,
        "win_rate": win_rate,
        "mse": mse,
    }

    if metadata:
        data["metadata"] = metadata

    log_metrics(data)


def log_agent_metrics(agent_id: str, action: str, confidence: float, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Log agent decision metrics.

    Args:
        agent_id: ID of the agent
        action: Action taken
        confidence: Confidence score (0-1)
        metadata: Optional additional metadata
    """
    data = {"type": "agent_metrics", "agent_id": agent_id, "action": action, "confidence": confidence}

    if metadata:
        data["metadata"] = metadata

    log_metrics(data)
