"""
Logging Module

This module provides centralized logging configuration and utilities for the Evolve Trading Platform:
- Logging setup and configuration
- Log level management
- Log file rotation and management
- Structured logging for different components
"""

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class LogManager:
    """Manages logging configuration and setup."""

    def __init__(self, log_dir: str = "logs", log_level: int = logging.INFO):
        """
        Initialize the log manager.

        Args:
            log_dir: Directory for log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_dir = Path(log_dir)
        self.log_level = log_level
        self.loggers = {}

        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup basic logging
        self._setup_basic_logging()

    def _setup_basic_logging(self):
        """Setup basic logging configuration."""
        # Configure root logger
        logging.basicConfig(
            level=self.log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.handlers.RotatingFileHandler(
                    self.log_dir / "evolve.log", maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
                ),
            ],
        )

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger for a specific component.

        Args:
            name: Logger name (usually __name__)

        Returns:
            logging.Logger: Configured logger instance
        """
        if name not in self.loggers:
            logger = logging.getLogger(name)
            logger.setLevel(self.log_level)

            # Add file handler for this component
            component_log_file = self.log_dir / f"{name.replace('.', '_')}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                component_log_file, maxBytes=5 * 1024 * 1024, backupCount=3  # 5MB
            )
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            logger.addHandler(file_handler)

            self.loggers[name] = logger

        return self.loggers[name]

    def setup_component_logging(self, component_name: str, log_file: Optional[str] = None) -> logging.Logger:
        """
        Setup logging for a specific component.

        Args:
            component_name: Name of the component
            log_file: Optional custom log file name

        Returns:
            logging.Logger: Configured logger for the component
        """
        if log_file is None:
            log_file = f"{component_name.lower()}.log"

        logger = logging.getLogger(component_name)
        logger.setLevel(self.log_level)

        # Clear existing handlers
        logger.handlers.clear()

        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(console_handler)

        # Add file handler
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / log_file, maxBytes=5 * 1024 * 1024, backupCount=3  # 5MB
        )
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

        self.loggers[component_name] = logger
        return logger

    def set_log_level(self, level: int):
        """
        Set the logging level for all loggers.

        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_level = level
        for logger in self.loggers.values():
            logger.setLevel(level)
        logging.getLogger().setLevel(level)

    def get_log_files(self) -> Dict[str, str]:
        """
        Get list of log files and their sizes.

        Returns:
            Dict: Mapping of log file names to their sizes
        """
        log_files = {}
        for log_file in self.log_dir.glob("*.log"):
            size = log_file.stat().st_size
            log_files[log_file.name] = f"{size / 1024:.1f} KB"
        return log_files

    def cleanup_old_logs(self, days: int = 30):
        """
        Clean up log files older than specified days.

        Args:
            days: Number of days to keep logs
        """
        cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
        deleted_count = 0

        for log_file in self.log_dir.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff_time:
                log_file.unlink()
                deleted_count += 1

        if deleted_count > 0:
            logging.info(f"Cleaned up {deleted_count} old log files")


class ModelLogger:
    """Specialized logger for model operations."""

    def __init__(self, log_dir: str = "logs/models"):
        """
        Initialize model logger.

        Args:
            log_dir: Directory for model logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("model_operations")
        self._setup_logger()

    def _setup_logger(self):
        """Setup the model logger."""
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Add handlers
        handlers = [
            logging.StreamHandler(sys.stdout),
            logging.handlers.RotatingFileHandler(
                self.log_dir / "model_operations.log", maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
            ),
        ]

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        for handler in handlers:
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_model_training(self, model_name: str, metrics: Dict[str, Any]):
        """Log model training information."""
        self.logger.info(f"Model training completed: {model_name}")
        self.logger.info(f"Training metrics: {metrics}")

    def log_model_prediction(self, model_name: str, prediction: Any, confidence: float):
        """Log model prediction information."""
        self.logger.info(f"Model prediction: {model_name} - Confidence: {confidence:.3f}")
        self.logger.debug(f"Prediction value: {prediction}")

    def log_model_error(self, model_name: str, error: Exception):
        """Log model error information."""
        self.logger.error(f"Model error in {model_name}: {error}")


class DataLogger:
    """Specialized logger for data operations."""

    def __init__(self, log_dir: str = "logs/data"):
        """
        Initialize data logger.

        Args:
            log_dir: Directory for data logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("data_operations")
        self._setup_logger()

    def _setup_logger(self):
        """Setup the data logger."""
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Add handlers
        handlers = [
            logging.StreamHandler(sys.stdout),
            logging.handlers.RotatingFileHandler(
                self.log_dir / "data_operations.log", maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
            ),
        ]

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        for handler in handlers:
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_data_load(self, symbol: str, records: int, date_range: str):
        """Log data loading information."""
        self.logger.info(f"Data loaded: {symbol} - {records} records - {date_range}")

    def log_data_error(self, symbol: str, error: Exception):
        """Log data error information."""
        self.logger.error(f"Data error for {symbol}: {error}")


class PerformanceLogger:
    """Specialized logger for performance metrics."""

    def __init__(self, log_dir: str = "logs/performance"):
        """
        Initialize performance logger.

        Args:
            log_dir: Directory for performance logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("performance")
        self._setup_logger()

    def _setup_logger(self):
        """Setup the performance logger."""
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Add handlers
        handlers = [
            logging.StreamHandler(sys.stdout),
            logging.handlers.RotatingFileHandler(
                self.log_dir / "performance.log", maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
            ),
        ]

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        for handler in handlers:
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_performance_metric(self, metric_name: str, value: float, unit: str = ""):
        """Log a performance metric."""
        self.logger.info(f"Performance metric: {metric_name} = {value}{unit}")

    def log_system_health(self, health_status: Dict[str, Any]):
        """Log system health information."""
        self.logger.info(f"System health: {health_status}")


def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO) -> LogManager:
    """
    Setup logging for the application.

    Args:
        log_dir: Directory for log files
        log_level: Logging level

    Returns:
        LogManager: Configured log manager instance
    """
    return LogManager(log_dir, log_level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)
