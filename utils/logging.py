"""
Logging Module

Enhanced with Batch 11 features: comprehensive file output + optional rotating log handler
support via logging.handlers with additional configuration options.

This module provides centralized logging configuration and utilities for the Evolve Trading Platform:
- Logging setup and configuration
- Log level management
- Log file rotation and management
- Structured logging for different components
- Enhanced file output with multiple formats
- Configurable rotating log handlers
"""

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class EnhancedLogManager:
    """Enhanced log manager with comprehensive file output and rotating handlers."""

    def __init__(
        self,
        log_dir: str = "logs",
        log_level: int = logging.INFO,
        enable_file_output: bool = True,
        enable_rotating_handlers: bool = True,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        enable_json_logging: bool = False
    ):
        """
        Initialize the enhanced log manager.

        Args:
            log_dir: Directory for log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            enable_file_output: Whether to enable file output
            enable_rotating_handlers: Whether to use rotating file handlers
            max_file_size: Maximum size for log files before rotation
            backup_count: Number of backup files to keep
            enable_json_logging: Whether to enable JSON format logging
        """
        self.log_dir = Path(log_dir)
        self.log_level = log_level
        self.enable_file_output = enable_file_output
        self.enable_rotating_handlers = enable_rotating_handlers
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_json_logging = enable_json_logging
        self.loggers = {}
        self.handlers = {}

        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup basic logging
        self._setup_enhanced_logging()

    def _setup_enhanced_logging(self):
        """Setup enhanced logging configuration."""
        # Configure root logger
        handlers = [logging.StreamHandler(sys.stdout)]

        if self.enable_file_output:
            if self.enable_rotating_handlers:
                file_handler = logging.handlers.RotatingFileHandler(
                    self.log_dir / "evolve.log",
                    maxBytes=self.max_file_size,
                    backupCount=self.backup_count,
                    encoding='utf-8'
                )
            else:
                file_handler = logging.FileHandler(
                    self.log_dir / "evolve.log",
                    encoding='utf-8'
                )

            if self.enable_json_logging:
                file_handler.setFormatter(logging.Formatter(
                    '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}'
                ))
            else:
                file_handler.setFormatter(logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                ))

            handlers.append(file_handler)

        logging.basicConfig(
            level=self.log_level,
            handlers=handlers,
            force=True
        )

    def get_enhanced_logger(
        self,
        name: str,
        log_file: Optional[str] = None,
        enable_console: bool = True,
        enable_file: bool = True,
        enable_json: Optional[bool] = None
    ) -> logging.Logger:
        """
        Get an enhanced logger for a specific component.

        Args:
            name: Logger name (usually __name__)
            log_file: Custom log file name
            enable_console: Whether to enable console output
            enable_file: Whether to enable file output
            enable_json: Whether to use JSON format (None = use global setting)

        Returns:
            logging.Logger: Configured logger instance
        """
        if name not in self.loggers:
            logger = logging.getLogger(name)
            logger.setLevel(self.log_level)

            # Clear existing handlers
            logger.handlers.clear()

            handlers = []

            # Add console handler if enabled
            if enable_console:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                ))
                handlers.append(console_handler)

            # Add file handler if enabled
            if enable_file and self.enable_file_output:
                if log_file is None:
                    log_file = f"{name.replace('.', '_')}.log"

                if self.enable_rotating_handlers:
                    file_handler = logging.handlers.RotatingFileHandler(
                        self.log_dir / log_file,
                        maxBytes=self.max_file_size,
                        backupCount=self.backup_count,
                        encoding='utf-8'
                    )
                else:
                    file_handler = logging.FileHandler(
                        self.log_dir / log_file,
                        encoding='utf-8'
                    )

                use_json = enable_json if enable_json is not None else self.enable_json_logging
                if use_json:
                    file_handler.setFormatter(logging.Formatter(
                        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}'
                    ))
                else:
                    file_handler.setFormatter(logging.Formatter(
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    ))

                handlers.append(file_handler)
                self.handlers[name] = file_handler

            # Add all handlers to logger
            for handler in handlers:
                logger.addHandler(handler)

            self.loggers[name] = logger

        return self.loggers[name]

    def setup_timed_rotating_handler(
        self,
        name: str,
        log_file: str,
        when: str = 'midnight',
        interval: int = 1,
        backup_count: int = 30
    ) -> logging.Logger:
        """
        Setup a logger with timed rotating file handler.

        Args:
            name: Logger name
            log_file: Log file name
            when: When to rotate ('S', 'M', 'H', 'D', 'W0'-'W6', 'midnight')
            interval: Interval between rotations
            backup_count: Number of backup files to keep

        Returns:
            logging.Logger: Configured logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)

        # Clear existing handlers
        logger.handlers.clear()

        # Create timed rotating file handler
        handler = logging.handlers.TimedRotatingFileHandler(
            self.log_dir / log_file,
            when=when,
            interval=interval,
            backupCount=backup_count,
            encoding='utf-8'
        )

        # Set formatter
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))

        logger.addHandler(handler)
        self.loggers[name] = logger
        self.handlers[name] = handler

        return logger

    def setup_memory_handler(
        self,
        name: str,
        capacity: int = 1000,
        flushLevel: int = logging.ERROR,
        target_handler: Optional[logging.Handler] = None
    ) -> logging.Logger:
        """
        Setup a logger with memory handler.

        Args:
            name: Logger name
            capacity: Buffer capacity
            flushLevel: Level at which to flush buffer
            target_handler: Target handler for flushing

        Returns:
            logging.Logger: Configured logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)

        # Clear existing handlers
        logger.handlers.clear()

        # Create memory handler
        if target_handler is None:
            target_handler = logging.StreamHandler(sys.stdout)

        handler = logging.handlers.MemoryHandler(
            capacity=capacity,
            flushLevel=flushLevel,
            target=target_handler
        )

        # Set formatter
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))

        logger.addHandler(handler)
        self.loggers[name] = logger
        self.handlers[name] = handler

        return logger

    def setup_queue_handler(
        self,
        name: str,
        queue_size: int = 1000
    ) -> logging.Logger:
        """
        Setup a logger with queue handler.

        Args:
            name: Logger name
            queue_size: Queue size

        Returns:
            logging.Logger: Configured logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)

        # Clear existing handlers
        logger.handlers.clear()

        # Create queue handler
        handler = logging.handlers.QueueHandler(queue_size)

        # Set formatter
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))

        logger.addHandler(handler)
        self.loggers[name] = logger
        self.handlers[name] = handler

        return logger

    def set_log_level(self, level: int):
        """Set log level for all loggers."""
        self.log_level = level
        for logger in self.loggers.values():
            logger.setLevel(level)

    def get_log_files(self) -> Dict[str, Dict[str, Any]]:
        """Get information about log files."""
        log_files = {}
        for log_file in self.log_dir.glob("*.log"):
            try:
                stat = log_file.stat()
                log_files[log_file.name] = {
                    "size_mb": stat.st_size / (1024 * 1024),
                    "modified": datetime.fromtimestamp(stat.st_mtime),
                    "created": datetime.fromtimestamp(stat.st_ctime)
                }
            except Exception:
                continue
        return log_files

    def cleanup_old_logs(self, days: int = 30, max_size_mb: Optional[int] = None):
        """Clean up old log files."""
        cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_date = cutoff_date.replace(day=cutoff_date.day - days)

        cleaned_count = 0
        for log_file in self.log_dir.glob("*.log*"):
            try:
                file_date = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_date < cutoff_date:
                    log_file.unlink()
                    cleaned_count += 1
            except Exception:
                continue

        # Size-based cleanup
        if max_size_mb:
            total_size = sum(
                f.stat().st_size for f in self.log_dir.glob("*.log*")
                if f.is_file()
            ) / (1024 * 1024)

            if total_size > max_size_mb:
                # Remove oldest files until under size limit
                log_files = []
                for log_file in self.log_dir.glob("*.log*"):
                    if log_file.is_file():
                        log_files.append((log_file, log_file.stat().st_mtime))

                log_files.sort(key=lambda x: x[1])  # Sort by modification time

                for log_file, _ in log_files:
                    if total_size <= max_size_mb:
                        break
                    try:
                        file_size = log_file.stat().st_size / (1024 * 1024)
                        log_file.unlink()
                        total_size -= file_size
                        cleaned_count += 1
                    except Exception:
                        continue

        return cleaned_count

    def get_logger_stats(self) -> Dict[str, Any]:
        """Get statistics about loggers."""
        return {
            "total_loggers": len(self.loggers),
            "total_handlers": len(self.handlers),
            "log_level": self.log_level,
            "enable_file_output": self.enable_file_output,
            "enable_rotating_handlers": self.enable_rotating_handlers,
            "enable_json_logging": self.enable_json_logging
        }

    def flush_all_handlers(self):
        """Flush all handlers."""
        for handler in self.handlers.values():
            try:
                handler.flush()
            except Exception:
                continue

    def close_all_handlers(self):
        """Close all handlers."""
        for handler in self.handlers.values():
            try:
                handler.close()
            except Exception:
                continue


class LogManager(EnhancedLogManager):
    """Backward compatibility alias for EnhancedLogManager."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ModelLogger:
    """Specialized logger for model operations."""

    def __init__(self, log_dir: str = "logs/models"):
        """Initialize model logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logger()

    def _setup_logger(self):
        """Setup the model logger."""
        self.logger = logging.getLogger("model_logger")
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Add file handler
        file_handler = logging.FileHandler(
            self.log_dir / "model_operations.log",
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        ))
        self.logger.addHandler(file_handler)

    def log_model_training(self, model_name: str, metrics: Dict[str, Any]):
        """Log model training metrics."""
        self.logger.info(f"Model training completed: {model_name}, metrics: {metrics}")

    def log_model_prediction(self, model_name: str, prediction: Any, confidence: float):
        """Log model prediction."""
        self.logger.info(f"Model prediction: {model_name}, prediction: {prediction}, confidence: {confidence}")

    def log_model_error(self, model_name: str, error: Exception):
        """Log model error."""
        self.logger.error(f"Model error: {model_name}, error: {error}")


class DataLogger:
    """Specialized logger for data operations."""

    def __init__(self, log_dir: str = "logs/data"):
        """Initialize data logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logger()

    def _setup_logger(self):
        """Setup the data logger."""
        self.logger = logging.getLogger("data_logger")
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Add file handler
        file_handler = logging.FileHandler(
            self.log_dir / "data_operations.log",
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        ))
        self.logger.addHandler(file_handler)

    def log_data_load(self, symbol: str, records: int, date_range: str):
        """Log data loading operation."""
        self.logger.info(f"Data loaded: {symbol}, records: {records}, range: {date_range}")

    def log_data_error(self, symbol: str, error: Exception):
        """Log data error."""
        self.logger.error(f"Data error: {symbol}, error: {error}")


class PerformanceLogger:
    """Specialized logger for performance metrics."""

    def __init__(self, log_dir: str = "logs/performance"):
        """Initialize performance logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logger()

    def _setup_logger(self):
        """Setup the performance logger."""
        self.logger = logging.getLogger("performance_logger")
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Add file handler
        file_handler = logging.FileHandler(
            self.log_dir / "performance_metrics.log",
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        ))
        self.logger.addHandler(file_handler)

    def log_performance_metric(self, metric_name: str, value: float, unit: str = ""):
        """Log performance metric."""
        self.logger.info(f"Performance metric: {metric_name} = {value} {unit}")

    def log_system_health(self, health_status: Dict[str, Any]):
        """Log system health status."""
        self.logger.info(f"System health: {health_status}")


def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO) -> EnhancedLogManager:
    """
    Setup enhanced logging.

    Args:
        log_dir: Directory for log files
        log_level: Logging level

    Returns:
        EnhancedLogManager instance
    """
    return EnhancedLogManager(log_dir=log_dir, log_level=log_level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name

    Returns:
        logging.Logger instance
    """
    return logging.getLogger(name)
