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
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List, Union


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
        
        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(console_handler)
        
        # Add timed rotating file handler
        file_handler = logging.handlers.TimedRotatingFileHandler(
            self.log_dir / log_file,
            when=when,
            interval=interval,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(file_handler)
        
        self.loggers[name] = logger
        self.handlers[name] = file_handler
        
        return logger

    def setup_memory_handler(
        self, 
        name: str,
        capacity: int = 1000,
        flushLevel: int = logging.ERROR,
        target_handler: Optional[logging.Handler] = None
    ) -> logging.Logger:
        """
        Setup a logger with memory handler for buffered logging.

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
        
        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(console_handler)
        
        # Add memory handler
        if target_handler is None:
            target_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / f"{name}_buffered.log",
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            target_handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ))
        
        memory_handler = logging.handlers.MemoryHandler(
            capacity=capacity,
            flushLevel=flushLevel,
            target=target_handler
        )
        logger.addHandler(memory_handler)
        
        self.loggers[name] = logger
        self.handlers[name] = memory_handler
        
        return logger

    def setup_queue_handler(
        self, 
        name: str,
        queue_size: int = 1000
    ) -> logging.Logger:
        """
        Setup a logger with queue handler for async logging.

        Args:
            name: Logger name
            queue_size: Queue size

        Returns:
            logging.Logger: Configured logger
        """
        try:
            import queue
            from logging.handlers import QueueHandler, QueueListener
            
            logger = logging.getLogger(name)
            logger.setLevel(self.log_level)
            
            # Clear existing handlers
            logger.handlers.clear()
            
            # Create queue and handler
            log_queue = queue.Queue(maxsize=queue_size)
            queue_handler = QueueHandler(log_queue)
            logger.addHandler(queue_handler)
            
            # Create listener with file handler
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / f"{name}_async.log",
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ))
            
            listener = QueueListener(log_queue, file_handler)
            listener.start()
            
            self.loggers[name] = logger
            self.handlers[name] = listener
            
            return logger
            
        except ImportError:
            # Fallback to regular handler if queue not available
            return self.get_enhanced_logger(name)

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

    def get_log_files(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed information about log files.

        Returns:
            Dict: Mapping of log file names to their details
        """
        log_files = {}
        for log_file in self.log_dir.glob("*.log*"):
            stat = log_file.stat()
            log_files[log_file.name] = {
                "size_kb": f"{stat.st_size / 1024:.1f}",
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
            }
        return log_files

    def cleanup_old_logs(self, days: int = 30, max_size_mb: Optional[int] = None):
        """
        Clean up old log files with size limit option.

        Args:
            days: Number of days to keep logs
            max_size_mb: Maximum total size in MB
        """
        cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
        deleted_count = 0
        deleted_size = 0

        # Get all log files sorted by modification time
        log_files = sorted(
            self.log_dir.glob("*.log*"),
            key=lambda x: x.stat().st_mtime
        )

        for log_file in log_files:
            if log_file.stat().st_mtime < cutoff_time:
                size = log_file.stat().st_size
                log_file.unlink()
                deleted_count += 1
                deleted_size += size

        # Size-based cleanup
        if max_size_mb:
            total_size = sum(f.stat().st_size for f in self.log_dir.glob("*.log*"))
            max_size_bytes = max_size_mb * 1024 * 1024
            
            if total_size > max_size_bytes:
                for log_file in log_files:
                    if total_size <= max_size_bytes:
                        break
                    size = log_file.stat().st_size
                    log_file.unlink()
                    deleted_count += 1
                    deleted_size += size
                    total_size -= size

        if deleted_count > 0:
            logging.info(f"Cleaned up {deleted_count} old log files ({deleted_size / 1024 / 1024:.1f} MB)")

    def get_logger_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all loggers.

        Returns:
            Dict: Logger statistics
        """
        stats = {
            "total_loggers": len(self.loggers),
            "log_files": self.get_log_files(),
            "handlers": {}
        }
        
        for name, handler in self.handlers.items():
            handler_type = type(handler).__name__
            stats["handlers"][name] = {
                "type": handler_type,
                "level": handler.level if hasattr(handler, 'level') else 'N/A'
            }
        
        return stats

    def flush_all_handlers(self):
        """Flush all handlers to ensure logs are written."""
        for handler in self.handlers.values():
            if hasattr(handler, 'flush'):
                handler.flush()
            elif hasattr(handler, 'target') and hasattr(handler.target, 'flush'):
                handler.target.flush()

    def close_all_handlers(self):
        """Close all handlers properly."""
        for handler in self.handlers.values():
            if hasattr(handler, 'close'):
                handler.close()
            elif hasattr(handler, 'target') and hasattr(handler.target, 'close'):
                handler.target.close()


# Legacy compatibility
class LogManager(EnhancedLogManager):
    """Legacy LogManager for backward compatibility."""
    pass


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
                self.log_dir / "model_operations.log",
                maxBytes=10 * 1024 * 1024,
                backupCount=5,  # 10MB
            ),
        ]

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        for handler in handlers:
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_model_training(self, model_name: str, metrics: Dict[str, Any]):
        """Log model training information."""
        self.logger.info(f"Model training completed: {model_name}")
        self.logger.info(f"Training metrics: {metrics}")

    def log_model_prediction(self, model_name: str, prediction: Any, confidence: float):
        """Log model prediction information."""
        self.logger.info(
            f"Model prediction: {model_name} - Confidence: {confidence:.3f}"
        )
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
                self.log_dir / "data_operations.log",
                maxBytes=10 * 1024 * 1024,
                backupCount=5,  # 10MB
            ),
        ]

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

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
                self.log_dir / "performance.log",
                maxBytes=10 * 1024 * 1024,
                backupCount=5,  # 10MB
            ),
        ]

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        for handler in handlers:
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_performance_metric(self, metric_name: str, value: float, unit: str = ""):
        """Log a performance metric."""
        self.logger.info(f"Performance metric: {metric_name} = {value}{unit}")

    def log_system_health(self, health_status: Dict[str, Any]):
        """Log system health information."""
        self.logger.info(f"System health: {health_status}")


def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO) -> EnhancedLogManager:
    """
    Setup logging for the application.

    Args:
        log_dir: Directory for log files
        log_level: Logging level

    Returns:
        EnhancedLogManager: Configured enhanced log manager instance
    """
    return EnhancedLogManager(log_dir, log_level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific component using the __name__ pattern.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)
