"""
Logging configuration and setup.

This module handles logging configuration, setup, and management for the application.
"""

import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .app_config import get_config


class LoggingConfig:
    """
    Logging configuration manager.

    This class handles the setup and configuration of logging for the application.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize logging configuration.

        Args:
            config: Logging configuration dictionary
        """
        self.config = config or self._get_default_config()
        self._setup_logging()

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default logging configuration.

        Returns:
            Default configuration dictionary
        """
        app_config = get_config()
        return {
            "level": app_config.logging.level,
            "format": app_config.logging.format,
            "file": app_config.logging.file,
            "max_size": app_config.logging.max_size,
            "backup_count": app_config.logging.backup_count,
            "console": app_config.logging.console,
            "json_format": app_config.logging.json_format,
        }

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        try:
            # Create logs directory if it doesn't exist
            log_file = Path(self.config["file"])
            log_file.parent.mkdir(parents=True, exist_ok=True)

            # Configure root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(self._get_log_level())

            # Clear existing handlers
            root_logger.handlers.clear()

            # Add console handler if enabled
            if self.config.get("console", True):
                console_handler = self._create_console_handler()
                root_logger.addHandler(console_handler)

            # Add file handler
            file_handler = self._create_file_handler()
            root_logger.addHandler(file_handler)

            # Add error file handler
            error_handler = self._create_error_handler()
            root_logger.addHandler(error_handler)

            # Add performance handler
            perf_handler = self._create_performance_handler()
            root_logger.addHandler(perf_handler)

            # Add audit handler
            audit_handler = self._create_audit_handler()
            root_logger.addHandler(audit_handler)

            logging.info("Logging configuration initialized successfully")

        except Exception as e:
            # Use basic logging for error since logging setup failed
            logging.basicConfig(
                level=logging.ERROR,
                format="%(asctime)s - %(levelname)s - %(message)s",
                handlers=[logging.StreamHandler(sys.stderr)],
            )
            logging.error(f"Error setting up logging: {e}")
            # Fallback to basic logging
            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    def _get_log_level(self) -> int:
        """
        Get log level from configuration.

        Returns:
            Log level integer
        """
        level_str = self.config.get("level", "INFO").upper()
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        return level_map.get(level_str, logging.INFO)

    def _create_console_handler(self) -> logging.StreamHandler:
        """
        Create console handler.

        Returns:
            Console handler
        """
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(self._get_log_level())

        if self.config.get("json_format", False):
            formatter = self._create_json_formatter()
        else:
            formatter = logging.Formatter(self.config.get("format"))

        handler.setFormatter(formatter)
        return handler

    def _create_file_handler(self) -> logging.handlers.RotatingFileHandler:
        """
        Create rotating file handler.

        Returns:
            Rotating file handler
        """
        handler = logging.handlers.RotatingFileHandler(
            filename=self.config["file"],
            maxBytes=self.config.get("max_size", 10485760),  # 10MB
            backupCount=self.config.get("backup_count", 5),
            encoding="utf-8",
        )
        handler.setLevel(self._get_log_level())

        if self.config.get("json_format", False):
            formatter = self._create_json_formatter()
        else:
            formatter = logging.Formatter(self.config.get("format"))

        handler.setFormatter(formatter)
        return handler

    def _create_error_handler(self) -> logging.handlers.RotatingFileHandler:
        """
        Create error file handler.

        Returns:
            Error file handler
        """
        error_file = Path(self.config["file"]).parent / "errors.log"
        handler = logging.handlers.RotatingFileHandler(
            filename=str(error_file),
            maxBytes=self.config.get("max_size", 10485760),
            backupCount=self.config.get("backup_count", 5),
            encoding="utf-8",
        )
        handler.setLevel(logging.ERROR)

        if self.config.get("json_format", False):
            formatter = self._create_json_formatter()
        else:
            formatter = logging.Formatter(self.config.get("format"))

        handler.setFormatter(formatter)
        return handler

    def _create_performance_handler(self) -> logging.handlers.RotatingFileHandler:
        """
        Create performance logging handler.

        Returns:
            Performance file handler
        """
        perf_file = Path(self.config["file"]).parent / "performance.log"
        handler = logging.handlers.RotatingFileHandler(
            filename=str(perf_file),
            maxBytes=self.config.get("max_size", 10485760),
            backupCount=self.config.get("backup_count", 5),
            encoding="utf-8",
        )
        handler.setLevel(logging.INFO)

        # Custom formatter for performance logs
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)

        # Create performance logger
        perf_logger = logging.getLogger("performance")
        perf_logger.addHandler(handler)
        perf_logger.setLevel(logging.INFO)
        perf_logger.propagate = False

        return handler

    def _create_audit_handler(self) -> logging.handlers.RotatingFileHandler:
        """
        Create audit logging handler.

        Returns:
            Audit file handler
        """
        audit_file = Path(self.config["file"]).parent / "audit.log"
        handler = logging.handlers.RotatingFileHandler(
            filename=str(audit_file),
            maxBytes=self.config.get("max_size", 10485760),
            backupCount=self.config.get("backup_count", 5),
            encoding="utf-8",
        )
        handler.setLevel(logging.INFO)

        # Custom formatter for audit logs
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)

        # Create audit logger
        audit_logger = logging.getLogger("audit")
        audit_logger.addHandler(handler)
        audit_logger.setLevel(logging.INFO)
        audit_logger.propagate = False

        return handler

    def _create_json_formatter(self) -> logging.Formatter:
        """
        Create JSON formatter.

        Returns:
            JSON formatter
        """

        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno,
                }

                # Add exception info if present
                if record.exc_info:
                    log_entry["exception"] = self.formatException(record.exc_info)

                # Add extra fields
                for key, value in record.__dict__.items():
                    if key not in [
                        "name",
                        "msg",
                        "args",
                        "levelname",
                        "levelno",
                        "pathname",
                        "filename",
                        "module",
                        "lineno",
                        "funcName",
                        "created",
                        "msecs",
                        "relativeCreated",
                        "thread",
                        "threadName",
                        "processName",
                        "process",
                        "getMessage",
                        "exc_info",
                        "exc_text",
                        "stack_info",
                    ]:
                        log_entry[key] = value

                return json.dumps(log_entry)

        return JSONFormatter()

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get logger with specified name.

        Args:
            name: Logger name

        Returns:
            Logger instance
        """
        return logging.getLogger(name)

    def get_performance_logger(self) -> logging.Logger:
        """
        Get performance logger.

        Returns:
            Performance logger
        """
        return logging.getLogger("performance")

    def get_audit_logger(self) -> logging.Logger:
        """
        Get audit logger.

        Returns:
            Audit logger
        """
        return logging.getLogger("audit")

    def set_level(self, level: str) -> None:
        """
        Set logging level.

        Args:
            level: Log level string
        """
        level_int = self._get_log_level()
        logging.getLogger().setLevel(level_int)

        # Update all handlers
        for handler in logging.getLogger().handlers:
            handler.setLevel(level_int)

    def add_handler(self, handler: logging.Handler) -> None:
        """
        Add custom handler to root logger.

        Args:
            handler: Logging handler
        """
        logging.getLogger().addHandler(handler)

    def remove_handler(self, handler: logging.Handler) -> None:
        """
        Remove handler from root logger.

        Args:
            handler: Logging handler
        """
        logging.getLogger().removeHandler(handler)

    def get_log_files(self) -> List[Path]:
        """
        Get list of log files.

        Returns:
            List of log file paths
        """
        log_dir = Path(self.config["file"]).parent
        log_files = []

        if log_dir.exists():
            for file in log_dir.glob("*.log*"):
                log_files.append(file)

        return sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)

    def cleanup_old_logs(self, days: int = 30) -> int:
        """
        Clean up old log files.

        Args:
            days: Number of days to keep logs

        Returns:
            Number of files removed
        """
        import time

        cutoff_time = time.time() - (days * 24 * 60 * 60)
        removed_count = 0

        for log_file in self.get_log_files():
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    removed_count += 1
                except Exception as e:
                    logging.warning(f"Failed to remove old log file {log_file}: {e}")

        return removed_count

    def get_log_stats(self) -> Dict[str, Any]:
        """
        Get logging statistics.

        Returns:
            Logging statistics dictionary
        """
        log_files = self.get_log_files()
        total_size = sum(f.stat().st_size for f in log_files)

        return {
            "total_files": len(log_files),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "files": [
                {
                    "name": f.name,
                    "size_bytes": f.stat().st_size,
                    "size_mb": f.stat().st_size / (1024 * 1024),
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                }
                for f in log_files
            ],
        }


# Global logging configuration instance
_logging_config: Optional[LoggingConfig] = None


def setup_logging(config: Optional[Dict[str, Any]] = None) -> LoggingConfig:
    """
    Setup logging configuration.

    Args:
        config: Logging configuration dictionary

    Returns:
        LoggingConfig instance
    """
    global _logging_config
    _logging_config = LoggingConfig(config)
    return _logging_config


def get_logging_config() -> LoggingConfig:
    """
    Get global logging configuration.

    Returns:
        LoggingConfig instance
    """
    global _logging_config
    if _logging_config is None:
        _logging_config = LoggingConfig()
    return _logging_config


def get_logger(name: str) -> logging.Logger:
    """
    Get logger with specified name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return get_logging_config().get_logger(name)


def get_performance_logger() -> logging.Logger:
    """
    Get performance logger.

    Returns:
        Performance logger
    """
    return get_logging_config().get_performance_logger()


def get_audit_logger() -> logging.Logger:
    """
    Get audit logger.

    Returns:
        Audit logger
    """
    return get_logging_config().get_audit_logger()


# Convenience functions for common logging patterns


def log_function_call(func_name: str, args: tuple = None, kwargs: dict = None, logger: logging.Logger = None) -> None:
    """
    Log function call.

    Args:
        func_name: Function name
        args: Function arguments
        kwargs: Function keyword arguments
        logger: Logger instance
    """
    if logger is None:
        logger = get_logger(__name__)

    args_str = str(args) if args else "()"
    kwargs_str = str(kwargs) if kwargs else "{}"
    logger.debug(f"Function call: {func_name}{args_str} {kwargs_str}")


def log_performance(operation: str, duration: float, details: dict = None, logger: logging.Logger = None) -> None:
    """
    Log performance metrics.

    Args:
        operation: Operation name
        duration: Duration in seconds
        details: Additional details
        logger: Logger instance
    """
    if logger is None:
        logger = get_performance_logger()

    log_data = {"operation": operation, "duration_seconds": duration, "timestamp": datetime.now().isoformat()}

    if details:
        log_data.update(details)

    logger.info(f"Performance: {json.dumps(log_data)}")


def log_audit_event(event_type: str, user: str = None, details: dict = None, logger: logging.Logger = None) -> None:
    """
    Log audit event.

    Args:
        event_type: Event type
        user: User performing the action
        details: Event details
        logger: Logger instance
    """
    if logger is None:
        logger = get_audit_logger()

    log_data = {"event_type": event_type, "user": user or "system", "timestamp": datetime.now().isoformat()}

    if details:
        log_data.update(details)

    logger.info(f"Audit: {json.dumps(log_data)}")


# Initialize logging on module import
setup_logging()
