"""
Enhanced Logging Utilities with Return Values

This module provides logging utilities that return status information
for better integration with the agentic architecture.
"""

import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class LoggingManager:
    """Manages logging configuration and provides status returns."""

    def __init__(self, log_dir: str = "logs", max_log_size: int = 10 * 1024 * 1024):
        """
        Initialize logging manager.

        Args:
            log_dir: Directory for log files
            max_log_size: Maximum log file size in bytes
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.max_log_size = max_log_size
        self.loggers = {}

    def setup_logger(
        self, name: str, level: str = "INFO", log_to_file: bool = True, log_to_console: bool = True
    ) -> Dict[str, Any]:
        """
        Setup a logger with file and console handlers.

        Args:
            name: Logger name
            level: Logging level
            log_to_file: Whether to log to file
            log_to_console: Whether to log to console

        Returns:
            Dictionary with setup status
        """
        try:
            if name in self.loggers:
                return {"success": True, "message": f"Logger {name} already exists", "logger": self.loggers[name]}

            logger = logging.getLogger(name)
            logger.setLevel(getattr(logging, level.upper()))

            # Clear existing handlers
            logger.handlers.clear()

            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

            # File handler
            if log_to_file:
                log_file = self.log_dir / f"{name}.log"
                file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=self.max_log_size, backupCount=5)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

            # Console handler
            if log_to_console:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)

            self.loggers[name] = logger

            return {
                "success": True,
                "message": f"Logger {name} setup successfully",
                "logger": logger,
                "log_file": str(log_file) if log_to_file else None,
            }

        except Exception as e:
            return {"success": False, "error": f"Failed to setup logger {name}: {str(e)}", "logger": None}

    def log_event(
        self, logger_name: str, level: str, message: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log an event with metadata.

        Args:
            logger_name: Name of the logger to use
            level: Logging level
            message: Log message
            metadata: Additional metadata

        Returns:
            Dictionary with logging status
        """
        try:
            if logger_name not in self.loggers:
                setup_result = self.setup_logger(logger_name)
                if not setup_result["success"]:
                    return setup_result

            logger = self.loggers[logger_name]
            log_func = getattr(logger, level.lower())

            # Format message with metadata
            if metadata:
                message_with_metadata = f"{message} | Metadata: {json.dumps(metadata)}"
            else:
                message_with_metadata = message

            log_func(message_with_metadata)

            return {
                "success": True,
                "message": "Event logged successfully",
                "timestamp": datetime.now().isoformat(),
                "level": level,
                "logger": logger_name,
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to log event: {str(e)}",
                "timestamp": datetime.now().isoformat(),
            }

    def get_log_stats(self, logger_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get logging statistics.

        Args:
            logger_name: Specific logger name (None for all)

        Returns:
            Dictionary with log statistics
        """
        try:
            stats = {"total_loggers": len(self.loggers), "log_files": [], "total_log_size": 0}

            loggers_to_check = [logger_name] if logger_name else self.loggers.keys()

            for name in loggers_to_check:
                if name in self.loggers:
                    log_file = self.log_dir / f"{name}.log"
                    if log_file.exists():
                        file_size = log_file.stat().st_size
                        stats["log_files"].append(
                            {"name": name, "size_mb": file_size / (1024 * 1024), "path": str(log_file)}
                        )
                        stats["total_log_size"] += file_size

            stats["total_log_size_mb"] = stats["total_log_size"] / (1024 * 1024)

            return {"success": True, "stats": stats, "timestamp": datetime.now().isoformat()}

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get log stats: {str(e)}",
                "timestamp": datetime.now().isoformat(),
            }

    def cleanup_old_logs(self, max_age_days: int = 30) -> Dict[str, Any]:
        """
        Clean up old log files.

        Args:
            max_age_days: Maximum age of log files to keep

        Returns:
            Dictionary with cleanup status
        """
        try:
            current_time = datetime.now()
            files_removed = 0
            size_freed = 0

            for log_file in self.log_dir.glob("*.log*"):
                if log_file.is_file():
                    file_age = current_time - datetime.fromtimestamp(log_file.stat().st_mtime)

                    if file_age.days > max_age_days:
                        file_size = log_file.stat().st_size
                        log_file.unlink()
                        files_removed += 1
                        size_freed += file_size

            return {
                "success": True,
                "message": f"Cleanup completed: {files_removed} files removed",
                "files_removed": files_removed,
                "size_freed_mb": size_freed / (1024 * 1024),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to cleanup logs: {str(e)}",
                "timestamp": datetime.now().isoformat(),
            }


# Global logging manager instance
logging_manager = LoggingManager()


def setup_logging(name: str, **kwargs) -> Dict[str, Any]:
    """Setup logging for a module."""
    return logging_manager.setup_logger(name, **kwargs)


def log_event(logger_name: str, level: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Log an event with status return."""
    return logging_manager.log_event(logger_name, level, message, metadata)


def get_logging_stats(logger_name: Optional[str] = None) -> Dict[str, Any]:
    """Get logging statistics."""
    return logging_manager.get_log_stats(logger_name)


def cleanup_logs(max_age_days: int = 30) -> Dict[str, Any]:
    """Clean up old log files."""
    return logging_manager.cleanup_old_logs(max_age_days)
