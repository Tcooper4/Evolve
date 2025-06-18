"""Logging utilities with structured logging and rotation support.

This module provides utilities for setting up and managing logging with
structured output, log rotation, and different log levels for different
components of the system.
"""

import logging
import logging.handlers
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Union
from datetime import datetime
import os

class StructuredFormatter(logging.Formatter):
    """Formatter for structured JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON formatted log string
        """
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)

class LogManager:
    """Manager for logging configuration and rotation."""
    
    def __init__(
        self,
        log_dir: str = "logs",
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        """Initialize the log manager.
        
        Args:
            log_dir: Directory for log files
            max_bytes: Maximum size of log file before rotation
            backup_count: Number of backup files to keep
        """
        self.log_dir = Path(log_dir)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        self._configure_root_logger()
    
    def _configure_root_logger(self) -> None:
        """Configure the root logger with basic settings."""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    def setup_logger(
        self,
        name: str,
        level: int = logging.INFO,
        log_to_file: bool = True,
        log_to_console: bool = True,
        structured: bool = True
    ) -> logging.Logger:
        """Set up a logger with specified configuration.
        
        Args:
            name: Name of the logger
            level: Logging level
            log_to_file: Whether to log to file
            log_to_console: Whether to log to console
            structured: Whether to use structured logging
            
        Returns:
            Configured logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create formatters
        if structured:
            file_formatter = StructuredFormatter()
            console_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
        
        # Add file handler
        if log_to_file:
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / f"{name}.log",
                maxBytes=self.max_bytes,
                backupCount=self.backup_count
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        # Add console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def get_logger(
        self,
        name: str,
        level: Optional[int] = None,
        log_to_file: Optional[bool] = None,
        log_to_console: Optional[bool] = None,
        structured: Optional[bool] = None
    ) -> logging.Logger:
        """Get a logger with specified configuration.
        
        Args:
            name: Name of the logger
            level: Logging level
            log_to_file: Whether to log to file
            log_to_console: Whether to log to console
            structured: Whether to use structured logging
            
        Returns:
            Configured logger
        """
        logger = logging.getLogger(name)
        
        # Only configure if not already configured
        if not logger.handlers:
            return self.setup_logger(
                name,
                level or logging.INFO,
                log_to_file if log_to_file is not None else True,
                log_to_console if log_to_console is not None else True,
                structured if structured is not None else True
            )
        
        return logger
    
    def set_level(self, name: str, level: int) -> None:
        """Set the logging level for a logger.
        
        Args:
            name: Name of the logger
            level: Logging level
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
    
    def add_handler(
        self,
        name: str,
        handler: logging.Handler,
        formatter: Optional[logging.Formatter] = None
    ) -> None:
        """Add a handler to a logger.
        
        Args:
            name: Name of the logger
            handler: Handler to add
            formatter: Formatter for the handler
        """
        logger = logging.getLogger(name)
        
        if formatter:
            handler.setFormatter(formatter)
        
        logger.addHandler(handler)
    
    def remove_handler(self, name: str, handler: logging.Handler) -> None:
        """Remove a handler from a logger.
        
        Args:
            name: Name of the logger
            handler: Handler to remove
        """
        logger = logging.getLogger(name)
        logger.removeHandler(handler)
    
    def clear_handlers(self, name: str) -> None:
        """Clear all handlers from a logger.
        
        Args:
            name: Name of the logger
        """
        logger = logging.getLogger(name)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

# Create singleton instance
log_manager = LogManager()

def get_logger(
    name: str,
    level: Optional[int] = None,
    log_to_file: Optional[bool] = None,
    log_to_console: Optional[bool] = None,
    structured: Optional[bool] = None
) -> logging.Logger:
    """Get a logger with specified configuration.
    
    Args:
        name: Name of the logger
        level: Logging level
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        structured: Whether to use structured logging
        
    Returns:
        Configured logger
    """
    return log_manager.get_logger(
        name,
        level,
        log_to_file,
        log_to_console,
        structured
    )

def log_config(logger: logging.Logger, config: Dict[str, Any]) -> None:
    """Log configuration dictionary.
    
    Args:
        logger: Logger instance
        config: Configuration dictionary
    """
    logger.info("Configuration:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"{key}:")
            for subkey, subvalue in value.items():
                logger.info(f"  {subkey}: {subvalue}")
        else:
            logger.info(f"{key}: {value}")

def save_log_config(config: Dict[str, Any], log_dir: str, name: str) -> None:
    """Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        log_dir: Directory to store config file
        name: Config file name
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_file = log_path / f"{name}_{timestamp}.json"
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)

def load_log_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file.
    
    Args:
        config_file: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config 