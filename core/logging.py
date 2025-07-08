"""
Core Logging System

This module provides a logging implementation that implements the ILogger interface
and integrates with the event system for comprehensive logging.
"""

import logging
import logging.handlers
from typing import Any, Dict, Optional
from datetime import datetime
import os
import json
from pathlib import Path

from .interfaces import ILogger, SystemEvent, EventType
from .events import get_event_bus

class SystemLogger(ILogger):
    """
    System logger implementation that integrates with the event system.
    
    Features:
    - Structured logging
    - Event integration
    - File and console output
    - Log rotation
    - Performance metrics
    """
    
    def __init__(self, name: str = "evolve", log_level: str = "INFO", log_dir: str = "logs"):
        """
        Initialize the system logger.
        
        Args:
            name: Logger name
            log_level: Logging level
            log_dir: Directory for log files
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        self._setup_console_handler()
        self._setup_file_handlers()
        
        # Performance tracking
        self.stats = {
            'info_count': 0,
            'warning_count': 0,
            'error_count': 0,
            'debug_count': 0,
            'event_count': 0
        }
    
    def _setup_console_handler(self) -> None:
        """Setup console logging handler."""
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
    
    def _setup_file_handlers(self) -> None:
        """Setup file logging handlers."""
        # General log file
        general_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "system.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        general_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        general_handler.setFormatter(formatter)
        
        self.logger.addHandler(general_handler)
        
        # Error log file
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "errors.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        
        self.logger.addHandler(error_handler)
        
        # Event log file
        self.event_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "events.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        self.event_handler.setLevel(logging.INFO)
        
        event_formatter = logging.Formatter(
            '%(asctime)s - EVENT - %(message)s'
        )
        self.event_handler.setFormatter(event_formatter)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.stats['info_count'] += 1
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.stats['warning_count'] += 1
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.stats['error_count'] += 1
        self.logger.error(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.stats['debug_count'] += 1
        self.logger.debug(message, extra=kwargs)
    
    def log_event(self, event: SystemEvent) -> None:
        """Log system event."""
        self.stats['event_count'] += 1
        
        # Log to event file
        event_message = self._format_event_message(event)
        self.event_handler.emit(logging.LogRecord(
            name=self.name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=event_message,
            args=(),
            exc_info=None
        ))
        
        # Also log to general log
        self.info(f"Event: {event.event_type.value} from {event.source}")
    
    def _format_event_message(self, event: SystemEvent) -> str:
        """Format event message for logging."""
        event_data = {
            'timestamp': event.timestamp.isoformat(),
            'event_type': event.event_type.value,
            'source': event.source,
            'data': event.data,
            'metadata': event.metadata
        }
        return json.dumps(event_data, default=str)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return {
            **self.stats,
            'log_files': [
                str(f) for f in self.log_dir.glob("*.log")
            ]
        }
    
    def cleanup(self) -> None:
        """Cleanup logging resources."""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

# Convenience functions
def get_logger(name: str = "evolve") -> SystemLogger:
    """Get a system logger instance."""
    return SystemLogger(name)

def log_info(message: str, **kwargs) -> None:
    """Log info message using global logger."""
    logger = get_logger()
    logger.info(message, **kwargs)

def log_warning(message: str, **kwargs) -> None:
    """Log warning message using global logger."""
    logger = get_logger()
    logger.warning(message, **kwargs)

def log_error(message: str, **kwargs) -> None:
    """Log error message using global logger."""
    logger = get_logger()
    logger.error(message, **kwargs)

def log_debug(message: str, **kwargs) -> None:
    """Log debug message using global logger."""
    logger = get_logger()
    logger.debug(message, **kwargs)

def log_event(event: SystemEvent) -> None:
    """Log event using global logger."""
    logger = get_logger()
    logger.log_event(event)

__all__ = [
    'SystemLogger', 'get_logger',
    'log_info', 'log_warning', 'log_error', 'log_debug', 'log_event'
] 