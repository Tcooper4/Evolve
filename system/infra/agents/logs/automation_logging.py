"""
Enhanced Logging System.

This module provides a structured logging system using structlog for the
agentic forecasting platform, including:
- Structured logging with context
- Log aggregation
- Performance monitoring
- Error tracking
"""

import structlog
import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict, Optional
import traceback
from pathlib import Path

# Configure structlog
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

class AutomationLogger:
    def __init__(
        self,
        name: str,
        log_dir: str = "logs",
        log_level: str = "INFO",
        enable_console: bool = True,
        enable_file: bool = True
    ):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_level = getattr(logging, log_level.upper())
        # Create logger
        self.logger = structlog.get_logger(name)
        # Configure handlers
        self._setup_handlers(enable_console, enable_file)

    def _setup_handlers(self, enable_console: bool, enable_file: bool) -> None:
        """Set up logging handlers."""
        # Create log directory if it doesn't exist
        if enable_file:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        # Remove existing handlers
        for handler in getattr(self.logger, 'handlers', [])[:]:
            self.logger.removeHandler(handler)
        # Add console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            self.logger.addHandler(console_handler)
        # Add file handler
        if enable_file:
            log_file = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(self.log_level)
            self.logger.addHandler(file_handler)

    def _format_context(self, **kwargs) -> Dict[str, Any]:
        """Format context for logging."""
        context = {
            "timestamp": datetime.now().isoformat(),
            "logger": self.name,
            **kwargs
        }
        return context

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(message, **self._format_context(**kwargs))

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.logger.info(message, **self._format_context(**kwargs))

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(message, **self._format_context(**kwargs))

    def error(self, message: str, exc_info: Optional[Exception] = None, **kwargs) -> None:
        """Log error message."""
        context = self._format_context(**kwargs)
        if exc_info:
            context["exception"] = {
                "type": type(exc_info).__name__,
                "message": str(exc_info),
                "traceback": traceback.format_exc()
            }
        self.logger.error(message, **context)

    def critical(self, message: str, exc_info: Optional[Exception] = None, **kwargs) -> None:
        """Log critical message."""
        context = self._format_context(**kwargs)
        if exc_info:
            context["exception"] = {
                "type": type(exc_info).__name__,
                "message": str(exc_info),
                "traceback": traceback.format_exc()
            }
        self.logger.critical(message, **context)

    def performance(self, operation: str, duration_ms: float, **kwargs) -> None:
        """Log performance metrics."""
        self.logger.info(
            f"Performance: {operation}",
            **self._format_context(
                operation=operation,
                duration_ms=duration_ms,
                **kwargs
            )
        )

    def security(self, event: str, **kwargs) -> None:
        """Log security events."""
        self.logger.warning(
            f"Security Event: {event}",
            **self._format_context(
                event_type="security",
                event=event,
                **kwargs
            )
        )

    def audit(self, action: str, user: str, **kwargs) -> None:
        """Log audit events."""
        self.logger.info(
            f"Audit: {action} by {user}",
            **self._format_context(
                event_type="audit",
                action=action,
                user=user,
                **kwargs
            )
        )

    def set_level(self, level: str) -> None:
        """Set logging level."""
        self.log_level = getattr(logging, level.upper())
        for handler in getattr(self.logger, 'handlers', []):
            handler.setLevel(self.log_level)

    def get_logger(self) -> structlog.BoundLogger:
        """Get the underlying logger."""
        return self.logger