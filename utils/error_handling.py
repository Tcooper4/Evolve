"""
Unified error handling system for EVOLVE trading system

Provides centralized error handling with categorization, severity levels,
and callback support.
"""

import logging
import traceback
from typing import Optional, Callable, Dict, Any, List
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories"""
    DATA = "data"
    BROKER = "broker"
    MODEL = "model"
    STRATEGY = "strategy"
    RISK = "risk"
    SYSTEM = "system"
    VALIDATION = "validation"
    NETWORK = "network"
    CONFIG = "config"


class TradingError(Exception):
    """Base exception for all trading errors"""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory,
        severity: ErrorSeverity,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.timestamp = datetime.now()
        self.traceback_str = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary"""
        return {
            'message': self.message,
            'category': self.category.value,
            'severity': self.severity.value,
            'context': self.context,
            'timestamp': self.timestamp.isoformat(),
            'traceback': self.traceback_str or traceback.format_exc(),
            'type': self.__class__.__name__,
        }
    
    def __str__(self) -> str:
        return f"[{self.category.value.upper()}] {self.severity.value.upper()}: {self.message}"


class DataError(TradingError):
    """Data-related errors"""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, ErrorCategory.DATA, severity, context)


class BrokerError(TradingError):
    """Broker-related errors"""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, ErrorCategory.BROKER, severity, context)


class ModelError(TradingError):
    """Model-related errors"""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, ErrorCategory.MODEL, severity, context)


class StrategyError(TradingError):
    """Strategy-related errors"""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, ErrorCategory.STRATEGY, severity, context)


class RiskError(TradingError):
    """Risk management errors"""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, ErrorCategory.RISK, severity, context)


class ValidationError(TradingError):
    """Validation errors"""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.LOW,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, ErrorCategory.VALIDATION, severity, context)


class ErrorHandler:
    """Centralized error handling"""
    
    def __init__(self):
        self.error_log: List[Dict[str, Any]] = []
        self.error_callbacks: List[Callable] = []
        self.max_log_size = 1000
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        default_category: ErrorCategory = ErrorCategory.SYSTEM,
        default_severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    ) -> None:
        """
        Handle an error with unified logging.
        
        Args:
            error: Exception to handle
            context: Additional context information
            default_category: Default category if error is not TradingError
            default_severity: Default severity if error is not TradingError
        """
        if not isinstance(error, TradingError):
            error = TradingError(
                message=str(error),
                category=default_category,
                severity=default_severity,
                context=context or {},
            )
            error.traceback_str = traceback.format_exc()
        
        error_dict = error.to_dict()
        
        # Log based on severity
        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL [{error.category.value}]: {error.message}", extra=error_dict)
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH [{error.category.value}]: {error.message}", extra=error_dict)
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"MEDIUM [{error.category.value}]: {error.message}", extra=error_dict)
        else:
            logger.info(f"LOW [{error.category.value}]: {error.message}", extra=error_dict)
        
        # Add to error log
        self.error_log.append(error_dict)
        
        # Keep log size manageable
        if len(self.error_log) > self.max_log_size:
            self.error_log = self.error_log[-self.max_log_size:]
        
        # Trigger callbacks
        for callback in self.error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")
    
    def register_callback(self, callback: Callable) -> None:
        """
        Register a callback to be called when errors occur.
        
        Args:
            callback: Callable that takes an error as argument
        """
        self.error_callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable) -> None:
        """Unregister a callback"""
        if callback in self.error_callbacks:
            self.error_callbacks.remove(callback)
    
    def get_recent_errors(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent errors.
        
        Args:
            count: Number of recent errors to return
            
        Returns:
            List of error dictionaries
        """
        return self.error_log[-count:]
    
    def get_errors_by_category(
        self, category: ErrorCategory, count: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent errors by category.
        
        Args:
            category: Error category to filter by
            count: Maximum number of errors to return
            
        Returns:
            List of error dictionaries
        """
        filtered = [e for e in self.error_log if e.get('category') == category.value]
        return filtered[-count:]
    
    def get_errors_by_severity(
        self, severity: ErrorSeverity, count: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent errors by severity.
        
        Args:
            severity: Error severity to filter by
            count: Maximum number of errors to return
            
        Returns:
            List of error dictionaries
        """
        filtered = [e for e in self.error_log if e.get('severity') == severity.value]
        return filtered[-count:]
    
    def clear_log(self) -> None:
        """Clear the error log"""
        self.error_log.clear()
        logger.info("Error log cleared")


# Global error handler instance
_error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance"""
    return _error_handler


def handle_errors(
    category: ErrorCategory = ErrorCategory.SYSTEM,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    reraise: bool = False,
):
    """
    Decorator for automatic error handling.
    
    Args:
        category: Error category for unhandled exceptions
        severity: Error severity for unhandled exceptions
        reraise: Whether to re-raise the exception after handling
    
    Example:
        @handle_errors(category=ErrorCategory.DATA, severity=ErrorSeverity.LOW)
        def fetch_data(symbol):
            # Errors automatically handled
            pass
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handler = get_error_handler()
                context = {
                    'function': func.__name__,
                    'module': func.__module__,
                    'args': str(args)[:100],  # Limit length
                    'kwargs': str(kwargs)[:100],  # Limit length
                }
                handler.handle_error(e, context, category, severity)
                if reraise:
                    raise
                return None
        
        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__module__ = func.__module__
        
        return wrapper
    return decorator

