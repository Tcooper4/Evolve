import logging
import traceback
import time
from datetime import datetime
from functools import wraps
from typing import Callable, Optional, Type, Any, Dict, List, Union
from dataclasses import dataclass, field

from trading.exceptions import TradingSystemError, ModelError, StrategyError

logger = logging.getLogger(__name__)

# Basic error classes
class TradingError(TradingSystemError):
    """Base class for trading system errors."""

class RoutingError(TradingError):
    """Raised when routing operations fail."""
    pass

@dataclass
class ErrorContext:
    """Context information for error handling."""
    function_name: str
    context_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def add_error(self, error_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Add an error to the context."""
        self.errors.append({
            "type": error_type,
            "message": message,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "function_name": self.function_name,
            "context_data": self.context_data,
            "timestamp": self.timestamp.isoformat(),
            "errors": self.errors
        }

class ErrorRecoveryStrategy:
    """Base class for error recovery strategies."""

    @staticmethod
    def retry(max_attempts: int = 3, delay: float = 1) -> Callable:
        """Create a retry strategy."""
        def strategy(error: Exception, context: ErrorContext) -> Dict[str, Any]:
            return {
                "action": "retry",
                "max_attempts": max_attempts,
                "delay": delay,
                "success": True
            }
        return strategy

    @staticmethod
    def fallback(fallback_function: str = None) -> Callable:
        """Create a fallback strategy."""
        def strategy(error: Exception, context: ErrorContext) -> Dict[str, Any]:
            return {
                "action": "fallback",
                "fallback_function": fallback_function,
                "success": True
            }
        return strategy

    @staticmethod
    def log_and_continue() -> Callable:
        """Create a log and continue strategy."""
        def strategy(error: Exception, context: ErrorContext) -> Dict[str, Any]:
            return {
                "action": "log_and_continue",
                "success": True
            }
        return strategy

class ErrorHandler:
    """Centralized error handler with recovery strategies."""

    def __init__(self):
        self.recovery_strategies: Dict[Type[Exception], Callable] = {}
        self.error_counts: Dict[str, int] = {}

    def register_recovery_strategy(self, exception_type: Type[Exception], strategy: Callable):
        """Register a recovery strategy for an exception type."""
        self.recovery_strategies[exception_type] = strategy

    def handle_error(self, error: Exception, context: ErrorContext) -> Dict[str, Any]:
        """Handle an error using registered strategies."""
        error_type = type(error)

        # Update error count
        error_name = error_type.__name__
        self.error_counts[error_name] = self.error_counts.get(error_name, 0) + 1
        # Add error to context
        context.add_error(error_type.__name__, str(error))

        # Check for registered strategy
        for exception_type, strategy in self.recovery_strategies.items():
            if isinstance(error, exception_type):
                return strategy(error, context)

        # Default strategy: log and fail
        return {
            "action": "log_and_fail",
            "success": False,
            "error_type": error_type.__name__,
            "message": str(error)
        }

# Global error handler instance
error_handler = ErrorHandler()

# Simple error logging decorator
def log_errors(
    logger: Optional[logging.Logger] = None,
    error_types: Optional[tuple[Type[Exception], ...]] = (Exception,)
) -> Callable:
    """Decorator to log errors in a function."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except error_types as e:
                log = logger or logging.getLogger(func.__module__)
                log.error(f"Error in {func.__name__}: {e}")
                log.error(traceback.format_exc())
                raise
        return wrapper
    return decorator

def retry_on_error(
    max_retries: int = 3,
    delay: float =1,
    retry_exceptions: tuple[Type[Exception], ...] = (Exception,),
    backoff_factor: float =10) -> Callable:
    """Decorator to retry functions on specific exceptions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retry_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        sleep_time = delay * (backoff_factor ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {sleep_time}s...")
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}: {e}")
                        raise

            raise last_exception
        return wrapper
    return decorator

def handle_routing_errors(func: Callable) -> Callable:
    """Decorator to handle routing-specific errors."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Routing error in {func.__name__}: {e}")

            # Create error context
            context = ErrorContext(
                function_name=func.__name__,
                context_data={
                    "args": str(args),
                    "kwargs": str(kwargs)
                }
            )

            # Handle with error handler
            result = error_handler.handle_error(e, context)

            # Return error response for routing functions
            return {
                "success": False,
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "context": context.to_dict(),
                    "recovery_action": result.get("action")
                }
            }

    return wrapper

def with_error_context(func: Callable) -> Callable:
    """Decorator to add error context to function calls."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        context = ErrorContext(
            function_name=func.__name__,
            context_data={
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys())
            }
        )

        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            # Add error to context
            context.add_error(type(e).__name__, str(e))

            # Handle with error handler
            recovery_result = error_handler.handle_error(e, context)

            # Log context for debugging
            logger.error(f"Error context for {func.__name__}: {context.to_dict()}")

            # Re-raise with context if needed
            if not recovery_result.get("success", False):
                raise e

        return wrapper

def safe_execute(
    func: Callable,
    default_return: Any = None,
    error_types: tuple[Type[Exception], ...] = (Exception,),
    logger: Optional[logging.Logger] = None
) -> Any:
    """Execute a function with error handling."""
    try:
        return func()
    except error_types as e:
        log = logger or logging.getLogger(func.__module__)
        log.error(f"Safe execution failed for {func.__name__}: {e}")
        return default_return

def validate_error_context(context: ErrorContext) -> bool:
    """Validate error context structure."""
    required_fields = ["function_name", "timestamp", "errors"]
    return all(hasattr(context, field) for field in required_fields)

def get_error_summary() -> Dict[str, Any]:
    """Get a summary of error statistics."""
    return {
        "error_counts": error_handler.error_counts.copy(),
        "registered_strategies": list(error_handler.recovery_strategies.keys()),
        "total_errors": sum(error_handler.error_counts.values())
    }

# Register default recovery strategies
error_handler.register_recovery_strategy(
    RoutingError,
    ErrorRecoveryStrategy.retry(max_attempts=2, delay=0.5)
)

error_handler.register_recovery_strategy(
    ModelError,
    ErrorRecoveryStrategy.fallback(fallback_function="fallback_model")
)

error_handler.register_recovery_strategy(
    StrategyError,
    ErrorRecoveryStrategy.log_and_continue()
) 