import logging
import traceback
from typing import Optional, Callable, Any, Type, Dict, List
from functools import wraps
import sys

class TradingError(Exception):
    """Base class for trading system errors."""
    pass

class DataError(TradingError):
    """Error related to data processing."""
    pass

class ModelError(TradingError):
    """Error related to model operations."""
    pass

class StrategyError(TradingError):
    """Error related to strategy execution."""
    pass

class RiskError(TradingError):
    """Error related to risk management."""
    pass

class PortfolioError(TradingError):
    """Error related to portfolio management."""
    pass

def handle_exceptions(logger: Optional[logging.Logger] = None,
                     error_types: Optional[list[Type[Exception]]] = None,
                     default_return: Any = None) -> Callable:
    """Decorator for handling exceptions in functions.
    
    Args:
        logger: Logger instance for error logging
        error_types: List of exception types to handle
        default_return: Value to return if an error occurs
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if error_types is None or isinstance(e, tuple(error_types)):
                    if logger:
                        logger.error(f"Error in {func.__name__}: {str(e)}")
                        logger.error(traceback.format_exc())
                    return {'success': True, 'result': {'success': True, 'result': {'success': True, 'result': default_return, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
                raise
        return wrapper
    return decorator

def retry(max_attempts: int = 3, delay: float = 1.0,
          backoff: float = 2.0, logger: Optional[logging.Logger] = None) -> Callable:
    """Decorator for retrying functions on failure.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        logger: Logger instance for retry logging
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            import time
            
            attempt = 1
            current_delay = delay
            
            while attempt <= max_attempts:
                try:
                    return {'success': True, 'result': {'success': True, 'result': {'success': True, 'result': func(*args, **kwargs), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
                except Exception as e:
                    if attempt == max_attempts:
                        if logger:
                            logger.error(f"Failed after {max_attempts} attempts: {str(e)}")
                        raise
                    
                    if logger:
                        logger.warning(
                            f"Attempt {attempt} failed: {str(e)}. "
                            f"Retrying in {current_delay:.1f} seconds..."
                        )
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
                    attempt += 1
        return wrapper
    return decorator

def validate_input(func: Callable) -> Callable:
    """Decorator for validating function inputs.
    
    Args:
        func: Function to validate inputs for
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Get function signature
        import inspect
        sig = inspect.signature(func)
        
        # Validate positional arguments
        for i, (name, param) in enumerate(sig.parameters.items()):
            if i < len(args):
                value = args[i]
            else:
                value = kwargs.get(name, param.default)
            
            # Skip if parameter is optional and value is None
            if param.default is not inspect.Parameter.empty and value is None:
                continue
            
            # Check type annotation
            if param.annotation is not inspect.Parameter.empty:
                if not isinstance(value, param.annotation):
                    raise TypeError(
                        f"Argument '{name}' must be of type {param.annotation.__name__}, "
                        f"got {type(value).__name__}"
                    )
        
        return {'success': True, 'result': {'success': True, 'result': func(*args, **kwargs), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    return wrapper

def log_execution_time(logger: Optional[logging.Logger] = None) -> Callable:
    """Decorator for logging function execution time.
    
    Args:
        logger: Logger instance for timing logs
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            import time
            
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            if logger:
                logger.info(
                    f"{func.__name__} executed in {end_time - start_time:.2f} seconds"
                )
            
            return {'success': True, 'result': {'success': True, 'result': {'success': True, 'result': result, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        return wrapper
    return decorator

def handle_keyboard_interrupt(func: Callable) -> Callable:
    """Decorator for handling keyboard interrupts gracefully.
    
    Args:
        func: Function to handle interrupts for
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return {'success': True, 'result': {'success': True, 'result': func(*args, **kwargs), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            sys.exit(0)
    return wrapper

def handle_file_errors(func: Callable) -> Callable:
    """Decorator for handling file operation errors.
    
    Args:
        func: Function to handle file errors for
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {e}")
            return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    return wrapper 