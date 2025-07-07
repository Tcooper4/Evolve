"""
Global Exception Handler for Trading System

This module provides comprehensive error handling and logging for the entire
trading system, capturing unhandled errors and providing recovery mechanisms.
"""

import sys
import traceback
import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
from functools import wraps
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

class GlobalErrorHandler:
    """Global error handler for capturing and logging all unhandled errors."""
    
    def __init__(self, log_file: str = "logs/global_errors.log"):
        """Initialize the global error handler.
        
        Args:
            log_file: Path to error log file
        """
        self.log_file = log_file
        self.error_count = 0
        self.recovery_attempts = {}
        self.error_history = []
        self.max_history_size = 1000
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Register global exception handlers
        self._register_global_handlers()
        
        logger.info("Global error handler initialized")
    
    def _setup_logging(self):
        """Setup error logging configuration."""
        # Create file handler for errors
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.ERROR)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to root logger
        logging.getLogger().addHandler(file_handler)
    
    def _register_global_handlers(self):
        """Register global exception handlers."""
        # Store original exception handlers
        self.original_excepthook = sys.excepthook
        self.original_thread_excepthook = threading.excepthook
        
        # Register custom handlers
        sys.excepthook = self._handle_uncaught_exception
        threading.excepthook = self._handle_thread_exception
    
    def _handle_uncaught_exception(self, exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions."""
        try:
            # Log the error
            error_info = {
                'timestamp': datetime.now().isoformat(),
                'type': exc_type.__name__,
                'message': str(exc_value),
                'traceback': traceback.format_tb(exc_traceback),
                'thread': threading.current_thread().name,
                'error_count': self.error_count + 1
            }
            
            self._log_error(error_info)
            self.error_count += 1
            
            # Attempt recovery
            self._attempt_recovery(error_info)
            
            # Call original handler
            self.original_excepthook(exc_type, exc_value, exc_traceback)
            
        except Exception as e:
            # Fallback to original handler if our handler fails
            logger.error(f"Error in global exception handler: {e}")
            self.original_excepthook(exc_type, exc_value, exc_traceback)
    
    def _handle_thread_exception(self, args):
        """Handle thread exceptions."""
        try:
            error_info = {
                'timestamp': datetime.now().isoformat(),
                'type': 'ThreadException',
                'message': str(args.exc_value),
                'traceback': traceback.format_tb(args.exc_traceback),
                'thread': args.thread.name,
                'error_count': self.error_count + 1
            }
            
            self._log_error(error_info)
            self.error_count += 1
            
            # Attempt recovery
            self._attempt_recovery(error_info)
            
        except Exception as e:
            logger.error(f"Error in thread exception handler: {e}")
    
    def _log_error(self, error_info: Dict[str, Any]):
        """Log error information."""
        try:
            # Add to history
            self.error_history.append(error_info)
            
            # Keep history size manageable
            if len(self.error_history) > self.max_history_size:
                self.error_history = self.error_history[-self.max_history_size:]
            
            # Log to file
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(error_info, indent=2) + '\n')
            
            # Log to console
            logger.error(f"Global error #{error_info['error_count']}: {error_info['type']} - {error_info['message']}")
            
        except Exception as e:
            logger.error(f"Error logging error: {e}")
    
    def _attempt_recovery(self, error_info: Dict[str, Any]):
        """Attempt to recover from error."""
        try:
            error_type = error_info['type']
            
            # Check if we have recovery attempts for this error type
            if error_type in self.recovery_attempts:
                recovery_func = self.recovery_attempts[error_type]
                recovery_func(error_info)
            else:
                # Default recovery actions
                self._default_recovery(error_info)
                
        except Exception as e:
            logger.error(f"Error in recovery attempt: {e}")
    
    def _default_recovery(self, error_info: Dict[str, Any]):
        """Default recovery actions."""
        try:
            # Log recovery attempt
            logger.info(f"Attempting default recovery for error #{error_info['error_count']}")
            
            # Clear any temporary files
            self._cleanup_temp_files()
            
            # Reset any corrupted state
            self._reset_corrupted_state()
            
            # Send alert if too many errors
            if self.error_count > 10:
                self._send_error_alert(error_info)
                
        except Exception as e:
            logger.error(f"Error in default recovery: {e}")
    
    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        try:
            temp_dirs = ['temp', 'cache', 'logs/temp']
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    for file in os.listdir(temp_dir):
                        file_path = os.path.join(temp_dir, file)
                        if os.path.isfile(file_path):
                            # Remove files older than 1 hour
                            if os.path.getmtime(file_path) < datetime.now().timestamp() - 3600:
                                os.remove(file_path)
                                
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {e}")
    
    def _reset_corrupted_state(self):
        """Reset any corrupted system state."""
        try:
            # Reset session state if corrupted
            if hasattr(sys, 'session_state'):
                corrupted_keys = []
                for key in sys.session_state:
                    try:
                        _ = sys.session_state[key]
                    except (AttributeError, TypeError, KeyError) as e:
                        corrupted_keys.append(key)
                        logger.debug(f"Detected corrupted session state key {key}: {e}")
                
                for key in corrupted_keys:
                    del sys.session_state[key]
                    logger.info(f"Removed corrupted session state key: {key}")
                    
        except Exception as e:
            logger.error(f"Error resetting corrupted state: {e}")
    
    def _send_error_alert(self, error_info: Dict[str, Any]):
        """Send error alert."""
        try:
            alert_message = f"High error count detected: {self.error_count} errors"
            logger.warning(alert_message)
            
            # Could integrate with external alerting systems here
            # For now, just log the alert
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    def register_recovery_handler(self, error_type: str, recovery_func: Callable):
        """Register a custom recovery handler for a specific error type.
        
        Args:
            error_type: Type of error to handle
            recovery_func: Function to call for recovery
        """
        self.recovery_attempts[error_type] = recovery_func
        logger.info(f"Registered recovery handler for error type: {error_type}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors."""
        try:
            if not self.error_history:
                return {'message': 'No errors recorded'}
            
            # Count error types
            error_types = {}
            for error in self.error_history:
                error_type = error['type']
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            # Get recent errors
            recent_errors = self.error_history[-10:] if len(self.error_history) >= 10 else self.error_history
            
            return {
                'total_errors': self.error_count,
                'error_types': error_types,
                'recent_errors': recent_errors,
                'log_file': self.log_file
            }
            
        except Exception as e:
            logger.error(f"Error getting error summary: {e}")
            return {'error': str(e)}
    
    def clear_error_history(self):
        """Clear error history."""
        self.error_history = []
        self.error_count = 0
        logger.info("Error history cleared")
    
    def handle_exceptions(self, func: Callable) -> Callable:
        """Decorator to handle exceptions in functions.
        
        Args:
            func: Function to wrap
            
        Returns:
            Wrapped function with exception handling
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_info = {
                    'timestamp': datetime.now().isoformat(),
                    'type': type(e).__name__,
                    'message': str(e),
                    'traceback': traceback.format_tb(sys.exc_info()[2]),
                    'function': func.__name__,
                    'error_count': self.error_count + 1
                }
                
                self._log_error(error_info)
                self.error_count += 1
                
                # Attempt recovery
                self._attempt_recovery(error_info)
                
                # Re-raise or return error response
                raise
                
        return wrapper
    
    def safe_execute(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Safely execute a function with error handling.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Dictionary with result or error information
        """
        try:
            result = func(*args, **kwargs)
            return {
                'success': True,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            error_info = {
                'timestamp': datetime.now().isoformat(),
                'type': type(e).__name__,
                'message': str(e),
                'traceback': traceback.format_tb(sys.exc_info()[2]),
                'function': func.__name__,
                'error_count': self.error_count + 1
            }
            
            self._log_error(error_info)
            self.error_count += 1
            
            # Attempt recovery
            self._attempt_recovery(error_info)
            
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'timestamp': datetime.now().isoformat()
            }

# Global instance
global_error_handler = GlobalErrorHandler()

def handle_exceptions(func: Callable) -> Callable:
    """Global decorator for exception handling."""
    return global_error_handler.handle_exceptions(func)

def safe_execute(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """Global safe execution function."""
    return global_error_handler.safe_execute(func, *args, **kwargs)

def get_error_summary() -> Dict[str, Any]:
    """Get global error summary."""
    return global_error_handler.get_error_summary()

def register_recovery_handler(error_type: str, recovery_func: Callable):
    """Register global recovery handler."""
    global_error_handler.register_recovery_handler(error_type, recovery_func) 