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
from typing import Dict, Optional, Union, Any, List
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
        
        return {
            'success': True,
            'message': 'Log Manager initialized successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    def _configure_root_logger(self) -> None:
        """Configure the root logger with basic settings."""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        return {
            'success': True,
            'message': 'Root logger configured successfully',
            'timestamp': datetime.now().isoformat()
        }
    
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
            return {'success': True, 'result': self.setup_logger(
                name,
                level or logging.INFO,
                log_to_file if log_to_file is not None else True,
                log_to_console if log_to_console is not None else True,
                structured if structured is not None else True
            )}
        
        return logger
    
    def set_level(self, name: str, level: int) -> Dict[str, Any]:
        """Set the logging level for a logger.
        
        Args:
            name: Name of the logger
            level: Logging level
            
        Returns:
            Dictionary with status and message
        """
        try:
            logger = logging.getLogger(name)
            logger.setLevel(level)
            return {"status": "success", "message": f"Log level set to {level} for {name}"}
        except Exception as e:
            return {'success': True, 'result': {"status": "error", "message": str(e), "logger_name": name}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def add_handler(
        self,
        name: str,
        handler: logging.Handler,
        formatter: Optional[logging.Formatter] = None
    ) -> Dict[str, Any]:
        """Add a handler to a logger.
        
        Args:
            name: Name of the logger
            handler: Handler to add
            formatter: Optional formatter for the handler
            
        Returns:
            Dictionary with status and message
        """
        try:
            logger = logging.getLogger(name)
            
            if formatter:
                handler.setFormatter(formatter)
            
            logger.addHandler(handler)
            
            return {'success': True, 'message': f'Handler added to logger {name}', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def remove_handler(
        self,
        name: str,
        handler: logging.Handler
    ) -> Dict[str, Any]:
        """Remove a handler from a logger.
        
        Args:
            name: Name of the logger
            handler: Handler to remove
            
        Returns:
            Dictionary with status and message
        """
        try:
            logger = logging.getLogger(name)
            logger.removeHandler(handler)
            
            return {'success': True, 'message': f'Handler removed from logger {name}', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def clear_handlers(self, name: str) -> Dict[str, Any]:
        """Clear all handlers from a logger.
        
        Args:
            name: Name of the logger
            
        Returns:
            Dictionary with status and message
        """
        try:
            logger = logging.getLogger(name)
            
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            
            return {'success': True, 'message': f'All handlers cleared from logger {name}', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

    def rotate_logs(self, name: str) -> Dict[str, Any]:
        """Manually rotate logs for a logger.
        
        Args:
            name: Name of the logger
            
        Returns:
            Dictionary with status and message
        """
        try:
            logger = logging.getLogger(name)
            
            for handler in logger.handlers:
                if isinstance(handler, logging.handlers.RotatingFileHandler):
                    handler.doRollover()
            
            return {'success': True, 'message': f'Logs rotated for logger {name}', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

    def get_log_stats(self, name: str) -> Dict[str, Any]:
        """Get statistics for a logger.
        
        Args:
            name: Name of the logger
            
        Returns:
            Dictionary with log statistics
        """
        try:
            logger = logging.getLogger(name)
            
            stats = {
                'name': name,
                'level': logger.level,
                'handlers_count': len(logger.handlers),
                'propagate': logger.propagate,
                'disabled': logger.disabled
            }
            
            # Get file sizes if file handlers exist
            file_sizes = {}
            for handler in logger.handlers:
                if isinstance(handler, logging.handlers.RotatingFileHandler):
                    log_file = Path(handler.baseFilename)
                    if log_file.exists():
                        file_sizes[log_file.name] = log_file.stat().st_size
            
            stats['file_sizes'] = file_sizes
            
            return {'success': True, 'result': stats, 'message': 'Log statistics retrieved', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

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
    try:
        # Create global log manager if not exists
        if not hasattr(get_logger, '_log_manager'):
            get_logger._log_manager = LogManager()
        
        logger = get_logger._log_manager.get_logger(
            name,
            level,
            log_to_file,
            log_to_console,
            structured
        )
        
        return {'success': True, 'result': logger, 'message': 'Logger retrieved successfully', 'timestamp': datetime.now().isoformat()}
        
    except Exception as e:
        return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

def log_config(logger: logging.Logger, config: Dict[str, Any]) -> Dict[str, Any]:
    """Log configuration information.
    
    Args:
        logger: Logger to use
        config: Configuration to log
        
    Returns:
        Dictionary with status and message
    """
    try:
        logger.info("Configuration loaded", extra={
            'config_keys': list(config.keys()),
            'config_size': len(str(config))
        })
        
        # Log sensitive information separately
        sensitive_keys = ['api_key', 'api_secret', 'password', 'token']
        for key in sensitive_keys:
            if key in config:
                logger.info(f"Configuration contains {key}", extra={
                    'key': key,
                    'value_length': len(str(config[key])) if config[key] else 0
                })
        
        return {'success': True, 'message': 'Configuration logged successfully', 'timestamp': datetime.now().isoformat()}
        
    except Exception as e:
        return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

def save_log_config(config: Dict[str, Any], log_dir: str, name: str) -> Dict[str, Any]:
    """Save logging configuration to file.
    
    Args:
        config: Logging configuration
        log_dir: Directory to save config
        name: Name of the config file
        
    Returns:
        Dictionary with status and message
    """
    try:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        config_file = log_path / f"{name}_log_config.json"
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        return {'success': True, 'message': f'Log config saved to {config_file}', 'timestamp': datetime.now().isoformat()}
        
    except Exception as e:
        return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

def load_log_config(config_file: str) -> Dict[str, Any]:
    """Load logging configuration from file.
    
    Args:
        config_file: Path to config file
        
    Returns:
        Dictionary with configuration
    """
    try:
        config_path = Path(config_file)
        
        if not config_path.exists():
            return {'success': False, 'error': f'Config file not found: {config_file}', 'timestamp': datetime.now().isoformat()}
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return {'success': True, 'result': config, 'message': 'Log config loaded successfully', 'timestamp': datetime.now().isoformat()}
        
    except Exception as e:
        return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

def setup_structured_logging(
    name: str,
    log_dir: str = "logs",
    level: int = logging.INFO
) -> Dict[str, Any]:
    """Set up structured logging for a component.
    
    Args:
        name: Name of the component
        log_dir: Directory for log files
        level: Logging level
        
    Returns:
        Dictionary with logger and status
    """
    try:
        log_manager = LogManager(log_dir)
        logger = log_manager.setup_logger(
            name,
            level=level,
            log_to_file=True,
            log_to_console=True,
            structured=True
        )
        
        return {'success': True, 'result': logger, 'message': 'Structured logging setup completed', 'timestamp': datetime.now().isoformat()}
        
    except Exception as e:
        return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

def log_performance_metrics(
    logger: logging.Logger,
    metrics: Dict[str, Any],
    component: str = "unknown"
) -> Dict[str, Any]:
    """Log performance metrics in structured format.
    
    Args:
        logger: Logger to use
        metrics: Performance metrics
        component: Component name
        
    Returns:
        Dictionary with status and message
    """
    try:
        logger.info("Performance metrics", extra={
            'component': component,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
        
        return {'success': True, 'message': 'Performance metrics logged successfully', 'timestamp': datetime.now().isoformat()}
        
    except Exception as e:
        return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()}

def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Log error with additional context.
    
    Args:
        logger: Logger to use
        error: Exception to log
        context: Additional context
        
    Returns:
        Dictionary with status and message
    """
    try:
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {}
        }
        
        logger.error("Error occurred", extra=error_data, exc_info=True)
        
        return {'success': True, 'message': 'Error logged with context', 'timestamp': datetime.now().isoformat()}
        
    except Exception as e:
        return {'success': False, 'error': str(e), 'timestamp': datetime.now().isoformat()} 