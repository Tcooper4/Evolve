"""Centralized error logging system for the trading platform."""

import logging
import traceback
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import sys
import os

# Configure logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

class ErrorLogger:
    """Centralized error logging with file and memory storage."""
    
    def __init__(self):
        """Initialize the error logger."""
        self.logger = logging.getLogger("trading_errors")
        self.logger.setLevel(logging.ERROR)
        
        # Create file handler
        log_file = LOG_DIR / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.ERROR)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        
        # Store last error
        self.last_error = None
        self.error_count = 0
    
    def log_error(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        traceback_str: Optional[str] = None

    ) -> None:
        """Log an error with context and traceback.
        
        Args:
            message: Error message
            context: Additional context information
            traceback_str: Optional traceback string
        """
        # Get traceback if not provided
        if not traceback_str:
            traceback_str = traceback.format_exc()
        
        # Prepare error data
        error_data = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'context': context or {},
            'traceback': traceback_str,
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform,
                'cwd': os.getcwd()
            }
        }
        
        # Log to file
        self.logger.error(
            f"Error: {message}\n"
            f"Context: {json.dumps(context or {})}\n"
            f"Traceback: {traceback_str}"
        )
        
        # Update last error
        self.last_error = error_data
        self.error_count += 1
        
        # Save to JSON file for UI access
        error_file = LOG_DIR / "last_error.json"
        with open(error_file, 'w') as f:
            json.dump(error_data, f, indent=2)
    
    def get_last_error(self) -> Optional[Dict[str, Any]]:
        """Get the last logged error."""
        return self.last_error
    
    def get_error_count(self) -> int:
        """Get total number of errors logged."""
        return self.error_count
    
    def clear_errors(self) -> None:
        """Clear error history."""
        self.last_error = None
        self.error_count = 0
        
        # Clear JSON file
        error_file = LOG_DIR / "last_error.json"
        if error_file.exists():
            error_file.unlink()

# Create singleton instance
error_logger = ErrorLogger() 