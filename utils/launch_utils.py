"""
Launch Utilities Module

This module provides common utilities for launching services and setting up logging.
Created to replace missing utils.launch_utils imports.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    service_name: Optional[str] = None,
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    enable_file_output: bool = True,
    enable_rotating_handlers: bool = True,
) -> logging.Logger:
    """
    Basic logging setup function.
    
    Args:
        service_name: Name of the service
        log_dir: Directory for log files
        log_level: Logging level
        enable_file_output: Whether to log to file
        enable_rotating_handlers: Whether to use rotating handlers
        
    Returns:
        Configured logger
    """
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Get or create logger
    logger_name = service_name or "service"
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if enabled
    if enable_file_output:
        log_file = Path(log_dir) / f"{logger_name}.log"
        if enable_rotating_handlers:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5
            )
        else:
            file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


__all__ = ["setup_logging"]
