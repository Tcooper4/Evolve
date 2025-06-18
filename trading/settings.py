"""Global settings for the trading system with agent tracking and persistence support."""

import os
import json
import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
from functools import wraps

# Default settings
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_METRIC_LOGGING = True
DEFAULT_METRICS_PATH = "logs/metrics.jsonl"
DEFAULT_SETTINGS_PATH = "config/settings.json"

# Global settings
LOG_LEVEL = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL)
METRIC_LOGGING_ENABLED = os.getenv("METRIC_LOGGING_ENABLED", str(DEFAULT_METRIC_LOGGING)).lower() == "true"
METRICS_PATH = os.getenv("METRICS_PATH", DEFAULT_METRICS_PATH)
SETTINGS_PATH = os.getenv("SETTINGS_PATH", DEFAULT_SETTINGS_PATH)
AGENT_ID = os.getenv("AGENT_ID", str(uuid.uuid4()))
SESSION_ID = os.getenv("SESSION_ID", str(uuid.uuid4()))

# Initialize logger
logger = logging.getLogger(__name__)

def log_setting_change(func):
    """Decorator to log setting changes with timestamps and reasons."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the reason from kwargs or use default
        reason = kwargs.pop('reason', 'No reason provided')
        
        # Get the old value before change
        old_value = globals()[func.__name__.split('_')[1].upper()]
        
        # Call the original function
        result = func(*args, **kwargs)
        
        # Get the new value after change
        new_value = globals()[func.__name__.split('_')[1].upper()]
        
        # Log the change
        logger.info(
            f"Setting changed: {func.__name__.split('_')[1]} "
            f"from {old_value} to {new_value} "
            f"by agent {AGENT_ID} "
            f"in session {SESSION_ID} "
            f"at {datetime.now().isoformat()} "
            f"Reason: {reason}"
        )
        
        # Save settings to disk
        save_settings()
        
        return result
    return wrapper

def load_settings() -> None:
    """Load settings from disk if they exist."""
    try:
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, 'r') as f:
                settings = json.load(f)
                
            # Update global settings
            global LOG_LEVEL, METRIC_LOGGING_ENABLED, METRICS_PATH, AGENT_ID, SESSION_ID
            LOG_LEVEL = settings.get('LOG_LEVEL', LOG_LEVEL)
            METRIC_LOGGING_ENABLED = settings.get('METRIC_LOGGING_ENABLED', METRIC_LOGGING_ENABLED)
            METRICS_PATH = settings.get('METRICS_PATH', METRICS_PATH)
            AGENT_ID = settings.get('AGENT_ID', AGENT_ID)
            SESSION_ID = settings.get('SESSION_ID', SESSION_ID)
            
            logger.info(f"Settings loaded from {SETTINGS_PATH}")
    except Exception as e:
        logger.error(f"Error loading settings: {str(e)}")

def save_settings() -> None:
    """Save current settings to disk."""
    try:
        # Ensure directory exists
        Path(SETTINGS_PATH).parent.mkdir(parents=True, exist_ok=True)
        
        # Get current settings
        settings = get_settings()
        
        # Add metadata
        settings.update({
            'last_modified': datetime.now().isoformat(),
            'modified_by_agent': AGENT_ID,
            'session_id': SESSION_ID
        })
        
        # Save to disk
        with open(SETTINGS_PATH, 'w') as f:
            json.dump(settings, f, indent=4)
            
        logger.info(f"Settings saved to {SETTINGS_PATH}")
    except Exception as e:
        logger.error(f"Error saving settings: {str(e)}")

@log_setting_change
def set_log_level(level: str, reason: str = "No reason provided") -> None:
    """Set the global log level.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        reason: Reason for the change
    """
    global LOG_LEVEL
    LOG_LEVEL = level.upper()
    
    # Update root logger
    logging.getLogger().setLevel(getattr(logging, LOG_LEVEL, "INFO"))
    
    # Update all existing loggers
    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, LOG_LEVEL, "INFO"))

@log_setting_change
def set_metric_logging(enabled: bool, reason: str = "No reason provided") -> None:
    """Enable or disable metric logging.
    
    Args:
        enabled: Whether to enable metric logging
        reason: Reason for the change
    """
    global METRIC_LOGGING_ENABLED
    METRIC_LOGGING_ENABLED = enabled

@log_setting_change
def set_metrics_path(path: str, reason: str = "No reason provided") -> None:
    """Set the path for metrics logging.
    
    Args:
        path: Path to metrics file
        reason: Reason for the change
    """
    global METRICS_PATH
    METRICS_PATH = path
    
    # Ensure directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)

@log_setting_change
def set_agent_id(agent_id: str, reason: str = "No reason provided") -> None:
    """Set the current agent ID.
    
    Args:
        agent_id: Unique identifier for the agent
        reason: Reason for the change
    """
    global AGENT_ID
    AGENT_ID = agent_id

@log_setting_change
def set_session_id(session_id: str, reason: str = "No reason provided") -> None:
    """Set the current session ID.
    
    Args:
        session_id: Unique identifier for the session
        reason: Reason for the change
    """
    global SESSION_ID
    SESSION_ID = session_id

def get_settings() -> Dict[str, Any]:
    """Get current settings with metadata.
    
    Returns:
        Dictionary of current settings with metadata
    """
    return {
        "LOG_LEVEL": LOG_LEVEL,
        "METRIC_LOGGING_ENABLED": METRIC_LOGGING_ENABLED,
        "METRICS_PATH": METRICS_PATH,
        "AGENT_ID": AGENT_ID,
        "SESSION_ID": SESSION_ID,
        "last_modified": datetime.now().isoformat()
    }

# Load settings at module import
load_settings() 