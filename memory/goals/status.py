# -*- coding: utf-8 -*-
"""Goal status tracking and management."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Constants
GOALS_DIR = Path("memory/goals")
STATUS_FILE = GOALS_DIR / "status.json"

def ensure_goals_directory():
    """Ensure the goals directory exists."""
    GOALS_DIR.mkdir(parents=True, exist_ok=True)

def load_goals() -> Dict[str, Any]:
    """
    Load current goal status from JSON file.
    
    Returns:
        Dictionary containing goal status and metrics
    """
    try:
        if not STATUS_FILE.exists():
            return {
                "status": "No Data",
                "message": "Goal status file not found",
                "timestamp": datetime.now().isoformat()
            }
        
        with open(STATUS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    except Exception as e:
        error_msg = f"Error loading goals: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "Error",
            "message": error_msg,
            "timestamp": datetime.now().isoformat()
        }

def save_goals(status: Dict[str, Any]) -> None:
    """
    Save goal status to JSON file.
    
    Args:
        status: Dictionary containing goal status and metrics
    """
    try:
        ensure_goals_directory()
        
        # Add timestamp if not present
        if "timestamp" not in status:
            status["timestamp"] = datetime.now().isoformat()
        
        with open(STATUS_FILE, 'w', encoding='utf-8') as f:
            json.dump(status, f, indent=4)
            
        logger.info("Goal status saved successfully")
        
    except Exception as e:
        error_msg = f"Error saving goals: {str(e)}"
        logger.error(error_msg)
        raise

def clear_goals() -> None:
    """Clear the goal status file."""
    if STATUS_FILE.exists():
        STATUS_FILE.unlink()
        logger.info("Goal status cleared") 