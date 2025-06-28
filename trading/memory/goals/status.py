# -*- coding: utf-8 -*-
"""Goal status tracking and management."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Constants
GOALS_DIR = Path("memory/goals")
STATUS_FILE = GOALS_DIR / "status.json"

@dataclass
class GoalStatus:
    """Goal status data structure"""
    status: str
    message: str
    timestamp: str
    metrics: Optional[Dict[str, Any]] = None
    progress: Optional[float] = None
    target_date: Optional[str] = None
    priority: Optional[str] = None

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

def get_status_summary() -> Dict[str, Any]:
    """
    Get a summary of current goal status for UI display.
    
    Returns:
        Dictionary with status summary including progress, metrics, and recommendations
    """
    try:
        goals_data = load_goals()
        
        # Create summary
        summary = {
            "current_status": goals_data.get("status", "Unknown"),
            "last_updated": goals_data.get("timestamp", "Unknown"),
            "message": goals_data.get("message", "No status message"),
            "progress": goals_data.get("progress", 0.0),
            "metrics": goals_data.get("metrics", {}),
            "recommendations": _generate_recommendations(goals_data),
            "alerts": _check_alerts(goals_data)
        }
        
        logger.info(f"Generated goal status summary: {summary['current_status']}")
        return summary
        
    except Exception as e:
        logger.error(f"Error generating status summary: {str(e)}")
        return {
            "current_status": "Error",
            "last_updated": datetime.now().isoformat(),
            "message": f"Error generating summary: {str(e)}",
            "progress": 0.0,
            "metrics": {},
            "recommendations": [],
            "alerts": []
        }

def _generate_recommendations(goals_data: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on current goal status."""
    recommendations = []
    
    status = goals_data.get("status", "").lower()
    progress = goals_data.get("progress", 0.0)
    
    if status == "behind_schedule":
        recommendations.append("Consider increasing resources or adjusting timeline")
        recommendations.append("Review bottlenecks in current workflow")
    elif status == "on_track":
        recommendations.append("Continue current approach")
        recommendations.append("Monitor progress regularly")
    elif status == "ahead_of_schedule":
        recommendations.append("Consider adding additional objectives")
        recommendations.append("Review quality standards")
    
    if progress < 0.25:
        recommendations.append("Focus on quick wins to build momentum")
    elif progress > 0.75:
        recommendations.append("Prepare for completion and next phase planning")
    
    return recommendations

def _check_alerts(goals_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Check for alerts based on goal status."""
    alerts = []
    
    status = goals_data.get("status", "").lower()
    progress = goals_data.get("progress", 0.0)
    target_date = goals_data.get("target_date")
    
    if status == "behind_schedule":
        alerts.append({
            "type": "warning",
            "message": "Goal is behind schedule",
            "severity": "medium"
        })
    
    if progress < 0.1 and status != "not_started":
        alerts.append({
            "type": "warning",
            "message": "Very low progress detected",
            "severity": "high"
        })
    
    if target_date:
        try:
            target_dt = datetime.fromisoformat(target_date)
            days_remaining = (target_dt - datetime.now()).days
            
            if days_remaining < 7:
                alerts.append({
                    "type": "urgent",
                    "message": f"Goal deadline approaching: {days_remaining} days remaining",
                    "severity": "high"
                })
            elif days_remaining < 30:
                alerts.append({
                    "type": "warning",
                    "message": f"Goal deadline in {days_remaining} days",
                    "severity": "medium"
                })
        except ValueError:
            pass
    
    return alerts

def update_goal_progress(progress: float, metrics: Optional[Dict[str, Any]] = None, 
                        status: Optional[str] = None, message: Optional[str] = None) -> None:
    """
    Update goal progress and metrics.
    
    Args:
        progress: Progress percentage (0.0 to 1.0)
        metrics: Optional metrics dictionary
        status: Optional status update
        message: Optional status message
    """
    try:
        current_goals = load_goals()
        
        # Update progress
        current_goals["progress"] = max(0.0, min(1.0, progress))
        
        # Update metrics if provided
        if metrics:
            if "metrics" not in current_goals:
                current_goals["metrics"] = {}
            current_goals["metrics"].update(metrics)
        
        # Update status if provided
        if status:
            current_goals["status"] = status
        
        # Update message if provided
        if message:
            current_goals["message"] = message
        
        # Save updated goals
        save_goals(current_goals)
        
        logger.info(f"Updated goal progress to {progress:.1%}")
        
    except Exception as e:
        logger.error(f"Error updating goal progress: {str(e)}")
        raise

def log_agent_contribution(agent_name: str, contribution: str, impact: str = "medium") -> None:
    """
    Log agent contribution to goal progress.
    
    Args:
        agent_name: Name of the agent
        contribution: Description of the contribution
        impact: Impact level (low, medium, high)
    """
    try:
        current_goals = load_goals()
        
        # Initialize contributions if not present
        if "contributions" not in current_goals:
            current_goals["contributions"] = []
        
        # Add contribution
        contribution_entry = {
            "agent": agent_name,
            "contribution": contribution,
            "impact": impact,
            "timestamp": datetime.now().isoformat()
        }
        
        current_goals["contributions"].append(contribution_entry)
        
        # Keep only last 50 contributions
        if len(current_goals["contributions"]) > 50:
            current_goals["contributions"] = current_goals["contributions"][-50:]
        
        save_goals(current_goals)
        
        logger.info(f"Logged contribution from {agent_name}: {contribution}")
        
    except Exception as e:
        logger.error(f"Error logging agent contribution: {str(e)}")

def get_agent_contributions(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get recent agent contributions.
    
    Args:
        limit: Maximum number of contributions to return
        
    Returns:
        List of recent contributions
    """
    try:
        current_goals = load_goals()
        contributions = current_goals.get("contributions", [])
        
        # Return most recent contributions
        return contributions[-limit:] if contributions else []
        
    except Exception as e:
        logger.error(f"Error getting agent contributions: {str(e)}")
        return [] 