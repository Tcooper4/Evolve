"""
Core Trading System Components

This module provides the core components for the trading system including
performance tracking, agent management, and fundamental trading operations.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from utils.launch_utils import setup_logging

from .agents import AgentManager, AgentStatus
from .performance import PerformanceMetrics, PerformanceTracker

# Configure logging


def setup_core_logging():
    """Set up logging for the service."""
    return setup_logging(service_name="report_service")


def load_agent_registry() -> Dict[str, Any]:
    """Load agent registry."""
    # Agent registry loading logic here


def save_agent_registry(registry: Dict[str, Any]) -> bool:
    """Save agent registry to file.

    Args:
        registry: Agent registry dictionary

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        registry_path = Path("data/agent_registry.json")
        registry_path.parent.mkdir(exist_ok=True)

        # Update timestamp
        registry["last_updated"] = datetime.now().isoformat()

        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)

        logging.info(f"Saved agent registry to {registry_path}")
        return True

    except Exception as e:
        logging.error(f"Failed to save agent registry: {e}")
        return False


def initialize_core_system():
    """Initialize the core trading system components."""
    logging.info("Initializing core trading system...")

    # Setup logging
    setup_core_logging()

    # Load agent registry
    agent_registry = load_agent_registry()

    # Create necessary directories
    directories = ["logs", "data", "models", "cache", "reports", "memory", "backups"]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

    logging.info("Core trading system initialization completed")

    return {
        "agent_registry": agent_registry,
        "initialized_at": datetime.now().isoformat(),
    }


# Global variables for system state
_system_initialized = False
_agent_registry = None
_initialization_data = None


def get_agent_registry() -> Optional[Dict[str, Any]]:
    """Get the current agent registry.

    Returns:
        Agent registry dictionary or None if not initialized
    """
    global _agent_registry
    if not _system_initialized:
        initialize_core_system()
    return _agent_registry


def update_agent_status(agent_name: str, status: str, **kwargs):
    """Update agent status in registry.

    Args:
        agent_name: Name of the agent
        status: New status
        **kwargs: Additional agent information
    """
    global _agent_registry

    if not _system_initialized:
        initialize_core_system()

    if _agent_registry and "agents" in _agent_registry:
        if agent_name in _agent_registry["agents"]:
            _agent_registry["agents"][agent_name]["status"] = status
            _agent_registry["agents"][agent_name][
                "last_used"
            ] = datetime.now().isoformat()

            # Update additional fields
            for key, value in kwargs.items():
                _agent_registry["agents"][agent_name][key] = value

            # Save updated registry
            save_agent_registry(_agent_registry)
            logging.info(f"Updated agent {agent_name} status to {status}")


def get_system_status() -> Dict[str, Any]:
    """Get overall system status.

    Returns:
        Dictionary containing system status information
    """
    global _system_initialized, _agent_registry

    status = {
        "initialized": _system_initialized,
        "initialization_time": (
            _initialization_data.get("initialized_at") if _initialization_data else None
        ),
        "agent_count": len(_agent_registry.get("agents", {})) if _agent_registry else 0,
        "active_agents": 0,
        "inactive_agents": 0,
    }

    if _agent_registry and "agents" in _agent_registry:
        for agent_info in _agent_registry["agents"].values():
            if agent_info.get("status") == "active":
                status["active_agents"] += 1
            else:
                status["inactive_agents"] += 1

    return status


# Initialize system on module import
try:
    _initialization_data = initialize_core_system()
    _agent_registry = _initialization_data["agent_registry"]
    _system_initialized = True
    logging.info("Core module initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize core module: {e}")
    _system_initialized = False

__all__ = [
    "PerformanceTracker",
    "PerformanceMetrics",
    "AgentManager",
    "AgentStatus",
    "initialize_core_system",
    "get_agent_registry",
    "update_agent_status",
    "get_system_status",
    "setup_logging",
    "load_agent_registry",
    "save_agent_registry",
]

__version__ = "1.0.0"
__author__ = "Evolve Trading System"
__description__ = "Core Trading System Components"
