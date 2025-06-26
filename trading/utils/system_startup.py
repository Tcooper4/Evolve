"""System startup utilities for the trading platform."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import streamlit as st

from trading.utils.auto_repair import auto_repair
from trading.utils.diagnostics import diagnostics
from trading.utils.error_logger import error_logger
from trading.llm.llm_interface import LLMInterface
from core.agents.router import AgentRouter
from trading.agents.updater import ModelUpdater
from trading.memory.performance_memory import PerformanceMemory
from trading.config import config

logger = logging.getLogger(__name__)

def run_system_checks() -> Dict[str, Any]:
    """Run system repair and health checks.
    
    Returns:
        Dict containing repair and health check results
    """
    results = {
        "repair": auto_repair.run_repair(),
        "health": diagnostics.run_health_check()
    }
    
    # Log any issues
    if results["repair"]["status"] != "success":
        error_logger.log_error(
            "System repair required",
            context=results["repair"]
        )
    
    if results["health"]["status"] != "healthy":
        error_logger.log_error(
            "Health check failed",
            context=results["health"]
        )
    
    return results

def initialize_components() -> Dict[str, Any]:
    """Initialize system components with error handling.
    
    Returns:
        Dict containing initialization results
    """
    results = {
        "llm": None,
        "router": None,
        "updater": None,
        "memory": None,
        "errors": []
    }
    
    try:
        results["llm"] = LLMInterface()
        st.session_state.llm = results["llm"]
    except Exception as e:
        error_msg = f"Failed to initialize LLM: {str(e)}"
        error_logger.log_error(error_msg)
        results["errors"].append(error_msg)
    
    try:
        results["router"] = AgentRouter()
        st.session_state.router = results["router"]
    except Exception as e:
        error_msg = f"Failed to initialize AgentRouter: {str(e)}"
        error_logger.log_error(error_msg)
        results["errors"].append(error_msg)
    
    try:
        results["updater"] = ModelUpdater()
        st.session_state.updater = results["updater"]
    except Exception as e:
        error_msg = f"Failed to initialize ModelUpdater: {str(e)}"
        error_logger.log_error(error_msg)
        results["errors"].append(error_msg)
    
    try:
        results["memory"] = PerformanceMemory()
        st.session_state.memory = results["memory"]
    except Exception as e:
        error_msg = f"Failed to initialize PerformanceMemory: {str(e)}"
        error_logger.log_error(error_msg)
        results["errors"].append(error_msg)
    
    return results

def get_system_status() -> Dict[str, Any]:
    """Get current system status.
    
    Returns:
        Dict containing system status information
    """
    return {
        "health_status": st.session_state.get("health_status", "unknown"),
        "repair_status": st.session_state.get("repair_status", "unknown"),
        "llm_provider": st.session_state.get("llm_provider", "unknown"),
        "model_count": len(list(Path(config.get("model_dir", "trading/models")).glob("*.pkl"))),
        "last_check": st.session_state.get("last_system_check", None)
    }

def clear_session_state() -> None:
    """Clear all session state variables."""
    for key in list(st.session_state.keys()):
        del st.session_state[key] 