"""
Session utilities for Streamlit app state management.
Replacement for the removed core.session_utils module.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


def safe_session_get(key: str, default: Any = None) -> Any:
    """Safely get a value from session state with fallback default."""
    try:
        value = st.session_state.get(key, default)
        # Additional check to ensure we don't return None for critical values
        if value is None and key in ["forecast", "market_data", "model_results"]:
            # Return appropriate default based on key
            if key == "forecast":
                return pd.DataFrame()
            elif key == "market_data":
                return pd.DataFrame()
            elif key == "model_results":
                return {}
        return value
    except Exception as e:
        logger.warning(f"Error getting session state for key {key}: {e}")
        return default


def safe_session_set(key: str, value: Any) -> bool:
    """Safely set a value in session state."""
    try:
        st.session_state[key] = value
        return True
    except Exception as e:
        logger.warning(f"Error setting session state for key {key}: {e}")
        return False


def update_last_updated() -> None:
    """Update the last updated timestamp."""
    safe_session_set("last_updated", datetime.now().isoformat())


def get_last_updated() -> Optional[str]:
    """Get the last updated timestamp."""
    return safe_session_get("last_updated")


def display_system_status(status: Dict[str, Any]) -> None:
    """Display system status in a formatted way."""
    if status.get("overall_status") == "healthy":
        st.success("🟢 System Status: Healthy")
    elif status.get("overall_status") == "degraded":
        st.warning("🟡 System Status: Degraded")
    else:
        st.error("🔴 System Status: Error")

    # Display component status
    for component, comp_status in status.get("components", {}).items():
        if comp_status == "healthy":
            st.success(f"✅ {component}")
        elif comp_status == "degraded":
            st.warning(f"⚠️ {component}")
        else:
            st.error(f"❌ {component}")


def initialize_session_state() -> None:
    """Initialize session state with default values."""
    defaults = {
        "chat_history": [],
        "current_action": None,
        "last_result": None,
        "system_status": "initializing",
        "last_updated": datetime.now().isoformat(),
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            safe_session_set(key, default_value)


def clear_session_state() -> None:
    """Clear all session state."""
    try:
        st.session_state.clear()
        initialize_session_state()
    except Exception as e:
        logger.error(f"Error clearing session state: {e}")


def get_session_summary() -> Dict[str, Any]:
    """Get a summary of current session state."""
    try:
        return {
            "session_keys": list(st.session_state.keys()),
            "last_updated": get_last_updated(),
            "chat_history_length": len(safe_session_get("chat_history", [])),
            "current_action": safe_session_get("current_action"),
            "system_status": safe_session_get("system_status"),
        }
    except Exception as e:
        logger.error(f"Error getting session summary: {e}")
        return {"error": str(e)}
