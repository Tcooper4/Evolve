"""
Interface Module for Evolve Trading Platform

This module provides the main user interface components for the trading
system, including the unified interface and various UI components.
"""

import logging
from typing import Any, Dict, Optional

__all__ = ["UnifiedInterface", "StreamlitInterface", "TerminalInterface", "APIInterface"]

logger = logging.getLogger(__name__)


def create_interface(interface_type: str = "streamlit") -> Optional[Any]:
    """
    Create an interface instance based on the specified type.

    Args:
        interface_type: Type of interface to create ('streamlit', 'terminal', 'api')

    Returns:
        Optional[Any]: Interface instance or None if creation fails
    """
    try:
        if interface_type.lower() == "streamlit":
            from .streamlit_interface import StreamlitInterface

            return StreamlitInterface()
        elif interface_type.lower() == "terminal":
            from .terminal_interface import TerminalInterface

            return TerminalInterface()
        elif interface_type.lower() == "api":
            from .api_interface import APIInterface

            return APIInterface()
        elif interface_type.lower() == "unified":
            from .unified_interface import UnifiedInterface

            return UnifiedInterface()
        else:
            logger.error(f"Unknown interface type: {interface_type}")
            return None

    except Exception as e:
        logger.error(f"Error creating interface {interface_type}: {e}")
        return None


def get_available_interfaces() -> Dict[str, str]:
    """
    Get list of available interface types.

    Returns:
        Dict[str, str]: Dictionary of interface types and descriptions
    """
    return {
        "streamlit": "Web-based interface using Streamlit",
        "terminal": "Command-line interface",
        "api": "REST API interface",
        "unified": "Unified interface with multiple access methods",
    }
