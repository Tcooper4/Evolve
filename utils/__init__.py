"""
Utilities package — import submodules directly (e.g. ``utils.dataframe_utils``).
The root module stays light to avoid pulling trading.utils and heavy deps on import.
"""

import logging
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)


def get_system_info() -> Dict[str, Any]:
    """Lightweight package metadata (optional callers)."""
    try:
        return {
            "module": "utils",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return {"module": "utils", "status": "error", "error": str(e)}


__all__ = ["get_system_info"]
