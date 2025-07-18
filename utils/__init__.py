"""
Utilities Module for Evolve Trading Platform

This module provides shared utilities and helper functions used across
the trading system. These utilities ensure consistency and reduce code
duplication throughout the platform.

Components:
- Data Utilities: Data processing and validation helpers
- Logging Utilities: Enhanced logging and monitoring functions
- Math Utilities: Mathematical and statistical helper functions
- Validation Utilities: Input validation and error checking
- File Utilities: File and directory management helpers
- Time Utilities: Date and time manipulation functions
"""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict

# Add trading directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "trading"))

# Import from trading.utils.data_utils
from trading.utils.data_utils import (
    DataPreprocessor,
    DataValidator,
    calculate_technical_indicators,
    prepare_forecast_data,
    resample_data,
    split_data,
)

# Import from config_loader
from .config_loader import ConfigLoader, config

# Import from math_utils
from .math_utils import (
    calculate_alpha,
    calculate_beta,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_sharpe_ratio,
    calculate_volatility,
    calculate_win_rate,
)

# Import from model_utils
from .model_utils import get_model_info, load_model_state, save_model_state

# Import from strategy_utils (only unique functions)
from .strategy_utils import (
    calculate_gain_to_pain_ratio,
    calculate_information_ratio,
    calculate_recovery_factor,
    calculate_returns,
    calculate_risk_metrics,
    calculate_sortino_ratio,
    calculate_ulcer_index,
)

# Import from system_status
from .system_status import SystemStatus, get_system_health

__all__ = [
    # Data utilities
    "DataValidator",
    "DataPreprocessor",
    "resample_data",
    "calculate_technical_indicators",
    "split_data",
    "prepare_forecast_data",
    # Math utilities
    "calculate_sharpe_ratio",
    "calculate_max_drawdown",
    "calculate_win_rate",
    "calculate_profit_factor",
    "calculate_calmar_ratio",
    "calculate_volatility",
    "calculate_beta",
    "calculate_alpha",
    # Config utilities
    "ConfigLoader",
    "config",
    # Strategy utilities
    "calculate_returns",
    "calculate_information_ratio",
    "calculate_sortino_ratio",
    "calculate_ulcer_index",
    "calculate_gain_to_pain_ratio",
    "calculate_recovery_factor",
    "calculate_risk_metrics",
    # Model utilities
    "load_model_state",
    "save_model_state",
    "get_model_info",
    # System utilities
    "SystemStatus",
    "get_system_health",
]

logger = logging.getLogger(__name__)


def get_system_info() -> Dict[str, Any]:
    """
    Get system information and utility status.

    Returns:
        Dict[str, Any]: System information
    """
    try:
        return {
            "module": "utils",
            "version": "1.0.0",
            "components": {
                "data_utils": "available",
                "logging_utils": "available",
                "math_utils": "available",
                "validation_utils": "available",
                "file_utils": "available",
                "time_utils": "available",
            },
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return {"module": "utils", "status": "error", "error": str(e)}
