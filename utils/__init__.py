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

from typing import Dict, Any, List, Optional, Union
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import from trading.utils.data_utils since that's where the actual functions are
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'trading'))
from trading.utils.data_utils import (
    DataValidator,
    DataPreprocessor,
    resample_data,
    calculate_technical_indicators,
    split_data,
    prepare_forecast_data
)

# Import from actual math_utils
from .math_utils import (
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_win_rate,
    calculate_profit_factor,
    calculate_calmar_ratio,
    calculate_volatility,
    calculate_beta,
    calculate_alpha
)

# Import from config_loader
from .config_loader import ConfigLoader, config

# Import from strategy_utils
from .strategy_utils import (
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_win_rate,
    calculate_profit_factor,
    calculate_volatility,
    calculate_beta,
    calculate_alpha,
    calculate_information_ratio,
    calculate_calmar_ratio,
    calculate_sortino_ratio,
    calculate_ulcer_index,
    calculate_gain_to_pain_ratio,
    calculate_recovery_factor,
    calculate_risk_metrics
)

# Import from model_utils
from .model_utils import load_model_state, save_model_state, get_model_info

# Import from system_status
from .system_status import SystemStatus, get_system_health

__all__ = [
    # Data utilities
    'DataValidator',
    'DataPreprocessor',
    'resample_data',
    'calculate_technical_indicators',
    'split_data',
    'prepare_forecast_data',
    
    # Math utilities
    'calculate_sharpe_ratio',
    'calculate_max_drawdown',
    'calculate_win_rate',
    'calculate_profit_factor',
    'calculate_calmar_ratio',
    'calculate_volatility',
    'calculate_beta',
    'calculate_alpha',
    
    # Config utilities
    'ConfigLoader',
    'config',
    
    # Strategy utilities
    'calculate_returns',
    'calculate_sharpe_ratio',
    'calculate_max_drawdown',
    'calculate_win_rate',
    'calculate_profit_factor',
    'calculate_volatility',
    'calculate_beta',
    'calculate_alpha',
    'calculate_information_ratio',
    'calculate_calmar_ratio',
    'calculate_sortino_ratio',
    'calculate_ulcer_index',
    'calculate_gain_to_pain_ratio',
    'calculate_recovery_factor',
    'calculate_risk_metrics',
    
    # Model utilities
    'load_model_state',
    'save_model_state',
    'get_model_info',
    
    # System utilities
    'SystemStatus',
    'get_system_health'
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
            'module': 'utils',
            'version': '1.0.0',
            'components': {
                'data_utils': 'available',
                'logging_utils': 'available',
                'math_utils': 'available',
                'validation_utils': 'available',
                'file_utils': 'available',
                'time_utils': 'available'
            },
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return {
            'module': 'utils',
            'status': 'error',
            'error': str(e)
        } 