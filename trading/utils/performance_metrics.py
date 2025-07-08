"""
Performance Metrics Module

This module provides performance metrics functions for trading strategies.
It re-exports functions from existing utility modules to maintain backward compatibility.
"""

# Import from existing utility modules
from utils.strategy_utils import (
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_volatility,
    calculate_win_rate,
    calculate_profit_factor,
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

from utils.math_utils import (
    calculate_sharpe_ratio as math_calculate_sharpe_ratio,
    calculate_max_drawdown as math_calculate_max_drawdown,
    calculate_volatility as math_calculate_volatility,
    calculate_win_rate as math_calculate_win_rate,
    calculate_profit_factor as math_calculate_profit_factor,
    calculate_calmar_ratio as math_calculate_calmar_ratio,
    calculate_beta as math_calculate_beta,
    calculate_alpha as math_calculate_alpha
)

# Re-export the functions (prefer strategy_utils versions for consistency)
__all__ = [
    'calculate_sharpe_ratio',
    'calculate_max_drawdown', 
    'calculate_volatility',
    'calculate_win_rate',
    'calculate_profit_factor',
    'calculate_beta',
    'calculate_alpha',
    'calculate_information_ratio',
    'calculate_calmar_ratio',
    'calculate_sortino_ratio',
    'calculate_ulcer_index',
    'calculate_gain_to_pain_ratio',
    'calculate_recovery_factor',
    'calculate_risk_metrics'
] 