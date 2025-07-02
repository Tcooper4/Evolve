"""
Backtesting Module

This module provides a comprehensive backtesting framework with support for:
- Multiple strategy backtesting
- Advanced position sizing methods
- Detailed trade logging and analysis
- Comprehensive performance metrics
- Advanced visualization capabilities
- Sophisticated risk management
"""

from .backtester import Backtester as BacktestEngine
from .position_sizing import PositionSizing, PositionSizingEngine
from .risk_metrics import RiskMetric, RiskMetricsEngine
from .trade_models import Trade, TradeType
from .performance_analysis import PerformanceAnalyzer
from .visualization import BacktestVisualizer

__all__ = [
    'BacktestEngine',
    'PositionSizing',
    'PositionSizingEngine', 
    'RiskMetric',
    'RiskMetricsEngine',
    'Trade',
    'TradeType',
    'PerformanceAnalyzer',
    'BacktestVisualizer'
]

__version__ = "1.0.0"
__author__ = "Evolve Trading System"
__description__ = "Comprehensive Backtesting Framework" 