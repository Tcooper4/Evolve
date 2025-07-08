"""
Backtesting Module

This module provides a comprehensive backtesting framework with support for:
- Multiple strategy backtesting
- Advanced position sizing methods
- Detailed trade logging and analysis
- Comprehensive performance metrics
- Advanced visualization capabilities
- Sophisticated risk management
- Enhanced backtesting with automatic signal integration
- Unified trade reporting integration
"""

from .backtester import Backtester as BacktestEngine
from .enhanced_backtester import EnhancedBacktester, run_forecast_backtest, run_multi_model_backtest, run_strategy_comparison
from .position_sizing import PositionSizing, PositionSizingEngine
from .risk_metrics import RiskMetric, RiskMetricsEngine
from .trade_models import Trade, TradeType
from .performance_analysis import PerformanceAnalyzer
from .visualization import BacktestVisualizer
from .edge_case_handler import EdgeCaseHandler

__all__ = [
    'BacktestEngine',
    'EnhancedBacktester',
    'run_forecast_backtest',
    'run_multi_model_backtest', 
    'run_strategy_comparison',
    'PositionSizing',
    'PositionSizingEngine', 
    'RiskMetric',
    'RiskMetricsEngine',
    'Trade',
    'TradeType',
    'PerformanceAnalyzer',
    'BacktestVisualizer',
    'EdgeCaseHandler'
]

__version__ = "2.0.0"
__author__ = "Evolve Trading System"
__description__ = "Comprehensive Backtesting Framework with Enhanced Features" 