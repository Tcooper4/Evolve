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

from trading.backtesting.backtester import Backtester as BacktestEngine
from trading.backtesting.edge_case_handler import EdgeCaseHandler
from trading.backtesting.enhanced_backtester import (
    EnhancedBacktester,
    run_forecast_backtest,
    run_multi_model_backtest,
    run_strategy_comparison,
)
from trading.backtesting.performance_analysis import PerformanceAnalyzer
from trading.backtesting.position_sizing import PositionSizing, PositionSizingEngine
from trading.backtesting.risk_metrics import RiskMetric, RiskMetricsEngine
from trading.backtesting.trade_models import Trade, TradeType
from trading.backtesting.visualization import BacktestVisualizer

__all__ = [
    "BacktestEngine",
    "EnhancedBacktester",
    "run_forecast_backtest",
    "run_multi_model_backtest",
    "run_strategy_comparison",
    "PositionSizing",
    "PositionSizingEngine",
    "RiskMetric",
    "RiskMetricsEngine",
    "Trade",
    "TradeType",
    "PerformanceAnalyzer",
    "BacktestVisualizer",
    "EdgeCaseHandler",
]

__version__ = "2.0.0"
__author__ = "Evolve Trading System"
__description__ = "Comprehensive Backtesting Framework with Enhanced Features"
