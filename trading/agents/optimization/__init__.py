"""
Optimization Agent Package

This package contains the modularized optimization agent components:
- Core optimizer agent
- Parameter validation
- Strategy optimization
- Backtesting integration
- Performance analysis
"""

from .backtest_integration import BacktestIntegration
from .optimizer_agent import OptimizerAgent, create_optimizer_agent
from .parameter_validator import OptimizationParameter, ParameterValidator
from .performance_analyzer import OptimizationResult, PerformanceAnalyzer
from .strategy_optimizer import (
    OptimizationMetric,
    OptimizationType,
    StrategyConfig,
    StrategyOptimizer,
)

__all__ = [
    "OptimizerAgent",
    "create_optimizer_agent",
    "ParameterValidator",
    "OptimizationParameter",
    "StrategyOptimizer",
    "StrategyConfig",
    "OptimizationType",
    "OptimizationMetric",
    "BacktestIntegration",
    "PerformanceAnalyzer",
    "OptimizationResult",
]
