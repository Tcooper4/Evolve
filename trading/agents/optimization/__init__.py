"""
Optimization Agent Package

This package contains the modularized optimization agent components:
- Core optimizer agent
- Parameter validation
- Strategy optimization
- Backtesting integration
- Performance analysis
"""

from .optimizer_agent import OptimizerAgent, create_optimizer_agent
from .parameter_validator import ParameterValidator, OptimizationParameter
from .strategy_optimizer import StrategyOptimizer, StrategyConfig, OptimizationType, OptimizationMetric
from .backtest_integration import BacktestIntegration
from .performance_analyzer import PerformanceAnalyzer, OptimizationResult

__all__ = [
    'OptimizerAgent',
    'create_optimizer_agent',
    'ParameterValidator',
    'OptimizationParameter',
    'StrategyOptimizer',
    'StrategyConfig',
    'OptimizationType',
    'OptimizationMetric',
    'BacktestIntegration',
    'PerformanceAnalyzer',
    'OptimizationResult'
]
