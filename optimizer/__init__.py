"""
Strategy Optimizer Framework.

This package provides a comprehensive framework for optimizing trading strategies
using various optimization methods (Grid Search, Bayesian Optimization, Genetic
Algorithms) and visualization tools.

Components:
- Core optimizers (Grid, Bayesian, Genetic)
- Strategy-specific optimizers
- Visualization tools
- Streamlit dashboard integration
"""

from optimizer.core.optimizer_factory import (
    BaseOptimizer,
    OptimizerFactory,
    load_optimizers
)
from optimizer.core.grid_optimizer import GridOptimizer
from optimizer.core.bayesian_optimizer import BayesianOptimizer
from optimizer.core.genetic_optimizer import GeneticOptimizer
from optimizer.strategies.strategy_optimizer import StrategyOptimizer
from optimizer.visualization.optimization_visualizer import OptimizationVisualizer

__all__ = [
    'BaseOptimizer',
    'OptimizerFactory',
    'load_optimizers',
    'GridOptimizer',
    'BayesianOptimizer',
    'GeneticOptimizer',
    'StrategyOptimizer',
    'OptimizationVisualizer'
]

# Load available optimizers
load_optimizers('optimizer/core') 