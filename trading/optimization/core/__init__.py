"""
Core Optimization Module

Base classes and core optimization algorithms.
"""

try:
    from ..base_optimizer import BaseOptimizer, OptimizationResult, OptimizerConfig
except ImportError:
    BaseOptimizer = None
    OptimizationResult = None
    OptimizerConfig = None

try:
    from ..bayesian_optimizer import BayesianOptimizer
except ImportError:
    BayesianOptimizer = None

try:
    from ..genetic_optimizer import GeneticOptimizer
except ImportError:
    GeneticOptimizer = None

try:
    from ..multi_objective_optimizer import MultiObjectiveOptimizer
except ImportError:
    MultiObjectiveOptimizer = None

__all__ = [
    'BaseOptimizer',
    'OptimizationResult',
    'OptimizerConfig',
    'BayesianOptimizer',
    'GeneticOptimizer',
    'MultiObjectiveOptimizer'
] 