"""
Trading Optimization Module

This is the central optimization module for the Evolve trading system.
All optimization functionality has been consolidated here.

Available optimizers:
- BaseOptimizer: Base class for all optimizers
- BayesianOptimizer: Bayesian optimization
- GeneticOptimizer: Genetic algorithm optimization
- GridOptimizer: Grid search optimization
- MultiObjectiveOptimizer: Multi-objective optimization
- RSIOptimizer: RSI strategy optimization
- StrategyOptimizer: General strategy optimization

Available utilities:
- OptimizationVisualizer: Visualization tools
- OptimizerFactory: Factory for creating optimizers
- PerformanceLogger: Performance tracking
- StrategySelectionAgent: Strategy selection
"""

# Import base classes
from .base_optimizer import BaseOptimizer, OptimizationResult, OptimizerConfig

# Import core optimizers
try:
    from .bayesian_optimizer import BayesianOptimizer
except ImportError:
    BayesianOptimizer = None

try:
    from .genetic_optimizer import GeneticOptimizer
except ImportError:
    GeneticOptimizer = None

try:
    from .multi_objective_optimizer import MultiObjectiveOptimizer
except ImportError:
    MultiObjectiveOptimizer = None

# Import strategy optimizers
try:
    from .rsi_optimizer import RSIOptimizer, RSIParameters
except ImportError:
    RSIOptimizer = None
    RSIParameters = None

try:
    from .strategy_optimizer import StrategyOptimizer
except ImportError:
    StrategyOptimizer = None

# Import utilities
try:
    from .optimization_visualizer import OptimizationVisualizer
except ImportError:
    OptimizationVisualizer = None

try:
    from .optimizer_factory import OptimizerFactory
except ImportError:
    OptimizerFactory = None

try:
    from .performance_logger import PerformanceLogger
except ImportError:
    PerformanceLogger = None

try:
    from .strategy_selection_agent import StrategySelectionAgent
except ImportError:
    StrategySelectionAgent = None

# Import utils
try:
    from .utils.consolidator import OptimizerConsolidator, run_optimizer_consolidation
except ImportError:
    OptimizerConsolidator = None
    run_optimizer_consolidation = None

# Define what's available
__all__ = [
    'BaseOptimizer',
    'OptimizationResult', 
    'OptimizerConfig',
    'BayesianOptimizer',
    'GeneticOptimizer',
    'MultiObjectiveOptimizer',
    'RSIOptimizer',
    'RSIParameters',
    'StrategyOptimizer',
    'OptimizationVisualizer',
    'OptimizerFactory',
    'PerformanceLogger',
    'StrategySelectionAgent',
    'OptimizerConsolidator',
    'run_optimizer_consolidation'
]

# Version info
__version__ = "2.0.0"
__author__ = "Evolve Trading System"
__description__ = "Centralized optimization module for trading strategies" 