# Trading Optimization Module

A comprehensive framework for optimizing trading strategies using various optimization methods and visualization tools. This module consolidates all optimization functionality from the previous optimizer, optimize, and optimizers directories.

## Features

- Multiple optimization methods:
  - Grid Search
  - Bayesian Optimization
  - Genetic Algorithms
  - Multi-Objective Optimization
- Strategy-specific parameter optimization (RSI, MACD, Bollinger, etc.)
- Interactive visualization dashboard
- Performance tracking and comparison
- Results export and import
- Regime-aware optimization
- Risk-adjusted metrics

## Installation

The module is part of the Evolve trading system and is automatically available when the system is installed.

## Usage

### Basic Usage

```python
from trading.optimization import StrategyOptimizer, RSIOptimizer, OptimizerFactory

# Initialize optimizer
optimizer = StrategyOptimizer()

# Optimize a strategy
results = optimizer.optimize_strategy(
    strategy="RSI",
    optimizer_type="Bayesian",
    param_space={
        "period": {"min": 5, "max": 30},
        "overbought": {"min": 70, "max": 90},
        "oversold": {"min": 10, "max": 30}
    },
    training_data=data
)
```

### RSI Optimization

```python
from trading.optimization import RSIOptimizer

# Initialize RSI optimizer
rsi_optimizer = RSIOptimizer(data)

# Optimize parameters
results = rsi_optimizer.optimize_rsi_parameters(
    objective='sharpe',
    n_top=3,
    regime_filter='trending'
)
```

### Using the Dashboard

1. Start the Streamlit dashboard:
```bash
streamlit run pages/7_Optimizer.py
```

2. Configure optimization settings in the sidebar
3. Select a strategy and optimizer type
4. Adjust parameter ranges
5. Start optimization
6. View and analyze results

## Optimization Methods

### Grid Search
- Exhaustive search through parameter space
- Suitable for small parameter spaces
- Parallel processing support

### Bayesian Optimization
- Uses Gaussian processes to model objective function
- Efficient for expensive evaluations
- Adapts to parameter space characteristics

### Genetic Algorithm
- Evolutionary optimization approach
- Good for complex, non-linear spaces
- Population-based search

### Multi-Objective Optimization
- Optimizes multiple objectives simultaneously
- Pareto frontier analysis
- Trade-off visualization

## Available Optimizers

- **BaseOptimizer**: Base class for all optimizers
- **BayesianOptimizer**: Bayesian optimization implementation
- **GeneticOptimizer**: Genetic algorithm optimization
- **GridOptimizer**: Grid search optimization
- **MultiObjectiveOptimizer**: Multi-objective optimization
- **RSIOptimizer**: RSI strategy optimization with regime awareness
- **StrategyOptimizer**: General strategy optimization

## Utilities

- **OptimizationVisualizer**: Visualization tools for optimization results
- **OptimizerFactory**: Factory for creating optimizer instances
- **PerformanceLogger**: Performance tracking and logging
- **StrategySelectionAgent**: Intelligent strategy selection
- **OptimizerConsolidator**: Consolidation utilities for managing optimizer organization

## Visualization

The framework provides various visualization tools:
- Optimization progress tracking
- Parameter importance analysis
- Parameter distribution analysis
- Correlation analysis
- Interactive plots using Plotly
- Regime analysis plots
- Equity curve visualization

## Integration

This module is fully integrated with:
- The main Evolve trading system
- Streamlit dashboard interface
- Unified interface for natural language commands
- Agent management system
- Performance tracking and reporting

## Consolidation Status

This module consolidates functionality from:
- `optimizer/` directory (now removed)
- `optimize/` directory (now removed)
- `optimizers/` directory (now removed)

All imports have been updated to use `trading.optimization.*` instead of the old directory structure.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 