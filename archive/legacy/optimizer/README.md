# Strategy Optimizer Framework

A comprehensive framework for optimizing trading strategies using various optimization methods and visualization tools.

## Features

- Multiple optimization methods:
  - Grid Search
  - Bayesian Optimization
  - Genetic Algorithms
- Strategy-specific parameter optimization
- Interactive visualization dashboard
- Performance tracking and comparison
- Results export and import

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Add the optimizer directory to your Python path:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/optimizer
```

## Usage

### Basic Usage

```python
from optimizer import StrategyOptimizer
from utils.strategy_switcher import StrategySwitcher
from utils.memory_logger import MemoryLogger

# Initialize components
strategy_switcher = StrategySwitcher()
memory_logger = MemoryLogger()
optimizer = StrategyOptimizer(strategy_switcher, memory_logger)

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

## Visualization

The framework provides various visualization tools:
- Optimization progress tracking
- Parameter importance analysis
- Parameter distribution analysis
- Correlation analysis
- Interactive plots using Plotly

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 