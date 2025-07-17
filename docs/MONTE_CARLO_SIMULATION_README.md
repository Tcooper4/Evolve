# Monte Carlo Simulation for Backtesting

## Overview

The Monte Carlo Simulation module provides comprehensive portfolio simulation capabilities for backtesting, using bootstrapped historical returns to generate multiple possible future scenarios. This allows for robust risk analysis and confidence interval estimation.

## Features

- **🎲 1000+ Simulation Paths**: Generate thousands of possible portfolio scenarios
- **📊 Bootstrap Methods**: Historical, block, and parametric bootstrapping
- **📈 Percentile Analysis**: Calculate 5th, 50th, 95th percentile bands
- **🎨 Visualization**: Interactive plots with confidence bands and individual paths
- **⚠️ Risk Metrics**: VaR, CVaR, probability of loss, and drawdown analysis
- **🔧 Configurable**: Customizable parameters for different simulation needs

## Quick Start

### 1. Basic Usage

```python
from trading.backtesting.monte_carlo import run_monte_carlo_analysis

# Generate sample returns
import numpy as np
import pandas as pd

returns = pd.Series(np.random.normal(0.0005, 0.02, 252))

# Run Monte Carlo analysis
results = run_monte_carlo_analysis(
    returns=returns,
    initial_capital=10000.0,
    n_simulations=1000,
    bootstrap_method="historical"
)

print(f"Mean Final Value: ${results['summary_statistics']['mean_final_value']:,.2f}")
print(f"95% VaR: {results['summary_statistics']['var_95']:.2%}")
```

### 2. Interactive Dashboard

```bash
streamlit run pages/Monte_Carlo_Simulation.py
```

### 3. Run Example Script

```bash
python examples/monte_carlo_simulation_example.py
```

## API Reference

### Core Classes

#### `MonteCarloConfig`

Configuration class for Monte Carlo simulation parameters.

**Parameters:**
- `n_simulations` (int): Number of simulation paths (default: 1000)
- `confidence_levels` (List[float]): Percentiles to calculate (default: [0.05, 0.50, 0.95])
- `bootstrap_method` (str): Bootstrap method ("historical", "block", "parametric")
- `block_size` (int): Block size for block bootstrap (default: 20)
- `random_seed` (int): Random seed for reproducibility (default: 42)
- `initial_capital` (float): Starting portfolio value (default: 10000.0)

**Example:**
```python
config = MonteCarloConfig(
    n_simulations=1000,
    confidence_levels=[0.05, 0.25, 0.50, 0.75, 0.95],
    bootstrap_method="historical",
    initial_capital=50000.0
)
```

#### `MonteCarloSimulator`

Main simulation class for running Monte Carlo analysis.

**Methods:**

##### `simulate_portfolio_paths()`

Simulate portfolio paths using bootstrapped historical returns.

```python
simulator = MonteCarloSimulator(config)
portfolio_paths = simulator.simulate_portfolio_paths(
    returns=returns,
    initial_capital=10000.0,
    n_simulations=1000
)
```

##### `calculate_percentiles()`

Calculate percentile bands for the simulated paths.

```python
percentiles = simulator.calculate_percentiles([0.05, 0.50, 0.95])
```

##### `get_summary_statistics()`

Get comprehensive summary statistics.

```python
stats = simulator.get_summary_statistics()
print(f"Mean Final Value: ${stats['mean_final_value']:,.2f}")
print(f"95% VaR: {stats['var_95']:.2%}")
```

##### `plot_simulation_results()`

Create visualization of simulation results.

```python
fig = simulator.plot_simulation_results(
    show_paths=True,
    n_paths_to_show=50,
    confidence_bands=True,
    save_path="simulation_results.png"
)
```

### Bootstrap Methods

#### 1. Historical Bootstrap

Samples historical returns with replacement, preserving the empirical distribution.

```python
config = MonteCarloConfig(bootstrap_method="historical")
```

**Advantages:**
- Preserves actual return distribution
- No distributional assumptions
- Captures fat tails and skewness

**Use Cases:**
- General portfolio analysis
- When historical data is representative

#### 2. Block Bootstrap

Preserves time series structure by sampling blocks of consecutive returns.

```python
config = MonteCarloConfig(
    bootstrap_method="block",
    block_size=20  # 20-day blocks
)
```

**Advantages:**
- Preserves autocorrelation structure
- Better for time series data
- Captures volatility clustering

**Use Cases:**
- Time series analysis
- When autocorrelation is important

#### 3. Parametric Bootstrap

Assumes normal distribution and generates returns parametrically.

```python
config = MonteCarloConfig(bootstrap_method="parametric")
```

**Advantages:**
- Fast computation
- Smooth distribution
- Good for large datasets

**Use Cases:**
- Quick analysis
- When normality assumption is reasonable

## Visualization Features

### Portfolio Paths Plot

The main visualization shows:

1. **Individual Paths**: Gray lines showing individual simulation paths
2. **Confidence Bands**: Shaded area between 5th and 95th percentiles
3. **Median Path**: Red line showing 50th percentile
4. **Mean Path**: Green dashed line showing average path

### Return Distribution

Histogram showing the distribution of final returns across all simulations, with key statistics marked:

- Mean return (red dashed line)
- 5th percentile (orange dashed line)
- 95th percentile (orange dashed line)

### Custom Visualizations

Additional plot types available:

- **Drawdown Analysis**: Distribution of maximum drawdowns
- **Volatility Over Time**: Rolling volatility trends
- **Performance Comparison**: Multiple strategy comparison

## Risk Metrics

### Value at Risk (VaR)

The maximum expected loss at a given confidence level.

```python
stats = simulator.get_summary_statistics()
var_95 = stats['var_95']  # 95% VaR
var_99 = stats['var_99']  # 99% VaR
```

### Conditional Value at Risk (CVaR)

The expected loss given that the loss exceeds VaR.

```python
cvar_95 = stats['cvar_95']  # 95% CVaR
cvar_99 = stats['cvar_99']  # 99% CVaR
```

### Probability Metrics

```python
prob_loss = stats['probability_of_loss']  # Probability of negative return
prob_20_loss = stats['probability_of_20_percent_loss']  # Probability of 20% loss
prob_50_loss = stats['probability_of_50_percent_loss']  # Probability of 50% loss
```

### Drawdown Analysis

```python
report = simulator.create_detailed_report()
drawdowns = report['percentile_analysis']['max_drawdowns']
max_dd_median = drawdowns['max_drawdown_P50']
```

## Integration Examples

### 1. Strategy Comparison

```python
def compare_strategies(strategies_returns):
    """Compare multiple strategies using Monte Carlo simulation."""
    
    results = {}
    
    for strategy_name, returns in strategies_returns.items():
        config = MonteCarloConfig(n_simulations=1000)
        simulator = MonteCarloSimulator(config)
        
        simulator.simulate_portfolio_paths(returns, 10000.0, 1000)
        simulator.calculate_percentiles()
        
        stats = simulator.get_summary_statistics()
        results[strategy_name] = {
            'mean_return': stats['mean_total_return'],
            'volatility': stats['std_total_return'],
            'sharpe_ratio': stats['mean_total_return'] / stats['std_total_return'],
            'var_95': stats['var_95'],
            'probability_of_loss': stats['probability_of_loss']
        }
    
    return pd.DataFrame(results).T
```

### 2. Risk Monitoring

```python
def monitor_portfolio_risk(returns, threshold_var=-0.10):
    """Monitor portfolio risk using Monte Carlo simulation."""
    
    results = run_monte_carlo_analysis(
        returns=returns,
        initial_capital=10000.0,
        n_simulations=1000
    )
    
    stats = results['summary_statistics']
    
    if stats['var_95'] < threshold_var:
        print(f"⚠️ Warning: 95% VaR ({stats['var_95']:.2%}) below threshold ({threshold_var:.2%})")
    
    return stats
```

### 3. Parameter Sensitivity Analysis

```python
def parameter_sensitivity_analysis(returns):
    """Analyze sensitivity to different parameters."""
    
    # Test different numbers of simulations
    n_simulations_list = [100, 500, 1000, 2000]
    
    for n_sim in n_simulations_list:
        results = run_monte_carlo_analysis(
            returns=returns,
            n_simulations=n_sim,
            plot_results=False
        )
        
        stats = results['summary_statistics']
        print(f"{n_sim:>4} simulations: VaR={stats['var_95']:.2%}")
```

## Performance Considerations

### Computational Efficiency

- **Historical Bootstrap**: Fastest method, suitable for large datasets
- **Block Bootstrap**: Moderate speed, preserves time series structure
- **Parametric Bootstrap**: Fastest, but assumes normal distribution

### Memory Usage

- Each simulation path requires memory proportional to the number of periods
- 1000 simulations × 252 days ≈ 2MB of memory
- Consider reducing number of simulations for very large datasets

### Accuracy vs Speed Trade-offs

- **High Accuracy**: 2000+ simulations, block bootstrap
- **Balanced**: 1000 simulations, historical bootstrap
- **Quick Analysis**: 500 simulations, parametric bootstrap

## Best Practices

### 1. Data Quality

```python
# Clean and validate returns data
returns = returns.dropna()
returns = returns[returns != 0]  # Remove zero returns if needed

# Check for sufficient data
if len(returns) < 100:
    warnings.warn("Limited historical data may affect simulation accuracy")
```

### 2. Bootstrap Method Selection

```python
# For general analysis
config = MonteCarloConfig(bootstrap_method="historical")

# For time series with autocorrelation
config = MonteCarloConfig(bootstrap_method="block", block_size=20)

# For quick analysis with large datasets
config = MonteCarloConfig(bootstrap_method="parametric")
```

### 3. Confidence Level Selection

```python
# Standard confidence levels
confidence_levels = [0.05, 0.50, 0.95]

# More detailed analysis
confidence_levels = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
```

### 4. Simulation Parameters

```python
# For production analysis
config = MonteCarloConfig(
    n_simulations=1000,
    random_seed=42,  # For reproducibility
    initial_capital=10000.0
)

# For exploratory analysis
config = MonteCarloConfig(
    n_simulations=500,  # Faster for testing
    random_seed=None    # Different results each time
)
```

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce number of simulations
   - Use parametric bootstrap for large datasets
   - Process data in chunks

2. **Slow Performance**
   - Use parametric bootstrap
   - Reduce number of simulations
   - Use block bootstrap with smaller block size

3. **Unrealistic Results**
   - Check data quality
   - Verify return calculations
   - Use appropriate bootstrap method

4. **Visualization Issues**
   - Ensure matplotlib is installed
   - Check file permissions for saving plots
   - Reduce number of paths shown

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run simulation with debug output
simulator = MonteCarloSimulator(config)
simulator.simulate_portfolio_paths(returns, 10000.0, 1000)
```

## Testing

### Run Tests

```bash
# Run all Monte Carlo tests
python -m pytest tests/test_monte_carlo_simulation.py -v

# Run specific test
python tests/test_monte_carlo_simulation.py
```

### Test Coverage

The test suite covers:

- ✅ Configuration initialization
- ✅ Bootstrap methods (historical, block, parametric)
- ✅ Portfolio path simulation
- ✅ Percentile calculations
- ✅ Summary statistics
- ✅ Error handling
- ✅ Visualization
- ✅ Parameter sensitivity

## Contributing

When contributing to the Monte Carlo simulation module:

1. Add tests for new functionality
2. Update documentation
3. Follow the existing code style
4. Test with different bootstrap methods
5. Verify visualization functionality

## License

This module is part of the evolve_clean project and follows the same licensing terms. 