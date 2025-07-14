# Async Strategy Runner

The Async Strategy Runner provides parallel execution of multiple trading strategies using asyncio, with result gathering and ensemble combination capabilities.

## Features

- **Parallel Strategy Execution**: Run multiple strategies concurrently using asyncio
- **Result Gathering**: Collect and process results from all strategies
- **Ensemble Combination**: Automatically combine strategy results using weighted averaging
- **Error Handling**: Robust error handling with timeouts and fallbacks
- **Performance Monitoring**: Track execution times and success rates
- **Configurable**: Customizable concurrency limits, timeouts, and ensemble methods

## Quick Start

### Basic Usage

```python
import asyncio
import pandas as pd
from trading.strategies.strategy_runner import AsyncStrategyRunner

# Initialize runner
runner = AsyncStrategyRunner({
    "max_concurrent_strategies": 3,
    "strategy_timeout": 30,
    "enable_ensemble": True,
    "ensemble_method": "weighted"
})

# Run strategies in parallel
results = await runner.run_strategies_parallel(
    strategies=["RSI", "MACD", "Bollinger Bands"],
    data=market_data,
    parameters={
        "RSI": {"period": 14, "overbought": 70, "oversold": 30},
        "MACD": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        "Bollinger Bands": {"period": 20, "std_dev": 2}
    }
)
```

### The Exact Pattern You Requested

```python
async def run_rsi():
    # RSI strategy execution
    return await runner.run_rsi_strategy(data, parameters)

async def run_macd():
    # MACD strategy execution
    return await runner.run_macd_strategy(data, parameters)

async def run_bb():
    # Bollinger Bands strategy execution
    return await runner.run_bollinger_bands_strategy(data, parameters)

# Execute all strategies in parallel
results = await asyncio.gather(run_rsi(), run_macd(), run_bb())
```

## Configuration

### Runner Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_concurrent_strategies` | int | 5 | Maximum number of strategies to run concurrently |
| `strategy_timeout` | int | 30 | Timeout per strategy in seconds |
| `enable_ensemble` | bool | True | Enable ensemble result combination |
| `ensemble_method` | str | "weighted" | Ensemble method: "weighted", "voting", "average" |
| `log_performance` | bool | True | Enable performance logging |

### Example Configuration

```python
config = {
    "max_concurrent_strategies": 5,
    "strategy_timeout": 60,
    "enable_ensemble": True,
    "ensemble_method": "weighted",
    "log_performance": True
}

runner = AsyncStrategyRunner(config)
```

## API Reference

### AsyncStrategyRunner

#### `__init__(config: Optional[Dict[str, Any]] = None)`

Initialize the async strategy runner.

**Parameters:**
- `config`: Configuration dictionary (see Configuration section)

#### `async run_strategies_parallel(strategies, data, parameters=None, ensemble_config=None)`

Run multiple strategies in parallel and optionally combine results.

**Parameters:**
- `strategies`: List of strategy names or strategy instances
- `data`: Market data DataFrame
- `parameters`: Dictionary mapping strategy names to parameters
- `ensemble_config`: Configuration for ensemble combination

**Returns:**
```python
{
    "success": bool,
    "individual_results": Dict[str, Dict],
    "ensemble_result": Optional[Dict],
    "execution_stats": Dict[str, Any],
    "timestamp": str
}
```

#### Individual Strategy Methods

```python
async run_rsi_strategy(data, parameters) -> Dict[str, Any]
async run_macd_strategy(data, parameters) -> Dict[str, Any]
async run_bollinger_bands_strategy(data, parameters) -> Dict[str, Any]
async run_momentum_strategy(data, parameters) -> Dict[str, Any]
async run_mean_reversion_strategy(data, parameters) -> Dict[str, Any]
```

### Result Structure

#### Individual Strategy Result

```python
{
    "success": bool,
    "strategy_name": str,
    "signals": pd.DataFrame,
    "performance_metrics": Dict[str, float],
    "metadata": Dict[str, Any],
    "execution_time": float,
    "parameters_used": Dict[str, Any],
    "timestamp": str
}
```

#### Performance Metrics

```python
{
    "total_return": float,
    "volatility": float,
    "sharpe_ratio": float,
    "max_drawdown": float,
    "win_rate": float
}
```

#### Ensemble Result

```python
{
    "success": bool,
    "combined_signals": pd.DataFrame,
    "performance_metrics": Dict[str, float],
    "ensemble_config": {
        "method": str,
        "weights": List[float],
        "num_strategies": int
    },
    "timestamp": str
}
```

## Examples

### Example 1: Basic Parallel Execution

```python
import asyncio
import pandas as pd
from trading.strategies.strategy_runner import AsyncStrategyRunner

async def main():
    # Initialize
    runner = AsyncStrategyRunner()
    data = pd.read_csv("market_data.csv")
    
    # Run strategies
    results = await runner.run_strategies_parallel(
        strategies=["RSI", "MACD", "Bollinger Bands"],
        data=data,
        parameters={
            "RSI": {"period": 14},
            "MACD": {"fast_period": 12, "slow_period": 26},
            "Bollinger Bands": {"period": 20}
        }
    )
    
    # Process results
    if results["success"]:
        for strategy_name, result in results["individual_results"].items():
            if result["success"]:
                print(f"{strategy_name}: Sharpe = {result['performance_metrics']['sharpe_ratio']:.3f}")

asyncio.run(main())
```

### Example 2: Custom Strategy Instances

```python
from trading.strategies.rsi_strategy import RSIStrategy
from trading.strategies.macd_strategy import MACDStrategy

# Create strategy instances
rsi = RSIStrategy()
macd = MACDStrategy()

# Run with instances
results = await runner.run_strategies_parallel(
    strategies=[rsi, macd],
    data=data
)
```

### Example 3: Ensemble Configuration

```python
# Run with custom ensemble configuration
results = await runner.run_strategies_parallel(
    strategies=["RSI", "MACD", "Bollinger Bands"],
    data=data,
    ensemble_config={
        "method": "weighted",
        "weights": [0.4, 0.4, 0.2]
    }
)

# Access ensemble result
if results["ensemble_result"]:
    ensemble_perf = results["ensemble_result"]["performance_metrics"]
    print(f"Ensemble Sharpe: {ensemble_perf['sharpe_ratio']:.3f}")
```

### Example 4: Error Handling

```python
# Run with error handling
results = await runner.run_strategies_parallel(
    strategies=["RSI", "InvalidStrategy", "MACD"],
    data=data
)

# Check which strategies succeeded/failed
for strategy_name, result in results["individual_results"].items():
    if result["success"]:
        print(f"✓ {strategy_name}: Success")
    else:
        print(f"✗ {strategy_name}: {result['error']}")

# Check overall stats
stats = results["execution_stats"]
print(f"Success rate: {stats['successful_strategies']}/{stats['total_strategies']}")
```

## Performance Considerations

### Concurrency Limits

- Set `max_concurrent_strategies` based on your system's capabilities
- Too many concurrent strategies may cause memory issues
- Too few may not utilize available resources efficiently

### Timeout Management

- Set appropriate `strategy_timeout` values
- Consider strategy complexity when setting timeouts
- Monitor execution times to optimize timeout settings

### Memory Usage

- Large datasets may require chunking
- Consider using data sampling for testing
- Monitor memory usage during parallel execution

## Best Practices

### 1. Strategy Registration

Ensure all strategies are properly registered in the strategy registry:

```python
from trading.strategies.registry import StrategyRegistry

registry = StrategyRegistry()
registry.register_strategy("MyStrategy", MyStrategyClass)
```

### 2. Parameter Validation

Validate strategy parameters before execution:

```python
def validate_parameters(parameters):
    required_params = ["period", "threshold"]
    for param in required_params:
        if param not in parameters:
            raise ValueError(f"Missing required parameter: {param}")
```

### 3. Error Handling

Implement proper error handling:

```python
try:
    results = await runner.run_strategies_parallel(strategies, data)
    if not results["success"]:
        logger.error(f"Execution failed: {results['error']}")
        return
except Exception as e:
    logger.error(f"Unexpected error: {e}")
```

### 4. Performance Monitoring

Monitor execution performance:

```python
# Get execution history
history = runner.get_execution_history()
for entry in history:
    print(f"Success rate: {entry['success_rate']:.1%}")

# Get strategy performance
performance = runner.get_strategy_performance()
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all strategy modules are properly imported
2. **Timeout Errors**: Increase `strategy_timeout` for complex strategies
3. **Memory Errors**: Reduce `max_concurrent_strategies` or use smaller datasets
4. **Registry Errors**: Verify strategies are registered in the strategy registry

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

runner = AsyncStrategyRunner({"log_performance": True})
```

### Performance Debugging

Monitor execution times and success rates:

```python
results = await runner.run_strategies_parallel(strategies, data)
stats = results["execution_stats"]

print(f"Total time: {stats['execution_time']:.2f}s")
print(f"Parallel efficiency: {stats['parallel_efficiency']:.2f}")
```

## Integration

### With Existing Systems

The Async Strategy Runner integrates seamlessly with existing trading systems:

```python
# Integrate with portfolio manager
from trading.portfolio.portfolio_manager import PortfolioManager

portfolio = PortfolioManager()
runner = AsyncStrategyRunner()

# Run strategies and update portfolio
results = await runner.run_strategies_parallel(strategies, data)
if results["ensemble_result"]:
    portfolio.update_positions(results["ensemble_result"]["combined_signals"])
```

### With Backtesting

```python
# Integrate with backtesting system
from trading.backtesting.backtester import Backtester

backtester = Backtester()
runner = AsyncStrategyRunner()

# Run strategies and backtest
results = await runner.run_strategies_parallel(strategies, data)
backtest_results = backtester.run_backtest(results["ensemble_result"]["combined_signals"])
```

## Future Enhancements

- **Dynamic Strategy Loading**: Load strategies from configuration files
- **Real-time Execution**: Support for real-time market data streaming
- **Advanced Ensemble Methods**: More sophisticated ensemble combination techniques
- **Distributed Execution**: Support for distributed strategy execution across multiple nodes
- **Strategy Optimization**: Built-in strategy parameter optimization
- **Performance Analytics**: Advanced performance analytics and reporting

## Contributing

To contribute to the Async Strategy Runner:

1. Follow the existing code style and patterns
2. Add comprehensive tests for new features
3. Update documentation for any API changes
4. Ensure backward compatibility when possible
5. Add performance benchmarks for new features 