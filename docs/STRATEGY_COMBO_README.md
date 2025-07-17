# Strategy Combo Creator

## Overview

The Strategy Combo Creator is an enhanced module that allows you to combine multiple trading strategies into powerful ensemble combinations. It provides a flexible and robust framework for creating, testing, and optimizing strategy combinations with multiple combination modes.

## Features

### 🎯 **Multiple Combination Modes**
- **Intersection**: Only generate signals when all strategies agree
- **Union**: Generate signals when any strategy triggers
- **Weighted**: Combine signals using weighted averages
- **Voting**: Use voting mechanism with configurable thresholds
- **Confidence-based**: Weight strategies by their confidence levels

### 🔧 **Advanced Configuration**
- Configurable strategy weights
- Confidence thresholds
- Signal validation and smoothing
- Performance tracking and optimization
- Conflict resolution mechanisms

### 🔗 **Seamless Integration**
- Backward compatibility with existing code
- Integration with existing trading strategies
- Streamlit UI for easy combo creation
- Comprehensive testing framework

## Installation

The Strategy Combo Creator is already integrated into the existing codebase. No additional installation is required.

## Quick Start

### Basic Usage

```python
from strategies.strategy_pipeline import create_strategy_combo

# Create a simple strategy combination
pipeline = create_strategy_combo(
    strategy_names=["RSI", "MACD", "Bollinger"],
    mode='intersection',
    weights=[0.4, 0.4, 0.2]
)

# Generate combined signals
combined_signal, metadata = pipeline.generate_combined_signals(data, ["RSI", "MACD", "Bollinger"])
```

### Advanced Configuration

```python
from strategies.strategy_pipeline import (
    StrategyPipeline, StrategyConfig, CombinationConfig
)

# Create strategy configurations
strategies = [
    StrategyConfig(name="RSI", weight=1.0, confidence_threshold=0.6),
    StrategyConfig(name="MACD", weight=1.5, confidence_threshold=0.7),
    StrategyConfig(name="Bollinger", weight=0.8, confidence_threshold=0.5)
]

# Create combination configuration
combination_config = CombinationConfig(
    mode='weighted',
    min_agreement=0.5,
    confidence_threshold=0.6,
    smoothing_window=5,
    enable_validation=True
)

# Create pipeline
pipeline = StrategyPipeline(strategies, combination_config)
```

## Combination Modes

### 1. Intersection Mode
Only generates signals when all strategies agree on the direction.

```python
# Conservative approach - all strategies must agree
combined = pipeline.combine_signals(signals_list, mode='intersection')
```

**Use case**: Conservative trading with high confidence requirements.

### 2. Union Mode
Generates signals when any strategy triggers a signal.

```python
# Aggressive approach - any strategy can trigger
combined = pipeline.combine_signals(signals_list, mode='union')
```

**Use case**: Capturing more trading opportunities with lower confidence requirements.

### 3. Weighted Mode
Combines signals using weighted averages based on strategy performance or importance.

```python
# Weighted combination
weights = [0.4, 0.4, 0.2]  # RSI, MACD, Bollinger weights
combined = pipeline.combine_signals(signals_list, mode='weighted', weights=weights)
```

**Use case**: Balancing multiple strategies based on their historical performance.

### 4. Voting Mode
Uses a voting mechanism with configurable thresholds.

```python
# Voting with minimum agreement threshold
combined = pipeline.combine_signals(signals_list, mode='voting')
```

**Use case**: Democratic approach where majority rules.

### 5. Confidence-based Mode
Weights strategies by their confidence levels.

```python
# Confidence-based weighting
combined = pipeline.combine_signals(signals_list, mode='confidence')
```

**Use case**: Dynamic weighting based on strategy confidence.

## Available Strategies

The following strategies are available out of the box:

- **RSI**: Relative Strength Index strategy
- **MACD**: Moving Average Convergence Divergence strategy
- **Bollinger**: Bollinger Bands strategy
- **SMA**: Simple Moving Average crossover strategy

### Adding Custom Strategies

```python
def custom_strategy(data: pd.DataFrame, **params) -> pd.Series:
    """Your custom strategy implementation."""
    # Your strategy logic here
    signals = pd.Series(0, index=data.index)
    # ... strategy implementation
    return signals

# Add to pipeline
pipeline.add_strategy(
    name="Custom",
    function=custom_strategy,
    weight=1.0,
    confidence_threshold=0.6,
    parameters={'param1': 10, 'param2': 0.5}
)
```

## Streamlit Interface

### Accessing the Combo Creator

1. Start the Streamlit app: `streamlit run app.py`
2. Navigate to "Strategy Combo Creator" in the sidebar
3. Select strategies and configure combination settings
4. Test and save your combinations

### Features in the UI

- **Strategy Selection**: Multi-select available strategies
- **Combination Mode**: Choose from all available modes
- **Weight Configuration**: Adjust strategy weights with sliders
- **Advanced Settings**: Configure confidence thresholds and smoothing
- **Data Source**: Choose between sample data, file upload, or live data
- **Performance Analysis**: View detailed performance metrics
- **Strategy Comparison**: Compare individual strategy performance
- **Save & Export**: Save combinations and export results

## Performance Analysis

The Strategy Combo Creator provides comprehensive performance analysis:

### Metrics Calculated
- **Total Return**: Overall strategy performance
- **Annualized Return**: Yearly performance rate
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst historical decline
- **Win Rate**: Percentage of profitable trades
- **Signal Agreement**: Agreement level among strategies

### Performance Visualization
- Combined signal charts with price data
- Performance comparison vs buy & hold
- Individual strategy performance comparison
- Signal distribution analysis

## Testing

### Running Tests

```bash
# Run all strategy combo tests
python tests/test_strategy_combo.py

# Run specific test
python -m pytest tests/test_strategy_combo.py::TestStrategyCombo::test_backward_compatibility
```

### Test Coverage

The test suite covers:
- Backward compatibility with existing code
- Strategy pipeline creation and configuration
- Signal combination modes
- Error handling and validation
- Integration with existing strategies
- Performance calculation accuracy

## Examples

### Example 1: Conservative RSI + MACD Combo

```python
from strategies.strategy_pipeline import create_strategy_combo

# Create conservative combo
pipeline = create_strategy_combo(
    strategy_names=["RSI", "MACD"],
    mode='intersection',
    weights=[1.0, 1.0]
)

# Generate signals
combined_signal, metadata = pipeline.generate_combined_signals(data, ["RSI", "MACD"])
```

### Example 2: Aggressive Multi-Strategy Combo

```python
# Create aggressive combo with multiple strategies
pipeline = create_strategy_combo(
    strategy_names=["RSI", "MACD", "Bollinger", "SMA"],
    mode='union',
    weights=[0.3, 0.3, 0.2, 0.2]
)

# Generate signals
combined_signal, metadata = pipeline.generate_combined_signals(data)
```

### Example 3: Performance-Based Weighting

```python
# Create pipeline with performance-based weights
strategies = [
    StrategyConfig(name="RSI", weight=0.4),      # Higher weight for better performance
    StrategyConfig(name="MACD", weight=0.4),     # Higher weight for better performance
    StrategyConfig(name="Bollinger", weight=0.2) # Lower weight for lower performance
]

combination_config = CombinationConfig(mode='weighted')
pipeline = StrategyPipeline(strategies, combination_config)

# Update weights based on performance
performance_metrics = {
    'RSI': {'sharpe_ratio': 1.2},
    'MACD': {'sharpe_ratio': 1.1},
    'Bollinger': {'sharpe_ratio': 0.8}
}

pipeline.update_weights(performance_metrics)
```

## Integration with Existing Code

### Backward Compatibility

The enhanced strategy pipeline maintains full backward compatibility:

```python
# Old way (still works)
from strategies.strategy_pipeline import combine_signals, rsi_strategy, macd_strategy

rsi_signals = rsi_strategy(data)
macd_signals = macd_strategy(data)
combined = combine_signals([rsi_signals, macd_signals], mode='intersection')
```

### Integration with Trading System

```python
# Use in existing trading pipeline
from strategies.strategy_pipeline import StrategyPipeline

# Create combo for trading
pipeline = create_strategy_combo(["RSI", "MACD"], mode='weighted')

# Generate signals for trading
signals, metadata = pipeline.generate_combined_signals(market_data)

# Use signals in trading logic
for timestamp, signal in signals.items():
    if signal == 1:
        # Execute buy order
        pass
    elif signal == -1:
        # Execute sell order
        pass
```

## Best Practices

### 1. Strategy Selection
- Choose complementary strategies that work well together
- Avoid strategies with high correlation
- Consider market conditions when selecting strategies

### 2. Weight Configuration
- Start with equal weights and adjust based on performance
- Use performance metrics to guide weight allocation
- Regularly rebalance weights based on recent performance

### 3. Combination Mode Selection
- Use **intersection** for conservative, low-frequency trading
- Use **union** for aggressive, high-frequency trading
- Use **weighted** for balanced, performance-based approaches
- Use **voting** for democratic, consensus-based approaches

### 4. Performance Monitoring
- Regularly monitor strategy performance
- Track agreement levels between strategies
- Adjust configuration based on market conditions

### 5. Risk Management
- Set appropriate confidence thresholds
- Use signal validation to filter out noise
- Implement proper position sizing based on signal strength

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```python
   # Ensure you're importing from the correct location
   from strategies.strategy_pipeline import StrategyPipeline
   ```

2. **Empty Signals**
   - Check that your data contains required columns (OHLCV)
   - Verify strategy parameters are appropriate for your data
   - Ensure strategies are generating valid signals

3. **Performance Issues**
   - Use signal validation to filter out noise
   - Adjust confidence thresholds
   - Consider using smoothing for noisy signals

4. **Strategy Conflicts**
   - Monitor agreement levels between strategies
   - Adjust weights to balance conflicting signals
   - Consider using different combination modes

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Create pipeline with debug output
pipeline = StrategyPipeline()
pipeline.combine_signals(signals_list, mode='intersection')
```

## Contributing

### Adding New Strategies

1. Implement your strategy function
2. Add it to the `STRATEGY_FUNCTIONS` dictionary
3. Update tests to include your strategy
4. Document your strategy in this README

### Adding New Combination Modes

1. Implement the combination logic in `StrategyPipeline`
2. Add the mode to `COMBINE_MODES` list
3. Update tests for the new mode
4. Document the mode in this README

## License

This module is part of the Evolve Trading Platform and follows the same licensing terms.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the test files for examples
3. Check the existing documentation
4. Create an issue in the project repository

---

**Note**: The Strategy Combo Creator is designed to enhance your trading strategies by combining multiple approaches. Always test thoroughly in a paper trading environment before using with real money. 