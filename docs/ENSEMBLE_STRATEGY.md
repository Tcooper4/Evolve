# Weighted Ensemble Strategy

## Overview

The `WeightedEnsembleStrategy` is a powerful tool that combines multiple trading strategy outputs using configurable weights to produce a final buy/sell signal. This approach helps reduce individual strategy biases and improves overall trading performance through diversification.

## Features

- **Multiple Combination Methods**: Supports both weighted average and voting methods
- **Configurable Weights**: Dynamic weight assignment for each strategy
- **Confidence Thresholds**: Filter signals based on confidence levels
- **Consensus Detection**: Identify when multiple strategies agree
- **Performance Metrics**: Track ensemble performance over time
- **Dynamic Weight Updates**: Adjust weights based on performance

## Quick Start

### Basic Usage

```python
from trading.strategies.ensemble import create_rsi_macd_bollinger_ensemble
from trading.strategies import get_signals

# Create ensemble strategy
ensemble = create_rsi_macd_bollinger_ensemble()

# Generate individual strategy signals
strategy_signals = {}
strategy_signals["rsi"] = get_signals("rsi", data)["result"]
strategy_signals["macd"] = get_signals("macd", data)["result"]
strategy_signals["bollinger"] = get_signals("bollinger", data)["result"]

# Combine signals
combined_signals = ensemble.combine_signals(strategy_signals)
```

### Using the Unified Interface

```python
from trading.strategies import get_signals

# Generate ensemble signals directly
ensemble_signals = get_signals("ensemble", data, 
                              strategy_weights={"rsi": 0.4, "macd": 0.4, "bollinger": 0.2})
```

## Configuration

### EnsembleConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy_weights` | Dict[str, float] | `{"rsi": 0.4, "macd": 0.4, "bollinger": 0.2}` | Weights for each strategy |
| `combination_method` | str | `"weighted_average"` | Method to combine signals |
| `confidence_threshold` | float | `0.6` | Minimum confidence for signal generation |
| `consensus_threshold` | float | `0.5` | Minimum agreement for consensus signals |
| `position_size_multiplier` | float | `1.0` | Multiplier for position sizes |
| `risk_adjustment` | bool | `True` | Enable risk-based adjustments |
| `dynamic_weighting` | bool | `False` | Enable dynamic weight updates |
| `rebalance_frequency` | str | `"daily"` | Frequency for weight rebalancing |

### Pre-configured Ensembles

```python
from trading.strategies.ensemble import (
    create_rsi_macd_bollinger_ensemble,
    create_balanced_ensemble,
    create_conservative_ensemble
)

# RSI-Heavy Ensemble (40% RSI, 40% MACD, 20% Bollinger)
ensemble1 = create_rsi_macd_bollinger_ensemble()

# Balanced Ensemble (33% each)
ensemble2 = create_balanced_ensemble()

# Conservative Ensemble (higher thresholds)
ensemble3 = create_conservative_ensemble()
```

## Combination Methods

### 1. Weighted Average

Combines signals by calculating the weighted average of individual strategy outputs.

```python
config = EnsembleConfig(
    strategy_weights={"rsi": 0.5, "macd": 0.3, "bollinger": 0.2},
    combination_method="weighted_average"
)
```

**Advantages:**
- Smooth signal transitions
- Preserves signal magnitude
- Good for trend-following strategies

### 2. Voting

Uses a voting mechanism where each strategy "votes" for buy/sell/hold.

```python
config = EnsembleConfig(
    strategy_weights={"rsi": 0.33, "macd": 0.33, "bollinger": 0.34},
    combination_method="voting"
)
```

**Advantages:**
- Clear signal direction
- Good for mean-reversion strategies
- Reduces noise

## Signal Output

The ensemble strategy produces a DataFrame with the following columns:

| Column | Description |
|--------|-------------|
| `signal` | Combined signal (-1: sell, 0: hold, 1: buy) |
| `confidence` | Weighted average confidence |
| `weighted_score` | Raw weighted score before thresholding |
| `consensus` | Proportion of strategies agreeing on direction |
| `strong_signal` | Signal with consensus threshold applied |

## Advanced Usage

### Dynamic Weight Updates

```python
# Update weights based on performance
new_weights = {"rsi": 0.6, "macd": 0.25, "bollinger": 0.15}
result = ensemble.update_weights(new_weights)
```

### Performance Analysis

```python
# Get performance metrics
metrics = ensemble.get_performance_metrics()
print(f"Total signals: {metrics['result']['total_signals']}")
print(f"Buy signals: {metrics['result']['buy_signals']}")
print(f"Average confidence: {metrics['result']['avg_confidence']:.3f}")
```

### Position Calculation

```python
# Calculate trading positions
positions = ensemble.calculate_positions(data)
print(f"Current position: {positions['position'].iloc[-1]:.3f}")
```

## Best Practices

### 1. Strategy Selection

Choose strategies that are:
- **Diverse**: Different market conditions and timeframes
- **Complementary**: Not highly correlated
- **Well-tested**: Proven individual performance

### 2. Weight Assignment

- **Performance-based**: Assign higher weights to better-performing strategies
- **Market regime**: Adjust weights based on market conditions
- **Risk tolerance**: Conservative weights for risk-averse traders

### 3. Threshold Tuning

- **Confidence threshold**: Higher values reduce false signals
- **Consensus threshold**: Higher values require more agreement
- **Monitor performance**: Adjust thresholds based on results

### 4. Regular Rebalancing

```python
# Rebalance weights monthly
if should_rebalance():
    new_weights = calculate_optimal_weights(performance_history)
    ensemble.update_weights(new_weights)
```

## Example Strategies

### Conservative Ensemble

```python
config = EnsembleConfig(
    strategy_weights={"rsi": 0.3, "macd": 0.3, "bollinger": 0.4},
    combination_method="weighted_average",
    confidence_threshold=0.7,
    consensus_threshold=0.6
)
```

### Aggressive Ensemble

```python
config = EnsembleConfig(
    strategy_weights={"rsi": 0.5, "macd": 0.4, "bollinger": 0.1},
    combination_method="voting",
    confidence_threshold=0.4,
    consensus_threshold=0.3
)
```

## Integration with Existing Systems

### Strategy Manager Integration

```python
from trading.strategies.strategy_manager import StrategyManager

# Add ensemble strategy to manager
manager = StrategyManager()
ensemble_strategy = create_rsi_macd_bollinger_ensemble()
manager.add_strategy("ensemble", ensemble_strategy)
```

### Backtesting Integration

```python
from trading.backtesting import Backtester

# Use ensemble in backtesting
backtester = Backtester()
results = backtester.run_backtest(
    data, 
    "ensemble", 
    strategy_weights={"rsi": 0.4, "macd": 0.4, "bollinger": 0.2}
)
```

## Troubleshooting

### Common Issues

1. **Missing Strategy Signals**
   - Ensure all referenced strategies are available
   - Check strategy names match exactly

2. **Zero Signals Generated**
   - Lower confidence threshold
   - Check individual strategy performance
   - Verify data quality

3. **Poor Performance**
   - Review strategy weights
   - Adjust combination method
   - Consider market regime changes

### Debug Mode

```python
import logging
logging.getLogger("trading.strategies.ensemble").setLevel(logging.DEBUG)

# Run ensemble with debug output
ensemble = create_rsi_macd_bollinger_ensemble()
combined = ensemble.combine_signals(strategy_signals)
```

## Performance Considerations

- **Computational Cost**: Linear with number of strategies
- **Memory Usage**: Moderate, scales with data size
- **Real-time Usage**: Suitable for live trading
- **Optimization**: Consider caching individual strategy results

## Future Enhancements

- **Machine Learning Integration**: Auto-optimize weights
- **Market Regime Detection**: Adaptive weight adjustment
- **Risk Parity**: Risk-based weight allocation
- **Multi-timeframe**: Combine signals from different timeframes 