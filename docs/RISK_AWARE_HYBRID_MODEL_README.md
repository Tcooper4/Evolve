# Risk-Aware Hybrid Model

## Overview

The Risk-Aware Hybrid Model is an enhanced ensemble forecasting system that uses risk-aware weighting instead of traditional MSE-based weighting. It allows users to select between Sharpe ratio, drawdown, or MSE as the primary weighting metric, making the ensemble more risk-aware and better suited for trading applications.

## Key Features

### 🎯 **Risk-Aware Weighting**

1. **Sharpe Ratio Weighting**
   - Formula: `weight = Sharpe / total_Sharpe`
   - Favors models with better risk-adjusted returns
   - Ideal for maximizing risk-adjusted performance

2. **Drawdown Weighting**
   - Formula: `weight = (1 + drawdown) / total`
   - Favors models with lower maximum drawdown
   - Ideal for risk-averse strategies

3. **MSE Weighting**
   - Formula: `weight = (1/MSE) / total(1/MSE)`
   - Traditional accuracy-based weighting
   - Favors models with lower prediction error

### 📊 **Enhanced Performance Metrics**

- **Sharpe Ratio**: Risk-adjusted return calculation
- **Win Rate**: Percentage of profitable predictions
- **Maximum Drawdown**: Worst peak-to-trough decline
- **MSE**: Mean squared error for accuracy
- **Total Return**: Cumulative return from predictions

### 🎛️ **User-Selectable Configuration**

- Interactive weighting metric selection
- Configurable thresholds and floors
- Real-time weight updates
- Performance visualization
- Configuration validation

## Architecture

### Core Components

```
trading/forecasting/
├── hybrid_model.py              # Enhanced hybrid model with risk-aware weighting
└── hybrid_model_selector.py     # Model selection utilities

trading/ui/
└── hybrid_model_config.py       # Streamlit UI for configuration
```

### Weighting Methods

1. **Risk-Aware** (Default): Single metric weighting
2. **Weighted Average**: Multi-metric composite scoring
3. **AHP**: Analytic Hierarchy Process
4. **Composite**: Trend-adjusted scoring

## Usage

### Basic Usage

```python
from trading.forecasting.hybrid_model import HybridModel

# Create models dictionary
models = {
    "LSTM": lstm_model,
    "XGBoost": xgb_model,
    "Transformer": transformer_model
}

# Initialize hybrid model
hybrid_model = HybridModel(models)

# Fit models and calculate performance
hybrid_model.fit(data)

# Set weighting metric
hybrid_model.set_weighting_metric("sharpe")  # "sharpe", "drawdown", or "mse"

# Make predictions
predictions = hybrid_model.predict(data)
```

### Streamlit Integration

```python
import streamlit as st
from trading.ui.hybrid_model_config import render_hybrid_model_config_sidebar

# Render configuration in sidebar
config = render_hybrid_model_config_sidebar(hybrid_model)

# Use the configured model
if st.button("Run Prediction"):
    prediction = hybrid_model.predict(data)
    st.line_chart(prediction)
```

## Weighting Metrics

### Sharpe Ratio Weighting

**Formula**: `weight = Sharpe / total_Sharpe`

**Advantages**:
- Maximizes risk-adjusted returns
- Accounts for both return and volatility
- Industry standard for performance evaluation

**Best For**:
- Long-term investment strategies
- Risk-adjusted performance optimization
- Institutional trading

**Configuration**:
```python
hybrid_model.scoring_config["sharpe_floor"] = 0.0  # Minimum Sharpe ratio
```

### Drawdown Weighting

**Formula**: `weight = (1 + drawdown) / total`

**Advantages**:
- Minimizes portfolio risk
- Protects against large losses
- Conservative approach

**Best For**:
- Risk-averse strategies
- Capital preservation
- Conservative portfolios

**Configuration**:
```python
hybrid_model.scoring_config["drawdown_ceiling"] = -0.5  # Maximum drawdown
```

### MSE Weighting

**Formula**: `weight = (1/MSE) / total(1/MSE)`

**Advantages**:
- Traditional accuracy-based approach
- Simple and intuitive
- Good for prediction accuracy

**Best For**:
- Pure forecasting applications
- Academic research
- When accuracy is primary concern

**Configuration**:
```python
hybrid_model.scoring_config["mse_ceiling"] = 1000.0  # Maximum MSE
```

## Configuration Options

### Scoring Configuration

```python
scoring_config = {
    "method": "risk_aware",           # "risk_aware", "weighted_average", "ahp", "composite"
    "weighting_metric": "sharpe",     # "sharpe", "drawdown", "mse"
    "min_performance_threshold": 0.1, # Minimum performance to avoid zero weights
    "recency_weight": 0.7,            # Weight for recent vs historical performance
    "risk_free_rate": 0.02,           # Risk-free rate for Sharpe calculations
    "sharpe_floor": 0.0,              # Minimum Sharpe ratio
    "drawdown_ceiling": -0.5,         # Maximum drawdown threshold
    "mse_ceiling": 1000.0             # Maximum MSE threshold
}
```

### Performance Metrics

```python
metrics = {
    "sharpe_ratio": {"weight": 0.4, "direction": "maximize"},
    "win_rate": {"weight": 0.3, "direction": "maximize"},
    "max_drawdown": {"weight": 0.2, "direction": "minimize"},
    "mse": {"weight": 0.1, "direction": "minimize"},
    "total_return": {"weight": 0.0, "direction": "maximize"}
}
```

## Performance Analysis

### Model Performance Summary

```python
summary = hybrid_model.get_model_performance_summary()

for model_name, model_info in summary.items():
    if model_info["status"] == "active":
        print(f"Model: {model_name}")
        print(f"  Weight: {model_info['current_weight']:.2%}")
        print(f"  Avg Sharpe: {model_info['avg_metrics']['sharpe_ratio']:.3f}")
        print(f"  Avg Win Rate: {model_info['avg_metrics']['win_rate']:.1%}")
        print(f"  Avg Max Drawdown: {model_info['avg_metrics']['max_drawdown']:.1%}")
```

### Weighting Metric Information

```python
info = hybrid_model.get_weighting_metric_info()

print(f"Current Metric: {info['current_metric']}")
print(f"Current Method: {info['current_method']}")

for metric, details in info['available_metrics'].items():
    print(f"{metric}: {details['description']}")
```

## UI Components

### Configuration Sidebar

The Streamlit sidebar component provides:

1. **Ensemble Method Selection**: Choose weighting method
2. **Weighting Metric Selection**: Select primary metric for risk-aware weighting
3. **Metric-Specific Parameters**: Configure thresholds and floors
4. **Advanced Settings**: Performance thresholds and recency weights
5. **Performance Summary**: Real-time model weights and metrics
6. **Configuration Validation**: Warnings for unreasonable parameters

### Performance Dashboard

The performance dashboard shows:

1. **Model Performance Table**: Comprehensive metrics for each model
2. **Weight Comparison**: Visual comparison of different weighting methods
3. **Performance Visualization**: Charts and graphs of model performance
4. **Recommendations**: AI-powered suggestions for configuration

## Example Scenarios

### Scenario 1: Conservative Portfolio

```python
# Use drawdown weighting for risk-averse strategy
hybrid_model.set_weighting_metric("drawdown")
hybrid_model.scoring_config["drawdown_ceiling"] = -0.3  # Conservative threshold

# This will favor models with lower maximum drawdown
```

### Scenario 2: Aggressive Growth

```python
# Use Sharpe ratio weighting for risk-adjusted growth
hybrid_model.set_weighting_metric("sharpe")
hybrid_model.scoring_config["sharpe_floor"] = 0.5  # Higher minimum Sharpe

# This will favor models with better risk-adjusted returns
```

### Scenario 3: Pure Forecasting

```python
# Use MSE weighting for accuracy-focused applications
hybrid_model.set_weighting_metric("mse")
hybrid_model.scoring_config["mse_ceiling"] = 500.0  # Lower MSE threshold

# This will favor models with lower prediction error
```

## Integration with Existing Systems

### Backtester Integration

```python
from trading.backtesting.backtester import Backtester

# Create hybrid model
hybrid_model = HybridModel(models)
hybrid_model.set_weighting_metric("sharpe")

# Use in backtester
backtester = Backtester(data=data, initial_cash=100000)
# ... run backtest with hybrid model predictions
```

### Strategy Pipeline Integration

```python
from trading.strategies.strategy_pipeline import StrategyPipeline

# Create strategy pipeline with hybrid model
pipeline = StrategyPipeline()
pipeline.add_hybrid_model(hybrid_model)

# Run with risk-aware weighting
results = pipeline.run_backtest(data)
```

## Best Practices

### 1. Weighting Metric Selection

- **Sharpe Ratio**: Use for long-term, risk-adjusted strategies
- **Drawdown**: Use for conservative, capital-preservation strategies
- **MSE**: Use for pure forecasting or academic applications

### 2. Configuration Tuning

- Start with default parameters
- Adjust thresholds based on your risk tolerance
- Monitor performance and adjust accordingly
- Use cross-validation to validate settings

### 3. Model Selection

- Include diverse model types (LSTM, XGBoost, Transformer)
- Ensure models have different characteristics
- Monitor model performance over time
- Remove consistently underperforming models

### 4. Performance Monitoring

- Track weight changes over time
- Monitor ensemble performance vs individual models
- Validate against out-of-sample data
- Adjust configuration based on market conditions

## Testing and Validation

### Unit Tests

```bash
python -m pytest tests/test_enhanced_cost_modeling.py -v
```

### Example Script

```bash
python examples/risk_aware_hybrid_model_example.py
```

### Validation Checklist

- [ ] Weighting metrics calculate correctly
- [ ] Weights sum to 1.0
- [ ] Performance metrics are reasonable
- [ ] UI components work properly
- [ ] Configuration validation functions
- [ ] Integration with existing systems

## Troubleshooting

### Common Issues

1. **Weights Not Updating**
   - Check if models have performance data
   - Verify weighting metric is set correctly
   - Ensure minimum performance threshold is appropriate

2. **Extreme Weight Values**
   - Check performance metric calculations
   - Verify thresholds and floors
   - Review model performance data

3. **Poor Ensemble Performance**
   - Check individual model performance
   - Verify weighting metric suitability
   - Consider adjusting thresholds

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger('trading.forecasting.hybrid_model').setLevel(logging.DEBUG)
```

## Future Enhancements

### Planned Features

1. **Dynamic Weighting**
   - Market condition-based metric selection
   - Adaptive threshold adjustment
   - Real-time performance monitoring

2. **Advanced Metrics**
   - Sortino ratio weighting
   - Calmar ratio weighting
   - Information ratio weighting

3. **Machine Learning Integration**
   - ML-based metric selection
   - Automated parameter optimization
   - Performance prediction

4. **Enhanced Visualization**
   - Interactive weight evolution charts
   - Performance attribution analysis
   - Risk decomposition

## Conclusion

The Risk-Aware Hybrid Model provides a sophisticated approach to ensemble forecasting that goes beyond traditional MSE-based weighting. By incorporating risk-aware metrics like Sharpe ratio and drawdown, it enables more intelligent model combination that aligns with trading objectives.

The user-selectable weighting metrics and comprehensive configuration options make it suitable for a wide range of applications, from conservative capital preservation to aggressive growth strategies. The integration with Streamlit provides an intuitive interface for configuration and monitoring.

For questions or support, please refer to the main documentation or create an issue in the project repository. 