# Enhanced Optuna Tuner for Trading Models

## Overview

The Enhanced Optuna Tuner is a comprehensive hyperparameter optimization system designed specifically for trading models. It uses **Sharpe ratio** as the primary objective function, making it ideal for financial forecasting and trading strategy optimization.

## Key Features

### 🎯 Sharpe Ratio Optimization
- **Primary Objective**: Maximizes Sharpe ratio instead of traditional error metrics
- **Trading-Focused**: Evaluates models based on trading performance
- **Risk-Adjusted Returns**: Considers both returns and volatility

### 🤖 Multi-Model Support
- **LSTM**: Optimizes `num_layers`, `dropout`, `learning_rate`, `lookback`
- **XGBoost**: Optimizes `max_depth`, `learning_rate`, `n_estimators`
- **Transformer**: Optimizes `d_model`, `num_heads`, `ff_dim`, `dropout`

### 🔄 Integration Ready
- **Forecasting Pipeline**: Seamless integration with existing forecasting system
- **Model Selection**: Automatic selection of best performing model
- **Performance Tracking**: Comprehensive result storage and analysis

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   OptunaTuner   │───▶│ ForecastingOpt   │───▶│ Model Selection │
│                 │    │                  │    │                 │
│ • LSTM Opt      │    │ • Data Prep      │    │ • Best Model    │
│ • XGBoost Opt   │    │ • Auto Selection │    │ • Parameters    │
│ • Transformer   │    │ • Integration    │    │ • Performance   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Installation

```bash
# Install required dependencies
pip install optuna pandas numpy scikit-learn xgboost torch

# For GPU support (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Basic Usage

```python
from trading.optimization.optuna_tuner import SharpeOptunaTuner
import pandas as pd

# Create tuner
tuner = SharpeOptunaTuner(
    study_name="my_optimization",
    n_trials=100,
    timeout=3600  # 1 hour
)

# Optimize LSTM
result = tuner.optimize_lstm(
    data=your_data,
    target_column='price',
    feature_columns=['volume', 'rsi', 'sma']
)

print(f"Best Sharpe ratio: {result['best_score']:.4f}")
print(f"Best parameters: {result['best_params']}")
```

### All Models Optimization

```python
# Optimize all model types
result = tuner.optimize_all_models(
    data=your_data,
    target_column='price',
    model_types=['lstm', 'xgboost', 'transformer']
)

print(f"Best model: {result['best_model']}")
print(f"Best Sharpe: {result['best_score']:.4f}")
```

### Forecasting Integration

```python
from trading.optimization.forecasting_integration import ForecastingOptimizer

# Create optimizer
optimizer = ForecastingOptimizer(
    optimization_config={
        'n_trials': 50,
        'model_types': ['lstm', 'xgboost'],
        'min_sharpe_threshold': 0.1
    }
)

# Optimize for forecasting
result = optimizer.optimize_for_forecasting(
    data=your_data,
    target_column='price',
    forecast_horizon=30
)

# Get optimized model
model, params = optimizer.get_optimized_model(
    model_type=result['recommendation']['recommended_model'],
    data=your_data,
    target_column='price'
)
```

## Model-Specific Optimization

### LSTM Optimization

```python
# LSTM hyperparameters
lstm_params = {
    'num_layers': [1, 2, 3, 4],           # Number of LSTM layers
    'hidden_size': [32, 64, 128, 256],    # Hidden layer size
    'dropout': [0.1, 0.2, 0.3, 0.5],     # Dropout rate
    'learning_rate': [1e-4, 1e-3, 1e-2], # Learning rate
    'lookback': [10, 20, 30, 60],        # Sequence length
    'batch_size': [16, 32, 64, 128],     # Batch size
    'epochs': [50, 100, 200]             # Training epochs
}

result = tuner.optimize_lstm(data, 'price')
```

### XGBoost Optimization

```python
# XGBoost hyperparameters
xgboost_params = {
    'max_depth': [3, 6, 9, 12],          # Maximum tree depth
    'learning_rate': [0.01, 0.1, 0.3],   # Learning rate
    'n_estimators': [50, 100, 300, 500], # Number of trees
    'subsample': [0.6, 0.8, 1.0],        # Subsample ratio
    'colsample_bytree': [0.6, 0.8, 1.0], # Column subsample ratio
    'reg_alpha': [0, 0.1, 1.0],          # L1 regularization
    'reg_lambda': [0, 0.1, 1.0]          # L2 regularization
}

result = tuner.optimize_xgboost(data, 'price')
```

### Transformer Optimization

```python
# Transformer hyperparameters
transformer_params = {
    'd_model': [64, 128, 256, 512],      # Model dimension
    'num_heads': [2, 4, 8, 16],          # Number of attention heads
    'ff_dim': [128, 256, 512, 1024],     # Feed-forward dimension
    'dropout': [0.1, 0.2, 0.3, 0.5],    # Dropout rate
    'num_layers': [1, 2, 4, 6],          # Number of layers
    'learning_rate': [1e-4, 1e-3, 1e-2], # Learning rate
    'batch_size': [16, 32, 64, 128]      # Batch size
}

result = tuner.optimize_transformer(data, 'price')
```

## Performance Metrics

The tuner evaluates models using comprehensive trading metrics:

### Primary Metrics
- **Sharpe Ratio**: Risk-adjusted return measure
- **Win Rate**: Percentage of profitable trades
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Total Return**: Cumulative return over the period

### Secondary Metrics
- **Directional Accuracy**: Correct direction predictions
- **MSE/RMSE/MAE**: Traditional error metrics
- **Volatility**: Return volatility

## Configuration Options

### Tuner Configuration

```python
tuner = OptunaTuner(
    study_name="custom_study",           # Study name
    n_trials=100,                       # Number of trials
    timeout=3600,                       # Timeout in seconds
    validation_split=0.2,               # Validation split
    random_state=42,                    # Random seed
    storage="sqlite:///optuna.db",      # Storage backend
    pruner=MedianPruner()               # Pruning strategy
)
```

### Optimization Configuration

```python
config = {
    'n_trials': 50,                     # Number of trials
    'timeout': 1800,                    # Timeout (30 min)
    'validation_split': 0.2,            # Validation split
    'model_types': ['lstm', 'xgboost'], # Models to optimize
    'min_sharpe_threshold': 0.1,        # Minimum Sharpe ratio
    'auto_optimize': True,              # Auto-optimization
    'save_results': True                # Save results
}
```

## Integration with Forecasting Pipeline

### Automatic Integration

```python
from trading.optimization.forecasting_integration import integrate_with_forecasting_pipeline

# Integrate with existing pipeline
result = integrate_with_forecasting_pipeline(
    data=your_data,
    target_column='price',
    forecast_horizon=30,
    auto_optimize=True
)

if result['success']:
    model = result['model']
    params = result['parameters']
    print(f"Selected model: {result['model_type']}")
```

### Manual Integration

```python
# Create optimizer
optimizer = ForecastingOptimizer()

# Optimize for specific horizon
result = optimizer.optimize_for_forecasting(
    data=your_data,
    target_column='price',
    forecast_horizon=30
)

# Get optimized model
model, params = optimizer.get_optimized_model(
    model_type=result['recommendation']['recommended_model'],
    data=your_data,
    target_column='price'
)

# Evaluate performance
performance = optimizer.evaluate_model_performance(
    model=model,
    data=test_data,
    target_column='price',
    model_type='lstm'
)
```

## Results Management

### Saving Results

```python
# Save optimization results
results_file = tuner.save_results("my_optimization_results.pkl")
print(f"Results saved to: {results_file}")
```

### Loading Results

```python
# Load previous results
loaded_results = tuner.load_results("my_optimization_results.pkl")

# Get best parameters
best_params = tuner.get_best_params('lstm')
best_score = tuner.get_best_score('lstm')
```

### Results Analysis

```python
# Get optimization summary
summary = optimizer.get_optimization_summary()
print(f"Total optimizations: {summary['total_optimizations']}")
print(f"Best models: {summary['best_models']}")
```

## Advanced Features

### Custom Objective Functions

```python
def custom_objective(trial):
    # Define custom hyperparameters
    params = {
        'param1': trial.suggest_float('param1', 0.1, 1.0),
        'param2': trial.suggest_int('param2', 1, 10)
    }
    
    # Train model and calculate custom metric
    model = create_model(params)
    metric = evaluate_model(model)
    
    return -metric  # Negative for maximization
```

### Pruning Strategies

```python
from optuna.pruners import MedianPruner, HyperbandPruner

# Use different pruning strategies
tuner = OptunaTuner(
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
)
```

### Storage Backends

```python
# SQLite storage
tuner = OptunaTuner(storage="sqlite:///optuna.db")

# PostgreSQL storage
tuner = OptunaTuner(storage="postgresql://user:pass@localhost/optuna")

# Redis storage
tuner = OptunaTuner(storage="redis://localhost:6379")
```

## Best Practices

### 1. Data Preparation
- Ensure sufficient data for validation split
- Handle missing values appropriately
- Scale features if necessary
- Add relevant technical indicators

### 2. Hyperparameter Ranges
- Start with wide ranges for exploration
- Narrow ranges based on initial results
- Consider model-specific constraints
- Use log-scale for learning rates

### 3. Optimization Strategy
- Use appropriate number of trials
- Set reasonable timeout limits
- Monitor optimization progress
- Save results regularly

### 4. Model Selection
- Consider multiple metrics, not just Sharpe ratio
- Validate on out-of-sample data
- Test model stability over time
- Monitor for overfitting

## Troubleshooting

### Common Issues

1. **Low Sharpe Ratios**
   - Check data quality and preprocessing
   - Verify feature engineering
   - Consider different model types
   - Adjust hyperparameter ranges

2. **Optimization Timeouts**
   - Reduce number of trials
   - Use faster model configurations
   - Implement early stopping
   - Use pruning strategies

3. **Memory Issues**
   - Reduce batch sizes
   - Use smaller model architectures
   - Implement data streaming
   - Use GPU acceleration

4. **Import Errors**
   - Install required dependencies
   - Check model import paths
   - Verify Python environment
   - Update package versions

### Performance Tips

1. **GPU Acceleration**
   ```python
   # Enable GPU for PyTorch models
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model.to(device)
   ```

2. **Parallel Optimization**
   ```python
   # Use multiple workers
   study.optimize(objective, n_trials=100, n_jobs=4)
   ```

3. **Caching**
   ```python
   # Cache model results
   @functools.lru_cache(maxsize=128)
   def cached_model_evaluation(params):
       return evaluate_model(params)
   ```

## Examples

See the `examples/optuna_tuner_example.py` file for comprehensive examples demonstrating:

- Individual model optimization
- All models optimization
- Forecasting integration
- Pipeline integration
- Results analysis
- Performance evaluation

## API Reference

### OptunaTuner

#### Methods
- `optimize_lstm(data, target_column, feature_columns)`
- `optimize_xgboost(data, target_column, feature_columns)`
- `optimize_transformer(data, target_column, feature_columns)`
- `optimize_all_models(data, target_column, model_types)`
- `save_results(filename)`
- `load_results(filepath)`
- `get_best_params(model_type)`
- `get_best_score(model_type)`
- `get_model_recommendation(data, target_column)`

### ForecastingOptimizer

#### Methods
- `optimize_for_forecasting(data, target_column, forecast_horizon)`
- `get_optimized_model(model_type, data, target_column)`
- `evaluate_model_performance(model, data, target_column, model_type)`
- `get_optimization_summary()`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:
- Check the documentation
- Review examples
- Open an issue on GitHub
- Contact the development team 