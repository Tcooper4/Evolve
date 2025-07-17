# Enhanced ARIMA Model with Auto_arima Optimization

## Overview

The enhanced ARIMA model (`trading/models/arima_model.py`) has been upgraded to use `pmdarima.auto_arima` for automatic parameter selection, with support for multiple optimization criteria and seasonal component control.

## Key Features

### ✅ Automatic Parameter Selection
- Uses `pmdarima.auto_arima` to automatically find optimal ARIMA parameters
- Replaces manual parameter specification with intelligent selection
- Handles both non-seasonal and seasonal ARIMA models

### ✅ Seasonal Component Control
- `seasonal=True/False` flag to control seasonal components
- Automatic seasonal period detection (default: 12 for monthly data)
- Configurable seasonal parameters (P, D, Q, m)

### ✅ Multiple Optimization Criteria
- **AIC (Akaike Information Criterion)** - Default, balances fit and complexity
- **BIC (Bayesian Information Criterion)** - More conservative than AIC
- **MSE (Mean Squared Error)** - Uses backtesting for optimization
- **RMSE (Root Mean Squared Error)** - Uses backtesting for optimization

### ✅ Backtesting for MSE/RMSE Optimization
- Performs grid search with backtesting for MSE/RMSE optimization
- Configurable backtest steps (default: 5)
- Robust error handling with fallback to default auto_arima

### ✅ Fallback Mechanisms
- Falls back to manual ARIMA if pmdarima is not available
- Falls back to non-seasonal ARIMA if seasonal fitting fails
- Comprehensive error handling and logging

## Configuration Options

### Basic Configuration
```python
config = {
    "use_auto_arima": True,        # Enable auto_arima (default: True)
    "seasonal": True,              # Enable seasonal components (default: True)
    "optimization_criterion": "aic", # AIC, BIC, MSE, or RMSE (default: AIC)
    "backtest_steps": 5,           # Steps for MSE/RMSE optimization (default: 5)
}
```

### Advanced Auto_arima Configuration
```python
config = {
    "use_auto_arima": True,
    "seasonal": True,
    "optimization_criterion": "aic",
    "auto_arima_config": {
        "start_p": 0,              # Starting p value
        "start_q": 0,              # Starting q value
        "max_p": 5,                # Maximum p value
        "max_q": 5,                # Maximum q value
        "max_d": 2,                # Maximum d value
        "m": 12,                   # Seasonal period
        "D": 1,                    # Seasonal differencing
        "trace": True,             # Show optimization progress
        "stepwise": True,          # Use stepwise search
        "random_state": 42         # For reproducibility
    }
}
```

## Usage Examples

### 1. AIC Optimization with Seasonal Components
```python
from trading.models.arima_model import ARIMAModel

config = {
    "use_auto_arima": True,
    "seasonal": True,
    "optimization_criterion": "aic",
    "auto_arima_config": {
        "max_p": 3, "max_q": 3, "max_d": 2,
        "trace": True
    }
}

model = ARIMAModel(config)
result = model.fit(data)

if result['success']:
    print(f"Order: {result['order']}")
    print(f"Seasonal Order: {result['seasonal_order']}")
    print(f"AIC: {result['aic']:.2f}")
```

### 2. BIC Optimization without Seasonal Components
```python
config = {
    "use_auto_arima": True,
    "seasonal": False,
    "optimization_criterion": "bic",
    "auto_arima_config": {
        "max_p": 3, "max_q": 3, "max_d": 2
    }
}

model = ARIMAModel(config)
result = model.fit(data)
```

### 3. MSE Optimization with Backtesting
```python
config = {
    "use_auto_arima": True,
    "seasonal": True,
    "optimization_criterion": "mse",
    "backtest_steps": 10,
    "auto_arima_config": {
        "max_p": 2, "max_q": 2, "max_d": 1
    }
}

model = ARIMAModel(config)
result = model.fit(data)
```

### 4. RMSE Optimization without Seasonal Components
```python
config = {
    "use_auto_arima": True,
    "seasonal": False,
    "optimization_criterion": "rmse",
    "backtest_steps": 10,
    "auto_arima_config": {
        "max_p": 2, "max_q": 2, "max_d": 1
    }
}

model = ARIMAModel(config)
result = model.fit(data)
```

### 5. Manual ARIMA (Fallback)
```python
config = {
    "use_auto_arima": False,
    "order": (1, 1, 1),
    "seasonal_order": None
}

model = ARIMAModel(config)
result = model.fit(data)
```

## Optimization Criteria Comparison

### AIC (Akaike Information Criterion)
- **Use case**: General model selection
- **Pros**: Balances model fit and complexity
- **Cons**: May overfit with large datasets
- **Best for**: Most general forecasting tasks

### BIC (Bayesian Information Criterion)
- **Use case**: Conservative model selection
- **Pros**: More conservative than AIC, less overfitting
- **Cons**: May underfit with small datasets
- **Best for**: Large datasets, avoiding overfitting

### MSE (Mean Squared Error)
- **Use case**: Direct forecast accuracy optimization
- **Pros**: Directly optimizes for forecast accuracy
- **Cons**: Computationally expensive (requires backtesting)
- **Best for**: When forecast accuracy is the primary concern

### RMSE (Root Mean Squared Error)
- **Use case**: Direct forecast accuracy optimization (scaled)
- **Pros**: Same scale as original data, direct accuracy optimization
- **Cons**: Computationally expensive (requires backtesting)
- **Best for**: When forecast accuracy is the primary concern

## Backtesting Process

When using MSE or RMSE optimization, the model performs the following steps:

1. **Grid Search**: Tests different parameter combinations
2. **Train/Test Split**: Uses last `backtest_steps` for validation
3. **Model Fitting**: Fits model on training data
4. **Prediction**: Makes predictions on test data
5. **Error Calculation**: Computes MSE or RMSE
6. **Parameter Selection**: Chooses parameters with lowest error

## Error Handling

The enhanced ARIMA model includes robust error handling:

- **pmdarima Import Error**: Falls back to manual ARIMA
- **Seasonal Fitting Error**: Falls back to non-seasonal ARIMA
- **Backtesting Error**: Falls back to default auto_arima
- **Parameter Validation**: Validates optimization criteria
- **Comprehensive Logging**: Detailed error messages and warnings

## Performance Considerations

### Computational Cost
- **AIC/BIC**: Fastest (uses built-in optimization)
- **MSE/RMSE**: Slowest (requires grid search with backtesting)
- **Seasonal**: More expensive than non-seasonal

### Memory Usage
- **Backtesting**: Higher memory usage due to multiple model fits
- **Large Parameter Ranges**: Increases memory usage
- **Long Time Series**: May require more memory

### Recommendations
- Use AIC/BIC for quick model selection
- Use MSE/RMSE for final model optimization
- Limit parameter ranges for faster computation
- Consider seasonal=False for large datasets

## Testing

Run the test script to verify functionality:

```bash
python test_enhanced_arima.py
```

Or run the example:

```bash
python examples/enhanced_arima_example.py
```

## Dependencies

Required packages:
- `pmdarima` - For auto_arima functionality
- `statsmodels` - For ARIMA model implementation
- `pandas` - For data handling
- `numpy` - For numerical operations

Install with:
```bash
pip install pmdarima statsmodels pandas numpy
```

## Migration from Legacy ARIMA

The enhanced ARIMA model is backward compatible. Existing code will continue to work, but you can now take advantage of the new features:

```python
# Legacy usage (still works)
model = ARIMAModel({"order": (1, 1, 1)})

# Enhanced usage
model = ARIMAModel({
    "use_auto_arima": True,
    "seasonal": True,
    "optimization_criterion": "aic"
})
```

## Future Enhancements

Potential future improvements:
- Cross-validation for more robust parameter selection
- Parallel processing for faster grid search
- Additional optimization criteria (MAE, MAPE)
- Automatic seasonal period detection
- Integration with other time series models 