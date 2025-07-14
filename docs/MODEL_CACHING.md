# Model Caching with Joblib

This document describes the joblib-based caching system implemented for long-running model operations in the Evolve trading platform.

## Overview

The caching system uses joblib's `Memory` class to cache expensive model operations like LSTM and XGBoost forecasts, significantly improving performance for repeated computations with the same inputs.

## Features

- **Automatic Caching**: Expensive model operations are automatically cached
- **Smart Key Generation**: Unique cache keys based on function name and arguments
- **DataFrame Support**: Proper handling of pandas DataFrames in cache keys
- **Cache Management**: Easy cache clearing and monitoring
- **Cross-Platform**: Works on Windows, macOS, and Linux

## Implementation

### Core Components

1. **ModelCache Class** (`utils/model_cache.py`)
   - Manages joblib Memory instance
   - Handles cache key generation
   - Provides cache management utilities

2. **Caching Decorators**
   - `@cache_model_operation`: Decorator for expensive functions
   - Automatic cache key generation
   - Transparent caching behavior

3. **Pre-built Cached Functions**
   - `cached_lstm_forecast()`: Cached LSTM forecasting
   - `cached_xgboost_forecast()`: Cached XGBoost forecasting
   - `cached_ensemble_forecast()`: Cached ensemble forecasting
   - `cached_tcn_forecast()`: Cached TCN forecasting

## Usage

### Basic Usage

```python
from utils.model_cache import cache_model_operation

@cache_model_operation
def expensive_model_operation(data, config, horizon=30):
    # Your expensive computation here
    return result
```

### Using Pre-built Cached Functions

```python
from utils.model_cache import cached_lstm_forecast, cached_xgboost_forecast

# LSTM forecast with caching
lstm_config = {
    'input_size': 4,
    'hidden_size': 32,
    'num_layers': 2,
    'dropout': 0.2,
    'sequence_length': 10,
    'feature_columns': ['close', 'volume', 'high', 'low'],
    'target_column': 'close'
}

result = cached_lstm_forecast(data, lstm_config, horizon=10)

# XGBoost forecast with caching
xgboost_config = {
    'auto_feature_engineering': False,
    'xgboost_params': {
        'n_estimators': 50,
        'max_depth': 4,
        'learning_rate': 0.1,
        'random_state': 42
    }
}

result = cached_xgboost_forecast(data, xgboost_config, horizon=10)
```

### Cache Management

```python
from utils.model_cache import get_model_cache, clear_model_cache, get_cache_info

# Get cache information
cache_info = get_cache_info()
print(f"Cache size: {cache_info['cache_size_mb']:.2f} MB")
print(f"Cache files: {cache_info['cache_files']}")

# Clear cache
clear_model_cache()

# Get cache instance for advanced usage
cache = get_model_cache()
```

## Model Integration

### LSTM Model

The LSTM model's `forecast()` method is automatically cached:

```python
from trading.models.lstm_model import LSTMForecaster

model = LSTMForecaster(config)
# First call - slow, cached
result1 = model.forecast(data, horizon=30)
# Second call - fast, uses cache
result2 = model.forecast(data, horizon=30)
```

### XGBoost Model

The XGBoost model's `forecast()` method is automatically cached:

```python
from trading.models.xgboost_model import XGBoostModel

model = XGBoostModel(config)
# First call - slow, cached
result1 = model.forecast(data, horizon=30)
# Second call - fast, uses cache
result2 = model.forecast(data, horizon=30)
```

### Ensemble Model

The ensemble model's `forecast()` method is automatically cached:

```python
from trading.models.ensemble_model import EnsembleModel

model = EnsembleModel(config)
# First call - slow, cached
result1 = model.forecast(data, horizon=30)
# Second call - fast, uses cache
result2 = model.forecast(data, horizon=30)
```

## Cache Configuration

### Cache Directory

By default, cache files are stored in `.cache/` directory. You can customize this:

```python
from utils.model_cache import ModelCache

# Custom cache directory
cache = ModelCache(cache_dir="my_cache", verbose=1)
```

### Cache Key Generation

Cache keys are automatically generated based on:
- Function name
- Function arguments (including DataFrames)
- Function keyword arguments
- Timestamp (for versioning)

### DataFrame Handling

DataFrames are properly hashed for cache key generation:
- Content-based hashing
- Handles NaN values
- Preserves data types
- Supports nested structures

## Performance Benefits

### Typical Performance Improvements

- **First Run**: Normal execution time
- **Cached Run**: 90-95% faster than first run
- **Memory Usage**: Minimal overhead for cache storage
- **Disk Usage**: Cache files typically 1-10 MB per cached operation

### When Caching Helps

- Repeated model training with same parameters
- Multiple forecasts with same data and configuration
- Hyperparameter tuning with repeated evaluations
- Backtesting with same model configurations

### When Caching Doesn't Help

- One-time model operations
- Constantly changing input data
- Very small datasets (overhead > benefit)
- Memory-constrained environments

## Best Practices

### Do's

- Use caching for expensive, deterministic operations
- Clear cache periodically to manage disk space
- Monitor cache size in production environments
- Use meaningful function names for better cache keys

### Don'ts

- Don't cache functions with side effects
- Don't cache functions with random components
- Don't cache very small operations (overhead > benefit)
- Don't rely on cache for critical data persistence

## Troubleshooting

### Common Issues

1. **Cache Not Working**
   - Check if cache directory exists and is writable
   - Verify function arguments are hashable
   - Check for import errors in cached functions

2. **Cache Size Too Large**
   - Clear cache periodically
   - Monitor cache usage with `get_cache_info()`
   - Consider using smaller cache directory

3. **Inconsistent Results**
   - Ensure cached functions are deterministic
   - Check for random number generation in cached functions
   - Verify input data consistency

### Debugging

```python
from utils.model_cache import get_model_cache

# Enable verbose logging
cache = get_model_cache()
cache.memory.verbose = 2

# Check cache hits/misses
print(cache.memory.cache_hits)
print(cache.memory.cache_misses)
```

## Future Enhancements

- **Distributed Caching**: Support for Redis or other distributed caches
- **Cache Compression**: Automatic compression of large cached objects
- **Cache Expiration**: Time-based cache invalidation
- **Cache Metrics**: Detailed performance metrics and monitoring
- **Cache Warming**: Pre-loading frequently used cached results

## Dependencies

- `joblib>=1.4.0`: Core caching functionality
- `pandas>=2.0.0`: DataFrame support
- `numpy>=1.20.0`: Array operations
- `hashlib`: Built-in Python module for hashing

## Installation

The caching system is included with the main requirements:

```bash
pip install joblib pandas numpy
```

No additional installation is required beyond the standard dependencies. 