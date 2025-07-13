# Safe JSON Saving

This document describes the safe JSON saving utilities that prevent accidental data loss by protecting against overwriting important historical data with empty or invalid data.

## Overview

The safe JSON saving utilities provide a defensive programming approach to data persistence, ensuring that critical historical data is never accidentally wiped by empty or invalid data structures.

## Key Features

### 1. Empty Data Protection
- Prevents saving `None`, empty dictionaries `{}`, or empty lists `[]`
- Configurable minimum data size requirements
- Validates data structure before saving

### 2. Automatic Backup Creation
- Creates `.backup` files before overwriting existing data
- Preserves previous versions for recovery
- Configurable backup behavior

### 3. Data Validation
- Built-in validation for historical data patterns
- Custom validation function support
- Comprehensive error reporting

### 4. Comprehensive Logging
- Detailed success/failure logging
- Error messages with context
- Debug information for troubleshooting

## Usage

### Basic Safe Saving

```python
from utils.safe_json_saver import safe_json_save

# Valid data - will save successfully
data = {"timestamp": "2024-01-15", "metrics": {"sharpe": 1.25}}
result = safe_json_save(data, "performance.json")
print(result)  # {'success': True, 'filepath': 'performance.json'}

# Empty data - will be prevented
empty_data = {}
result = safe_json_save(empty_data, "performance.json")
print(result)  # {'success': False, 'error': 'Empty data detected - skipping save to prevent data loss'}
```

### Historical Data Saving

```python
from utils.safe_json_saver import safe_save_historical_data

historical_data = {
    "performance_history": [
        {
            "timestamp": "2024-01-15T10:30:00",
            "metric_name": "sharpe_ratio",
            "value": 1.25
        }
    ],
    "trends": {"sharpe_ratio": {"direction": "improving"}}
}

result = safe_save_historical_data(historical_data, "data/performance_history.json")
```

### Custom Validation

```python
from utils.safe_json_saver import safe_json_save_with_validation

def validate_trading_data(data):
    """Custom validation for trading data."""
    if not data:
        return {'valid': False, 'error': 'Data is empty'}
    
    required_fields = ['symbol', 'price', 'volume']
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        return {'valid': False, 'error': f'Missing required fields: {missing_fields}'}
    
    return {'valid': True}

trading_data = {"symbol": "AAPL", "price": 150.25, "volume": 1000}
result = safe_json_save_with_validation(
    trading_data, 
    "trading_data.json",
    validation_func=validate_trading_data
)
```

## API Reference

### `safe_json_save(data, filepath, **kwargs)`

Safely save JSON data with protection against overwriting with empty data.

**Parameters:**
- `data`: Data to save (will be converted to JSON)
- `filepath`: Path to save the JSON file
- `indent`: JSON indentation (default: 2)
- `default`: Function to handle non-serializable objects
- `backup_existing`: Whether to backup existing file (default: True)
- `min_data_size`: Minimum size of data to consider non-empty (default: 1)

**Returns:**
- Dictionary with operation result including success status and error messages

### `safe_save_historical_data(data, filepath, **kwargs)`

Safely save historical data with built-in validation.

**Parameters:**
- `data`: Historical data to save
- `filepath`: Path to save the JSON file
- `**kwargs`: Additional arguments for `safe_json_save`

**Returns:**
- Dictionary with operation result

### `safe_json_save_with_validation(data, filepath, validation_func, **kwargs)`

Save JSON data with additional custom validation.

**Parameters:**
- `data`: Data to save
- `filepath`: Path to save the JSON file
- `validation_func`: Optional function to validate data before saving
- `**kwargs`: Additional arguments for `safe_json_save`

**Returns:**
- Dictionary with operation result

### `validate_historical_data(data)`

Built-in validation function for historical data.

**Parameters:**
- `data`: Data to validate

**Returns:**
- Dictionary with validation result: `{'valid': bool, 'error': str}`

## Implementation in Critical Files

The safe JSON saving utilities have been integrated into the following critical files:

### 1. Model Monitoring (`trading/memory/model_monitor.py`)
- Protects model parameter history and drift alerts
- Prevents loss of model performance tracking data

### 2. Performance Tracking (`trading/memory/long_term_performance_tracker.py`)
- Safeguards performance metrics and trends
- Protects historical performance analysis data

### 3. Agent Leaderboard (`trading/agents/agent_leaderboard.py`)
- Protects agent performance rankings
- Safeguards agent history and statistics

### 4. Strategy Logging (`trading/memory/strategy_logger.py`)
- Protects strategy-regime mappings
- Safeguards strategy decision history

## Error Handling

The utilities provide comprehensive error handling:

```python
result = safe_save_historical_data(data, "important_data.json")

if result['success']:
    print(f"Data saved successfully to {result['filepath']}")
else:
    print(f"Failed to save data: {result['error']}")
    # Handle the error appropriately
```

## Best Practices

### 1. Always Check Results
```python
result = safe_json_save(data, filepath)
if not result['success']:
    logger.error(f"Failed to save data: {result['error']}")
    # Implement appropriate error handling
```

### 2. Use Appropriate Validation
```python
# For historical data
safe_save_historical_data(data, filepath)

# For custom data types
safe_json_save_with_validation(data, filepath, custom_validation_func)
```

### 3. Configure Backup Settings
```python
# Disable backups for temporary data
safe_json_save(data, filepath, backup_existing=False)

# Enable backups for critical data (default)
safe_json_save(data, filepath, backup_existing=True)
```

### 4. Set Minimum Data Size Requirements
```python
# Require at least 5 items for meaningful data
safe_json_save(data, filepath, min_data_size=5)
```

## Migration Guide

To migrate existing code to use safe JSON saving:

### Before (Unsafe)
```python
with open("data.json", "w") as f:
    json.dump(data, f, indent=2)
```

### After (Safe)
```python
from utils.safe_json_saver import safe_json_save

result = safe_json_save(data, "data.json")
if not result['success']:
    logger.error(f"Failed to save data: {result['error']}")
```

## Testing

Run the example script to see the utilities in action:

```bash
python examples/safe_json_saving_example.py
```

This will demonstrate:
- Basic safe saving functionality
- Historical data validation
- Custom validation functions
- Backup protection
- Minimum data size requirements

## Benefits

1. **Data Protection**: Prevents accidental loss of critical historical data
2. **Error Prevention**: Catches data issues before they cause problems
3. **Recovery Options**: Automatic backups provide recovery capabilities
4. **Debugging**: Comprehensive logging helps identify issues
5. **Flexibility**: Configurable validation and backup options

## Future Enhancements

Potential future improvements:
- Database integration for backup storage
- Compression for backup files
- Scheduled backup rotation
- Data integrity checksums
- Integration with monitoring systems 