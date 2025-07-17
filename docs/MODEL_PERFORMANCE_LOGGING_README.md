# Model Performance Logging System

## Overview

The Model Performance Logging System provides comprehensive tracking and analysis of model performance metrics across different tickers and time periods. It automatically stores performance data in both JSON and CSV formats, tracks best historical models per ticker, and provides a rich UI for analysis and visualization.

## Features

- **📊 Performance Logging**: Log model performance metrics with timestamps
- **🏆 Best Models Tracking**: Automatically track best models for each metric per ticker
- **📈 Historical Analysis**: Query and filter performance history by ticker, model, and time period
- **🎯 UI Dashboard**: Comprehensive Streamlit dashboard for visualization and analysis
- **📋 Data Export**: Export performance data in CSV format
- **🔍 Advanced Filtering**: Filter by ticker, model, date range, and performance metrics

## Quick Start

### 1. Basic Usage

```python
from memory.model_log import log_model_performance

# Log model performance
log_model_performance(
    model_name="LSTM_v1",
    ticker="AAPL",
    sharpe=1.85,
    mse=0.0234,
    drawdown=-0.12,
    total_return=0.25,
    win_rate=0.68,
    accuracy=0.72,
    notes="LSTM model with 50 epochs"
)
```

### 2. View Performance Dashboard

```bash
streamlit run pages/Model_Performance_Dashboard.py
```

### 3. Run Example Script

```bash
python examples/model_performance_logging_example.py
```

## API Reference

### Core Functions

#### `log_model_performance()`

Log model performance metrics to both JSON and CSV formats.

**Parameters:**
- `model_name` (str): Name of the model
- `ticker` (str): Stock ticker symbol
- `sharpe` (float, optional): Sharpe ratio
- `mse` (float, optional): Mean squared error
- `drawdown` (float, optional): Maximum drawdown
- `total_return` (float, optional): Total return percentage
- `win_rate` (float, optional): Win rate percentage
- `accuracy` (float, optional): Model accuracy
- `notes` (str, optional): Additional notes

**Returns:**
- `dict`: Dictionary containing the logged performance data

**Example:**
```python
result = log_model_performance(
    model_name="XGBoost_v2",
    ticker="GOOGL",
    sharpe=2.1,
    mse=0.0189,
    drawdown=-0.08,
    total_return=0.31,
    win_rate=0.75,
    accuracy=0.78,
    notes="XGBoost with hyperparameter tuning"
)
```

#### `get_model_performance_history()`

Get model performance history as a DataFrame.

**Parameters:**
- `ticker` (str, optional): Filter by ticker symbol
- `model_name` (str, optional): Filter by model name
- `days_back` (int, optional): Filter by number of days back

**Returns:**
- `pd.DataFrame`: DataFrame containing performance history

**Example:**
```python
# Get all history
all_history = get_model_performance_history()

# Get history for specific ticker
aapl_history = get_model_performance_history(ticker="AAPL")

# Get history for specific model
lstm_history = get_model_performance_history(model_name="LSTM_v1")

# Get recent history
recent_history = get_model_performance_history(days_back=30)
```

#### `get_best_models()`

Get best models for each metric per ticker.

**Parameters:**
- `ticker` (str, optional): Filter by ticker symbol

**Returns:**
- `dict`: Dictionary containing best models for each metric

**Example:**
```python
# Get best models for specific ticker
best_models = get_best_models("AAPL")

# Get best models for all tickers
all_best_models = get_best_models()
```

#### `get_available_tickers()`

Get list of available tickers with performance data.

**Returns:**
- `list`: List of ticker symbols

**Example:**
```python
tickers = get_available_tickers()
print(f"Available tickers: {tickers}")
```

#### `get_available_models()`

Get list of available models with performance data.

**Parameters:**
- `ticker` (str, optional): Filter by ticker symbol

**Returns:**
- `list`: List of model names

**Example:**
```python
# Get all available models
all_models = get_available_models()

# Get models for specific ticker
aapl_models = get_available_models("AAPL")
```

### Utility Functions

#### `clear_model_performance_log()`

Clear all model performance logs.

**Example:**
```python
clear_model_performance_log()
```

## Data Storage

### File Structure

The system stores data in the following files:

```
memory/logs/
├── model_performance.json    # JSON format performance data
├── model_performance.csv     # CSV format performance data
└── best_models.json         # Best models tracking data
```

### Data Format

#### Performance Record Structure

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "ticker": "AAPL",
  "model_name": "LSTM_v1",
  "sharpe": 1.85,
  "mse": 0.0234,
  "drawdown": -0.12,
  "total_return": 0.25,
  "win_rate": 0.68,
  "accuracy": 0.72,
  "notes": "LSTM model with 50 epochs"
}
```

#### Best Models Structure

```json
{
  "AAPL": {
    "best_sharpe": {
      "model": "XGBoost_v2",
      "value": 2.1,
      "timestamp": "2024-01-15T10:30:00"
    },
    "best_mse": {
      "model": "XGBoost_v2",
      "value": 0.0189,
      "timestamp": "2024-01-15T10:30:00"
    },
    "best_drawdown": {
      "model": "XGBoost_v2",
      "value": -0.08,
      "timestamp": "2024-01-15T10:30:00"
    },
    "best_total_return": {
      "model": "XGBoost_v2",
      "value": 0.31,
      "timestamp": "2024-01-15T10:30:00"
    },
    "best_win_rate": {
      "model": "XGBoost_v2",
      "value": 0.75,
      "timestamp": "2024-01-15T10:30:00"
    },
    "best_accuracy": {
      "model": "XGBoost_v2",
      "value": 0.78,
      "timestamp": "2024-01-15T10:30:00"
    }
  }
}
```

## UI Dashboard

### Features

The Streamlit dashboard (`pages/Model_Performance_Dashboard.py`) provides:

1. **📈 Performance Dashboard Tab**
   - Interactive performance trends
   - Model comparison charts
   - Best models display
   - Raw data table with export

2. **🏆 Best Models Summary Tab**
   - Summary of best models across all tickers
   - Performance metrics comparison

3. **📊 Quick Analytics Tab**
   - Overall statistics
   - Top performing models
   - Recent activity
   - Performance trends

### Usage

```bash
# Run the dashboard
streamlit run pages/Model_Performance_Dashboard.py
```

### Dashboard Features

- **Sidebar Actions**:
  - Add sample data for testing
  - Clear all data
  - Manual performance logging

- **Filtering Options**:
  - Ticker selection
  - Model selection
  - Date range filtering
  - Days back slider

- **Visualizations**:
  - Time series charts
  - Performance comparison heatmaps
  - Model leaderboards
  - Trend analysis

## Integration Examples

### 1. Integration with Model Training

```python
from memory.model_log import log_model_performance

def train_and_log_model(model_name, ticker, training_data):
    """Train model and log performance."""
    
    # Train your model here
    model = train_model(training_data)
    
    # Evaluate model
    predictions = model.predict(test_data)
    
    # Calculate performance metrics
    sharpe = calculate_sharpe_ratio(predictions, actual)
    mse = calculate_mse(predictions, actual)
    drawdown = calculate_max_drawdown(predictions)
    total_return = calculate_total_return(predictions)
    win_rate = calculate_win_rate(predictions, actual)
    accuracy = calculate_accuracy(predictions, actual)
    
    # Log performance
    log_model_performance(
        model_name=model_name,
        ticker=ticker,
        sharpe=sharpe,
        mse=mse,
        drawdown=drawdown,
        total_return=total_return,
        win_rate=win_rate,
        accuracy=accuracy,
        notes=f"Trained on {len(training_data)} samples"
    )
    
    return model
```

### 2. Model Selection Based on Performance

```python
from memory.model_log import get_best_models, get_model_performance_history

def select_best_model(ticker, metric="sharpe"):
    """Select the best model for a given ticker and metric."""
    
    best_models = get_best_models(ticker)
    
    if not best_models:
        return None
    
    metric_key = f"best_{metric}"
    if metric_key in best_models and best_models[metric_key]["model"]:
        return best_models[metric_key]["model"]
    
    return None

def get_model_performance_summary(ticker):
    """Get performance summary for all models of a ticker."""
    
    history = get_model_performance_history(ticker=ticker)
    
    if history.empty:
        return {}
    
    summary = history.groupby('model_name').agg({
        'sharpe': ['mean', 'std', 'count'],
        'mse': ['mean', 'std'],
        'total_return': ['mean', 'std'],
        'win_rate': ['mean', 'std']
    }).round(4)
    
    return summary
```

### 3. Performance Monitoring

```python
from memory.model_log import get_model_performance_history
import pandas as pd

def monitor_model_performance(model_name, ticker, threshold_sharpe=1.0):
    """Monitor model performance and alert if below threshold."""
    
    history = get_model_performance_history(
        ticker=ticker, 
        model_name=model_name,
        days_back=30
    )
    
    if history.empty:
        print(f"No recent performance data for {model_name}")
        return
    
    recent_sharpe = history['sharpe'].iloc[-1]
    
    if recent_sharpe < threshold_sharpe:
        print(f"⚠️ Warning: {model_name} Sharpe ratio ({recent_sharpe:.3f}) below threshold ({threshold_sharpe})")
    else:
        print(f"✅ {model_name} performing well: Sharpe ratio = {recent_sharpe:.3f}")
```

## Testing

### Run Tests

```bash
# Run all tests
python -m pytest tests/test_model_performance_logging.py -v

# Run specific test
python tests/test_model_performance_logging.py
```

### Test Coverage

The test suite covers:

- ✅ Performance logging functionality
- ✅ Data retrieval and filtering
- ✅ Best models tracking
- ✅ Available tickers and models queries
- ✅ Data persistence
- ✅ Optional parameters handling
- ✅ Data clearing functionality

## Best Practices

### 1. Consistent Model Naming

Use consistent naming conventions for models:

```python
# Good examples
model_names = [
    "LSTM_v1",
    "XGBoost_v2", 
    "Transformer_v1",
    "RandomForest_v3"
]

# Avoid
model_names = [
    "lstm_model",
    "XGBoost_2024_01_15",
    "transformer_final"
]
```

### 2. Regular Performance Logging

Log performance after each model training or evaluation:

```python
# After model training
log_model_performance(
    model_name="LSTM_v1",
    ticker="AAPL",
    sharpe=sharpe_ratio,
    mse=mean_squared_error,
    # ... other metrics
    notes="Model trained with latest data"
)
```

### 3. Meaningful Notes

Include useful information in the notes field:

```python
notes = f"Trained on {len(train_data)} samples, {epochs} epochs, lr={learning_rate}"
```

### 4. Performance Monitoring

Regularly check model performance:

```python
# Weekly performance check
def weekly_performance_check():
    tickers = get_available_tickers()
    
    for ticker in tickers:
        best_models = get_best_models(ticker)
        print(f"Best models for {ticker}:")
        for metric, data in best_models.items():
            if data.get("model"):
                print(f"  {metric}: {data['model']} ({data['value']})")
```

## Troubleshooting

### Common Issues

1. **No data in dashboard**
   - Ensure you've logged some performance data
   - Check that the log directory exists
   - Verify file permissions

2. **Import errors**
   - Ensure the project root is in your Python path
   - Check that all dependencies are installed

3. **Performance issues with large datasets**
   - Use filtering to reduce data size
   - Consider archiving old data
   - Use days_back parameter to limit history

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from memory.model_log import log_model_performance
# ... your code
```

## Contributing

When contributing to the model performance logging system:

1. Add tests for new functionality
2. Update documentation
3. Follow the existing code style
4. Test with the example scripts
5. Verify UI functionality

## License

This module is part of the evolve_clean project and follows the same licensing terms. 