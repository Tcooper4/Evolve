# Evolve System Unit Tests

This directory contains comprehensive unit tests for the Evolve trading system, covering all major components including forecasting models, strategy signals, agents, and backtesting functionality.

## 📁 Test Structure

```
tests/unit/
├── test_arima_forecaster.py      # ARIMA forecasting model tests
├── test_xgboost_forecaster.py    # XGBoost forecasting model tests
├── test_lstm_forecaster.py       # LSTM forecasting model tests
├── test_prophet_forecaster.py    # Prophet forecasting model tests
├── test_hybrid_forecaster.py     # Hybrid forecasting model tests
├── test_rsi_signals.py           # RSI strategy signal tests
├── test_macd_signals.py          # MACD strategy signal tests
├── test_bollinger_signals.py     # Bollinger Bands strategy tests
├── test_prompt_agent.py          # Prompt agent tests
├── test_backtester.py            # Backtesting functionality tests
├── run_tests.py                  # Test runner script
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Python Dependencies**: Ensure you have the required packages installed:
   ```bash
   pip install pytest pytest-cov pandas numpy scikit-learn
   ```

2. **Project Structure**: Make sure you're running from the project root directory where the `trading/` and `utils/` modules are located.

### Running Tests

#### Option 1: Run All Tests
```bash
# From project root directory
python tests/unit/run_tests.py
```

#### Option 2: Run with pytest directly
```bash
# Run all unit tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/unit/ --cov=trading --cov=utils --cov-report=html

# Run specific test file
pytest tests/unit/test_arima_forecaster.py -v

# Run specific test class
pytest tests/unit/test_arima_forecaster.py::TestARIMAModel -v

# Run specific test method
pytest tests/unit/test_arima_forecaster.py::TestARIMAModel::test_model_instantiation -v
```

#### Option 3: Run individual test suites
```bash
# Forecasting models
pytest tests/unit/test_*_forecaster.py -v

# Strategy signals
pytest tests/unit/test_*_signals.py -v

# Agents and backtesting
pytest tests/unit/test_prompt_agent.py tests/unit/test_backtester.py -v
```

## 📊 Test Coverage

### Forecasting Models (`test_*_forecaster.py`)

Each forecasting model test suite includes:

- **Model Instantiation**: Tests that models can be created correctly
- **Data Fitting**: Tests model training with synthetic time series data
- **Forecast Generation**: Tests prediction functionality with various horizons
- **Edge Cases**: 
  - Short time series (< 10 points)
  - Constant time series
  - NaN values
  - Empty data
- **Model Validation**: Tests model save/load, performance metrics, and validation
- **Configuration**: Tests different model parameters and configurations

**Synthetic Data**: Uses realistic time series with trends, seasonality, and noise.

### Strategy Signals (`test_*_signals.py`)

Each strategy test suite includes:

- **Signal Generation**: Tests signal creation from price data
- **Signal Validation**: Ensures signals are valid (1=buy, -1=sell, 0=hold)
- **Technical Indicators**: Tests calculation of RSI, MACD, Bollinger Bands
- **Returns Calculation**: Tests strategy returns and cumulative returns
- **Edge Cases**:
  - Short price data
  - Constant prices
  - NaN values
  - Missing columns
- **Performance Metrics**: Tests win rate, Sharpe ratio, and other metrics

**Synthetic Data**: Uses sine wave + noise for realistic price movements.

### Prompt Agent (`test_prompt_agent.py`)

Tests the natural language processing agent:

- **Prompt Processing**: Tests various prompt types and formats
- **Action Selection**: Tests correct action identification (forecast, strategy, etc.)
- **Model Selection**: Tests appropriate model selection for forecasting
- **Strategy Selection**: Tests strategy identification from prompts
- **LLM Integration**: Mocked tests for OpenAI and HuggingFace integration
- **Error Handling**: Tests invalid prompts and edge cases
- **Context Awareness**: Tests conversation context maintenance

**Mock Data**: Uses simulated prompts and mocked LLM responses.

### Backtester (`test_backtester.py`)

Tests the backtesting engine:

- **Basic Backtesting**: Tests core backtesting functionality
- **Equity Curve**: Tests equity curve calculation and validation
- **Performance Metrics**: Tests calculation of:
  - Total return
  - Sharpe ratio
  - Maximum drawdown
  - Win rate
- **Risk Management**: Tests stop-loss, take-profit, and trailing stops
- **Transaction Costs**: Tests impact of different cost structures
- **Position Sizing**: Tests various position sizing strategies
- **Edge Cases**:
  - Empty data
  - No signals
  - Single data point
- **Benchmark Comparison**: Tests strategy vs benchmark performance

**Synthetic Data**: Uses realistic price data with predefined signals.

## 🧪 Test Data

All tests use synthetic data to ensure:
- **Reproducibility**: Fixed random seeds for consistent results
- **Realism**: Data patterns that mimic real market conditions
- **Coverage**: Various scenarios including trends, volatility, and edge cases
- **Independence**: No external dependencies on live data or APIs

### Data Types

1. **Time Series Data**: 
   - Length: 50-100 points
   - Features: OHLCV (Open, High, Low, Close, Volume)
   - Patterns: Trends, seasonality, noise

2. **Signal Data**:
   - Values: 1 (buy), -1 (sell), 0 (hold)
   - Frequency: Realistic trading patterns
   - Validation: No NaN values, proper signal logic

3. **Prompt Data**:
   - Natural language commands
   - Various formats and complexity levels
   - Edge cases and error conditions

## 🔧 Test Configuration

### Pytest Configuration

Tests use the following pytest configuration:
- **Verbose output**: `-v` flag for detailed test results
- **Coverage reporting**: `--cov` for code coverage analysis
- **Warning suppression**: `--disable-warnings` for cleaner output
- **Timeout protection**: 60-second timeout for individual tests
- **Parallel execution**: Support for concurrent test execution

### Mock Configuration

External dependencies are mocked:
- **API Calls**: yfinance, Alpha Vantage, etc.
- **LLM Services**: OpenAI, HuggingFace
- **Database Connections**: Redis, PostgreSQL
- **File I/O**: Network storage, cloud services

## 📈 Coverage Goals

Target coverage percentages:
- **Forecasting Models**: 90%+
- **Strategy Signals**: 85%+
- **Prompt Agent**: 80%+
- **Backtester**: 90%+
- **Overall System**: 85%+

## 🚨 Error Handling

Tests validate proper error handling for:
- **Invalid Input**: Malformed data, missing columns
- **Edge Cases**: Empty data, single points, NaN values
- **Resource Limits**: Memory, timeouts, API limits
- **Configuration Errors**: Invalid parameters, missing dependencies

## 🔄 Continuous Integration

Tests are designed for CI/CD pipelines:
- **Fast Execution**: Most tests complete in < 1 second
- **Isolation**: No shared state between tests
- **Deterministic**: Same results on every run
- **Parallel Safe**: Can run concurrently without conflicts

## 📝 Adding New Tests

### Guidelines for New Test Files

1. **File Naming**: `test_<module_name>.py`
2. **Class Naming**: `Test<ClassName>`
3. **Method Naming**: `test_<functionality>`
4. **Documentation**: Include docstrings for all test methods
5. **Data Fixtures**: Use pytest fixtures for test data
6. **Mocking**: Mock external dependencies
7. **Edge Cases**: Include boundary condition tests

### Example Test Structure

```python
"""
Unit tests for NewModule.

Tests new module functionality with synthetic data and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

class TestNewModule:
    """Test suite for NewModule."""
    
    @pytest.fixture
    def test_data(self):
        """Create test data fixture."""
        # Create synthetic test data
        return pd.DataFrame(...)
    
    def test_basic_functionality(self, test_data):
        """Test basic functionality."""
        # Test implementation
        pass
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test edge cases
        pass
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from project root
2. **Missing Dependencies**: Install required packages
3. **Test Failures**: Check if production code has changed
4. **Timeout Errors**: Increase timeout or optimize slow tests
5. **Coverage Issues**: Add tests for uncovered code paths

### Debug Mode

Run tests in debug mode for detailed output:
```bash
pytest tests/unit/ -v -s --tb=long
```

## 📚 Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Python Testing Best Practices](https://realpython.com/python-testing/)
- [Mock Testing Guide](https://docs.python.org/3/library/unittest.mock.html)

## 🤝 Contributing

When adding new features to the Evolve system:

1. **Write Tests First**: Follow TDD principles
2. **Maintain Coverage**: Ensure new code is well-tested
3. **Update This README**: Document new test files and functionality
4. **Run Full Suite**: Verify all tests pass before submitting

---

**Last Updated**: December 2024  
**Test Count**: 200+ individual test methods  
**Coverage Target**: 85%+ overall system coverage 