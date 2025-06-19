# Evolve Trading System Tests

This directory contains the test suite for the Evolve trading system. The tests cover all critical functionality including forecasting, strategy generation, agent behavior, and system integration.

## Directory Structure

```
tests/
├── __init__.py
├── test_app_smoke.py          # Application smoke tests
├── test_router.py             # Router functionality tests
├── test_forecasting/          # Forecasting model tests
│   ├── test_arima.py
│   ├── test_lstm.py
│   ├── test_prophet.py
│   └── test_hybrid.py
├── test_strategies/           # Trading strategy tests
│   ├── test_rsi.py
│   ├── test_macd.py
│   ├── test_bollinger.py
│   ├── test_sma.py
│   └── fixtures.py
├── test_agents/               # Agent behavior tests
│   ├── test_self_improving_agent.py
│   ├── test_goal_planner.py
│   └── test_router_intent_detection.py
├── test_backtester.py         # Backtesting functionality
├── test_feature_engineering.py # Feature engineering tests
├── test_utils.py              # Utility function tests
└── conftest.py                # Shared test fixtures
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with detailed output
pytest -v

# Run with coverage report
pytest --cov=trading

# Run specific test file
pytest tests/test_strategies/test_rsi.py

# Run tests matching a pattern
pytest -k "test_rsi"
```

### Advanced Options

```bash
# Run tests in parallel
pytest -n auto

# Stop after 3 failures
pytest --maxfail=3

# Show traceback for failures
pytest --tb=short

# Generate HTML coverage report
pytest --cov=trading --cov-report=html
```

## Test Categories

### Unit Tests
- Strategy tests (`test_strategies/`)
- Utility function tests (`test_utils.py`)
- Feature engineering tests (`test_feature_engineering.py`)

### Integration Tests
- Agent behavior tests (`test_agents/`)
- Router functionality tests (`test_router.py`)
- Backtesting tests (`test_backtester.py`)

### System Tests
- Application smoke tests (`test_app_smoke.py`)
- End-to-end tests (in `test_agents/`)

## Adding New Tests

1. Create a new test file in the appropriate directory
2. Follow the naming convention: `test_*.py`
3. Use the shared fixtures from `conftest.py`
4. Add appropriate test cases and assertions
5. Update this README if adding new test categories

## Test Fixtures

Common test fixtures are defined in `conftest.py`:
- `sample_price_data`: Mock price data for testing
- `mock_forecaster`: Mock forecasting model
- `strategy_config`: Shared strategy configurations
- `mock_agent`: Mock agent for testing
- `mock_router`: Mock router for testing
- `mock_performance_logger`: Mock performance logger

## Best Practices

1. Use descriptive test names
2. Include docstrings for test functions
3. Use appropriate fixtures for common setup
4. Mock external dependencies
5. Test both success and failure cases
6. Keep tests independent and isolated
7. Use appropriate assertions for the test case

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure the project root is in PYTHONPATH
   - Check for circular imports
   - Verify all required packages are installed

2. **Test Failures**
   - Check test data and fixtures
   - Verify mock objects are properly configured
   - Ensure test environment matches production

3. **Performance Issues**
   - Use appropriate test markers
   - Skip slow tests when appropriate
   - Use parallel test execution

### Getting Help

For issues with the test suite:
1. Check the test logs
2. Review the test documentation
3. Consult the project documentation
4. Contact the development team 