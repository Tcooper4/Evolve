# Trading System Test Suite

This directory contains the test suite for the trading system. The tests are organized into several categories:

## Directory Structure

- `analysis/`: Tests for market analysis components
- `benchmark/`: Performance benchmarking tests
- `fixtures/`: Test data and fixtures
- `implementation/`: Implementation-specific tests
- `integration/`: Integration tests
- `nlp/`: Natural language processing tests
- `optimization/`: Strategy optimization tests
- `report/`: Report generation tests
- `risk/`: Risk management tests
- `strategies/`: Trading strategy tests
- `test_nlp/`: Additional NLP-specific tests
- `unit/`: Unit tests

## Running Tests

### Prerequisites

- Python 3.10
- All required packages installed (see requirements.txt)
- Environment variables set (if needed)

### Basic Test Execution

To run all tests:
```bash
python -m pytest tests/ -v
```

To run specific test categories:
```bash
python -m pytest tests/analysis/ -v  # Run analysis tests
python -m pytest tests/strategies/ -v  # Run strategy tests
```

### Test Health Check

A test health check script is available to verify the test suite:
```bash
python scripts/check_tests.py
```

This script will:
- Check for missing `__init__.py` files
- Verify test discovery
- Report potential import issues
- List all test files

## Test Categories

### Unit Tests
- Located in `unit/`
- Test individual components in isolation
- Use mocks and fixtures extensively

### Integration Tests
- Located in `integration/`
- Test component interactions
- Use real data and minimal mocking

### Strategy Tests
- Located in `strategies/`
- Test trading strategies
- Include parameter validation and edge cases

### Analysis Tests
- Located in `analysis/`
- Test market analysis components
- Include data processing and indicator calculations

### Risk Tests
- Located in `risk/`
- Test risk management components
- Include metrics calculation and validation

### Optimization Tests
- Located in `optimization/`
- Test strategy optimization
- Include parameter space exploration

### Report Tests
- Located in `report/`
- Test report generation
- Include formatting and data validation

## Test Data

Test data is stored in `fixtures/data/`. The main sample data file is `sample_market_data.csv`.

## Adding New Tests

1. Create test file with `test_` prefix
2. Add appropriate `__init__.py` files
3. Use existing fixtures where possible
4. Follow the established test patterns
5. Run the health check script to verify

## Best Practices

1. Use descriptive test names
2. Include docstrings explaining test purpose
3. Use appropriate fixtures and setup/teardown
4. Clean up any resources after tests
5. Mock external dependencies
6. Test edge cases and error conditions
7. Keep tests independent and isolated

## Troubleshooting

If tests fail to run:
1. Check Python version (3.10 required)
2. Verify all dependencies are installed
3. Run from project root directory
4. Check for missing `__init__.py` files
5. Run the health check script
6. Check import paths in test files 