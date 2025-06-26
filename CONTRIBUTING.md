# Contributing to Evolve

Thank you for your interest in contributing to Evolve! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

We welcome contributions from the community! Here are the main ways you can contribute:

### 🐛 Bug Reports
- Use the GitHub issue tracker
- Provide detailed reproduction steps
- Include error messages and stack traces
- Specify your environment (OS, Python version, etc.)

### 💡 Feature Requests
- Describe the feature clearly
- Explain the use case and benefits
- Consider implementation complexity
- Check if similar features already exist

### 🔧 Code Contributions
- Fork the repository
- Create a feature branch
- Make your changes
- Add tests for new functionality
- Ensure all tests pass
- Submit a pull request

## 🛠️ Development Setup

### Prerequisites
- Python 3.8 or higher
- Git
- (Optional) Docker

### Local Development

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Evolve.git
   cd Evolve
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## 📝 Code Style Guidelines

### Python Code Style
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions focused and under 50 lines when possible

### Example Code Style
```python
from typing import List, Optional, Dict
import pandas as pd


def calculate_technical_indicators(
    data: pd.DataFrame,
    indicators: List[str],
    params: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Calculate technical indicators for the given data.
    
    Args:
        data: DataFrame containing OHLCV data
        indicators: List of indicator names to calculate
        params: Optional parameters for indicators
        
    Returns:
        DataFrame with calculated indicators
        
    Raises:
        ValueError: If invalid indicator name is provided
    """
    if params is None:
        params = {}
    
    result = data.copy()
    
    for indicator in indicators:
        if indicator == "rsi":
            result = _add_rsi(result, params.get("rsi_period", 14))
        elif indicator == "macd":
            result = _add_macd(result, params.get("macd_params", {}))
        else:
            raise ValueError(f"Unknown indicator: {indicator}")
    
    return result
```

### File Naming Conventions
- Use snake_case for Python files and functions
- Use PascalCase for class names
- Use UPPER_CASE for constants
- Prefix test files with `test_`

### Directory Structure
```
trading/
├── models/          # ML models
├── strategies/      # Trading strategies
├── data/           # Data processing
├── backtesting/    # Backtesting engine
├── visualization/  # Charts and plots
└── utils/          # Utility functions
```

## 🧪 Testing Guidelines

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=trading --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/strategies/

# Run tests in parallel
pytest -n auto
```

### Writing Tests
- Write tests for all new functionality
- Use descriptive test names
- Follow the Arrange-Act-Assert pattern
- Mock external dependencies
- Test both success and failure cases

### Example Test
```python
import pytest
import pandas as pd
from trading.strategies.rsi_strategy import RSIStrategy


class TestRSIStrategy:
    """Test cases for RSI strategy implementation."""
    
    def test_rsi_calculation(self):
        """Test RSI calculation with known values."""
        # Arrange
        data = pd.DataFrame({
            'close': [44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.1, 44.09, 44.15, 44.33]
        })
        strategy = RSIStrategy(period=14)
        
        # Act
        result = strategy.calculate_rsi(data)
        
        # Assert
        assert len(result) == len(data)
        assert all(0 <= rsi <= 100 for rsi in result.dropna())
    
    def test_signal_generation(self):
        """Test buy/sell signal generation."""
        # Arrange
        strategy = RSIStrategy(period=14, overbought=70, oversold=30)
        
        # Act
        signals = strategy.generate_signals(pd.DataFrame({'rsi': [25, 75, 50]}))
        
        # Assert
        assert signals.iloc[0] == 'buy'  # RSI < 30
        assert signals.iloc[1] == 'sell'  # RSI > 70
        assert signals.iloc[2] == 'hold'  # RSI between 30-70
```

## 📊 Performance Considerations

### Code Performance
- Profile code for bottlenecks
- Use vectorized operations with NumPy/Pandas
- Implement caching for expensive computations
- Consider async operations for I/O-bound tasks

### Memory Management
- Avoid memory leaks in long-running processes
- Use generators for large datasets
- Implement proper cleanup in destructors
- Monitor memory usage in production

## 🔒 Security Guidelines

### Data Security
- Never commit API keys or sensitive data
- Use environment variables for configuration
- Validate all user inputs
- Implement proper error handling

### Code Security
- Use parameterized queries for database operations
- Sanitize user inputs
- Implement rate limiting for APIs
- Follow OWASP security guidelines

## 📚 Documentation

### Code Documentation
- Write clear docstrings for all public functions
- Include type hints
- Provide usage examples
- Document complex algorithms

### API Documentation
- Document all public APIs
- Include request/response examples
- Specify error codes and messages
- Keep documentation up to date

## 🚀 Pull Request Process

### Before Submitting
1. Ensure all tests pass
2. Update documentation if needed
3. Add tests for new functionality
4. Follow code style guidelines
5. Update CHANGELOG.md if applicable

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes
```

## 🏷️ Version Control

### Commit Messages
- Use conventional commit format
- Write clear, descriptive messages
- Reference issues when applicable

### Branch Naming
- `feature/feature-name` for new features
- `bugfix/bug-description` for bug fixes
- `hotfix/urgent-fix` for critical fixes
- `docs/documentation-update` for documentation

## 📞 Getting Help

### Communication Channels
- GitHub Issues for bug reports and feature requests
- GitHub Discussions for general questions
- Pull Request reviews for code feedback

### Resources
- [Python Documentation](https://docs.python.org/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Project Wiki](https://github.com/Tcooper4/Evolve/wiki)

## 🎉 Recognition

Contributors will be recognized in:
- Project README
- Release notes
- Contributor hall of fame
- GitHub contributors page

Thank you for contributing to Evolve! 🚀 