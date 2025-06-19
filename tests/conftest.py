"""Pytest configuration and shared fixtures."""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add all necessary directories to Python path
for directory in ['core', 'trading', 'models', 'utils', 'optimizer', 'market_analysis']:
    dir_path = project_root / directory
    if dir_path.exists():
        sys.path.insert(0, str(dir_path))

@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'open': np.random.normal(100, 2, 100),
        'high': np.random.normal(102, 2, 100),
        'low': np.random.normal(98, 2, 100),
        'close': np.random.normal(100, 2, 100),
        'volume': np.random.normal(1000000, 100000, 100)
    }, index=dates)
    return data

@pytest.fixture
def mock_forecaster():
    """Create a mock forecaster for testing."""
    mock = Mock()
    mock.predict.return_value = pd.DataFrame({
        'forecast': np.random.normal(100, 2, 30),
        'lower_bound': np.random.normal(98, 2, 30),
        'upper_bound': np.random.normal(102, 2, 30)
    }, index=pd.date_range(start='2024-04-01', periods=30, freq='D'))
    return mock

@pytest.fixture
def strategy_config():
    """Shared strategy configuration for testing."""
    return {
        'sma': {
            'short_window': 20,
            'long_window': 50
        },
        'rsi': {
            'window': 14,
            'overbought': 70,
            'oversold': 30
        },
        'macd': {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9
        },
        'bollinger': {
            'window': 20,
            'num_std': 2
        }
    }

@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    mock = Mock()
    mock.process_goal.return_value = {
        'action': 'buy',
        'confidence': 0.8,
        'reasoning': 'Test reasoning'
    }
    return mock

@pytest.fixture
def mock_router():
    """Create a mock router for testing."""
    mock = Mock()
    mock.route_intent.return_value = {
        'intent': 'buy',
        'entity': 'AAPL',
        'confidence': 0.9
    }
    return mock

@pytest.fixture
def mock_performance_logger():
    """Create a mock performance logger for testing."""
    mock = Mock()
    mock.log_performance.return_value = {
        'accuracy': 0.8,
        'sharpe_ratio': 1.5,
        'max_drawdown': -0.1
    }
    return mock

def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers",
        "requires_gpu: mark test as requiring GPU"
    ) 