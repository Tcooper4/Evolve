"""Test configuration and fixtures."""

import pytest
import pandas as pd
import numpy as np
import torch
from typing import Dict, Any
from tests.fixtures.mock_data import setup_mock_env, teardown_mock_env, MOCK_CONFIG, MOCK_USERS

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100),
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99
    }, index=dates)
    return data

@pytest.fixture
def model_config():
    """Create a standard model configuration."""
    return {
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'dropout': 0.1,
        'batch_size': 32,
        'learning_rate': 0.001,
        'sequence_length': 10
    }

@pytest.fixture
def device():
    """Get the device to use for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files."""
    return tmp_path

@pytest.fixture(autouse=True)
def mock_env():
    """Set up and tear down mock environment variables for all tests."""
    setup_mock_env()
    yield
    teardown_mock_env()

@pytest.fixture
def mock_config():
    """Provide mock configuration for tests."""
    return MOCK_CONFIG

@pytest.fixture
def mock_users():
    """Provide mock user data for tests."""
    return MOCK_USERS

@pytest.fixture
def mock_email_config():
    """Provide mock email configuration for tests."""
    return MOCK_CONFIG['email']

@pytest.fixture
def mock_slack_config():
    """Provide mock Slack configuration for tests."""
    return MOCK_CONFIG['slack']

@pytest.fixture
def mock_security_config():
    """Provide mock security configuration for tests."""
    return MOCK_CONFIG['security'] 