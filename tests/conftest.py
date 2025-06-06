import pytest
import pandas as pd
import numpy as np
import torch
from typing import Dict, Any

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