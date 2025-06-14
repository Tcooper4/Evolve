"""Tests for the ModelUpdater class."""

import pytest
from unittest.mock import Mock, patch
import numpy as np
from trading.agents.updater import ModelUpdater
from trading.memory.performance_memory import PerformanceMemory

@pytest.fixture
def mock_model():
    model = Mock()
    model.get_weights.return_value = np.array([0.5, 0.5])
    model.set_weights = Mock()
    return model

@pytest.fixture
def mock_memory(tmp_path):
    memory = PerformanceMemory(str(tmp_path))
    return memory

@pytest.fixture
def updater(mock_model, mock_memory):
    return ModelUpdater(mock_model, mock_memory, interval=1)

def test_initialization(updater, mock_model, mock_memory):
    """Test ModelUpdater initialization."""
    assert updater.model == mock_model
    assert updater.memory == mock_memory
    assert updater.interval == 1
    assert not updater.is_running

def test_update_model_weights(updater, mock_model, mock_memory):
    """Test weight updates based on performance metrics."""
    # Add some test metrics
    mock_memory.update_metrics("AAPL", {
        "accuracy": 0.8,
        "sharpe_ratio": 1.5,
        "timestamp": "2024-01-01"
    })
    
    # Mock the weight calculation
    with patch.object(updater, '_calculate_new_weights') as mock_calc:
        mock_calc.return_value = np.array([0.6, 0.4])
        updater.update_model_weights("accuracy")
        
        # Verify weight update
        mock_model.set_weights.assert_called_once_with(np.array([0.6, 0.4]))

def test_periodic_updates(updater):
    """Test periodic update functionality."""
    with patch.object(updater, 'update_model_weights') as mock_update:
        updater.start_periodic_updates()
        assert updater.is_running
        
        # Simulate some time passing
        updater._run()
        
        # Verify update was called
        mock_update.assert_called_once()
        
        # Stop updates
        updater.stop()
        assert not updater.is_running

def test_weight_calculation(updater, mock_memory):
    """Test weight calculation based on metrics."""
    # Add test metrics
    mock_memory.update_metrics("AAPL", {
        "accuracy": 0.8,
        "sharpe_ratio": 1.5,
        "timestamp": "2024-01-01"
    })
    
    # Calculate new weights
    new_weights = updater._calculate_new_weights("accuracy")
    
    # Verify weight properties
    assert isinstance(new_weights, np.ndarray)
    assert len(new_weights) == 2
    assert np.all(new_weights >= 0)
    assert np.all(new_weights <= 1)
    assert np.isclose(np.sum(new_weights), 1.0)

def test_error_handling(updater, mock_model, mock_memory):
    """Test error handling during updates."""
    # Simulate model error
    mock_model.set_weights.side_effect = Exception("Model error")
    
    # Should not raise exception
    updater.update_model_weights("accuracy")
    
    # Verify error was logged
    assert not updater.is_running

def test_memory_interaction(updater, mock_memory):
    """Test interaction with performance memory."""
    # Add test metrics
    mock_memory.update_metrics("AAPL", {
        "accuracy": 0.8,
        "sharpe_ratio": 1.5,
        "timestamp": "2024-01-01"
    })
    
    # Update weights
    updater.update_model_weights("accuracy")
    
    # Verify memory was accessed
    metrics = mock_memory.get_metrics("AAPL")
    assert "accuracy" in metrics
    assert metrics["accuracy"] == 0.8 