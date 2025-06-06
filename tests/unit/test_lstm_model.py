import pytest
import torch
import numpy as np
import pandas as pd
from trading.models.lstm_model import LSTMForecaster
from tests.unit.base_test import BaseModelTest

def make_sample_data():
    """Create sample data for testing."""
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    return data

class TestLSTMModel(BaseModelTest):
    """Test suite for LSTM model."""
    
    @pytest.fixture
    def model_class(self):
        return LSTMForecaster
    
    def test_lstm_specific_features(self, model_class, model_config, sample_data):
        """Test LSTM-specific features."""
        model = model_class(config=model_config)
        model.fit(sample_data, epochs=2, batch_size=4)
        
        # Test bidirectional functionality
        assert model.model.bidirectional == model_config.bidirectional
        
        # Test dropout
        assert model.model.dropout == model_config.dropout
        
        # Test hidden size
        assert model.model.hidden_size == model_config.hidden_size
    
    def test_lstm_sequence_handling(self, model_class, model_config, sample_data):
        """Test LSTM sequence handling."""
        model = model_class(config=model_config)
        
        # Test with different sequence lengths
        short_seq = sample_data.iloc[:5]
        long_seq = sample_data.iloc[:20]
        
        model.fit(short_seq, epochs=1, batch_size=2)
        short_pred = model.predict(short_seq)
        
        model.fit(long_seq, epochs=1, batch_size=4)
        long_pred = model.predict(long_seq)
        
        assert len(short_pred['predictions']) == len(short_seq)
        assert len(long_pred['predictions']) == len(long_seq)
    
    def test_lstm_gradient_flow(self, model_class, model_config, sample_data):
        """Test LSTM gradient flow."""
        model = model_class(config=model_config)
        model.fit(sample_data, epochs=1, batch_size=4)
        
        # Check if gradients are flowing
        for param in model.model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()
                assert not torch.isinf(param.grad).any()
    
    def test_lstm_state_management(self, model_class, model_config, sample_data):
        """Test LSTM state management."""
        model = model_class(config=model_config)
        
        # Test state reset between batches
        model.fit(sample_data, epochs=1, batch_size=4)
        
        # Check if hidden state is properly initialized
        assert model.model.hidden is None or (
            isinstance(model.model.hidden, tuple) and
            all(h is None or isinstance(h, torch.Tensor) for h in model.model.hidden)
        )

def test_lstm_model_instantiation():
    """Test LSTM model instantiation."""
    model = LSTMForecaster()
    assert model is not None
    assert model.model is not None

def test_lstm_model_fit():
    """Test LSTM model fitting."""
    model = LSTMForecaster()
    data = make_sample_data()
    model.fit(data, epochs=2, batch_size=4)
    assert len(model.history) > 0

def test_lstm_model_predict():
    """Test LSTM model prediction."""
    model = LSTMForecaster()
    data = make_sample_data()
    model.fit(data, epochs=2, batch_size=4)
    predictions = model.predict(data)
    assert 'predictions' in predictions
    assert len(predictions['predictions']) > 0

def test_lstm_model_save_load():
    """Test LSTM model saving and loading."""
    import tempfile
    import os
    
    # Create and train model
    model = LSTMForecaster()
    data = make_sample_data()
    model.fit(data, epochs=2, batch_size=4)
    
    # Save model
    save_path = tempfile.mkdtemp()
    model_path = os.path.join(save_path, 'model')
    model.save(model_path)
    
    # Create new model and load
    loaded_model = LSTMForecaster()
    loaded_model.load(model_path)
    
    # Compare predictions
    original_pred = model.predict(data)
    loaded_pred = loaded_model.predict(data)
    
    np.testing.assert_allclose(
        original_pred['predictions'],
        loaded_pred['predictions'],
        rtol=1e-2, atol=1e-2
    )
    
    # Cleanup
    import shutil
    shutil.rmtree(save_path) 