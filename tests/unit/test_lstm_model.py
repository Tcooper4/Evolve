import pytest
import torch
import numpy as np
import pandas as pd
from trading.models.lstm_model import LSTMForecaster
from tests.unit.base_test import BaseModelTest

class TestLSTMModel(BaseModelTest):
    """Test suite for LSTM model."""
    
    @pytest.fixture
    def model_class(self):
        return LSTMForecaster
        
    @pytest.fixture
    def model_config(self):
        return {
            'input_size': 2,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'epochs': 10,
            'patience': 5,
            'use_lr_scheduler': True
        }
    
    def test_lstm_specific_features(self, model_class, model_config, sample_data):
        """Test LSTM-specific features."""
        model = model_class(config=model_config)
        model.fit(sample_data, epochs=2)
        
        # Test dropout
        assert model.config['dropout'] == model_config['dropout']
        
        # Test hidden size
        assert model.config['hidden_size'] == model_config['hidden_size']
    
    def test_lstm_sequence_handling(self, model_class, model_config, sample_data):
        """Test LSTM sequence handling."""
        model = model_class(config=model_config)
        
        # Test with different sequence lengths
        short_seq = sample_data.iloc[:5]
        long_seq = sample_data.iloc[:20]
        
        model.fit(short_seq, epochs=1)
        short_pred = model.predict(short_seq)
        
        model.fit(long_seq, epochs=1)
        long_pred = model.predict(long_seq)
        
        assert len(short_pred['predictions']) == len(short_seq) - 1
        assert len(long_pred['predictions']) == len(long_seq) - 1
    
    def test_lstm_gradient_flow(self, model_class, model_config, sample_data):
        """Test LSTM gradient flow."""
        model = model_class(config=model_config)
        model.fit(sample_data, epochs=1)
        
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
        model.fit(sample_data, epochs=1)
        
        # Check if hidden state is properly initialized
        assert not hasattr(model.model, 'hidden') or model.model.hidden is None

    def test_lstm_model_learning_rate_scheduler(self, model_class, model_config, sample_data):
        """Test LSTM learning rate scheduler."""
        model = model_class(config=model_config)
        
        # Create data with extremely high variance to ensure high loss
        high_var_data = pd.DataFrame({
            'close': np.random.randn(100) * 10000,  # Even higher variance
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        # Train for more epochs with smaller batch size
        model.fit(high_var_data, epochs=10, batch_size=2)
        initial_lr = model.optimizer.param_groups[0]['lr']
        
        # Train for even more epochs to ensure scheduler triggers
        model.fit(high_var_data, epochs=20, batch_size=2)
        final_lr = model.optimizer.param_groups[0]['lr']
        
        assert final_lr != initial_lr

# Removed duplicate standalone test functions as they are now covered by the base test class 