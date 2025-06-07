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
            'sequence_length': 10,
            'feature_columns': ['close', 'volume'],
            'target_column': 'close',
            'use_lr_scheduler': True,
            'learning_rate': 0.001,
            'scheduler_patience': 1,
            'scheduler_factor': 0.1
        }
    
    @pytest.fixture
    def sample_data(self):
        # Create sample time series data
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        return data
    
    def test_lstm_specific_features(self, model_class, model_config, sample_data):
        """Test LSTM-specific features."""
        model = model_class(config=model_config)
        
        # Test model architecture
        assert model.model is not None
        assert model.model.lstm.input_size == model_config['input_size']
        assert model.model.lstm.hidden_size == model_config['hidden_size']
        assert model.model.lstm.num_layers == model_config['num_layers']
        
        # Test configuration validation
        assert model.sequence_length == model_config['sequence_length']
        assert model.feature_columns == model_config['feature_columns']
        assert model.target_column == model_config['target_column']
    
    def test_batch_normalization(self, model_class, model_config, sample_data):
        """Test batch normalization functionality."""
        config = model_config.copy()
        config['use_batch_norm'] = True
        model = model_class(config=config)
        
        # Test batch norm initialization
        assert model.batch_norm is not None
        assert isinstance(model.batch_norm, torch.nn.BatchNorm1d)
        assert model.batch_norm.num_features == config['hidden_size']
        
        # Test forward pass with batch norm
        X, y = model._prepare_data(sample_data, is_training=True)
        output = model(X)
        assert output.shape == (X.shape[0], 1)
    
    def test_layer_normalization(self, model_class, model_config, sample_data):
        """Test layer normalization functionality."""
        config = model_config.copy()
        config['use_layer_norm'] = True
        model = model_class(config=config)
        
        # Test layer norm initialization
        assert model.layer_norm is not None
        assert isinstance(model.layer_norm, torch.nn.LayerNorm)
        assert model.layer_norm.normalized_shape == (config['hidden_size'],)
        
        # Test forward pass with layer norm
        X, y = model._prepare_data(sample_data, is_training=True)
        output = model(X)
        assert output.shape == (X.shape[0], 1)
    
    def test_attention_mechanism(self, model_class, model_config, sample_data):
        """Test attention mechanism functionality."""
        config = model_config.copy()
        config['use_attention'] = True
        config['num_attention_heads'] = 4
        config['attention_dropout'] = 0.1
        model = model_class(config=config)
        
        # Test attention initialization
        assert model.attention is not None
        assert isinstance(model.attention, torch.nn.MultiheadAttention)
        assert model.attention.embed_dim == config['hidden_size']
        assert model.attention.num_heads == config['num_attention_heads']
        
        # Test forward pass with attention
        X, y = model._prepare_data(sample_data, is_training=True)
        output = model(X)
        assert output.shape == (X.shape[0], 1)
    
    def test_residual_connections(self, model_class, model_config, sample_data):
        """Test residual connections functionality."""
        config = model_config.copy()
        config['use_residual'] = True
        model = model_class(config=config)
        
        # Test residual connection flag
        assert model.use_residual is True
        
        # Test forward pass with residual connection
        X, y = model._prepare_data(sample_data, is_training=True)
        output = model(X)
        assert output.shape == (X.shape[0], 1)
    
    def test_additional_dropout(self, model_class, model_config, sample_data):
        """Test additional dropout functionality."""
        config = model_config.copy()
        config['additional_dropout'] = 0.2
        model = model_class(config=config)
        
        # Test dropout initialization
        assert model.dropout_layer is not None
        assert isinstance(model.dropout_layer, torch.nn.Dropout)
        assert model.dropout_layer.p == config['additional_dropout']
        
        # Test forward pass with additional dropout
        X, y = model._prepare_data(sample_data, is_training=True)
        output = model(X)
        assert output.shape == (X.shape[0], 1)
    
    def test_weight_initialization(self, model_class, model_config, sample_data):
        """Test weight initialization."""
        model = model_class(config=model_config)
        
        # Test weight initialization
        for name, param in model.model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                # Check if weights are initialized with Xavier uniform
                assert torch.all(param != 0)  # Weights should not be zero
                assert torch.all(torch.isfinite(param))  # Weights should be finite
            elif 'bias' in name:
                # Check if biases are initialized to zero
                assert torch.all(param == 0)
    
    def test_combined_features(self, model_class, model_config, sample_data):
        """Test combination of multiple features."""
        config = model_config.copy()
        config.update({
            'use_batch_norm': True,
            'use_layer_norm': True,
            'use_attention': True,
            'use_residual': True,
            'additional_dropout': 0.2
        })
        model = model_class(config=config)
        
        # Test all features are initialized
        assert model.batch_norm is not None
        assert model.layer_norm is not None
        assert model.attention is not None
        assert model.use_residual is True
        assert model.dropout_layer is not None
        
        # Test forward pass with all features
        X, y = model._prepare_data(sample_data, is_training=True)
        output = model(X)
        assert output.shape == (X.shape[0], 1)
        
        # Test training with all features
        model.fit(sample_data, epochs=2, batch_size=4)
        assert len(model.history) > 0
    
    def test_model_persistence(self, model_class, model_config, sample_data, tmp_path):
        """Test model persistence with all features."""
        config = model_config.copy()
        config.update({
            'use_batch_norm': True,
            'use_layer_norm': True,
            'use_attention': True,
            'use_residual': True,
            'additional_dropout': 0.2
        })
        model = model_class(config=config)
        
        # Train model
        model.fit(sample_data, epochs=2, batch_size=4)
        
        # Save model
        save_path = tmp_path / "lstm_model.pt"
        model.save(str(save_path))
        
        # Load model
        new_model = model_class(config=config)
        new_model.load(str(save_path))
        
        # Compare predictions
        original_pred = model.predict(sample_data)
        loaded_pred = new_model.predict(sample_data)
        
        assert np.allclose(original_pred['predictions'], loaded_pred['predictions'])

# Removed duplicate standalone test functions as they are now covered by the base test class 