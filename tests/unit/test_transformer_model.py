import pytest
import torch
import numpy as np
import pandas as pd
from trading.models.advanced.transformer.time_series_transformer import TransformerForecaster
from tests.unit.base_test import BaseModelTest

class TestTransformerModel(BaseModelTest):
    """Test suite for transformer model."""
    
    @pytest.fixture
    def model_class(self):
        return TransformerForecaster
        
    @pytest.fixture
    def model_config(self):
        return {
            'input_dim': 20,
            'd_model': 256,
            'nhead': 8,
            'num_encoder_layers': 4,
            'dim_feedforward': 1024,
            'dropout': 0.1,
            'activation': 'gelu',
            'batch_first': True,
            'sequence_length': 10,
            'feature_columns': [
                'Close', 'Volume', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower',
                'Volume_MA_5', 'Volume_MA_10', 'Fourier_Sin_7', 'Fourier_Cos_7',
                'Close_Lag_1', 'Close_Return_Lag_1', 'Volume_Lag_1',
                'Volume_Return_Lag_1', 'HL_Range_Lag_1', 'ROC_5', 'Momentum_5',
                'Stoch_K', 'Stoch_D', 'Volume_Trend'
            ],
            'target_column': 'Close',
            'use_batch_norm': False,
            'use_lr_scheduler': True,
            'learning_rate': 0.01,
            'scheduler_factor': 0.1,
            'scheduler_patience': 1,
            'scheduler_threshold': 0.1,
            'scheduler_min_lr': 1e-6
        }
    
    def test_prepare_data(self, model_class, model_config, sample_data):
        """Test data preparation for transformer model."""
        model = model_class(config=model_config)
        
        # Test data preparation
        prepared_data = model._prepare_data(sample_data)
        
        # Check output structure
        assert isinstance(prepared_data, dict)
        assert 'X' in prepared_data
        assert 'y' in prepared_data
        
        # Check tensor shapes
        X = prepared_data['X']
        y = prepared_data['y']
        
        expected_seq_len = model_config['sequence_length']
        expected_features = len(model_config['feature_columns'])
        expected_samples = len(sample_data) - expected_seq_len
        
        assert X.shape == (expected_samples, expected_seq_len, expected_features)
        assert y.shape == (expected_samples, 1)
        
        # Check data normalization
        assert torch.allclose(X.mean(dim=(0, 1)), torch.zeros(expected_features), atol=1e-6)
        assert torch.allclose(X.std(dim=(0, 1)), torch.ones(expected_features), atol=1e-6)
        assert torch.allclose(y.mean(), torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(y.std(), torch.tensor(1.0), atol=1e-6)
    
    def test_prepare_data_validation(self, model_class, model_config, sample_data):
        """Test data preparation validation."""
        model = model_class(config=model_config)
        
        # Test invalid input type
        with pytest.raises(ValueError, match="Input data must be a pandas DataFrame"):
            model._prepare_data(np.array(sample_data))
        
        # Test missing feature columns
        invalid_config = model_config.copy()
        invalid_config['feature_columns'] = []
        model = model_class(config=invalid_config)
        with pytest.raises(ValueError, match="Feature columns must be specified in config"):
            model._prepare_data(sample_data)
        
        # Test missing target column
        invalid_data = sample_data.copy()
        invalid_data = invalid_data.drop(columns=[model_config['target_column']])
        model = model_class(config=model_config)
        with pytest.raises(ValueError, match=f"Target column '{model_config['target_column']}' not found in data"):
            model._prepare_data(invalid_data)
        
        # Test missing values
        invalid_data = sample_data.copy()
        invalid_data.loc[0, model_config['target_column']] = np.nan
        with pytest.raises(ValueError, match="Input data contains missing values"):
            model._prepare_data(invalid_data)
    
    def test_transformer_specific_features(self, model_class, model_config, sample_data):
        """Test transformer-specific features."""
        model = model_class(config=model_config)
        
        # Test model initialization
        assert model.input_dim == model_config['input_dim']
        assert model.d_model == model_config['d_model']
        assert model.nhead == model_config['nhead']
        assert model.num_encoder_layers == model_config['num_encoder_layers']
        assert model.sequence_length == model_config['sequence_length']
        
        # Test model components
        assert isinstance(model.input_projection, torch.nn.Linear)
        assert isinstance(model.pos_encoder, torch.nn.Module)
        assert isinstance(model.transformer_encoder, torch.nn.TransformerEncoder)
        assert isinstance(model.output_projection, torch.nn.Linear)
        
        # Test forward pass
        prepared_data = model._prepare_data(sample_data)
        output = model(prepared_data['X'])
        assert output.shape == (prepared_data['X'].shape[0], model_config['sequence_length'], 1)
    
    def test_transformer_attention_patterns(self, model_class, model_config, sample_data):
        """Test transformer attention patterns."""
        model = model_class(config=model_config)
        prepared_data = model._prepare_data(sample_data)
        
        # Test attention mask generation
        mask = model._generate_square_subsequent_mask(model_config['sequence_length'])
        assert mask.shape == (model_config['sequence_length'], model_config['sequence_length'])
        assert torch.all(mask.diag() == 0)  # Diagonal should be 0
        assert torch.all(mask.triu() == 0)  # Upper triangle should be 0
        assert torch.all(mask.tril() == float('-inf'))  # Lower triangle should be -inf
    
    def test_transformer_prediction_intervals(self, model_class, model_config, sample_data):
        """Test transformer prediction intervals."""
        model = model_class(config=model_config)
        prepared_data = model._prepare_data(sample_data)
        
        # Train model
        model.fit(sample_data, epochs=2, batch_size=4)
        
        # Test predictions
        predictions = model.predict(sample_data)
        assert 'predictions' in predictions
        assert len(predictions['predictions']) > 0
        
        # Test prediction shape
        assert predictions['predictions'].shape[0] == len(sample_data) - model_config['sequence_length']
        assert predictions['predictions'].shape[1] == 1
    
    def test_transformer_batch_normalization(self, model_class, model_config, sample_data):
        """Test transformer batch normalization."""
        config = model_config.copy()
        config['use_batch_norm'] = True
        model = model_class(config=config)
        
        # Test model training with batch normalization
        model.fit(sample_data, epochs=2, batch_size=4)
        assert model.config['use_batch_norm']
        
        # Test predictions with batch normalization
        predictions = model.predict(sample_data)
        assert 'predictions' in predictions
        assert len(predictions['predictions']) > 0
    
    def test_transformer_positional_encoding(self, model_class, model_config, sample_data):
        """Test transformer positional encoding."""
        model = model_class(config=model_config)
        prepared_data = model._prepare_data(sample_data)
        
        # Test positional encoding
        pos_encoding = model.pos_encoder.pe
        assert pos_encoding.shape[0] == model_config['sequence_length']
        assert pos_encoding.shape[2] == model_config['d_model']
        
        # Test positional encoding properties
        assert torch.allclose(pos_encoding.mean(), torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(pos_encoding.std(), torch.tensor(1.0), atol=1e-6)
    
    def test_transformer_weight_initialization(self, model_class, model_config, sample_data):
        """Test transformer weight initialization."""
        model = model_class(config=model_config)
        
        # Test weight initialization
        for name, param in model.named_parameters():
            if param.dim() > 1:
                # Check if weights are initialized with Xavier uniform
                assert torch.all(param != 0)  # Weights should not be zero
                assert torch.all(torch.isfinite(param))  # Weights should be finite 