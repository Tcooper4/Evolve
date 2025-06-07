import pytest
import torch
import numpy as np
import pandas as pd
from trading.models.tcn_model import TCNModel
from tests.unit.base_test import BaseModelTest

class TestTCNModel(BaseModelTest):
    """Test suite for TCN model."""
    
    @pytest.fixture
    def model_class(self):
        return TCNModel
    
    @pytest.fixture
    def model_config(self):
        return {
            'input_size': 2,
            'output_size': 1,
            'num_channels': [64, 128, 256],
            'kernel_size': 3,
            'dropout': 0.2,
            'sequence_length': 10,
            'feature_columns': ['close', 'volume'],
            'target_column': 'close',
            'learning_rate': 0.001,
            'use_lr_scheduler': True
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
    
    def test_tcn_specific_features(self, model_class, model_config, sample_data):
        """Test TCN-specific features."""
        model = model_class(config=model_config)
        
        # Test model architecture
        assert model.model is not None
        assert len(model.tcn) == len(model_config['num_channels'])
        assert model.tcn[0].conv1.kernel_size[0] == model_config['kernel_size']
        
        # Test configuration validation
        assert model.sequence_length == model_config['sequence_length']
        assert model.feature_columns == model_config['feature_columns']
        assert model.target_column == model_config['target_column']
    
    def test_tcn_dilation(self, model_class, model_config, sample_data):
        """Test TCN dilation mechanism."""
        model = model_class(config=model_config)
        
        # Check dilation factors
        for i, layer in enumerate(model.tcn):
            assert layer.conv1.dilation[0] == 2 ** i
    
    def test_tcn_residual_connections(self, model_class, model_config, sample_data):
        """Test TCN residual connections."""
        model = model_class(config=model_config)
        
        # Test residual connections
        x = torch.randn(1, model_config['sequence_length'], model_config['input_size'])
        output = model(x)
        
        # Check output shape
        assert output.shape == (1, model_config['output_size'])
    
    def test_tcn_sequence_handling(self, model_class, model_config, sample_data):
        """Test TCN sequence handling."""
        model = model_class(config=model_config)
        
        # Test with different sequence lengths
        short_seq = sample_data.iloc[:model_config['sequence_length'] + 5]
        long_seq = sample_data.iloc[:model_config['sequence_length'] + 20]
        
        # Test data preparation
        X_short, y_short = model._prepare_data(short_seq, is_training=True)
        X_long, y_long = model._prepare_data(long_seq, is_training=True)
        
        # Check shapes
        assert X_short.shape[1] == model_config['sequence_length']
        assert X_long.shape[1] == model_config['sequence_length']
        assert X_short.shape[2] == model_config['input_size']
        assert y_short.shape[1] == model_config['output_size']
    
    def test_tcn_data_normalization(self, model_class, model_config, sample_data):
        """Test TCN data normalization."""
        model = model_class(config=model_config)
        
        # Prepare data
        X, y = model._prepare_data(sample_data, is_training=True)
        
        # Check normalization
        assert torch.allclose(X.mean(dim=(0, 1)), torch.zeros(model_config['input_size']), atol=1e-6)
        assert torch.allclose(X.std(dim=(0, 1)), torch.ones(model_config['input_size']), atol=1e-6)
        assert torch.allclose(y.mean(), torch.zeros(1), atol=1e-6)
        assert torch.allclose(y.std(), torch.ones(1), atol=1e-6)
    
    def test_tcn_config_validation(self, model_class, model_config):
        """Test TCN configuration validation."""
        # Test invalid sequence length
        invalid_config = model_config.copy()
        invalid_config['sequence_length'] = 1
        with pytest.raises(ValueError, match="Sequence length must be at least 2"):
            model_class(config=invalid_config)
        
        # Test invalid feature columns
        invalid_config = model_config.copy()
        invalid_config['feature_columns'] = ['close']
        with pytest.raises(ValueError, match="Number of feature columns"):
            model_class(config=invalid_config)
        
        # Test invalid target column
        invalid_config = model_config.copy()
        invalid_config['target_column'] = 'invalid'
        with pytest.raises(ValueError, match="Target column"):
            model_class(config=invalid_config)
    
    def test_tcn_missing_columns(self, model_class, model_config):
        """Test TCN handling of missing columns."""
        model = model_class(config=model_config)
        
        # Create data with missing columns
        data = pd.DataFrame({
            'close': np.random.randn(100),
            'missing': np.random.randn(100)
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            model._prepare_data(data, is_training=True)
    
    def test_tcn_model_save_load(self, model_class, model_config, sample_data, tmp_path):
        """Test TCN model saving and loading."""
        model = model_class(config=model_config)
        model.fit(sample_data, epochs=2, batch_size=4)
        
        # Save model
        save_path = tmp_path / "tcn_model.pt"
        model.save(str(save_path))
        
        # Load model
        new_model = model_class(config=model_config)
        new_model.load(str(save_path))
        
        # Compare predictions
        original_pred = model.predict(sample_data)
        loaded_pred = new_model.predict(sample_data)
        
        assert np.allclose(original_pred['predictions'], loaded_pred['predictions']) 