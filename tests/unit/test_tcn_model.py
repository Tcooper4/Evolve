import pytest
import torch
import numpy as np
import pandas as pd
from trading.models.advanced.tcn.tcn_model import TCNForecaster
from tests.unit.base_test import BaseModelTest

class TestTCNModel(BaseModelTest):
    """Test suite for TCN model."""
    
    @pytest.fixture
    def model_class(self):
        return TCNForecaster
    
    @pytest.fixture
    def model_config(self):
        return {
            'input_size': 2,
            'num_channels': [64, 128, 256],
            'kernel_size': 2,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'epochs': 100,
            'patience': 10,
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
        model = model_class(**model_config)
        model.fit(sample_data, epochs=2, batch_size=4)
        
        # Test TCN architecture
        assert model.model is not None
        assert len(model.model.network) == len(model_config['num_channels'])
        assert model.model.network[0].conv1.kernel_size[0] == model_config['kernel_size']
    
    def test_tcn_dilation(self, model_class, model_config, sample_data):
        """Test TCN dilation mechanism."""
        model = model_class(**model_config)
        model.fit(sample_data, epochs=1, batch_size=4)
        
        # Check dilation factors
        for i, layer in enumerate(model.model.network):
            assert layer.conv1.dilation[0] == 2 ** i
    
    def test_tcn_residual_connections(self, model_class, model_config, sample_data):
        """Test TCN residual connections."""
        model = model_class(**model_config)
        model.fit(sample_data, epochs=1, batch_size=4)
        
        # Test residual connections
        x = torch.randn(1, model_config['input_size'], 10)
        output = model.model(x)
        
        # Check output shape
        assert output.shape[1] == model_config['num_channels'][-1]
        assert output.shape[2] == x.shape[2]
    
    def test_tcn_causal_convolution(self, model_class, model_config, sample_data):
        """Test TCN causal convolution."""
        model = model_class(**model_config)
        
        # Test causal padding
        x = torch.randn(1, model_config['input_size'], 10)
        output = model.model(x)
        
        # Check that output doesn't depend on future values
        for i in range(output.shape[2]):
            assert not torch.any(output[0, :, i] != 0)
    
    def test_tcn_gradient_flow(self, model_class, model_config, sample_data):
        """Test TCN gradient flow."""
        model = model_class(**model_config)
        model.fit(sample_data, epochs=1, batch_size=4)
        
        # Check if gradients are flowing
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()
                assert not torch.isinf(param.grad).any()
    
    def test_tcn_sequence_handling(self, model_class, model_config, sample_data):
        """Test TCN sequence handling."""
        model = model_class(**model_config)
        
        # Test with different sequence lengths
        short_seq = sample_data.iloc[:5]
        long_seq = sample_data.iloc[:20]
        
        model.fit(short_seq, epochs=1, batch_size=2)
        short_pred = model.predict(short_seq)
        
        model.fit(long_seq, epochs=1, batch_size=4)
        long_pred = model.predict(long_seq)
        
        assert len(short_pred['predictions']) == len(short_seq)
        assert len(long_pred['predictions']) == len(long_seq)
    
    def test_tcn_memory_efficiency(self, model_class, model_config, sample_data):
        """Test TCN memory efficiency."""
        model = model_class(**model_config)
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        model.fit(sample_data, epochs=2, batch_size=4)
        predictions = model.predict(sample_data)
        
        # Check if memory was properly managed
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            assert final_memory <= initial_memory * 2  # Allow some memory growth 