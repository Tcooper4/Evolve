import pytest
import torch
import numpy as np
import pandas as pd
from trading.models.advanced.tcn_model import TCNForecaster
from tests.unit.base_test import BaseModelTest

class TestTCNModel(BaseModelTest):
    """Test suite for TCN model."""
    
    @pytest.fixture
    def model_class(self):
        return TCNForecaster
    
    def test_tcn_specific_features(self, model_class, model_config, sample_data):
        """Test TCN-specific features."""
        model = model_class(config=model_config)
        model.fit(sample_data, epochs=2, batch_size=4)
        
        # Test TCN architecture
        assert model.model.tcn is not None
        assert model.model.tcn.num_channels[0] == model_config.num_channels[0]
        assert model.model.tcn.kernel_size == model_config.kernel_size
        assert model.model.tcn.dropout == model_config.dropout
    
    def test_tcn_dilation(self, model_class, model_config, sample_data):
        """Test TCN dilation mechanism."""
        model = model_class(config=model_config)
        model.fit(sample_data, epochs=1, batch_size=4)
        
        # Check dilation factors
        for i, layer in enumerate(model.model.tcn.layers):
            assert layer.dilation == 2 ** i
    
    def test_tcn_residual_connections(self, model_class, model_config, sample_data):
        """Test TCN residual connections."""
        model = model_class(config=model_config)
        model.fit(sample_data, epochs=1, batch_size=4)
        
        # Test residual connections
        x = torch.randn(1, model_config.num_channels[0], 10)
        output = model.model.tcn(x)
        
        # Check output shape
        assert output.shape[1] == model_config.num_channels[-1]
        assert output.shape[2] == x.shape[2]
    
    def test_tcn_causal_convolution(self, model_class, model_config, sample_data):
        """Test TCN causal convolution."""
        model = model_class(config=model_config)
        
        # Test causal padding
        x = torch.randn(1, model_config.num_channels[0], 10)
        output = model.model.tcn(x)
        
        # Check that output doesn't depend on future values
        for i in range(output.shape[2]):
            assert not torch.any(output[0, :, i] != 0)
    
    def test_tcn_gradient_flow(self, model_class, model_config, sample_data):
        """Test TCN gradient flow."""
        model = model_class(config=model_config)
        model.fit(sample_data, epochs=1, batch_size=4)
        
        # Check if gradients are flowing
        for param in model.model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()
                assert not torch.isinf(param.grad).any()
    
    def test_tcn_sequence_handling(self, model_class, model_config, sample_data):
        """Test TCN sequence handling."""
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
    
    def test_tcn_memory_efficiency(self, model_class, model_config, sample_data):
        """Test TCN memory efficiency."""
        model = model_class(config=model_config)
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        model.fit(sample_data, epochs=2, batch_size=4)
        predictions = model.predict(sample_data)
        
        # Check if memory was properly managed
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            assert final_memory <= initial_memory * 2  # Allow some memory growth 