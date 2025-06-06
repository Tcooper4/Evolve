import pytest
import torch
import numpy as np
import pandas as pd
from trading.models.advanced.transformer_model import TransformerForecaster
from tests.unit.base_test import BaseModelTest

def make_sample_data():
    """Create sample data for testing."""
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    return data

def test_transformer_config():
    """Test transformer configuration."""
    config = {
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'dropout': 0.1
    }
    model = TransformerForecaster(config=config)
    assert model.config['d_model'] == 64
    assert model.config['nhead'] == 4

def test_transformer_forecaster_instantiation():
    """Test transformer forecaster instantiation."""
    model = TransformerForecaster()
    assert model is not None
    assert model.model is not None

def test_transformer_forecaster_fit():
    """Test transformer forecaster fitting."""
    model = TransformerForecaster()
    data = make_sample_data()
    model.fit(data, epochs=2, batch_size=4)
    assert len(model.history) > 0

def test_transformer_forecaster_predict():
    """Test transformer forecaster prediction."""
    model = TransformerForecaster()
    data = make_sample_data()
    model.fit(data, epochs=2, batch_size=4)
    predictions = model.predict(data)
    assert 'predictions' in predictions
    assert len(predictions['predictions']) > 0

def test_transformer_forecaster_save_load():
    """Test transformer forecaster saving and loading."""
    import tempfile
    import os
    
    # Create and train model
    model = TransformerForecaster()
    data = make_sample_data()
    model.fit(data, epochs=2, batch_size=4)
    
    # Save model
    save_path = tempfile.mkdtemp()
    model_path = os.path.join(save_path, 'model')
    model.save(model_path)
    
    # Create new model and load
    loaded_model = TransformerForecaster()
    loaded_model.load(model_path)
    
    # Set both models to eval mode
    model.model.eval()
    loaded_model.model.eval()
    
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

def test_transformer_model_learning_rate_scheduler():
    """Test learning rate scheduler functionality."""
    # Create model with scheduler enabled
    model = TransformerForecaster(use_lr_scheduler=True)
    data = make_sample_data()
    
    # Initialize model by fitting
    model.fit(data, epochs=1, batch_size=4)
    
    # Get initial learning rate
    initial_lr = model.optimizer.param_groups[0]['lr']
    
    # Train for a few more epochs
    model.fit(data, epochs=2, batch_size=4)
    
    # Check that scheduler was created
    assert model.scheduler is not None
    
    # Check that learning rate changed
    final_lr = model.optimizer.param_groups[0]['lr']
    assert final_lr != initial_lr 

class TestTransformerModel(BaseModelTest):
    """Test suite for Transformer model."""
    
    @pytest.fixture
    def model_class(self):
        return TransformerForecaster
    
    def test_transformer_specific_features(self, model_class, model_config, sample_data):
        """Test Transformer-specific features."""
        model = model_class(config=model_config)
        model.fit(sample_data, epochs=2, batch_size=4)
        
        # Test attention mechanism
        assert model.model.transformer_encoder is not None
        assert model.model.transformer_encoder.layers[0].self_attn is not None
        
        # Test positional encoding
        assert model.model.pos_encoder is not None
        
        # Test model dimensions
        assert model.model.d_model == model_config.d_model
        assert model.model.nhead == model_config.nhead
    
    def test_transformer_attention_patterns(self, model_class, model_config, sample_data):
        """Test Transformer attention patterns."""
        model = model_class(config=model_config)
        model.fit(sample_data, epochs=1, batch_size=4)
        
        # Get attention weights
        with torch.no_grad():
            x = model._prepare_data(sample_data, is_training=False)[0]
            attn_output, attn_weights = model.model.transformer_encoder.layers[0].self_attn(
                x, x, x, need_weights=True
            )
        
        # Check attention weights
        assert attn_weights is not None
        assert attn_weights.shape[0] == model_config.nhead
        assert not torch.isnan(attn_weights).any()
        assert not torch.isinf(attn_weights).any()
    
    def test_transformer_prediction_intervals(self, model_class, model_config, sample_data):
        """Test Transformer prediction intervals."""
        model = model_class(config=model_config)
        model.fit(sample_data, epochs=2, batch_size=4)
        
        predictions = model.predict(
            sample_data,
            return_prediction_intervals=True,
            n_samples=100
        )
        
        assert 'predictions' in predictions
        assert 'lower_bound' in predictions
        assert 'upper_bound' in predictions
        
        # Check interval validity
        assert np.all(predictions['lower_bound'] <= predictions['predictions'])
        assert np.all(predictions['predictions'] <= predictions['upper_bound'])
    
    def test_transformer_batch_normalization(self, model_class, model_config, sample_data):
        """Test Transformer batch normalization."""
        model_config.use_batch_norm = True
        model = model_class(config=model_config)
        model.fit(sample_data, epochs=2, batch_size=4)
        
        assert model.model.batch_norm is not None
        
        # Check batch norm statistics
        assert model.model.batch_norm.running_mean is not None
        assert model.model.batch_norm.running_var is not None
    
    def test_transformer_learning_rate_scheduler(self, model_class, model_config, sample_data):
        """Test Transformer learning rate scheduler."""
        model = model_class(config=model_config)
        model.fit(sample_data, epochs=1, batch_size=4)
        
        initial_lr = model.optimizer.param_groups[0]['lr']
        model.fit(sample_data, epochs=2, batch_size=4)
        final_lr = model.optimizer.param_groups[0]['lr']
        
        assert final_lr != initial_lr
        assert model.scheduler is not None 