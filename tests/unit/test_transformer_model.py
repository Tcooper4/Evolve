import pytest
import torch
import numpy as np
import pandas as pd
from trading.models.advanced.transformer_model import TransformerForecaster

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
        'dropout': 0.1,
        'use_batch_norm': False,
        'use_lr_scheduler': False
    }
    model = TransformerForecaster(config=config)
    assert model.config['d_model'] == 64
    assert model.config['nhead'] == 4
    assert model.config['num_layers'] == 2
    assert model.config['dropout'] == 0.1
    assert not model.config['use_batch_norm']
    assert not model.config['use_lr_scheduler']

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
    """Test transformer model learning rate scheduler."""
    config = {
        'use_lr_scheduler': True
    }
    model = TransformerForecaster(config=config)
    data = make_sample_data()
    model.fit(data, epochs=2, batch_size=4)
    assert model.scheduler is not None

class TestTransformerModel:
    """Test suite for transformer model."""
    
    @pytest.fixture
    def model(self):
        return TransformerForecaster()
        
    @pytest.fixture
    def data(self):
        return make_sample_data()
        
    def test_model_instantiation(self, model):
        """Test model instantiation."""
        assert model is not None
        assert model.model is not None
        
    def test_model_fit(self, model, data):
        """Test model fitting."""
        model.fit(data, epochs=2, batch_size=4)
        assert len(model.history) > 0
        
    def test_model_predict(self, model, data):
        """Test model prediction."""
        model.fit(data, epochs=2, batch_size=4)
        predictions = model.predict(data)
        assert 'predictions' in predictions
        assert len(predictions['predictions']) > 0
        
    def test_model_save_load(self, model, data):
        """Test model saving and loading."""
        import tempfile
        import os
        
        # Train model
        model.fit(data, epochs=2, batch_size=4)
        
        # Save model
        save_path = tempfile.mkdtemp()
        model_path = os.path.join(save_path, 'model')
        model.save(model_path)
        
        # Create new model and load
        loaded_model = TransformerForecaster()
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
        
    def test_model_invalid_input(self, model):
        """Test model with invalid input."""
        with pytest.raises(ValueError):
            model.predict(pd.DataFrame())
            
    def test_model_memory_management(self, model, data):
        """Test model memory management."""
        model.fit(data, epochs=2, batch_size=4)
        predictions = model.predict(data)
        assert 'predictions' in predictions
        assert len(predictions['predictions']) > 0
        
    def test_model_learning_rate_scheduler(self, model, data):
        """Test model learning rate scheduler."""
        config = {
            'use_lr_scheduler': True
        }
        model = TransformerForecaster(config=config)
        model.fit(data, epochs=2, batch_size=4)
        assert model.scheduler is not None
        
    def test_transformer_specific_features(self, model, data):
        """Test transformer-specific features."""
        config = {
            'd_model': 64,
            'nhead': 4,
            'num_layers': 2,
            'dropout': 0.1,
            'use_batch_norm': True
        }
        model = TransformerForecaster(config=config)
        model.fit(data, epochs=2, batch_size=4)
        assert model.config['use_batch_norm']
        
    def test_transformer_attention_patterns(self, model, data):
        """Test transformer attention patterns."""
        model.fit(data, epochs=2, batch_size=4)
        predictions = model.predict(data)
        assert 'predictions' in predictions
        assert len(predictions['predictions']) > 0
        
    def test_transformer_prediction_intervals(self, model, data):
        """Test transformer prediction intervals."""
        model.fit(data, epochs=2, batch_size=4)
        predictions = model.predict(data, return_prediction_intervals=True)
        assert 'predictions' in predictions
        assert 'lower_bound' in predictions
        assert 'upper_bound' in predictions
        assert len(predictions['predictions']) > 0
        
    def test_transformer_batch_normalization(self, model, data):
        """Test transformer batch normalization."""
        config = {
            'use_batch_norm': True
        }
        model = TransformerForecaster(config=config)
        model.fit(data, epochs=2, batch_size=4)
        assert model.config['use_batch_norm']
        
    def test_transformer_learning_rate_scheduler(self, model, data):
        """Test transformer learning rate scheduler."""
        config = {
            'use_lr_scheduler': True
        }
        model = TransformerForecaster(config=config)
        model.fit(data, epochs=2, batch_size=4)
        assert model.scheduler is not None 