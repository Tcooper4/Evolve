import pytest
import torch
import numpy as np
from typing import Dict, Any, Type
from abc import ABC

class BaseModelTest(ABC):
    """Base class for model testing."""
    
    @pytest.fixture
    def model_class(self) -> Type:
        """Return the model class to test."""
        raise NotImplementedError
    
    def test_model_instantiation(self, model_class, model_config):
        """Test model instantiation."""
        model = model_class(config=model_config)
        assert model is not None
        assert model.model is not None
    
    def test_model_fit(self, model_class, model_config, sample_data):
        """Test model fitting."""
        model = model_class(config=model_config)
        model.fit(sample_data, epochs=2, batch_size=4)
        assert len(model.history) > 0
    
    def test_model_predict(self, model_class, model_config, sample_data):
        """Test model prediction."""
        model = model_class(config=model_config)
        model.fit(sample_data, epochs=2, batch_size=4)
        predictions = model.predict(sample_data)
        assert 'predictions' in predictions
        assert len(predictions['predictions']) > 0
    
    def test_model_save_load(self, model_class, model_config, sample_data, temp_dir):
        """Test model saving and loading."""
        # Create and train model
        model = model_class(config=model_config)
        model.fit(sample_data, epochs=2, batch_size=4)
        
        # Save model
        model_path = temp_dir / 'model'
        model.save(str(model_path))
        
        # Create new model and load
        loaded_model = model_class(config=model_config)
        loaded_model.load(str(model_path))
        
        # Set both models to eval mode
        model.model.eval()
        loaded_model.model.eval()
        
        # Compare predictions
        original_pred = model.predict(sample_data)
        loaded_pred = loaded_model.predict(sample_data)
        
        np.testing.assert_allclose(
            original_pred['predictions'],
            loaded_pred['predictions'],
            rtol=1e-2, atol=1e-2
        )
    
    def test_model_invalid_input(self, model_class, model_config):
        """Test model with invalid input."""
        model = model_class(config=model_config)
        with pytest.raises(ValueError):
            model.fit(pd.DataFrame())
    
    def test_model_memory_management(self, model_class, model_config, sample_data):
        """Test model memory management."""
        model = model_class(config=model_config)
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        model.fit(sample_data, epochs=2, batch_size=4)
        predictions = model.predict(sample_data)
        
        # Check if memory was properly managed
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            assert final_memory <= initial_memory * 2  # Allow some memory growth
    
    def test_model_learning_rate_scheduler(self, model_class, model_config, sample_data):
        """Test learning rate scheduler functionality."""
        model = model_class(config=model_config)
        model.fit(sample_data, epochs=1, batch_size=4)
        
        initial_lr = model.optimizer.param_groups[0]['lr']
        model.fit(sample_data, epochs=2, batch_size=4)
        final_lr = model.optimizer.param_groups[0]['lr']
        
        assert final_lr != initial_lr 