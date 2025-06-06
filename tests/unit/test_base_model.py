import pytest
import torch
import numpy as np
import pandas as pd
from trading.models.base_model import BaseModel
from tests.unit.base_test import BaseModelTest

class TestBaseModel(BaseModelTest):
    """Test suite for Base Model."""
    
    @pytest.fixture
    def model_class(self):
        return BaseModel
    
    def test_base_model_initialization(self, model_class, model_config):
        """Test base model initialization."""
        model = model_class(config=model_config)
        
        # Test basic attributes
        assert model.config == model_config
        assert model.device == torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert model.model is None
        assert model.optimizer is None
        assert model.scheduler is None
        assert model.scaler is None
        assert model.history == []
    
    def test_base_model_data_preparation(self, model_class, model_config, sample_data):
        """Test data preparation methods."""
        model = model_class(config=model_config)
        
        # Test data validation
        with pytest.raises(ValueError):
            model._validate_data(pd.DataFrame())
        
        # Test data scaling
        scaled_data = model._scale_data(sample_data)
        assert isinstance(scaled_data, pd.DataFrame)
        assert not scaled_data.isna().any().any()
        
        # Test data unscaling
        unscaled_data = model._unscale_data(scaled_data)
        assert isinstance(unscaled_data, pd.DataFrame)
        assert not unscaled_data.isna().any().any()
    
    def test_base_model_device_management(self, model_class, model_config):
        """Test device management."""
        model = model_class(config=model_config)
        
        # Test device placement
        tensor = torch.randn(10)
        tensor = model._to_device(tensor)
        assert tensor.device == model.device
    
    def test_base_model_optimizer_setup(self, model_class, model_config):
        """Test optimizer setup."""
        model = model_class(config=model_config)
        model.model = torch.nn.Linear(10, 1)
        
        # Test optimizer initialization
        model._setup_optimizer()
        assert model.optimizer is not None
        assert isinstance(model.optimizer, torch.optim.Optimizer)
    
    def test_base_model_scheduler_setup(self, model_class, model_config):
        """Test learning rate scheduler setup."""
        model = model_class(config=model_config)
        model.model = torch.nn.Linear(10, 1)
        model._setup_optimizer()
        
        # Test scheduler initialization
        model._setup_scheduler()
        assert model.scheduler is not None
        assert isinstance(model.scheduler, torch.optim.lr_scheduler._LRScheduler)
    
    def test_base_model_checkpoint_management(self, model_class, model_config, temp_dir):
        """Test checkpoint management."""
        model = model_class(config=model_config)
        model.model = torch.nn.Linear(10, 1)
        model._setup_optimizer()
        model._setup_scheduler()
        
        # Test checkpoint saving
        checkpoint_path = temp_dir / 'checkpoint.pt'
        model._save_checkpoint(str(checkpoint_path))
        assert checkpoint_path.exists()
        
        # Test checkpoint loading
        loaded_model = model_class(config=model_config)
        loaded_model.model = torch.nn.Linear(10, 1)
        loaded_model._setup_optimizer()
        loaded_model._setup_scheduler()
        loaded_model._load_checkpoint(str(checkpoint_path))
        
        # Compare model states
        for p1, p2 in zip(model.model.parameters(), loaded_model.model.parameters()):
            assert torch.allclose(p1, p2)
    
    def test_base_model_early_stopping(self, model_class, model_config, sample_data):
        """Test early stopping mechanism."""
        model = model_class(config=model_config)
        model.model = torch.nn.Linear(10, 1)
        model._setup_optimizer()
        
        # Test early stopping
        model._setup_early_stopping(patience=2)
        assert model.early_stopping is not None
        
        # Simulate training with early stopping
        for epoch in range(5):
            loss = 1.0 / (epoch + 1)  # Decreasing loss
            should_stop = model._check_early_stopping(loss)
            if epoch >= 2:  # Should stop after patience period
                assert should_stop
            else:
                assert not should_stop
    
    def test_base_model_metrics_tracking(self, model_class, model_config, sample_data):
        """Test metrics tracking."""
        model = model_class(config=model_config)
        model.model = torch.nn.Linear(10, 1)
        model._setup_optimizer()
        
        # Test metrics tracking
        for epoch in range(3):
            metrics = {
                'loss': 1.0 / (epoch + 1),
                'val_loss': 1.0 / (epoch + 2)
            }
            model._update_history(metrics)
        
        assert len(model.history) == 3
        assert all('loss' in metrics for metrics in model.history)
        assert all('val_loss' in metrics for metrics in model.history)
    
    def test_base_model_error_handling(self, model_class, model_config):
        """Test error handling."""
        model = model_class(config=model_config)
        
        # Test invalid model state
        with pytest.raises(ValueError):
            model.predict(pd.DataFrame())
        
        # Test invalid data type
        with pytest.raises(TypeError):
            model._validate_data("invalid_data")
        
        # Test invalid device
        with pytest.raises(RuntimeError):
            model._to_device(torch.randn(10), device='invalid_device') 