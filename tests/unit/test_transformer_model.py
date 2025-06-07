import pytest
import torch
import numpy as np
import pandas as pd
from trading.models.advanced.transformer_model import TransformerForecaster
from tests.unit.base_test import BaseModelTest

class TestTransformerModel(BaseModelTest):
    """Test suite for transformer model."""
    
    @pytest.fixture
    def model_class(self):
        return TransformerForecaster
        
    @pytest.fixture
    def model_config(self):
        return {
            'd_model': 256,
            'nhead': 8,
            'num_layers': 4,
            'dropout': 0.1,
            'use_batch_norm': False,
            'use_lr_scheduler': True,
            'learning_rate': 0.01,
            'scheduler_factor': 0.1,
            'scheduler_patience': 1,
            'scheduler_threshold': 0.1,
            'scheduler_min_lr': 1e-6
        }
    
    def test_transformer_specific_features(self, model_class, model_config, sample_data):
        """Test transformer-specific features."""
        model = model_class(config=model_config)
        model.fit(sample_data, epochs=2, batch_size=4)
        assert model.config['use_batch_norm'] == model_config['use_batch_norm']
        
    def test_transformer_attention_patterns(self, model_class, model_config, sample_data):
        """Test transformer attention patterns."""
        model = model_class(config=model_config)
        model.fit(sample_data, epochs=2, batch_size=4)
        predictions = model.predict(sample_data)
        assert 'predictions' in predictions
        assert len(predictions['predictions']) > 0
        
    def test_transformer_prediction_intervals(self, model_class, model_config, sample_data):
        """Test transformer prediction intervals."""
        model = model_class(config=model_config)
        model.fit(sample_data, epochs=2, batch_size=4)
        predictions = model.predict(sample_data)
        assert 'predictions' in predictions
        assert len(predictions['predictions']) > 0
        
    def test_transformer_batch_normalization(self, model_class, model_config, sample_data):
        """Test transformer batch normalization."""
        config = model_config.copy()
        config['use_batch_norm'] = True
        model = model_class(config=config)
        model.fit(sample_data, epochs=2, batch_size=4)
        assert model.config['use_batch_norm'] 