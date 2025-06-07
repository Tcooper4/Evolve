import pytest
import numpy as np
import torch
import pandas as pd
from trading.models.advanced.ensemble.ensemble_model import EnsembleForecaster
from trading.models.lstm_model import LSTMForecaster
from tests.unit.base_test import BaseModelTest

class TestEnsembleModel(BaseModelTest):
    """Test suite for ensemble model."""
    
    @pytest.fixture
    def model_class(self):
        return EnsembleForecaster
        
    @pytest.fixture
    def model_config(self):
        return {
            'models': [LSTMForecaster(), LSTMForecaster()],
            'use_lr_scheduler': True
        }
    
    def test_ensemble_specific_features(self, model_class, model_config, sample_data):
        """Test ensemble-specific features."""
        model = model_class(config=model_config)
        model.fit(sample_data, epochs=2, batch_size=4)
        assert len(model.models) == len(model_config['models'])
        
    def test_ensemble_prediction_aggregation(self, model_class, model_config, sample_data):
        """Test ensemble prediction aggregation."""
        model = model_class(config=model_config)
        model.fit(sample_data, epochs=2, batch_size=4)
        predictions = model.predict(sample_data)
        assert 'predictions' in predictions
        assert len(predictions['predictions']) > 0 