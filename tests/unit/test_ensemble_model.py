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

    def test_predict_with_confidence_alpha(self, model_class, model_config, sample_data):
        """Test confidence interval width changes with alpha."""
        model = model_class(config=model_config)
        model.fit(sample_data, epochs=2, batch_size=4)

        X, _ = model._prepare_data(sample_data, is_training=False)

        preds_95 = model.predict_with_confidence(X, alpha=0.05)
        preds_90 = model.predict_with_confidence(X, alpha=0.10)

        width_95 = (preds_95['upper'] - preds_95['lower']).mean().item()
        width_90 = (preds_90['upper'] - preds_90['lower']).mean().item()

        assert width_95 > width_90
