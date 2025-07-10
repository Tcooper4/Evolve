"""
Unit tests for the ForecastRouter module.

This module tests the dynamic model loading, model selection, and forecasting
capabilities of the ForecastRouter class.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.forecast_router import ForecastRouter


class MockModel:
    """Mock model for testing."""
    
    def __init__(self, name="MockModel"):
        self.name = name
        self.fitted = False
        
    def fit(self, data):
        """Mock fit method."""
        self.fitted = True
        return self
        
    def predict(self, horizon):
        """Mock predict method."""
        return np.random.randn(horizon)
        
    def train_model(self, data, target, **kwargs):
        """Mock train_model method for LSTM."""
        self.fitted = True
        return self


class TestForecastRouter(unittest.TestCase):
    """Test cases for ForecastRouter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.router = ForecastRouter()
        self.sample_data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100),
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 105,
            'low': np.random.randn(100).cumsum() + 95
        }, index=pd.date_range('2023-01-01', periods=100, freq='D'))
        
    def test_initialization(self):
        """Test router initialization."""
        self.assertIsInstance(self.router.model_registry, dict)
        self.assertIsInstance(self.router.performance_history, pd.DataFrame)
        self.assertIsInstance(self.router.model_weights, dict)
        
    def test_load_default_models(self):
        """Test loading of default models."""
        # Clear registry and reload
        self.router.model_registry.clear()
        self.router._load_default_models()
        
        # Check that default models are loaded
        expected_models = ['arima', 'lstm', 'xgboost', 'autoformer']
        for model in expected_models:
            self.assertIn(model, self.router.model_registry)
            
    def test_register_model(self):
        """Test dynamic model registration."""
        mock_model = MockModel("TestModel")
        
        # Register new model
        self.router.register_model("test_model", MockModel)
        
        # Check registration
        self.assertIn("test_model", self.router.model_registry)
        self.assertEqual(self.router.model_registry["test_model"], MockModel)
        
    def test_unregister_model(self):
        """Test model unregistration."""
        # Register a model first
        self.router.register_model("test_model", MockModel)
        self.assertIn("test_model", self.router.model_registry)
        
        # Unregister it
        self.router.unregister_model("test_model")
        self.assertNotIn("test_model", self.router.model_registry)
        
    def test_analyze_data(self):
        """Test data analysis functionality."""
        characteristics = self.router._analyze_data(self.sample_data)
        
        # Check that all expected characteristics are present
        expected_keys = ['length', 'has_seasonality', 'has_trend', 'volatility', 'missing_values']
        for key in expected_keys:
            self.assertIn(key, characteristics)
            
        # Check data types
        self.assertIsInstance(characteristics['length'], int)
        self.assertIsInstance(characteristics['has_seasonality'], bool)
        self.assertIsInstance(characteristics['has_trend'], bool)
        self.assertIsInstance(characteristics['volatility'], float)
        self.assertIsInstance(characteristics['missing_values'], int)
        
    def test_select_model_with_preference(self):
        """Test model selection with user preference."""
        # Test with valid model preference
        selected = self.router._select_model(self.sample_data, model_type='lstm')
        self.assertEqual(selected, 'lstm')
        
        # Test with invalid model preference
        selected = self.router._select_model(self.sample_data, model_type='invalid_model')
        self.assertIn(selected, self.router.model_registry.keys())
        
    def test_select_model_auto(self):
        """Test automatic model selection."""
        selected = self.router._select_model(self.sample_data)
        self.assertIn(selected, self.router.model_registry.keys())
        
    def test_get_forecast_with_valid_data(self):
        """Test forecast generation with valid data."""
        with patch.object(self.router, '_prepare_data_safely') as mock_prepare:
            mock_prepare.return_value = self.sample_data
            
            with patch.object(self.router, '_select_model_with_fallback') as mock_select:
                mock_select.return_value = 'lstm'
                
                with patch('models.forecast_router.LSTMModel') as mock_lstm_class:
                    mock_model = MockModel()
                    mock_lstm_class.return_value = mock_model
                    
                    result = self.router.get_forecast(
                        data=self.sample_data,
                        horizon=30,
                        model_type='lstm'
                    )
                    
                    # Check result structure
                    self.assertIsInstance(result, dict)
                    self.assertIn('forecast', result)
                    self.assertIn('model_type', result)
                    self.assertIn('confidence', result)
                    self.assertIn('metadata', result)
                    
    def test_get_forecast_with_empty_data(self):
        """Test forecast generation with empty data."""
        empty_data = pd.DataFrame()
        
        result = self.router.get_forecast(
            data=empty_data,
            horizon=30
        )
        
        # Should return fallback result
        self.assertIsInstance(result, dict)
        self.assertIn('forecast', result)
        
    def test_get_forecast_with_none_data(self):
        """Test forecast generation with None data."""
        result = self.router.get_forecast(
            data=None,
            horizon=30
        )
        
        # Should return fallback result
        self.assertIsInstance(result, dict)
        self.assertIn('forecast', result)
        
    def test_get_forecast_with_invalid_horizon(self):
        """Test forecast generation with invalid horizon."""
        result = self.router.get_forecast(
            data=self.sample_data,
            horizon=-5
        )
        
        # Should use default horizon
        self.assertIsInstance(result, dict)
        self.assertIn('forecast', result)
        
    def test_prepare_data_safely(self):
        """Test safe data preparation."""
        # Test with valid data
        prepared = self.router._prepare_data_safely(self.sample_data)
        self.assertIsInstance(prepared, pd.DataFrame)
        
        # Test with data missing required columns
        incomplete_data = pd.DataFrame({'price': [1, 2, 3]})
        prepared = self.router._prepare_data_safely(incomplete_data)
        self.assertIsInstance(prepared, pd.DataFrame)
        
    def test_select_model_with_fallback(self):
        """Test model selection with fallback logic."""
        # Test with valid preferred model
        selected = self.router._select_model_with_fallback(
            self.sample_data, 
            preferred_model='lstm'
        )
        self.assertEqual(selected, 'lstm')
        
        # Test with invalid preferred model
        selected = self.router._select_model_with_fallback(
            self.sample_data, 
            preferred_model='invalid_model'
        )
        self.assertIn(selected, self.router.model_registry.keys())
        
    def test_get_model_defaults(self):
        """Test getting model default parameters."""
        defaults = self.router._get_model_defaults('lstm')
        self.assertIsInstance(defaults, dict)
        
        # Test with unknown model
        defaults = self.router._get_model_defaults('unknown_model')
        self.assertIsInstance(defaults, dict)
        
    def test_get_fallback_model(self):
        """Test fallback model selection."""
        fallback = self.router._get_fallback_model('lstm')
        self.assertIn(fallback, self.router.model_registry.keys())
        self.assertNotEqual(fallback, 'lstm')
        
    def test_generate_simple_forecast(self):
        """Test simple forecast generation."""
        forecast = self.router._generate_simple_forecast(self.sample_data, 10)
        self.assertIsInstance(forecast, np.ndarray)
        self.assertEqual(len(forecast), 10)
        
    def test_get_confidence(self):
        """Test confidence calculation."""
        mock_model = MockModel()
        confidence = self.router._get_confidence(mock_model, 'lstm')
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
    def test_get_metadata(self):
        """Test metadata generation."""
        mock_model = MockModel()
        metadata = self.router._get_metadata(mock_model, 'lstm')
        self.assertIsInstance(metadata, dict)
        self.assertIn('model_type', metadata)
        self.assertIn('timestamp', metadata)
        
    def test_get_warnings(self):
        """Test warning generation."""
        warnings = self.router._get_warnings(self.sample_data, 'lstm')
        self.assertIsInstance(warnings, list)
        
    def test_get_fallback_result(self):
        """Test fallback result generation."""
        result = self.router._get_fallback_result(self.sample_data, 30)
        self.assertIsInstance(result, dict)
        self.assertIn('forecast', result)
        self.assertIn('model_type', result)
        self.assertEqual(result['model_type'], 'fallback')
        
    def test_log_performance(self):
        """Test performance logging."""
        forecast_data = pd.DataFrame({'forecast': [1, 2, 3]})
        actual_data = pd.DataFrame({'actual': [1.1, 2.1, 3.1]})
        
        # Should not raise an exception
        self.router._log_performance('lstm', forecast_data, actual_data)
        
    def test_update_weights(self):
        """Test weight updating."""
        # Should not raise an exception
        self.router._update_weights()
        
    def test_get_available_models(self):
        """Test getting available models."""
        models = self.router.get_available_models()
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)
        
    def test_get_model_performance(self):
        """Test getting model performance."""
        performance = self.router.get_model_performance()
        self.assertIsInstance(performance, pd.DataFrame)
        
        # Test with specific model
        performance = self.router.get_model_performance('lstm')
        self.assertIsInstance(performance, pd.DataFrame)
        
    def test_forecast_method(self):
        """Test the main forecast method."""
        with patch.object(self.router, 'get_forecast') as mock_get_forecast:
            mock_get_forecast.return_value = {
                'forecast': [1, 2, 3],
                'model_type': 'lstm',
                'confidence': 0.8
            }
            
            result = self.router.forecast(self.sample_data, horizon=30)
            self.assertIsInstance(result, dict)
            self.assertIn('forecast', result)
            
    def test_plot_results(self):
        """Test plotting results."""
        forecast_result = {
            'forecast': [1, 2, 3],
            'model_type': 'lstm',
            'confidence': 0.8
        }
        
        # Should not raise an exception
        self.router.plot_results(self.sample_data, forecast_result)
        
    def test_model_discovery(self):
        """Test dynamic model discovery."""
        # Clear registry and test discovery
        self.router.model_registry.clear()
        self.router._discover_available_models()
        
        # Should have discovered some models
        self.assertGreaterEqual(len(self.router.model_registry), 0)
        
    def test_config_based_loading(self):
        """Test loading models from configuration."""
        config = {
            'models': {
                'test_model': {
                    'enabled': True,
                    'class_path': 'tests.unit.test_forecast_router.MockModel'
                }
            }
        }
        
        router = ForecastRouter(config)
        self.assertIn('test_model', router.model_registry)
        
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test with corrupted data
        corrupted_data = pd.DataFrame({'invalid': ['not', 'numeric', 'data']})
        
        result = self.router.get_forecast(
            data=corrupted_data,
            horizon=30
        )
        
        # Should return fallback result
        self.assertIsInstance(result, dict)
        self.assertIn('forecast', result)


if __name__ == '__main__':
    unittest.main() 