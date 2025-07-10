"""Tests for the ARIMA forecasting model."""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Try to import ARIMA model with fallback
try:
    from trading.models.arima_model import ARIMAModel
except ImportError:
    # Create a mock if the module doesn't exist
    from unittest.mock import Mock
    ARIMAModel = Mock()
    print("Warning: ARIMAModel not available, using mock")

class TestARIMAModel:
    @pytest.fixture
    def model(self):
        """Create an ARIMA model instance for testing."""
        return ARIMAModel(p=2, d=1, q=2)

    @pytest.fixture
    def sample_data(self, sample_price_data):
        """Create sample price data for testing."""
        return sample_price_data

    def test_model_initialization(self, model):
        """Test that model initializes with correct parameters."""
        assert model.p == 2
        assert model.d == 1
        assert model.q == 2
        assert model.name == 'ARIMA'

    def test_data_preprocessing(self, model, sample_data):
        """Test that data is preprocessed correctly."""
        processed_data = model.preprocess_data(sample_data['close'])
        
        assert isinstance(processed_data, pd.Series)
        assert not processed_data.isnull().any()
        assert len(processed_data) == len(sample_data)

    def test_model_fitting(self, model, sample_data):
        """Test that model fits to data correctly."""
        model.fit(sample_data['close'])
        
        assert hasattr(model, 'fitted_model')
        assert model.is_fitted

    def test_forecast_generation(self, model, sample_data):
        """Test that forecasts are generated correctly."""
        model.fit(sample_data['close'])
        forecast = model.forecast(steps=5)
        
        assert isinstance(forecast, pd.Series)
        assert len(forecast) == 5
        assert not forecast.isnull().any()

    def test_forecast_confidence_intervals(self, model, sample_data):
        """Test that confidence intervals are calculated correctly."""
        model.fit(sample_data['close'])
        forecast, lower, upper = model.forecast_with_confidence(steps=5)
        
        assert isinstance(forecast, pd.Series)
        assert isinstance(lower, pd.Series)
        assert isinstance(upper, pd.Series)
        assert len(forecast) == len(lower) == len(upper) == 5
        assert (lower <= forecast).all()
        assert (forecast <= upper).all()

    def test_model_evaluation(self, model, sample_data):
        """Test that model evaluation metrics are calculated correctly."""
        model.fit(sample_data['close'])
        metrics = model.evaluate(sample_data['close'])
        
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert all(isinstance(v, float) for v in metrics.values())

    def test_parameter_validation(self):
        """Test that model parameters are validated."""
        with pytest.raises(ValueError):
            ARIMAModel(p=-1, d=1, q=2)  # Invalid p
        with pytest.raises(ValueError):
            ARIMAModel(p=2, d=-1, q=2)  # Invalid d
        with pytest.raises(ValueError):
            ARIMAModel(p=2, d=1, q=-1)  # Invalid q

    def test_empty_data_handling(self, model):
        """Test that model handles empty data correctly."""
        empty_data = pd.Series([])
        with pytest.raises(ValueError):
            model.fit(empty_data)

    def test_missing_data_handling(self, model):
        """Test that model handles missing data correctly."""
        data = pd.Series([100, np.nan, 101, 102])
        with pytest.raises(ValueError):
            model.fit(data)

    def test_forecast_horizon_validation(self, model, sample_data):
        """Test that forecast horizon is validated."""
        model.fit(sample_data['close'])
        with pytest.raises(ValueError):
            model.forecast(steps=0)  # Invalid forecast horizon

    def test_model_persistence(self, model, sample_data, tmp_path):
        """Test that model can be saved and loaded."""
        # Fit model
        model.fit(sample_data['close'])
        
        # Save model
        model_path = tmp_path / "arima_model.pkl"
        model.save(model_path)
        
        # Load model
        loaded_model = ARIMAModel.load(model_path)
        
        # Verify loaded model
        assert loaded_model.p == model.p
        assert loaded_model.d == model.d
        assert loaded_model.q == model.q
        assert loaded_model.is_fitted

    def test_forecast_consistency(self, model, sample_data):
        """Test that forecasts are consistent across multiple calls."""
        model.fit(sample_data['close'])
        forecast1 = model.forecast(steps=5)
        forecast2 = model.forecast(steps=5)
        
        pd.testing.assert_series_equal(forecast1, forecast2)

    def test_trend_detection(self, model):
        """Test that model correctly identifies trends."""
        # Create data with clear trend
        trend_data = pd.Series(np.linspace(100, 200, 100))
        model.fit(trend_data)
        forecast = model.forecast(steps=5)
        
        # Forecast should continue the trend
        assert (forecast.diff().dropna() > 0).all()

    def test_seasonality_handling(self, model):
        """Test that model handles seasonal data correctly."""
        # Create seasonal data
        t = np.linspace(0, 4*np.pi, 100)
        seasonal_data = pd.Series(100 + 10*np.sin(t))
        model.fit(seasonal_data)
        forecast = model.forecast(steps=5)
        
        # Forecast should capture seasonality
        assert len(forecast) == 5
        assert not forecast.isnull().any()

    def test_constant_series_handling(self, model):
        """Test that model handles constant series correctly."""
        # Create constant data
        constant_data = pd.Series([100.0] * 50)
        
        # Should handle constant series gracefully
        result = model.fit(constant_data)
        assert result['success'] is True or 'constant' in result.get('error', '').lower()
        
        if result['success']:
            forecast = model.predict(steps=5)
            assert forecast['success'] is True
            assert len(forecast['predictions']) == 5

    def test_short_series_handling(self, model):
        """Test that model handles very short time series correctly."""
        # Test with less than 3 datapoints
        short_data = pd.Series([100, 101])
        
        result = model.fit(short_data)
        # Should fail gracefully with clear error message
        assert result['success'] is False
        assert 'insufficient' in result.get('error', '').lower() or 'at least' in result.get('error', '').lower()
        
        # Test with exactly 3 datapoints (minimum for ARIMA)
        minimal_data = pd.Series([100, 101, 102])
        result = model.fit(minimal_data)
        # Should work or fail with clear message
        assert isinstance(result, dict)
        assert 'success' in result

    def test_sarima_support(self, model):
        """Test that model supports SARIMA with seasonal orders."""
        # Create seasonal data
        t = np.linspace(0, 8*np.pi, 200)
        seasonal_data = pd.Series(100 + 10*np.sin(t) + np.random.normal(0, 1, 200))
        
        # Test with seasonal order
        sarima_config = {
            'order': (1, 1, 1),
            'seasonal_order': (1, 1, 1, 12)  # Monthly seasonality
        }
        sarima_model = ARIMAModel(config=sarima_config)
        
        result = sarima_model.fit(seasonal_data)
        assert result['success'] is True
        
        if result['success']:
            forecast = sarima_model.predict(steps=5)
            assert forecast['success'] is True
            assert len(forecast['predictions']) == 5 