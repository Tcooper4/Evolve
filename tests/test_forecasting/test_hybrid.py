"""Tests for the Hybrid forecasting model."""

import pytest
import pandas as pd
import numpy as np
from trading.forecasting.hybrid_model import HybridModel
from trading.forecasting.arima_model import ARIMAModel
from trading.forecasting.lstm_model import LSTMModel
from trading.forecasting.prophet_model import ProphetModel

class TestHybridModel:
    @pytest.fixture
    def model(self):
        """Create a Hybrid model instance for testing."""
        return HybridModel(
            arima_model=ARIMAModel(p=2, d=1, q=2),
            lstm_model=LSTMModel(sequence_length=10, n_units=50),
            prophet_model=ProphetModel()
        )

    @pytest.fixture
    def sample_data(self, sample_price_data):
        """Create sample price data for testing."""
        # Ensure data has datetime index for Prophet
        data = sample_price_data.copy()
        data.index = pd.date_range(start='2020-01-01', periods=len(data), freq='D')
        return data

    def test_model_initialization(self, model):
        """Test that model initializes with correct parameters."""
        assert isinstance(model.arima_model, ARIMAModel)
        assert isinstance(model.lstm_model, LSTMModel)
        assert isinstance(model.prophet_model, ProphetModel)
        assert model.name == 'Hybrid'

    def test_model_fitting(self, model, sample_data):
        """Test that model fits to data correctly."""
        model.fit(sample_data['close'])
        
        assert model.is_fitted
        assert model.arima_model.is_fitted
        assert model.lstm_model.is_fitted
        assert model.prophet_model.is_fitted

    def test_forecast_generation(self, model, sample_data):
        """Test that forecasts are generated correctly."""
        model.fit(sample_data['close'])
        forecast = model.forecast(steps=5)
        
        assert isinstance(forecast, pd.Series)
        assert len(forecast) == 5
        assert not forecast.isnull().any()

    def test_forecast_components(self, model, sample_data):
        """Test that forecast components are calculated correctly."""
        model.fit(sample_data['close'])
        forecast, components = model.forecast_with_components(steps=5)
        
        assert isinstance(forecast, pd.Series)
        assert isinstance(components, pd.DataFrame)
        assert 'arima' in components.columns
        assert 'lstm' in components.columns
        assert 'prophet' in components.columns
        assert len(forecast) == len(components) == 5

    def test_model_evaluation(self, model, sample_data):
        """Test that model evaluation metrics are calculated correctly."""
        model.fit(sample_data['close'])
        metrics = model.evaluate(sample_data['close'])
        
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert all(isinstance(v, float) for v in metrics.values())

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
        model_path = tmp_path / "hybrid_model"
        model.save(model_path)
        
        # Load model
        loaded_model = HybridModel.load(model_path)
        
        # Verify loaded model
        assert isinstance(loaded_model.arima_model, ARIMAModel)
        assert isinstance(loaded_model.lstm_model, LSTMModel)
        assert isinstance(loaded_model.prophet_model, ProphetModel)
        assert loaded_model.is_fitted

    def test_ensemble_weights(self, model, sample_data):
        """Test that ensemble weights are calculated correctly."""
        model.fit(sample_data['close'])
        weights = model.calculate_ensemble_weights(sample_data['close'])
        
        assert isinstance(weights, dict)
        assert 'arima' in weights
        assert 'lstm' in weights
        assert 'prophet' in weights
        assert sum(weights.values()) == pytest.approx(1.0)
        assert all(0 <= w <= 1 for w in weights.values())

    def test_individual_forecasts(self, model, sample_data):
        """Test that individual model forecasts are generated correctly."""
        model.fit(sample_data['close'])
        forecasts = model.generate_individual_forecasts(steps=5)
        
        assert isinstance(forecasts, dict)
        assert 'arima' in forecasts
        assert 'lstm' in forecasts
        assert 'prophet' in forecasts
        assert all(len(f) == 5 for f in forecasts.values())
        assert all(not f.isnull().any() for f in forecasts.values())

    def test_forecast_consistency(self, model, sample_data):
        """Test that forecasts are consistent across multiple calls."""
        model.fit(sample_data['close'])
        forecast1 = model.forecast(steps=5)
        forecast2 = model.forecast(steps=5)
        
        pd.testing.assert_series_equal(forecast1, forecast2)

    def test_model_adaptation(self, model, sample_data):
        """Test that model adapts to changing patterns."""
        # Create data with changing patterns
        dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
        data = pd.Series(index=dates)
        
        # First 100 days: linear trend
        data.iloc[:100] = np.linspace(100, 200, 100)
        # Next 100 days: seasonal pattern
        data.iloc[100:] = 150 + 20*np.sin(np.linspace(0, 4*np.pi, 100))
        
        model.fit(data)
        forecast = model.forecast(steps=5)
        
        assert len(forecast) == 5
        assert not forecast.isnull().any()

    def test_error_handling(self, model, sample_data):
        """Test that model handles errors in individual models correctly."""
        # Corrupt one of the models
        model.arima_model = None
        
        with pytest.raises(ValueError):
            model.fit(sample_data['close']) 