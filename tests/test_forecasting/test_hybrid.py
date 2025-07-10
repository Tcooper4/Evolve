"""Tests for the Hybrid forecasting model."""

import pytest
import pandas as pd
import numpy as np
from trading.models.ensemble_model import EnsembleModel as HybridModel
from trading.models.arima_model import ARIMAModel
from trading.models.lstm_model import LSTMForecaster
from trading.models.prophet_model import ProphetModel

class TestHybridModel:
    @pytest.fixture
    def model(self):
        """Create a Hybrid model instance for testing."""
        return HybridModel(
            arima_model=ARIMAModel(config={'order': (2, 1, 2)}),
            lstm_model=LSTMForecaster(config={'sequence_length': 10, 'hidden_dim': 50}),
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

    def test_weight_normalization(self, model, sample_data):
        """Test that ensemble weights are properly normalized."""
        model.fit(sample_data['close'])
        
        # Test with unnormalized weights
        unnormalized_weights = {'arima': 0.3, 'lstm': 0.2, 'prophet': 0.1}
        normalized_weights = model._normalize_weights(unnormalized_weights)
        
        # Check normalization
        assert sum(normalized_weights.values()) == pytest.approx(1.0)
        assert all(0 <= w <= 1 for w in normalized_weights.values())
        
        # Check relative proportions are maintained
        assert normalized_weights['arima'] > normalized_weights['lstm'] > normalized_weights['prophet']
        
        # Test with zero weights
        zero_weights = {'arima': 0.0, 'lstm': 0.0, 'prophet': 0.0}
        with pytest.raises(ValueError):
            model._normalize_weights(zero_weights)
        
        # Test with negative weights
        negative_weights = {'arima': -0.1, 'lstm': 0.5, 'prophet': 0.6}
        with pytest.raises(ValueError):
            model._normalize_weights(negative_weights)

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
    
    def test_model_failure_fallback(self, model, sample_data):
        """Test that model handles individual model failures with fallbacks."""
        # Test with one model throwing exception
        original_lstm_predict = model.lstm_model.predict
        
        def failing_predict(*args, **kwargs):
            raise RuntimeError("LSTM model failed")
        
        model.lstm_model.predict = failing_predict
        
        # Should still work with other models
        model.fit(sample_data['close'])
        forecast = model.forecast(steps=5)
        
        assert isinstance(forecast, pd.Series)
        assert len(forecast) == 5
        assert not forecast.isnull().any()
        
        # Restore original method
        model.lstm_model.predict = original_lstm_predict

    def test_ensemble_outperforms_weakest_model(self, model, sample_data):
        """Test that ensemble outperforms the weakest individual model."""
        from sklearn.metrics import mean_squared_error
        
        model.fit(sample_data['close'])
        
        # Get individual model forecasts
        individual_forecasts = model.generate_individual_forecasts(steps=10)
        
        # Get ensemble forecast
        ensemble_forecast = model.forecast(steps=10)
        
        # Calculate MSE for each model
        actual_values = sample_data['close'].tail(10)
        model_mses = {}
        
        for model_name, forecast in individual_forecasts.items():
            mse = mean_squared_error(actual_values, forecast)
            model_mses[model_name] = mse
        
        # Calculate ensemble MSE
        ensemble_mse = mean_squared_error(actual_values, ensemble_forecast)
        
        # Find the worst individual model MSE
        worst_individual_mse = max(model_mses.values())
        
        # Ensemble should perform better than the worst individual model
        assert ensemble_mse <= worst_individual_mse, (
            f"Ensemble MSE ({ensemble_mse:.4f}) should be <= worst individual MSE ({worst_individual_mse:.4f}). "
            f"Individual MSEs: {model_mses}"
        )
        
        # Log performance comparison
        print(f"Ensemble MSE: {ensemble_mse:.4f}")
        print(f"Individual MSEs: {model_mses}")
        print(f"Improvement over worst model: {((worst_individual_mse - ensemble_mse) / worst_individual_mse * 100):.2f}%")

    def test_dynamic_weighting_performance(self, model, sample_data):
        """Test that dynamic weighting improves performance over static weights."""
        from sklearn.metrics import mean_squared_error
        
        model.fit(sample_data['close'])
        
        # Test with static equal weights
        static_weights = {'arima': 0.33, 'lstm': 0.33, 'prophet': 0.34}
        static_forecast = model._weighted_forecast(static_weights, steps=10)
        
        # Test with dynamic weights
        model._update_weights(sample_data)  # Update weights based on performance
        dynamic_weights = model.weights
        dynamic_forecast = model._weighted_forecast(dynamic_weights, steps=10)
        
        # Calculate MSE for both approaches
        actual_values = sample_data['close'].tail(10)
        static_mse = mean_squared_error(actual_values, static_forecast)
        dynamic_mse = mean_squared_error(actual_values, dynamic_forecast)
        
        # Dynamic weighting should not perform worse than static weighting
        # (allowing for some tolerance due to randomness)
        assert dynamic_mse <= static_mse * 1.1, (
            f"Dynamic weighting MSE ({dynamic_mse:.4f}) should not be much worse than "
            f"static weighting MSE ({static_mse:.4f})"
        )
        
        print(f"Static weights MSE: {static_mse:.4f}")
        print(f"Dynamic weights MSE: {dynamic_mse:.4f}")
        print(f"Weight improvement: {((static_mse - dynamic_mse) / static_mse * 100):.2f}%")

    def test_ensemble_robustness(self, model, sample_data):
        """Test ensemble robustness to individual model failures."""
        model.fit(sample_data['close'])
        
        # Test with different subsets of models failing
        original_models = {
            'arima': model.arima_model,
            'lstm': model.lstm_model,
            'prophet': model.prophet_model
        }
        
        for failing_model in ['arima', 'lstm', 'prophet']:
            # Temporarily disable one model
            if failing_model == 'arima':
                model.arima_model = None
            elif failing_model == 'lstm':
                model.lstm_model = None
            elif failing_model == 'prophet':
                model.prophet_model = None
            
            try:
                # Ensemble should still work with remaining models
                forecast = model.forecast(steps=5)
                assert len(forecast) == 5
                assert not forecast.isnull().any()
                print(f"Ensemble works correctly when {failing_model} model is disabled")
                
            except Exception as e:
                # Should fail gracefully with clear error message
                assert "insufficient" in str(e).lower() or "no models" in str(e).lower(), (
                    f"Ensemble should fail gracefully when {failing_model} is disabled, got: {e}"
                )
            
            finally:
                # Restore original model
                if failing_model == 'arima':
                    model.arima_model = original_models['arima']
                elif failing_model == 'lstm':
                    model.lstm_model = original_models['lstm']
                elif failing_model == 'prophet':
                    model.prophet_model = original_models['prophet'] 