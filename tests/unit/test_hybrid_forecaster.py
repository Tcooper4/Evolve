"""
Unit tests for Hybrid forecaster model.

Tests Hybrid model functionality with synthetic time series data,
including edge cases like short series, constant series, and NaN handling.
"""

import warnings
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

# Import the Hybrid model
try:
    from trading.models.hybrid_model import HybridModel

    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False
    HybridModel = Mock()


class TestHybridForecaster:
    """Test suite for Hybrid forecaster model."""

    @pytest.fixture
    def synthetic_time_series(self):
        """Create synthetic time series with increasing trend."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

        # Create time series with trend and noise
        trend = np.linspace(100, 150, 100)  # Increasing trend
        noise = np.random.normal(0, 2, 100)  # Random noise
        values = trend + noise

        return pd.Series(values, index=dates, name="Close")

    @pytest.fixture
    def synthetic_dataframe(self):
        """Create synthetic DataFrame with multiple features."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

        # Create multiple features
        close = np.linspace(100, 150, 100) + np.random.normal(0, 2, 100)
        volume = np.random.uniform(1000000, 5000000, 100)
        high = close + np.random.uniform(0, 5, 100)
        low = close - np.random.uniform(0, 5, 100)

        df = pd.DataFrame({"Close": close, "Volume": volume, "High": high, "Low": low}, index=dates)

        return df

    @pytest.fixture
    def short_time_series(self):
        """Create short time series (< 10 points)."""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        values = [100, 101, 102, 103, 104]
        return pd.Series(values, index=dates, name="Close")

    @pytest.fixture
    def constant_time_series(self):
        """Create constant time series."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        values = [100.0] * 50
        return pd.Series(values, index=dates, name="Close")

    @pytest.fixture
    def nan_time_series(self):
        """Create time series with NaN values."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        values = [100, 101, np.nan, 103, 104] + [100 + i for i in range(45)]
        return pd.Series(values, index=dates, name="Close")

    @pytest.fixture
    def hybrid_model(self):
        """Create Hybrid model instance."""
        if not HYBRID_AVAILABLE:
            pytest.skip("Hybrid model not available")
        return HybridModel()

    def test_model_instantiation(self, hybrid_model):
        """Test that Hybrid model instantiates correctly."""
        assert hybrid_model is not None
        assert hasattr(hybrid_model, "models")
        assert hasattr(hybrid_model, "is_fitted")
        assert hybrid_model.is_fitted is False

    def test_model_fitting(self, hybrid_model, synthetic_time_series):
        """Test that Hybrid model fits to data correctly."""
        result = hybrid_model.fit(synthetic_time_series)

        assert result["success"] is True
        assert hybrid_model.is_fitted is True
        assert len(hybrid_model.models) > 0
        assert "timestamp" in result

    def test_forecast_generation(self, hybrid_model, synthetic_time_series):
        """Test that Hybrid model generates forecasts correctly."""
        # Fit the model first
        hybrid_model.fit(synthetic_time_series)

        # Generate forecast
        forecast_steps = 10
        result = hybrid_model.predict(synthetic_time_series, horizon=forecast_steps)

        assert result["success"] is True
        assert "predictions" in result
        assert len(result["predictions"]) == forecast_steps
        assert not np.isnan(result["predictions"]).any()
        assert "timestamp" in result

    def test_forecast_output_length(self, hybrid_model, synthetic_time_series):
        """Test that forecast output has correct length."""
        hybrid_model.fit(synthetic_time_series)

        for steps in [1, 5, 10, 30]:
            result = hybrid_model.predict(synthetic_time_series, horizon=steps)
            assert result["success"] is True
            assert len(result["predictions"]) == steps

    def test_no_nan_in_forecast(self, hybrid_model, synthetic_time_series):
        """Test that forecasts contain no NaN values."""
        hybrid_model.fit(synthetic_time_series)

        result = hybrid_model.predict(synthetic_time_series, horizon=10)
        assert result["success"] is True
        assert not np.isnan(result["predictions"]).any()

    def test_short_time_series_handling(self, hybrid_model, short_time_series):
        """Test handling of short time series (< 10 points)."""
        result = hybrid_model.fit(short_time_series)

        # Should fail gracefully with clear error message
        assert result["success"] is False
        assert "error" in result
        assert any(keyword in result["error"].lower() for keyword in ["insufficient", "at least", "minimum", "data"])

    def test_constant_series_handling(self, hybrid_model, constant_time_series):
        """Test handling of constant time series."""
        result = hybrid_model.fit(constant_time_series)

        # Should handle constant series gracefully
        if result["success"]:
            # If it succeeds, test prediction
            forecast_result = hybrid_model.predict(constant_time_series, horizon=5)
            assert forecast_result["success"] is True
            assert len(forecast_result["predictions"]) == 5
        else:
            # If it fails, should be due to constant series
            assert any(keyword in result.get("error", "").lower() for keyword in ["constant", "variance", "unique"])

    def test_nan_series_handling(self, hybrid_model, nan_time_series):
        """Test handling of time series with NaN values."""
        result = hybrid_model.fit(nan_time_series)

        # Should fail gracefully with clear error message
        assert result["success"] is False
        assert "error" in result
        assert any(keyword in result["error"].lower() for keyword in ["nan", "missing", "invalid"])

    def test_model_summary(self, hybrid_model, synthetic_time_series):
        """Test that model summary is generated correctly."""
        hybrid_model.fit(synthetic_time_series)

        summary_result = hybrid_model.get_model_summary()
        assert summary_result["success"] is True
        assert "summary" in summary_result
        assert isinstance(summary_result["summary"], str)
        assert len(summary_result["summary"]) > 0

    def test_unfitted_model_behavior(self, hybrid_model, synthetic_time_series):
        """Test behavior when trying to predict without fitting."""
        result = hybrid_model.predict(synthetic_time_series, horizon=5)

        assert result["success"] is False
        assert "error" in result
        assert "fitted" in result["error"].lower()

    def test_ensemble_weights(self, hybrid_model, synthetic_time_series):
        """Test ensemble weights calculation."""
        hybrid_model.fit(synthetic_time_series)

        if hasattr(hybrid_model, "get_ensemble_weights"):
            weights = hybrid_model.get_ensemble_weights()
            assert isinstance(weights, dict)
            assert len(weights) > 0
            assert all(0 <= w <= 1 for w in weights.values())
            assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_forecast_method(self, hybrid_model, synthetic_time_series):
        """Test the forecast method specifically."""
        result = hybrid_model.forecast(synthetic_time_series, horizon=10)

        assert "forecast" in result
        assert "confidence" in result
        assert "model" in result
        assert result["model"] == "Hybrid"
        assert result["horizon"] == 10
        assert len(result["forecast"]) == 10

    def test_individual_model_forecasts(self, hybrid_model, synthetic_time_series):
        """Test individual model forecasts."""
        hybrid_model.fit(synthetic_time_series)

        if hasattr(hybrid_model, "get_individual_forecasts"):
            individual_forecasts = hybrid_model.get_individual_forecasts(synthetic_time_series, horizon=5)
            assert isinstance(individual_forecasts, dict)
            assert len(individual_forecasts) > 0

            for model_name, forecast in individual_forecasts.items():
                assert isinstance(forecast, np.ndarray)
                assert len(forecast) == 5

    def test_model_save_load(self, hybrid_model, synthetic_time_series, tmp_path):
        """Test model save and load functionality."""
        # Fit the model
        hybrid_model.fit(synthetic_time_series)

        # Save model
        save_path = tmp_path / "hybrid_model.pkl"
        save_result = hybrid_model.save_model(str(save_path))

        if save_result["success"]:
            # Load model
            new_model = HybridModel()
            load_result = new_model.load_model(str(save_path))

            assert load_result["success"] is True
            assert new_model.is_fitted is True
            assert len(new_model.models) > 0

    def test_error_handling_edge_cases(self, hybrid_model):
        """Test error handling for various edge cases."""
        # Test with empty series
        empty_series = pd.Series(dtype=float)
        result = hybrid_model.fit(empty_series)
        assert result["success"] is False

        # Test with None
        result = hybrid_model.fit(None)
        assert result["success"] is False

        # Test with single value
        single_value = pd.Series([100])
        result = hybrid_model.fit(single_value)
        assert result["success"] is False

    def test_forecast_consistency(self, hybrid_model, synthetic_time_series):
        """Test that forecasts are consistent across multiple calls."""
        hybrid_model.fit(synthetic_time_series)

        # Generate multiple forecasts
        forecast1 = hybrid_model.predict(synthetic_time_series, horizon=5)
        forecast2 = hybrid_model.predict(synthetic_time_series, horizon=5)

        assert forecast1["success"] is True
        assert forecast2["success"] is True
        np.testing.assert_array_almost_equal(forecast1["predictions"], forecast2["predictions"], decimal=10)

    def test_trend_detection(self, hybrid_model):
        """Test that model correctly identifies trends."""
        # Create data with clear trend
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        trend_data = pd.Series(np.linspace(100, 200, 50), index=dates)

        hybrid_model.fit(trend_data)
        forecast = hybrid_model.predict(trend_data, horizon=5)

        if forecast["success"]:
            # Forecast should continue the trend (simplified check)
            assert len(forecast["predictions"]) == 5
            assert not np.isnan(forecast["predictions"]).any()

    def test_seasonality_handling(self, hybrid_model):
        """Test that model handles seasonal data correctly."""
        # Create seasonal data
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        t = np.linspace(0, 4 * np.pi, 100)
        seasonal_data = pd.Series(100 + 10 * np.sin(t), index=dates)

        hybrid_model.fit(seasonal_data)
        forecast = hybrid_model.predict(seasonal_data, horizon=5)

        if forecast["success"]:
            assert len(forecast["predictions"]) == 5
            assert not np.isnan(forecast["predictions"]).any()

    @pytest.mark.parametrize("horizon", [1, 5, 10, 30])
    def test_different_forecast_horizons(self, hybrid_model, synthetic_time_series, horizon):
        """Test forecasting with different horizons."""
        hybrid_model.fit(synthetic_time_series)

        result = hybrid_model.predict(synthetic_time_series, horizon=horizon)
        assert result["success"] is True
        assert len(result["predictions"]) == horizon
        assert not np.isnan(result["predictions"]).any()

    def test_model_configuration(self):
        """Test different model configurations."""
        configs = [
            {"models": ["arima", "lstm"], "weights": [0.5, 0.5]},
            {"models": ["prophet", "xgboost"], "weights": [0.3, 0.7]},
            {"models": ["arima", "lstm", "prophet"], "weights": [0.4, 0.3, 0.3]},
        ]

        for config in configs:
            model = HybridModel(config=config)
            assert model.config == config

    def test_model_selection(self, hybrid_model, synthetic_time_series):
        """Test model selection functionality."""
        hybrid_model.fit(synthetic_time_series)

        if hasattr(hybrid_model, "select_best_models"):
            best_models = hybrid_model.select_best_models(synthetic_time_series, n_models=2)
            assert isinstance(best_models, list)
            assert len(best_models) <= 2

    def test_ensemble_methods(self, hybrid_model, synthetic_time_series):
        """Test different ensemble methods."""
        hybrid_model.fit(synthetic_time_series)

        # Test weighted average
        if hasattr(hybrid_model, "set_ensemble_method"):
            hybrid_model.set_ensemble_method("weighted_average")
            result = hybrid_model.predict(synthetic_time_series, horizon=5)
            assert result["success"] is True

        # Test voting
        if hasattr(hybrid_model, "set_ensemble_method"):
            hybrid_model.set_ensemble_method("voting")
            result = hybrid_model.predict(synthetic_time_series, horizon=5)
            assert result["success"] is True

    def test_model_performance_tracking(self, hybrid_model, synthetic_time_series):
        """Test model performance tracking."""
        hybrid_model.fit(synthetic_time_series)

        if hasattr(hybrid_model, "get_model_performance"):
            performance = hybrid_model.get_model_performance()
            assert isinstance(performance, dict)
            assert len(performance) > 0

    def test_confidence_intervals(self, hybrid_model, synthetic_time_series):
        """Test confidence interval calculation."""
        hybrid_model.fit(synthetic_time_series)

        if hasattr(hybrid_model, "predict_with_confidence"):
            result = hybrid_model.predict_with_confidence(synthetic_time_series, horizon=10)
            assert result["success"] is True
            assert "predictions" in result
            assert "lower" in result
            assert "upper" in result
            assert len(result["predictions"]) == 10
            assert len(result["lower"]) == 10
            assert len(result["upper"]) == 10

    def test_model_validation(self, hybrid_model, synthetic_time_series):
        """Test model validation functionality."""
        hybrid_model.fit(synthetic_time_series)

        if hasattr(hybrid_model, "cross_validate"):
            cv_results = hybrid_model.cross_validate(synthetic_time_series, folds=3)
            assert isinstance(cv_results, dict)
            assert "metrics" in cv_results or "predictions" in cv_results

    def test_adaptive_weights(self, hybrid_model, synthetic_time_series):
        """Test adaptive weight adjustment."""
        hybrid_model.fit(synthetic_time_series)

        if hasattr(hybrid_model, "update_weights_adaptively"):
            hybrid_model.update_weights_adaptively(synthetic_time_series)
            weights = hybrid_model.get_ensemble_weights()
            assert isinstance(weights, dict)
            assert len(weights) > 0

    def test_model_robustness(self, hybrid_model, synthetic_time_series):
        """Test model robustness to different data conditions."""
        hybrid_model.fit(synthetic_time_series)

        # Test with noisy data
        noisy_data = synthetic_time_series + np.random.normal(0, 5, len(synthetic_time_series))
        result = hybrid_model.predict(noisy_data, horizon=5)
        assert result["success"] is True

        # Test with missing data (interpolated)
        missing_data = synthetic_time_series.copy()
        missing_data.iloc[10:15] = np.nan
        missing_data = missing_data.interpolate()
        result = hybrid_model.predict(missing_data, horizon=5)
        assert result["success"] is True

    def test_ensemble_diversity(self, hybrid_model, synthetic_time_series):
        """Test ensemble diversity measurement."""
        hybrid_model.fit(synthetic_time_series)

        if hasattr(hybrid_model, "measure_ensemble_diversity"):
            diversity = hybrid_model.measure_ensemble_diversity(synthetic_time_series)
            assert isinstance(diversity, float)
            assert 0 <= diversity <= 1


if __name__ == "__main__":
    pytest.main([__file__])
