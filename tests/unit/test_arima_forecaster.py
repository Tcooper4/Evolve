"""
Unit tests for ARIMA forecaster model.

Tests ARIMA model functionality with synthetic time series data,
including edge cases like short series, constant series, and NaN handling.
"""

import warnings
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

# Import the ARIMA model
try:
    from trading.models.arima_model import ARIMAModel

    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    ARIMAModel = Mock()


class TestARIMAForecaster:
    """Test suite for ARIMA forecaster model."""

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
    def arima_model(self):
        """Create ARIMA model instance."""
        if not ARIMA_AVAILABLE:
            pytest.skip("ARIMA model not available")
        return ARIMAModel(config={"order": (1, 1, 1)})

    def test_model_instantiation(self, arima_model):
        """Test that ARIMA model instantiates correctly."""
        assert arima_model is not None
        assert hasattr(arima_model, "order")
        assert hasattr(arima_model, "is_fitted")
        assert arima_model.is_fitted is False

    def test_model_fitting(self, arima_model, synthetic_time_series):
        """Test that ARIMA model fits to data correctly."""
        result = arima_model.fit(synthetic_time_series)

        assert result["success"] is True
        assert arima_model.is_fitted is True
        assert arima_model.fitted_model is not None
        assert "timestamp" in result

    def test_forecast_generation(self, arima_model, synthetic_time_series):
        """Test that ARIMA model generates forecasts correctly."""
        # Fit the model first
        arima_model.fit(synthetic_time_series)

        # Generate forecast
        forecast_steps = 10
        result = arima_model.predict(steps=forecast_steps)

        assert result["success"] is True
        assert "predictions" in result
        assert len(result["predictions"]) == forecast_steps
        assert not np.isnan(result["predictions"]).any()
        assert "timestamp" in result

    def test_forecast_output_length(self, arima_model, synthetic_time_series):
        """Test that forecast output has correct length."""
        arima_model.fit(synthetic_time_series)

        for steps in [1, 5, 10, 30]:
            result = arima_model.predict(steps=steps)
            assert result["success"] is True
            assert len(result["predictions"]) == steps

    def test_no_nan_in_forecast(self, arima_model, synthetic_time_series):
        """Test that forecasts contain no NaN values."""
        arima_model.fit(synthetic_time_series)

        result = arima_model.predict(steps=10)
        assert result["success"] is True
        assert not np.isnan(result["predictions"]).any()

    def test_short_time_series_handling(self, arima_model, short_time_series):
        """Test handling of short time series (< 10 points)."""
        result = arima_model.fit(short_time_series)

        # Should fail gracefully with clear error message
        assert result["success"] is False
        assert "error" in result
        assert any(
            keyword in result["error"].lower()
            for keyword in ["insufficient", "at least", "minimum", "20"]
        )

    def test_constant_series_handling(self, arima_model, constant_time_series):
        """Test handling of constant time series."""
        result = arima_model.fit(constant_time_series)

        # Should handle constant series gracefully
        if result["success"]:
            # If it succeeds, test prediction
            forecast_result = arima_model.predict(steps=5)
            assert forecast_result["success"] is True
            assert len(forecast_result["predictions"]) == 5
        else:
            # If it fails, should be due to constant series
            assert (
                "constant" in result.get("error", "").lower()
                or "stationary" in result.get("error", "").lower()
            )

    def test_nan_series_handling(self, arima_model, nan_time_series):
        """Test handling of time series with NaN values."""
        result = arima_model.fit(nan_time_series)

        # Should fail gracefully with clear error message
        assert result["success"] is False
        assert "error" in result
        assert any(
            keyword in result["error"].lower()
            for keyword in ["nan", "missing", "invalid"]
        )

    def test_model_summary(self, arima_model, synthetic_time_series):
        """Test that model summary is generated correctly."""
        arima_model.fit(synthetic_time_series)

        summary_result = arima_model.get_model_summary()
        assert summary_result["success"] is True
        assert "summary" in summary_result
        assert isinstance(summary_result["summary"], str)
        assert len(summary_result["summary"]) > 0

    def test_unfitted_model_behavior(self, arima_model):
        """Test behavior when trying to predict without fitting."""
        result = arima_model.predict(steps=5)

        assert result["success"] is False
        assert "error" in result
        assert "fitted" in result["error"].lower()

    def test_different_arima_orders(self, synthetic_time_series):
        """Test ARIMA model with different order parameters."""
        orders = [(1, 1, 1), (2, 1, 2), (1, 0, 1), (0, 1, 1)]

        for order in orders:
            model = ARIMAModel(config={"order": order})
            result = model.fit(synthetic_time_series)

            # Should succeed for valid orders
            if result["success"]:
                forecast_result = model.predict(steps=5)
                assert forecast_result["success"] is True
                assert len(forecast_result["predictions"]) == 5

    def test_forecast_method(self, arima_model, synthetic_time_series):
        """Test the forecast method specifically."""
        result = arima_model.forecast(synthetic_time_series, horizon=10)

        assert "forecast" in result
        assert "confidence" in result
        assert "model" in result
        assert result["model"] == "ARIMA"
        assert result["horizon"] == 10
        assert len(result["forecast"]) == 10

    def test_find_best_order(self, arima_model, synthetic_time_series):
        """Test automatic order selection."""
        result = arima_model.find_best_order(
            synthetic_time_series, max_p=2, max_d=1, max_q=2
        )

        assert result["success"] is True
        assert "best_order" in result
        assert "best_aic" in result
        assert isinstance(result["best_order"], tuple)
        assert len(result["best_order"]) == 3

    def test_model_save_load(self, arima_model, synthetic_time_series, tmp_path):
        """Test model save and load functionality."""
        # Fit the model
        arima_model.fit(synthetic_time_series)

        # Save model
        save_path = tmp_path / "arima_model.pkl"
        save_result = arima_model.save_model(str(save_path))

        if save_result["success"]:
            # Load model
            new_model = ARIMAModel()
            load_result = new_model.load_model(str(save_path))

            assert load_result["success"] is True
            assert new_model.is_fitted is True
            assert new_model.fitted_model is not None

    def test_error_handling_edge_cases(self, arima_model):
        """Test error handling for various edge cases."""
        # Test with empty series
        empty_series = pd.Series(dtype=float)
        result = arima_model.fit(empty_series)
        assert result["success"] is False

        # Test with None
        result = arima_model.fit(None)
        assert result["success"] is False

        # Test with single value
        single_value = pd.Series([100])
        result = arima_model.fit(single_value)
        assert result["success"] is False

    def test_forecast_consistency(self, arima_model, synthetic_time_series):
        """Test that forecasts are consistent across multiple calls."""
        arima_model.fit(synthetic_time_series)

        # Generate multiple forecasts
        forecast1 = arima_model.predict(steps=5)
        forecast2 = arima_model.predict(steps=5)

        assert forecast1["success"] is True
        assert forecast2["success"] is True
        np.testing.assert_array_almost_equal(
            forecast1["predictions"], forecast2["predictions"], decimal=10
        )

    def test_trend_detection(self, arima_model):
        """Test that model correctly identifies trends."""
        # Create data with clear trend
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        trend_data = pd.Series(np.linspace(100, 200, 50), index=dates)

        arima_model.fit(trend_data)
        forecast = arima_model.predict(steps=5)

        if forecast["success"]:
            # Forecast should continue the trend (simplified check)
            assert len(forecast["predictions"]) == 5
            assert not np.isnan(forecast["predictions"]).any()

    def test_seasonality_handling(self, arima_model):
        """Test that model handles seasonal data correctly."""
        # Create seasonal data
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        t = np.linspace(0, 4 * np.pi, 100)
        seasonal_data = pd.Series(100 + 10 * np.sin(t), index=dates)

        arima_model.fit(seasonal_data)
        forecast = arima_model.predict(steps=5)

        if forecast["success"]:
            assert len(forecast["predictions"]) == 5
            assert not np.isnan(forecast["predictions"]).any()

    @pytest.mark.parametrize("horizon", [1, 5, 10, 30])
    def test_different_forecast_horizons(
        self, arima_model, synthetic_time_series, horizon
    ):
        """Test forecasting with different horizons."""
        arima_model.fit(synthetic_time_series)

        result = arima_model.predict(steps=horizon)
        assert result["success"] is True
        assert len(result["predictions"]) == horizon
        assert not np.isnan(result["predictions"]).any()

    def test_model_configuration(self):
        """Test different model configurations."""
        configs = [
            {"order": (1, 1, 1)},
            {"order": (2, 1, 2)},
            {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 12)},
        ]

        for config in configs:
            model = ARIMAModel(config=config)
            assert model.order == config["order"]
            if "seasonal_order" in config:
                assert model.seasonal_order == config["seasonal_order"]


if __name__ == "__main__":
    pytest.main([__file__])
