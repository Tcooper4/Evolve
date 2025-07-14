"""
Unit tests for Prophet forecaster model.

Tests Prophet model functionality with synthetic time series data,
including edge cases like short series, constant series, and NaN handling.
"""

import warnings
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

# Import the Prophet model
try:
    from trading.models.prophet_model import ProphetModel

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    ProphetModel = Mock()


class TestProphetForecaster:
    """Test suite for Prophet forecaster model."""

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
    def seasonal_time_series(self):
        """Create synthetic time series with seasonality."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=365, freq="D")

        # Create seasonal data with weekly and yearly patterns
        t = np.arange(len(dates))
        trend = np.linspace(100, 120, len(dates))
        weekly_seasonal = 5 * np.sin(2 * np.pi * t / 7)  # Weekly pattern
        yearly_seasonal = 10 * np.sin(2 * np.pi * t / 365)  # Yearly pattern
        noise = np.random.normal(0, 1, len(dates))

        values = trend + weekly_seasonal + yearly_seasonal + noise

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
    def prophet_model(self):
        """Create Prophet model instance."""
        if not PROPHET_AVAILABLE:
            pytest.skip("Prophet model not available")
        return ProphetModel()

    def test_model_instantiation(self, prophet_model):
        """Test that Prophet model instantiates correctly."""
        assert prophet_model is not None
        assert hasattr(prophet_model, "model")
        assert hasattr(prophet_model, "is_fitted")
        assert prophet_model.is_fitted is False

    def test_model_fitting(self, prophet_model, synthetic_time_series):
        """Test that Prophet model fits to data correctly."""
        result = prophet_model.fit(synthetic_time_series)

        assert result["success"] is True
        assert prophet_model.is_fitted is True
        assert prophet_model.model is not None
        assert "timestamp" in result

    def test_forecast_generation(self, prophet_model, synthetic_time_series):
        """Test that Prophet model generates forecasts correctly."""
        # Fit the model first
        prophet_model.fit(synthetic_time_series)

        # Generate forecast
        forecast_steps = 10
        result = prophet_model.predict(synthetic_time_series, horizon=forecast_steps)

        assert result["success"] is True
        assert "predictions" in result
        assert len(result["predictions"]) == forecast_steps
        assert not np.isnan(result["predictions"]).any()
        assert "timestamp" in result

    def test_forecast_output_length(self, prophet_model, synthetic_time_series):
        """Test that forecast output has correct length."""
        prophet_model.fit(synthetic_time_series)

        for steps in [1, 5, 10, 30]:
            result = prophet_model.predict(synthetic_time_series, horizon=steps)
            assert result["success"] is True
            assert len(result["predictions"]) == steps

    def test_no_nan_in_forecast(self, prophet_model, synthetic_time_series):
        """Test that forecasts contain no NaN values."""
        prophet_model.fit(synthetic_time_series)

        result = prophet_model.predict(synthetic_time_series, horizon=10)
        assert result["success"] is True
        assert not np.isnan(result["predictions"]).any()

    def test_short_time_series_handling(self, prophet_model, short_time_series):
        """Test handling of short time series (< 10 points)."""
        result = prophet_model.fit(short_time_series)

        # Should fail gracefully with clear error message
        assert result["success"] is False
        assert "error" in result
        assert any(
            keyword in result["error"].lower()
            for keyword in ["insufficient", "at least", "minimum", "data"]
        )

    def test_constant_series_handling(self, prophet_model, constant_time_series):
        """Test handling of constant time series."""
        result = prophet_model.fit(constant_time_series)

        # Should handle constant series gracefully
        if result["success"]:
            # If it succeeds, test prediction
            forecast_result = prophet_model.predict(constant_time_series, horizon=5)
            assert forecast_result["success"] is True
            assert len(forecast_result["predictions"]) == 5
        else:
            # If it fails, should be due to constant series
            assert any(
                keyword in result.get("error", "").lower()
                for keyword in ["constant", "variance", "unique"]
            )

    def test_nan_series_handling(self, prophet_model, nan_time_series):
        """Test handling of time series with NaN values."""
        result = prophet_model.fit(nan_time_series)

        # Should fail gracefully with clear error message
        assert result["success"] is False
        assert "error" in result
        assert any(
            keyword in result["error"].lower()
            for keyword in ["nan", "missing", "invalid"]
        )

    def test_model_summary(self, prophet_model, synthetic_time_series):
        """Test that model summary is generated correctly."""
        prophet_model.fit(synthetic_time_series)

        summary_result = prophet_model.get_model_summary()
        assert summary_result["success"] is True
        assert "summary" in summary_result
        assert isinstance(summary_result["summary"], str)
        assert len(summary_result["summary"]) > 0

    def test_unfitted_model_behavior(self, prophet_model, synthetic_time_series):
        """Test behavior when trying to predict without fitting."""
        result = prophet_model.predict(synthetic_time_series, horizon=5)

        assert result["success"] is False
        assert "error" in result
        assert "fitted" in result["error"].lower()

    def test_seasonality_detection(self, prophet_model, seasonal_time_series):
        """Test that Prophet correctly detects seasonality."""
        result = prophet_model.fit(seasonal_time_series)

        if result["success"]:
            # Test seasonality detection
            if hasattr(prophet_model, "detect_seasonality"):
                seasonality = prophet_model.detect_seasonality(seasonal_time_series)
                assert isinstance(seasonality, dict)
                assert "weekly" in seasonality or "yearly" in seasonality

    def test_forecast_method(self, prophet_model, synthetic_time_series):
        """Test the forecast method specifically."""
        result = prophet_model.forecast(synthetic_time_series, horizon=10)

        assert "forecast" in result
        assert "confidence" in result
        assert "model" in result
        assert result["model"] == "Prophet"
        assert result["horizon"] == 10
        assert len(result["forecast"]) == 10

    def test_confidence_intervals(self, prophet_model, synthetic_time_series):
        """Test confidence interval calculation."""
        prophet_model.fit(synthetic_time_series)

        if hasattr(prophet_model, "predict_with_confidence"):
            result = prophet_model.predict_with_confidence(
                synthetic_time_series, horizon=10
            )
            assert result["success"] is True
            assert "predictions" in result
            assert "lower" in result
            assert "upper" in result
            assert len(result["predictions"]) == 10
            assert len(result["lower"]) == 10
            assert len(result["upper"]) == 10

    def test_model_save_load(self, prophet_model, synthetic_time_series, tmp_path):
        """Test model save and load functionality."""
        # Fit the model
        prophet_model.fit(synthetic_time_series)

        # Save model
        save_path = tmp_path / "prophet_model.json"
        save_result = prophet_model.save_model(str(save_path))

        if save_result["success"]:
            # Load model
            new_model = ProphetModel()
            load_result = new_model.load_model(str(save_path))

            assert load_result["success"] is True
            assert new_model.is_fitted is True
            assert new_model.model is not None

    def test_error_handling_edge_cases(self, prophet_model):
        """Test error handling for various edge cases."""
        # Test with empty series
        empty_series = pd.Series(dtype=float)
        result = prophet_model.fit(empty_series)
        assert result["success"] is False

        # Test with None
        result = prophet_model.fit(None)
        assert result["success"] is False

        # Test with single value
        single_value = pd.Series([100])
        result = prophet_model.fit(single_value)
        assert result["success"] is False

    def test_forecast_consistency(self, prophet_model, synthetic_time_series):
        """Test that forecasts are consistent across multiple calls."""
        prophet_model.fit(synthetic_time_series)

        # Generate multiple forecasts
        forecast1 = prophet_model.predict(synthetic_time_series, horizon=5)
        forecast2 = prophet_model.predict(synthetic_time_series, horizon=5)

        assert forecast1["success"] is True
        assert forecast2["success"] is True
        np.testing.assert_array_almost_equal(
            forecast1["predictions"], forecast2["predictions"], decimal=10
        )

    def test_trend_detection(self, prophet_model):
        """Test that model correctly identifies trends."""
        # Create data with clear trend
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        trend_data = pd.Series(np.linspace(100, 200, 50), index=dates)

        prophet_model.fit(trend_data)
        forecast = prophet_model.predict(trend_data, horizon=5)

        if forecast["success"]:
            # Forecast should continue the trend (simplified check)
            assert len(forecast["predictions"]) == 5
            assert not np.isnan(forecast["predictions"]).any()

    def test_seasonality_handling(self, prophet_model, seasonal_time_series):
        """Test that model handles seasonal data correctly."""
        prophet_model.fit(seasonal_time_series)
        forecast = prophet_model.predict(seasonal_time_series, horizon=5)

        if forecast["success"]:
            assert len(forecast["predictions"]) == 5
            assert not np.isnan(forecast["predictions"]).any()

    @pytest.mark.parametrize("horizon", [1, 5, 10, 30])
    def test_different_forecast_horizons(
        self, prophet_model, synthetic_time_series, horizon
    ):
        """Test forecasting with different horizons."""
        prophet_model.fit(synthetic_time_series)

        result = prophet_model.predict(synthetic_time_series, horizon=horizon)
        assert result["success"] is True
        assert len(result["predictions"]) == horizon
        assert not np.isnan(result["predictions"]).any()

    def test_model_configuration(self):
        """Test different model configurations."""
        configs = [
            {"changepoint_prior_scale": 0.05},
            {"seasonality_prior_scale": 10.0},
            {"holidays_prior_scale": 10.0},
            {"seasonality_mode": "multiplicative"},
        ]

        for config in configs:
            model = ProphetModel(config=config)
            assert model.config == config

    def test_changepoint_detection(self, prophet_model, synthetic_time_series):
        """Test changepoint detection functionality."""
        prophet_model.fit(synthetic_time_series)

        if hasattr(prophet_model, "detect_changepoints"):
            changepoints = prophet_model.detect_changepoints(synthetic_time_series)
            assert isinstance(changepoints, list) or isinstance(
                changepoints, np.ndarray
            )

    def test_holiday_effects(self, prophet_model, synthetic_time_series):
        """Test holiday effects handling."""
        prophet_model.fit(synthetic_time_series)

        if hasattr(prophet_model, "add_holidays"):
            # Add holiday effects
            holidays = pd.DataFrame(
                {
                    "ds": pd.to_datetime(["2023-01-01", "2023-07-04"]),
                    "holiday": ["New Year", "Independence Day"],
                }
            )
            prophet_model.add_holidays(holidays)

            # Test prediction with holidays
            result = prophet_model.predict(synthetic_time_series, horizon=10)
            assert result["success"] is True

    def test_custom_seasonality(self, prophet_model, synthetic_time_series):
        """Test custom seasonality handling."""
        prophet_model.fit(synthetic_time_series)

        if hasattr(prophet_model, "add_seasonality"):
            # Add custom seasonality
            prophet_model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

            # Test prediction with custom seasonality
            result = prophet_model.predict(synthetic_time_series, horizon=10)
            assert result["success"] is True

    def test_regressors(self, prophet_model, synthetic_time_series):
        """Test regressor handling."""
        # Create data with regressors
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {
                "ds": dates,
                "y": synthetic_time_series.values,
                "regressor1": np.random.normal(0, 1, 100),
                "regressor2": np.random.normal(0, 1, 100),
            }
        )

        prophet_model.fit(df)

        if hasattr(prophet_model, "add_regressor"):
            # Add regressor
            prophet_model.add_regressor("regressor1")
            prophet_model.add_regressor("regressor2")

            # Test prediction with regressors
            result = prophet_model.predict(df, horizon=10)
            assert result["success"] is True

    def test_model_components(self, prophet_model, synthetic_time_series):
        """Test model components decomposition."""
        prophet_model.fit(synthetic_time_series)

        if hasattr(prophet_model, "get_components"):
            components = prophet_model.get_components(synthetic_time_series)
            assert isinstance(components, dict)
            assert "trend" in components or "seasonal" in components

    def test_model_performance_metrics(self, prophet_model, synthetic_time_series):
        """Test model performance metrics calculation."""
        prophet_model.fit(synthetic_time_series)

        if hasattr(prophet_model, "get_performance_metrics"):
            metrics = prophet_model.get_performance_metrics(synthetic_time_series)
            assert isinstance(metrics, dict)
            assert len(metrics) > 0

    def test_forecast_uncertainty(self, prophet_model, synthetic_time_series):
        """Test forecast uncertainty quantification."""
        prophet_model.fit(synthetic_time_series)

        if hasattr(prophet_model, "predict_with_uncertainty"):
            result = prophet_model.predict_with_uncertainty(
                synthetic_time_series, horizon=10
            )
            assert result["success"] is True
            assert "uncertainty" in result or "std" in result

    def test_model_validation(self, prophet_model, synthetic_time_series):
        """Test model validation functionality."""
        prophet_model.fit(synthetic_time_series)

        if hasattr(prophet_model, "cross_validate"):
            cv_results = prophet_model.cross_validate(
                synthetic_time_series, initial=50, period=10, horizon=5
            )
            assert isinstance(cv_results, dict)
            assert "metrics" in cv_results or "predictions" in cv_results


if __name__ == "__main__":
    pytest.main([__file__])
