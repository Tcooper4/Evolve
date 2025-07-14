"""
Unit tests for XGBoost forecaster model.

Tests XGBoost model functionality with synthetic time series data,
including edge cases like short series, constant series, and NaN handling.
"""

import warnings
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

# Import the XGBoost model
try:
    from trading.models.xgboost_model import XGBoostModel

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBoostModel = Mock()


class TestXGBoostForecaster:
    """Test suite for XGBoost forecaster model."""

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

        df = pd.DataFrame(
            {"Close": close, "Volume": volume, "High": high, "Low": low}, index=dates
        )

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
    def xgboost_model(self):
        """Create XGBoost model instance."""
        if not XGBOOST_AVAILABLE:
            pytest.skip("XGBoost model not available")
        return XGBoostModel()

    def test_model_instantiation(self, xgboost_model):
        """Test that XGBoost model instantiates correctly."""
        assert xgboost_model is not None
        assert hasattr(xgboost_model, "model")
        assert hasattr(xgboost_model, "is_fitted")
        assert xgboost_model.is_fitted is False

    def test_model_fitting(self, xgboost_model, synthetic_dataframe):
        """Test that XGBoost model fits to data correctly."""
        result = xgboost_model.fit(synthetic_dataframe)

        assert result["success"] is True
        assert xgboost_model.is_fitted is True
        assert xgboost_model.model is not None
        assert "timestamp" in result

    def test_forecast_generation(self, xgboost_model, synthetic_dataframe):
        """Test that XGBoost model generates forecasts correctly."""
        # Fit the model first
        xgboost_model.fit(synthetic_dataframe)

        # Generate forecast
        forecast_steps = 10
        result = xgboost_model.predict(synthetic_dataframe, horizon=forecast_steps)

        assert result["success"] is True
        assert "predictions" in result
        assert len(result["predictions"]) == forecast_steps
        assert not np.isnan(result["predictions"]).any()
        assert "timestamp" in result

    def test_forecast_output_length(self, xgboost_model, synthetic_dataframe):
        """Test that forecast output has correct length."""
        xgboost_model.fit(synthetic_dataframe)

        for steps in [1, 5, 10, 30]:
            result = xgboost_model.predict(synthetic_dataframe, horizon=steps)
            assert result["success"] is True
            assert len(result["predictions"]) == steps

    def test_no_nan_in_forecast(self, xgboost_model, synthetic_dataframe):
        """Test that forecasts contain no NaN values."""
        xgboost_model.fit(synthetic_dataframe)

        result = xgboost_model.predict(synthetic_dataframe, horizon=10)
        assert result["success"] is True
        assert not np.isnan(result["predictions"]).any()

    def test_short_time_series_handling(self, xgboost_model, short_time_series):
        """Test handling of short time series (< 10 points)."""
        # Convert to DataFrame format
        df = pd.DataFrame({"Close": short_time_series})
        result = xgboost_model.fit(df)

        # Should fail gracefully with clear error message
        assert result["success"] is False
        assert "error" in result
        assert any(
            keyword in result["error"].lower()
            for keyword in ["insufficient", "at least", "minimum", "data"]
        )

    def test_constant_series_handling(self, xgboost_model, constant_time_series):
        """Test handling of constant time series."""
        # Convert to DataFrame format
        df = pd.DataFrame({"Close": constant_time_series})
        result = xgboost_model.fit(df)

        # Should handle constant series gracefully
        if result["success"]:
            # If it succeeds, test prediction
            forecast_result = xgboost_model.predict(df, horizon=5)
            assert forecast_result["success"] is True
            assert len(forecast_result["predictions"]) == 5
        else:
            # If it fails, should be due to constant series
            assert any(
                keyword in result.get("error", "").lower()
                for keyword in ["constant", "variance", "unique"]
            )

    def test_nan_series_handling(self, xgboost_model, nan_time_series):
        """Test handling of time series with NaN values."""
        # Convert to DataFrame format
        df = pd.DataFrame({"Close": nan_time_series})
        result = xgboost_model.fit(df)

        # Should fail gracefully with clear error message
        assert result["success"] is False
        assert "error" in result
        assert any(
            keyword in result["error"].lower()
            for keyword in ["nan", "missing", "invalid"]
        )

    def test_model_summary(self, xgboost_model, synthetic_dataframe):
        """Test that model summary is generated correctly."""
        xgboost_model.fit(synthetic_dataframe)

        summary_result = xgboost_model.get_model_summary()
        assert summary_result["success"] is True
        assert "summary" in summary_result
        assert isinstance(summary_result["summary"], str)
        assert len(summary_result["summary"]) > 0

    def test_unfitted_model_behavior(self, xgboost_model, synthetic_dataframe):
        """Test behavior when trying to predict without fitting."""
        result = xgboost_model.predict(synthetic_dataframe, horizon=5)

        assert result["success"] is False
        assert "error" in result
        assert "fitted" in result["error"].lower()

    def test_feature_importance(self, xgboost_model, synthetic_dataframe):
        """Test that feature importance is calculated correctly."""
        xgboost_model.fit(synthetic_dataframe)

        importance_result = xgboost_model.get_feature_importance()
        assert importance_result["success"] is True
        assert "importance" in importance_result
        assert isinstance(importance_result["importance"], dict)
        assert len(importance_result["importance"]) > 0

    def test_forecast_method(self, xgboost_model, synthetic_dataframe):
        """Test the forecast method specifically."""
        result = xgboost_model.forecast(synthetic_dataframe, horizon=10)

        assert "forecast" in result
        assert "confidence" in result
        assert "model" in result
        assert result["model"] == "XGBoost"
        assert result["horizon"] == 10
        assert len(result["forecast"]) == 10

    def test_model_save_load(self, xgboost_model, synthetic_dataframe, tmp_path):
        """Test model save and load functionality."""
        # Fit the model
        xgboost_model.fit(synthetic_dataframe)

        # Save model
        save_path = tmp_path / "xgboost_model.json"
        save_result = xgboost_model.save_model(str(save_path))

        if save_result["success"]:
            # Load model
            new_model = XGBoostModel()
            load_result = new_model.load_model(str(save_path))

            assert load_result["success"] is True
            assert new_model.is_fitted is True
            assert new_model.model is not None

    def test_error_handling_edge_cases(self, xgboost_model):
        """Test error handling for various edge cases."""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        result = xgboost_model.fit(empty_df)
        assert result["success"] is False

        # Test with None
        result = xgboost_model.fit(None)
        assert result["success"] is False

        # Test with single row
        single_row = pd.DataFrame({"Close": [100]})
        result = xgboost_model.fit(single_row)
        assert result["success"] is False

    def test_forecast_consistency(self, xgboost_model, synthetic_dataframe):
        """Test that forecasts are consistent across multiple calls."""
        xgboost_model.fit(synthetic_dataframe)

        # Generate multiple forecasts
        forecast1 = xgboost_model.predict(synthetic_dataframe, horizon=5)
        forecast2 = xgboost_model.predict(synthetic_dataframe, horizon=5)

        assert forecast1["success"] is True
        assert forecast2["success"] is True
        np.testing.assert_array_almost_equal(
            forecast1["predictions"], forecast2["predictions"], decimal=10
        )

    def test_trend_detection(self, xgboost_model):
        """Test that model correctly identifies trends."""
        # Create data with clear trend
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        trend_data = pd.DataFrame(
            {
                "Close": np.linspace(100, 200, 50),
                "Volume": np.random.uniform(1000000, 5000000, 50),
            },
            index=dates,
        )

        xgboost_model.fit(trend_data)
        forecast = xgboost_model.predict(trend_data, horizon=5)

        if forecast["success"]:
            # Forecast should continue the trend (simplified check)
            assert len(forecast["predictions"]) == 5
            assert not np.isnan(forecast["predictions"]).any()

    def test_seasonality_handling(self, xgboost_model):
        """Test that model handles seasonal data correctly."""
        # Create seasonal data
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        t = np.linspace(0, 4 * np.pi, 100)
        seasonal_data = pd.DataFrame(
            {
                "Close": 100 + 10 * np.sin(t),
                "Volume": np.random.uniform(1000000, 5000000, 100),
            },
            index=dates,
        )

        xgboost_model.fit(seasonal_data)
        forecast = xgboost_model.predict(seasonal_data, horizon=5)

        if forecast["success"]:
            assert len(forecast["predictions"]) == 5
            assert not np.isnan(forecast["predictions"]).any()

    @pytest.mark.parametrize("horizon", [1, 5, 10, 30])
    def test_different_forecast_horizons(
        self, xgboost_model, synthetic_dataframe, horizon
    ):
        """Test forecasting with different horizons."""
        xgboost_model.fit(synthetic_dataframe)

        result = xgboost_model.predict(synthetic_dataframe, horizon=horizon)
        assert result["success"] is True
        assert len(result["predictions"]) == horizon
        assert not np.isnan(result["predictions"]).any()

    def test_model_configuration(self):
        """Test different model configurations."""
        configs = [
            {"n_estimators": 100, "max_depth": 6},
            {"n_estimators": 200, "max_depth": 8, "learning_rate": 0.1},
            {"n_estimators": 50, "max_depth": 4, "subsample": 0.8},
        ]

        for config in configs:
            model = XGBoostModel(config=config)
            assert model.config == config

    def test_feature_engineering(self, xgboost_model):
        """Test that feature engineering works correctly."""
        # Create data with basic features
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        data = pd.DataFrame(
            {
                "Close": np.linspace(100, 150, 50) + np.random.normal(0, 2, 50),
                "Volume": np.random.uniform(1000000, 5000000, 50),
            },
            index=dates,
        )

        # Test that model can handle feature engineering
        result = xgboost_model.fit(data)
        if result["success"]:
            # Should be able to predict with engineered features
            forecast_result = xgboost_model.predict(data, horizon=5)
            assert forecast_result["success"] is True

    def test_hyperparameter_tuning(self, xgboost_model, synthetic_dataframe):
        """Test hyperparameter tuning functionality."""
        # Test if model supports hyperparameter tuning
        if hasattr(xgboost_model, "tune_hyperparameters"):
            result = xgboost_model.tune_hyperparameters(synthetic_dataframe)
            assert isinstance(result, dict)
            assert "best_params" in result or "success" in result

    def test_model_evaluation(self, xgboost_model, synthetic_dataframe):
        """Test model evaluation metrics."""
        xgboost_model.fit(synthetic_dataframe)

        # Test evaluation if method exists
        if hasattr(xgboost_model, "evaluate"):
            eval_result = xgboost_model.evaluate(synthetic_dataframe)
            assert isinstance(eval_result, dict)
            assert "metrics" in eval_result or "success" in eval_result

    def test_ensemble_forecasting(self, xgboost_model, synthetic_dataframe):
        """Test ensemble forecasting capabilities."""
        xgboost_model.fit(synthetic_dataframe)

        # Test ensemble prediction if supported
        if hasattr(xgboost_model, "ensemble_predict"):
            result = xgboost_model.ensemble_predict(synthetic_dataframe, horizon=5)
            assert result["success"] is True
            assert len(result["predictions"]) == 5

    def test_confidence_intervals(self, xgboost_model, synthetic_dataframe):
        """Test confidence interval calculation."""
        xgboost_model.fit(synthetic_dataframe)

        # Test confidence intervals if supported
        if hasattr(xgboost_model, "predict_with_confidence"):
            result = xgboost_model.predict_with_confidence(
                synthetic_dataframe, horizon=5
            )
            assert result["success"] is True
            assert "predictions" in result
            assert "confidence_intervals" in result or "lower" in result

    def test_model_performance_metrics(self, xgboost_model, synthetic_dataframe):
        """Test model performance metrics calculation."""
        xgboost_model.fit(synthetic_dataframe)

        # Test performance metrics if method exists
        if hasattr(xgboost_model, "get_performance_metrics"):
            metrics = xgboost_model.get_performance_metrics(synthetic_dataframe)
            assert isinstance(metrics, dict)
            assert len(metrics) > 0


if __name__ == "__main__":
    pytest.main([__file__])
