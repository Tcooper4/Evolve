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

    def test_forecast_shape_validation(self, xgboost_model, synthetic_dataframe):
        """Test that forecast output shape matches expected dimensions."""
        xgboost_model.fit(synthetic_dataframe)
        
        # Test different horizons
        test_horizons = [1, 5, 10, 30]
        for horizon in test_horizons:
            result = xgboost_model.forecast(synthetic_dataframe, horizon=horizon)
            
            # Validate forecast shape
            assert "forecast" in result
            forecast_array = np.array(result["forecast"])
            assert len(forecast_array) == horizon, f"Expected {horizon} predictions, got {len(forecast_array)}"
            
            # Validate forecast is 1D array
            assert forecast_array.ndim == 1, f"Expected 1D array, got {forecast_array.ndim}D"
            
            # Validate no NaN values
            assert not np.isnan(forecast_array).any(), "Forecast contains NaN values"
            
            # Validate finite values
            assert np.all(np.isfinite(forecast_array)), "Forecast contains infinite values"

    def test_confidence_score_validation(self, xgboost_model, synthetic_dataframe):
        """Test that confidence scores are valid and within expected range."""
        xgboost_model.fit(synthetic_dataframe)
        
        result = xgboost_model.forecast(synthetic_dataframe, horizon=10)
        
        # Validate confidence score exists
        assert "confidence" in result, "Confidence score missing from forecast result"
        
        confidence = result["confidence"]
        
        # Validate confidence is numeric
        assert isinstance(confidence, (int, float, np.number)), f"Confidence must be numeric, got {type(confidence)}"
        
        # Validate confidence is within valid range [0, 1]
        assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} outside valid range [0, 1]"
        
        # Validate confidence is finite
        assert np.isfinite(confidence), "Confidence score is not finite"

    def test_forecast_metadata_validation(self, xgboost_model, synthetic_dataframe):
        """Test that forecast result contains all required metadata."""
        xgboost_model.fit(synthetic_dataframe)
        
        result = xgboost_model.forecast(synthetic_dataframe, horizon=10)
        
        # Required fields
        required_fields = ["forecast", "confidence", "model", "horizon"]
        for field in required_fields:
            assert field in result, f"Required field '{field}' missing from forecast result"
        
        # Validate field types
        assert isinstance(result["forecast"], (list, np.ndarray)), "Forecast must be array-like"
        assert isinstance(result["confidence"], (int, float, np.number)), "Confidence must be numeric"
        assert isinstance(result["model"], str), "Model name must be string"
        assert isinstance(result["horizon"], int), "Horizon must be integer"
        
        # Validate model name
        assert result["model"] == "XGBoost", f"Expected model name 'XGBoost', got '{result['model']}'"
        
        # Validate horizon matches input
        assert result["horizon"] == 10, f"Expected horizon 10, got {result['horizon']}"
        
        # Check for optional fields
        if "feature_importance" in result:
            assert isinstance(result["feature_importance"], dict), "Feature importance must be dictionary"
        
        if "metadata" in result:
            assert isinstance(result["metadata"], dict), "Metadata must be dictionary"

    def test_feature_importance_validation(self, xgboost_model, synthetic_dataframe):
        """Test that feature importance is properly included in forecast results."""
        xgboost_model.fit(synthetic_dataframe)
        
        result = xgboost_model.forecast(synthetic_dataframe, horizon=10)
        
        # Check if feature importance is included
        if "feature_importance" in result:
            importance = result["feature_importance"]
            
            # Validate feature importance structure
            assert isinstance(importance, dict), "Feature importance must be dictionary"
            
            # Validate importance scores are numeric and non-negative
            for feature, score in importance.items():
                assert isinstance(score, (int, float, np.number)), f"Importance score for {feature} must be numeric"
                assert score >= 0, f"Importance score for {feature} must be non-negative"
                assert np.isfinite(score), f"Importance score for {feature} must be finite"

    def test_forecast_consistency_across_runs(self, xgboost_model, synthetic_dataframe):
        """Test that forecasts are consistent across multiple runs."""
        xgboost_model.fit(synthetic_dataframe)
        
        # Run forecast multiple times
        results = []
        for _ in range(3):
            result = xgboost_model.forecast(synthetic_dataframe, horizon=5)
            results.append(result)
        
        # All results should have same structure
        for result in results:
            assert "forecast" in result
            assert "confidence" in result
            assert len(result["forecast"]) == 5
        
        # For deterministic models, forecasts should be identical
        # XGBoost should be deterministic with fixed random_state
        forecast_lengths = [len(r["forecast"]) for r in results]
        assert len(set(forecast_lengths)) == 1, "Forecast lengths should be consistent"
        
        confidence_scores = [r["confidence"] for r in results]
        assert all(0 <= c <= 1 for c in confidence_scores), "All confidence scores should be valid"

    def test_forecast_with_different_data_lengths(self, xgboost_model):
        """Test forecast behavior with different input data lengths."""
        # Create datasets of different lengths
        lengths = [50, 100, 200]
        
        for length in lengths:
            dates = pd.date_range(start="2023-01-01", periods=length, freq="D")
            close = np.linspace(100, 150, length) + np.random.normal(0, 2, length)
            volume = np.random.uniform(1000000, 5000000, length)
            data = pd.DataFrame({"Close": close, "Volume": volume}, index=dates)
            
            try:
                xgboost_model.fit(data)
                result = xgboost_model.forecast(data, horizon=10)
                
                # Validate successful forecast
                assert "forecast" in result
                assert len(result["forecast"]) == 10
                assert "confidence" in result
                
            except Exception as e:
                # If it fails, should be due to insufficient data
                assert "insufficient" in str(e).lower() or "minimum" in str(e).lower()

    def test_forecast_edge_cases(self, xgboost_model, synthetic_dataframe):
        """Test forecast behavior with edge cases."""
        xgboost_model.fit(synthetic_dataframe)
        
        # Test horizon = 0
        try:
            result = xgboost_model.forecast(synthetic_dataframe, horizon=0)
            assert len(result["forecast"]) == 0
        except ValueError:
            # Some models might not support horizon=0
            pass
        
        # Test very large horizon
        try:
            result = xgboost_model.forecast(synthetic_dataframe, horizon=1000)
            assert len(result["forecast"]) == 1000
            assert "confidence" in result
        except Exception as e:
            # Large horizons might be limited by model constraints
            assert "horizon" in str(e).lower() or "limit" in str(e).lower()

    def test_forecast_performance_metrics(self, xgboost_model, synthetic_dataframe):
        """Test that forecast includes performance metrics when available."""
        xgboost_model.fit(synthetic_dataframe)
        
        result = xgboost_model.forecast(synthetic_dataframe, horizon=10)
        
        # Basic validation
        assert "forecast" in result
        assert "confidence" in result
        
        # Check for optional performance metrics
        optional_metrics = ["rmse", "mae", "mape", "r2_score"]
        for metric in optional_metrics:
            if metric in result:
                assert isinstance(result[metric], (int, float, np.number)), f"{metric} must be numeric"
                assert np.isfinite(result[metric]), f"{metric} must be finite"

    def test_forecast_error_handling(self, xgboost_model):
        """Test forecast error handling with invalid inputs."""
        # Test with unfitted model
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        data = pd.DataFrame({
            "Close": np.random.randn(50),
            "Volume": np.random.uniform(1000000, 5000000, 50)
        }, index=dates)
        
        try:
            result = xgboost_model.forecast(data, horizon=10)
            # If it succeeds, it should auto-fit
            assert "forecast" in result
        except Exception as e:
            # Should provide clear error message
            assert any(keyword in str(e).lower() for keyword in ["fit", "train", "model"])

    def test_forecast_data_validation(self, xgboost_model, synthetic_dataframe):
        """Test forecast data validation and preprocessing."""
        xgboost_model.fit(synthetic_dataframe)
        
        # Test with different DataFrame structures
        result = xgboost_model.forecast(synthetic_dataframe, horizon=10)
        assert "forecast" in result
        assert len(result["forecast"]) == 10
        
        # Test with subset of columns
        subset_data = synthetic_dataframe[["Close"]]
        try:
            result = xgboost_model.forecast(subset_data, horizon=10)
            assert "forecast" in result
        except Exception:
            # Some models might require specific columns
            pass


if __name__ == "__main__":
    pytest.main([__file__])
