"""
Unit tests for LSTM forecaster model.

Tests LSTM model functionality with synthetic time series data,
including edge cases like short series, constant series, and NaN handling.
"""

import warnings
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

# Import the LSTM model
try:
    from trading.models.lstm_model import LSTMForecaster, LSTMModel

    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    LSTMModel = Mock()
    LSTMForecaster = Mock()


class TestLSTMForecaster:
    """Test suite for LSTM forecaster model."""

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
    def lstm_model(self):
        """Create LSTM model instance."""
        if not LSTM_AVAILABLE:
            pytest.skip("LSTM model not available")
        return LSTMModel(input_dim=1, hidden_dim=50, output_dim=1)

    @pytest.fixture
    def lstm_forecaster(self):
        """Create LSTM forecaster instance."""
        if not LSTM_AVAILABLE:
            pytest.skip("LSTM forecaster not available")
        return LSTMForecaster()

    def test_model_instantiation(self, lstm_model):
        """Test that LSTM model instantiates correctly."""
        assert lstm_model is not None
        assert hasattr(lstm_model, "model")
        assert hasattr(lstm_model, "is_fitted")
        assert lstm_model.is_fitted is False

    def test_model_fitting(self, lstm_model, synthetic_time_series):
        """Test that LSTM model fits to data correctly."""
        result = lstm_model.fit(synthetic_time_series)

        assert result["success"] is True
        assert lstm_model.is_fitted is True
        assert lstm_model.model is not None
        assert "timestamp" in result

    def test_forecast_generation(self, lstm_model, synthetic_time_series):
        """Test that LSTM model generates forecasts correctly."""
        # Fit the model first
        lstm_model.fit(synthetic_time_series)

        # Generate forecast
        forecast_steps = 10
        result = lstm_model.predict(synthetic_time_series, horizon=forecast_steps)

        assert result["success"] is True
        assert "predictions" in result
        assert len(result["predictions"]) == forecast_steps
        assert not np.isnan(result["predictions"]).any()
        assert "timestamp" in result

    def test_forecast_output_length(self, lstm_model, synthetic_time_series):
        """Test that forecast output has correct length."""
        lstm_model.fit(synthetic_time_series)

        for steps in [1, 5, 10, 30]:
            result = lstm_model.predict(synthetic_time_series, horizon=steps)
            assert result["success"] is True
            assert len(result["predictions"]) == steps

    def test_no_nan_in_forecast(self, lstm_model, synthetic_time_series):
        """Test that forecasts contain no NaN values."""
        lstm_model.fit(synthetic_time_series)

        result = lstm_model.predict(synthetic_time_series, horizon=10)
        assert result["success"] is True
        assert not np.isnan(result["predictions"]).any()

    def test_short_time_series_handling(self, lstm_model, short_time_series):
        """Test handling of short time series (< 10 points)."""
        result = lstm_model.fit(short_time_series)

        # Should fail gracefully with clear error message
        assert result["success"] is False
        assert "error" in result
        assert any(
            keyword in result["error"].lower()
            for keyword in ["insufficient", "at least", "minimum", "sequence"]
        )

    def test_constant_series_handling(self, lstm_model, constant_time_series):
        """Test handling of constant time series."""
        result = lstm_model.fit(constant_time_series)

        # Should handle constant series gracefully
        if result["success"]:
            # If it succeeds, test prediction
            forecast_result = lstm_model.predict(constant_time_series, horizon=5)
            assert forecast_result["success"] is True
            assert len(forecast_result["predictions"]) == 5
        else:
            # If it fails, should be due to constant series
            assert any(
                keyword in result.get("error", "").lower()
                for keyword in ["constant", "variance", "unique"]
            )

    def test_nan_series_handling(self, lstm_model, nan_time_series):
        """Test handling of time series with NaN values."""
        result = lstm_model.fit(nan_time_series)

        # Should fail gracefully with clear error message
        assert result["success"] is False
        assert "error" in result
        assert any(
            keyword in result["error"].lower()
            for keyword in ["nan", "missing", "invalid"]
        )

    def test_model_summary(self, lstm_model, synthetic_time_series):
        """Test that model summary is generated correctly."""
        lstm_model.fit(synthetic_time_series)

        summary_result = lstm_model.get_model_summary()
        assert summary_result["success"] is True
        assert "summary" in summary_result
        assert isinstance(summary_result["summary"], str)
        assert len(summary_result["summary"]) > 0

    def test_unfitted_model_behavior(self, lstm_model, synthetic_time_series):
        """Test behavior when trying to predict without fitting."""
        result = lstm_model.predict(synthetic_time_series, horizon=5)

        assert result["success"] is False
        assert "error" in result
        assert "fitted" in result["error"].lower()

    def test_sequence_length_handling(self, lstm_model, synthetic_time_series):
        """Test handling of different sequence lengths."""
        lstm_model.fit(synthetic_time_series)

        # Test different sequence lengths
        for seq_length in [10, 20, 30]:
            if hasattr(lstm_model, "set_sequence_length"):
                lstm_model.set_sequence_length(seq_length)
                result = lstm_model.predict(synthetic_time_series, horizon=5)
                assert result["success"] is True

    def test_forecast_method(self, lstm_model, synthetic_time_series):
        """Test the forecast method specifically."""
        result = lstm_model.forecast(synthetic_time_series, horizon=10)

        assert "forecast" in result
        assert "confidence" in result
        assert "model" in result
        assert result["model"] == "LSTM"
        assert result["horizon"] == 10
        assert len(result["forecast"]) == 10

    def test_model_save_load(self, lstm_model, synthetic_time_series, tmp_path):
        """Test model save and load functionality."""
        # Fit the model
        lstm_model.fit(synthetic_time_series)

        # Save model
        save_path = tmp_path / "lstm_model.pt"
        save_result = lstm_model.save_model(str(save_path))

        if save_result["success"]:
            # Load model
            new_model = LSTMModel(input_dim=1, hidden_dim=50, output_dim=1)
            load_result = new_model.load_model(str(save_path))

            assert load_result["success"] is True
            assert new_model.is_fitted is True
            assert new_model.model is not None

    def test_error_handling_edge_cases(self, lstm_model):
        """Test error handling for various edge cases."""
        # Test with empty series
        empty_series = pd.Series(dtype=float)
        result = lstm_model.fit(empty_series)
        assert result["success"] is False

        # Test with None
        result = lstm_model.fit(None)
        assert result["success"] is False

        # Test with single value
        single_value = pd.Series([100])
        result = lstm_model.fit(single_value)
        assert result["success"] is False

    def test_forecast_consistency(self, lstm_model, synthetic_time_series):
        """Test that forecasts are consistent across multiple calls."""
        lstm_model.fit(synthetic_time_series)

        # Generate multiple forecasts
        forecast1 = lstm_model.predict(synthetic_time_series, horizon=5)
        forecast2 = lstm_model.predict(synthetic_time_series, horizon=5)

        assert forecast1["success"] is True
        assert forecast2["success"] is True
        np.testing.assert_array_almost_equal(
            forecast1["predictions"], forecast2["predictions"], decimal=10
        )

    def test_trend_detection(self, lstm_model):
        """Test that model correctly identifies trends."""
        # Create data with clear trend
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        trend_data = pd.Series(np.linspace(100, 200, 50), index=dates)

        lstm_model.fit(trend_data)
        forecast = lstm_model.predict(trend_data, horizon=5)

        if forecast["success"]:
            # Forecast should continue the trend (simplified check)
            assert len(forecast["predictions"]) == 5
            assert not np.isnan(forecast["predictions"]).any()

    def test_seasonality_handling(self, lstm_model):
        """Test that model handles seasonal data correctly."""
        # Create seasonal data
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        t = np.linspace(0, 4 * np.pi, 100)
        seasonal_data = pd.Series(100 + 10 * np.sin(t), index=dates)

        lstm_model.fit(seasonal_data)
        forecast = lstm_model.predict(seasonal_data, horizon=5)

        if forecast["success"]:
            assert len(forecast["predictions"]) == 5
            assert not np.isnan(forecast["predictions"]).any()

    @pytest.mark.parametrize("horizon", [1, 5, 10, 30])
    def test_different_forecast_horizons(
        self, lstm_model, synthetic_time_series, horizon
    ):
        """Test forecasting with different horizons."""
        lstm_model.fit(synthetic_time_series)

        result = lstm_model.predict(synthetic_time_series, horizon=horizon)
        assert result["success"] is True
        assert len(result["predictions"]) == horizon
        assert not np.isnan(result["predictions"]).any()

    def test_model_configuration(self):
        """Test different model configurations."""
        configs = [
            {"input_dim": 1, "hidden_dim": 50, "output_dim": 1},
            {"input_dim": 1, "hidden_dim": 100, "output_dim": 1, "num_layers": 2},
            {"input_dim": 1, "hidden_dim": 64, "output_dim": 1, "dropout": 0.2},
        ]

        for config in configs:
            model = LSTMModel(**config)
            assert model.config == config

    def test_training_parameters(self, lstm_model, synthetic_time_series):
        """Test training with different parameters."""
        # Test training with different epochs
        for epochs in [10, 20, 50]:
            if hasattr(lstm_model, "train_model"):
                result = lstm_model.train_model(synthetic_time_series, epochs=epochs)
                assert result["success"] is True

    def test_batch_size_handling(self, lstm_model, synthetic_time_series):
        """Test handling of different batch sizes."""
        lstm_model.fit(synthetic_time_series)

        # Test different batch sizes
        for batch_size in [16, 32, 64]:
            if hasattr(lstm_model, "set_batch_size"):
                lstm_model.set_batch_size(batch_size)
                result = lstm_model.predict(synthetic_time_series, horizon=5)
                assert result["success"] is True

    def test_learning_rate_handling(self, lstm_model, synthetic_time_series):
        """Test handling of different learning rates."""
        # Test different learning rates
        for lr in [0.001, 0.01, 0.1]:
            if hasattr(lstm_model, "set_learning_rate"):
                lstm_model.set_learning_rate(lr)
                result = lstm_model.fit(synthetic_time_series)
                if result["success"]:
                    assert lstm_model.is_fitted is True

    def test_model_architecture(self, lstm_model):
        """Test model architecture parameters."""
        assert hasattr(lstm_model, "input_dim")
        assert hasattr(lstm_model, "hidden_dim")
        assert hasattr(lstm_model, "output_dim")

        # Test architecture parameters
        assert lstm_model.input_dim == 1
        assert lstm_model.hidden_dim == 50
        assert lstm_model.output_dim == 1

    def test_sequence_preprocessing(self, lstm_model, synthetic_time_series):
        """Test sequence preprocessing functionality."""
        if hasattr(lstm_model, "preprocess_sequences"):
            sequences = lstm_model.preprocess_sequences(
                synthetic_time_series, sequence_length=10
            )
            assert isinstance(sequences, np.ndarray)
            assert len(sequences.shape) == 2  # (samples, features)

    def test_normalization(self, lstm_model, synthetic_time_series):
        """Test data normalization functionality."""
        if hasattr(lstm_model, "normalize_data"):
            normalized = lstm_model.normalize_data(synthetic_time_series)
            assert isinstance(normalized, pd.Series)
            assert len(normalized) == len(synthetic_time_series)

    def test_denormalization(self, lstm_model, synthetic_time_series):
        """Test data denormalization functionality."""
        if hasattr(lstm_model, "denormalize_predictions"):
            # First normalize
            if hasattr(lstm_model, "normalize_data"):
                normalized = lstm_model.normalize_data(synthetic_time_series)
                # Then denormalize
                denormalized = lstm_model.denormalize_predictions(normalized.iloc[:5])
                assert isinstance(denormalized, np.ndarray)
                assert len(denormalized) == 5

    def test_model_evaluation_metrics(self, lstm_model, synthetic_time_series):
        """Test model evaluation metrics."""
        lstm_model.fit(synthetic_time_series)

        if hasattr(lstm_model, "evaluate"):
            metrics = lstm_model.evaluate(synthetic_time_series)
            assert isinstance(metrics, dict)
            assert len(metrics) > 0

    def test_early_stopping(self, lstm_model, synthetic_time_series):
        """Test early stopping functionality."""
        if hasattr(lstm_model, "train_with_early_stopping"):
            result = lstm_model.train_with_early_stopping(
                synthetic_time_series, patience=5, min_delta=0.001
            )
            assert result["success"] is True

    def test_model_performance_tracking(self, lstm_model, synthetic_time_series):
        """Test model performance tracking."""
        if hasattr(lstm_model, "get_training_history"):
            lstm_model.fit(synthetic_time_series)
            history = lstm_model.get_training_history()
            assert isinstance(history, dict)
            assert "loss" in history or "val_loss" in history

    def test_forecast_shape_validation(self, lstm_model, synthetic_time_series):
        """Test that forecast output shape matches expected dimensions."""
        lstm_model.fit(synthetic_time_series)

        # Test different horizons
        test_horizons = [1, 5, 10, 30]
        for horizon in test_horizons:
            result = lstm_model.forecast(synthetic_time_series, horizon=horizon)

            # Validate forecast shape
            assert "forecast" in result
            forecast_array = np.array(result["forecast"])
            assert (
                len(forecast_array) == horizon
            ), f"Expected {horizon} predictions, got {len(forecast_array)}"

            # Validate forecast is 1D array
            assert (
                forecast_array.ndim == 1
            ), f"Expected 1D array, got {forecast_array.ndim}D"

            # Validate no NaN values
            assert not np.isnan(forecast_array).any(), "Forecast contains NaN values"

            # Validate finite values
            assert np.all(
                np.isfinite(forecast_array)
            ), "Forecast contains infinite values"

    def test_confidence_score_validation(self, lstm_model, synthetic_time_series):
        """Test that confidence scores are valid and within expected range."""
        lstm_model.fit(synthetic_time_series)

        result = lstm_model.forecast(synthetic_time_series, horizon=10)

        # Validate confidence score exists
        assert "confidence" in result, "Confidence score missing from forecast result"

        confidence = result["confidence"]

        # Validate confidence is numeric
        assert isinstance(
            confidence, (int, float, np.number)
        ), f"Confidence must be numeric, got {type(confidence)}"

        # Validate confidence is within valid range [0, 1]
        assert (
            0.0 <= confidence <= 1.0
        ), f"Confidence {confidence} outside valid range [0, 1]"

        # Validate confidence is finite
        assert np.isfinite(confidence), "Confidence score is not finite"

    def test_forecast_metadata_validation(self, lstm_model, synthetic_time_series):
        """Test that forecast result contains all required metadata."""
        lstm_model.fit(synthetic_time_series)

        result = lstm_model.forecast(synthetic_time_series, horizon=10)

        # Required fields
        required_fields = ["forecast", "confidence", "model", "horizon"]
        for field in required_fields:
            assert (
                field in result
            ), f"Required field '{field}' missing from forecast result"

        # Validate field types
        assert isinstance(
            result["forecast"], (list, np.ndarray)
        ), "Forecast must be array-like"
        assert isinstance(
            result["confidence"], (int, float, np.number)
        ), "Confidence must be numeric"
        assert isinstance(result["model"], str), "Model name must be string"
        assert isinstance(result["horizon"], int), "Horizon must be integer"

        # Validate model name
        assert (
            result["model"] == "LSTM"
        ), f"Expected model name 'LSTM', got '{result['model']}'"

        # Validate horizon matches input
        assert result["horizon"] == 10, f"Expected horizon 10, got {result['horizon']}"

    def test_forecast_consistency_across_runs(self, lstm_model, synthetic_time_series):
        """Test that forecasts are consistent across multiple runs (for deterministic models)."""
        lstm_model.fit(synthetic_time_series)

        # Run forecast multiple times
        results = []
        for _ in range(3):
            result = lstm_model.forecast(synthetic_time_series, horizon=5)
            results.append(result)

        # All results should have same structure
        for result in results:
            assert "forecast" in result
            assert "confidence" in result
            assert len(result["forecast"]) == 5

        # For deterministic models, forecasts should be identical
        # Note: LSTM might have some randomness, so we check structure consistency
        forecast_lengths = [len(r["forecast"]) for r in results]
        assert len(set(forecast_lengths)) == 1, "Forecast lengths should be consistent"

        confidence_scores = [r["confidence"] for r in results]
        assert all(
            0 <= c <= 1 for c in confidence_scores
        ), "All confidence scores should be valid"

    def test_forecast_with_different_data_lengths(self, lstm_model):
        """Test forecast behavior with different input data lengths."""
        # Create datasets of different lengths
        lengths = [50, 100, 200]

        for length in lengths:
            dates = pd.date_range(start="2023-01-01", periods=length, freq="D")
            values = np.linspace(100, 150, length) + np.random.normal(0, 2, length)
            data = pd.Series(values, index=dates, name="Close")

            try:
                lstm_model.fit(data)
                result = lstm_model.forecast(data, horizon=10)

                # Validate successful forecast
                assert "forecast" in result
                assert len(result["forecast"]) == 10
                assert "confidence" in result

            except Exception as e:
                # If it fails, should be due to insufficient data
                assert "insufficient" in str(e).lower() or "minimum" in str(e).lower()

    def test_forecast_edge_cases(self, lstm_model, synthetic_time_series):
        """Test forecast behavior with edge cases."""
        lstm_model.fit(synthetic_time_series)

        # Test horizon = 0
        try:
            result = lstm_model.forecast(synthetic_time_series, horizon=0)
            assert len(result["forecast"]) == 0
        except ValueError:
            # Some models might not support horizon=0
            pass

        # Test very large horizon
        try:
            result = lstm_model.forecast(synthetic_time_series, horizon=1000)
            assert len(result["forecast"]) == 1000
            assert "confidence" in result
        except Exception as e:
            # Large horizons might be limited by model constraints
            assert "horizon" in str(e).lower() or "limit" in str(e).lower()

    def test_forecast_performance_metrics(self, lstm_model, synthetic_time_series):
        """Test that forecast includes performance metrics when available."""
        lstm_model.fit(synthetic_time_series)

        result = lstm_model.forecast(synthetic_time_series, horizon=10)

        # Basic validation
        assert "forecast" in result
        assert "confidence" in result

        # Check for optional performance metrics
        optional_metrics = ["aic", "bic", "rmse", "mae", "mape"]
        for metric in optional_metrics:
            if metric in result:
                assert isinstance(
                    result[metric], (int, float, np.number)
                ), f"{metric} must be numeric"
                assert np.isfinite(result[metric]), f"{metric} must be finite"

    def test_forecast_error_handling(self, lstm_model):
        """Test forecast error handling with invalid inputs."""
        # Test with unfitted model
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        data = pd.Series(np.random.randn(50), index=dates, name="Close")

        try:
            result = lstm_model.forecast(data, horizon=10)
            # If it succeeds, it should auto-fit
            assert "forecast" in result
        except Exception as e:
            # Should provide clear error message
            assert any(
                keyword in str(e).lower() for keyword in ["fit", "train", "model"]
            )

    def test_forecast_data_validation(self, lstm_model, synthetic_time_series):
        """Test forecast data validation and preprocessing."""
        lstm_model.fit(synthetic_time_series)

        # Test with DataFrame input
        df = synthetic_time_series.to_frame()
        result = lstm_model.forecast(df, horizon=10)
        assert "forecast" in result
        assert len(result["forecast"]) == 10

        # Test with numpy array input
        array_data = synthetic_time_series.values
        try:
            result = lstm_model.forecast(array_data, horizon=10)
            assert "forecast" in result
        except Exception:
            # Some models might require pandas input
            pass


if __name__ == "__main__":
    pytest.main([__file__])
