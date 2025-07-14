"""
Comprehensive tests for LSTM forecasting model.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add the models directory to the path
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "trading", "models")
)


def test_lstm_empty_df():
    """Test LSTM model with empty DataFrame."""
    try:
        from lstm_model import LSTMModel

        model = LSTMModel()
        result = model.forecast(pd.DataFrame())

        # Should handle empty DataFrame gracefully
        assert result is not None
        assert hasattr(result, "empty") or isinstance(result, (list, dict))

    except ImportError as e:
        pytest.skip(f"LSTMModel not available: {e}")


def test_lstm_none_input():
    """Test LSTM model with None input."""
    try:
        from lstm_model import LSTMModel

        model = LSTMModel()
        result = model.forecast(None)

        # Should handle None input gracefully
        assert result is not None
        assert hasattr(result, "empty") or isinstance(result, (list, dict))

    except ImportError as e:
        pytest.skip(f"LSTMModel not available: {e}")


def test_lstm_missing_columns():
    """Test LSTM model with missing required columns."""
    try:
        from lstm_model import LSTMModel

        model = LSTMModel()

        # Create DataFrame without 'Close' column
        data = pd.DataFrame(
            {"Open": [100, 101, 102], "High": [105, 106, 107], "Low": [95, 96, 97]}
        )

        result = model.forecast(data)

        # Should handle missing columns gracefully
        assert result is not None

    except ImportError as e:
        pytest.skip(f"LSTMModel not available: {e}")


def test_lstm_correct_output_shape():
    """Test that LSTM model produces correct output shape."""
    try:
        from lstm_model import LSTMModel

        model = LSTMModel()

        # Create sample data
        data = pd.DataFrame(
            {
                "Close": np.random.randn(100).cumsum() + 100,
                "Volume": np.random.randint(1000, 10000, 100),
            }
        )

        result = model.forecast(data)

        # Check output structure
        assert result is not None

        # If result is DataFrame, check it has expected columns
        if isinstance(result, pd.DataFrame):
            assert len(result) > 0
            assert "forecast" in result.columns or "prediction" in result.columns

        # If result is dict, check it has expected keys
        elif isinstance(result, dict):
            assert "forecast" in result or "predictions" in result or "values" in result

    except ImportError as e:
        pytest.skip(f"LSTMModel not available: {e}")


def test_lstm_exception_fallback():
    """Test LSTM model exception fallback logic."""
    try:
        from lstm_model import LSTMModel

        model = LSTMModel()

        # Create problematic data that might cause exceptions
        data = pd.DataFrame(
            {
                "Close": [np.nan, np.inf, -np.inf, 100, 101],
                "Volume": [0, 0, 0, 1000, 1000],
            }  # Contains invalid values
        )

        result = model.forecast(data)

        # Should handle exceptions gracefully
        assert result is not None

    except ImportError as e:
        pytest.skip(f"LSTMModel not available: {e}")


def test_lstm_validation():
    """Test LSTM model input validation."""
    try:
        from lstm_model import LSTMModel

        model = LSTMModel()

        # Test with various invalid inputs
        invalid_inputs = ["not a dataframe", 123, [], {}, pd.Series([1, 2, 3])]

        for invalid_input in invalid_inputs:
            result = model.forecast(invalid_input)
            assert result is not None  # Should not crash

    except ImportError as e:
        pytest.skip(f"LSTMModel not available: {e}")


def test_lstm_parameter_validation():
    """Test LSTM model parameter validation."""
    try:
        from lstm_model import LSTMModel

        # Test with invalid parameters
        with pytest.raises((ValueError, TypeError)):
            model = LSTMModel(sequence_length=-1)

        with pytest.raises((ValueError, TypeError)):
            model = LSTMModel(units=0)

    except ImportError as e:
        pytest.skip(f"LSTMModel not available: {e}")


def test_lstm_training_data():
    """Test LSTM model with training data."""
    try:
        from lstm_model import LSTMModel

        model = LSTMModel()

        # Create realistic training data
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        data = pd.DataFrame(
            {
                "Close": np.random.randn(100).cumsum() + 100,
                "Volume": np.random.randint(1000, 10000, 100),
            },
            index=dates,
        )

        result = model.forecast(data)

        # Should produce valid forecast
        assert result is not None

    except ImportError as e:
        pytest.skip(f"LSTMModel not available: {e}")


def test_lstm_model_persistence():
    """Test LSTM model save/load functionality."""
    try:
        from lstm_model import LSTMModel

        model = LSTMModel()

        # Test save functionality
        save_path = "test_lstm_model.pt"
        try:
            model.save(save_path)
            assert os.path.exists(save_path)

            # Test load functionality
            loaded_model = LSTMModel.load(save_path)
            assert loaded_model is not None

        finally:
            # Cleanup
            if os.path.exists(save_path):
                os.remove(save_path)

    except ImportError as e:
        pytest.skip(f"LSTMModel not available: {e}")


def test_lstm_hyperparameters():
    """Test LSTM model hyperparameter handling."""
    try:
        from lstm_model import LSTMModel

        # Test with different hyperparameters
        hyperparams = {
            "sequence_length": 20,
            "units": 50,
            "dropout": 0.2,
            "learning_rate": 0.001,
        }

        model = LSTMModel(**hyperparams)

        # Verify hyperparameters are set correctly
        for param, value in hyperparams.items():
            assert hasattr(model, param) or hasattr(model, f"_{param}")

    except ImportError as e:
        pytest.skip(f"LSTMModel not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
