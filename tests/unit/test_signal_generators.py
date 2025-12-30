"""
Unit tests for signal generators.

Tests signal generation edge cases including overlapping buy signals,
missing 'Close' column, and other validation scenarios.
"""

import warnings
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

# Import signal generation utilities
try:
    from utils.strategy_utils import validate_signal_schema
    from utils.validators import SignalSchemaValidator

    SIGNAL_UTILS_AVAILABLE = True
except ImportError:
    SIGNAL_UTILS_AVAILABLE = False
    validate_signal_schema = Mock()
    SignalSchemaValidator = Mock()


class TestSignalGenerators:
    """Test suite for signal generator edge cases."""

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data with OHLCV columns."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        return pd.DataFrame(
            {
                "Open": np.random.uniform(100, 110, 100),
                "High": np.random.uniform(110, 120, 100),
                "Low": np.random.uniform(90, 100, 100),
                "Close": np.random.uniform(100, 110, 100),
                "Volume": np.random.uniform(1000000, 5000000, 100),
            },
            index=dates,
        )

    @pytest.fixture
    def data_without_close(self):
        """Create data without 'Close' column."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        return pd.DataFrame(
            {
                "Open": np.random.uniform(100, 110, 50),
                "High": np.random.uniform(110, 120, 50),
                "Low": np.random.uniform(90, 100, 50),
                "Volume": np.random.uniform(1000000, 5000000, 50),
            },
            index=dates,
        )

    @pytest.fixture
    def overlapping_signals_data(self):
        """Create data that would generate overlapping buy signals."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        # Create price data that would trigger multiple buy signals
        close_prices = [100] * 50
        # Add some volatility to trigger signals
        for i in range(10, 50, 5):
            close_prices[i : i + 3] = [95, 94, 93]  # Oversold conditions
        return pd.DataFrame(
            {
                "Close": close_prices,
                "Volume": [1000000] * 50,
            },
            index=dates,
        )

    def test_missing_close_column(self, data_without_close):
        """Test handling of data without 'Close' column."""
        if not SIGNAL_UTILS_AVAILABLE:
            pytest.skip("Signal utilities not available")

        # Test schema validation
        assert not validate_signal_schema(data_without_close)

        # Test SignalSchemaValidator
        assert not SignalSchemaValidator.validate(data_without_close)

        with pytest.raises(ValueError):
            SignalSchemaValidator.assert_valid(data_without_close)

    def test_overlapping_buy_signals(self, overlapping_signals_data):
        """Test handling of overlapping buy signals."""
        if not SIGNAL_UTILS_AVAILABLE:
            pytest.skip("Signal utilities not available")

        # Create overlapping buy signals
        signals = pd.Series(0, index=overlapping_signals_data.index)
        signals.iloc[10:15] = 1  # First buy signal
        signals.iloc[12:17] = 1  # Overlapping buy signal

        # Check for overlapping signals
        overlapping_count = 0
        for i in range(len(signals) - 1):
            if signals.iloc[i] == 1 and signals.iloc[i + 1] == 1:
                overlapping_count += 1

        assert overlapping_count > 0, "Should have overlapping buy signals"

        # Test signal validation
        signal_df = pd.DataFrame(
            {
                "Close": overlapping_signals_data["Close"],
                "SignalType": signals,
            },
            index=overlapping_signals_data.index,
        )

        # Should pass basic schema validation
        assert SignalSchemaValidator.validate(signal_df)

    def test_signal_schema_validation(self, sample_price_data):
        """Test signal schema validation with valid data."""
        if not SIGNAL_UTILS_AVAILABLE:
            pytest.skip("Signal utilities not available")

        # Create valid signal DataFrame
        signals = pd.Series([1, -1, 0, 1, -1] * 20, index=sample_price_data.index)
        signal_df = pd.DataFrame(
            {
                "Close": sample_price_data["Close"],
                "SignalType": signals,
                "Confidence": np.random.uniform(0.5, 1.0, len(signals)),
            },
            index=sample_price_data.index,
        )

        # Should pass validation
        assert validate_signal_schema(signal_df)
        assert SignalSchemaValidator.validate(signal_df)
        SignalSchemaValidator.assert_valid(signal_df)

    def test_signal_schema_invalid_index(self):
        """Test signal schema validation with invalid index."""
        if not SIGNAL_UTILS_AVAILABLE:
            pytest.skip("Signal utilities not available")

        # Create DataFrame with None index
        signal_df = pd.DataFrame(
            {
                "Close": [100, 101, 102],
                "SignalType": [1, -1, 0],
            }
        )
        signal_df.index = [None, None, None]

        # Should fail validation
        assert not SignalSchemaValidator.validate(signal_df)

    def test_signal_schema_missing_required_columns(self):
        """Test signal schema validation with missing required columns."""
        if not SIGNAL_UTILS_AVAILABLE:
            pytest.skip("Signal utilities not available")

        # Create DataFrame missing required columns
        signal_df = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [110, 111, 112],
            }
        )

        # Should fail validation
        assert not validate_signal_schema(signal_df)
        assert not SignalSchemaValidator.validate(signal_df)

    def test_signal_registration(self):
        """Test dynamic signal class registration."""
        if not SIGNAL_UTILS_AVAILABLE:
            pytest.skip("Signal utilities not available")

        try:
            from utils.strategy_utils import get_signal_class, register_signal

            # Mock signal class
            class MockSignalGenerator:
                def __init__(self):
                    self.name = "MockSignal"

            # Test registration
            register_signal("MockSignal", MockSignalGenerator)

            # Test retrieval
            retrieved_class = get_signal_class("MockSignal")
            assert retrieved_class == MockSignalGenerator

            # Test non-existent signal
            assert get_signal_class("NonExistentSignal") is None

        except ImportError:
            pytest.skip("Signal registration utilities not available")
