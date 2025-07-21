"""Tests for the RSI strategy using pandas-ta."""

import json
import os
import sys
from unittest.mock import mock_open, patch

import numpy as np
import pandas as pd
import pytest

from trading.strategies.rsi_signals import generate_rsi_signals

# Add project root to path for imports
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


class TestRSISignals:
    """Test RSI signal generation functionality."""

    def test_rsi_calculation_with_pandas_ta(self, sample_price_data):
        """Test that RSI is calculated correctly using pandas-ta."""
        # Generate RSI signals
        result = generate_rsi_signals(sample_price_data, period=14)

        # Check that RSI column exists and has correct properties
        assert "rsi" in result.columns
        assert isinstance(result["rsi"], pd.Series)
        assert len(result["rsi"]) == len(sample_price_data)

        # RSI should be between 0 and 100
        rsi_values = result["rsi"].dropna()
        assert (rsi_values >= 0).all() and (rsi_values <= 100).all()

        # First few values should be NaN (due to RSI calculation window)
        assert result["rsi"].iloc[:13].isna().all()
        assert not result["rsi"].iloc[14:].isna().all()

    def test_signal_generation(self, sample_price_data):
        """Test that trading signals are generated correctly."""
        result = generate_rsi_signals(sample_price_data, period=14)

        # Check signal column exists
        assert "signal" in result.columns
        assert isinstance(result["signal"], pd.Series)
        assert len(result["signal"]) == len(sample_price_data)

        # Signals should be 1 (buy), -1 (sell), or 0 (hold)
        valid_signals = result["signal"].dropna()
        assert valid_signals.isin([1, 0, -1]).all()

    def test_buy_signal_generation(self, sample_price_data):
        """Test that oversold condition triggers buy signal."""
        # Create data that will generate low RSI (oversold)
        low_rsi_data = sample_price_data.copy()
        # Force downward trend to create oversold condition
        low_rsi_data["Close"] = np.linspace(100, 90, len(low_rsi_data))

        result = generate_rsi_signals(low_rsi_data, period=14, buy_threshold=30)

        # Should have at least one buy signal (1)
        assert (
            result["signal"] == 1
        ).any(), "No buy signals generated for oversold condition"

    def test_sell_signal_generation(self, sample_price_data):
        """Test that overbought condition triggers sell signal."""
        # Create data that will generate high RSI (overbought)
        high_rsi_data = sample_price_data.copy()
        # Force upward trend to create overbought condition
        high_rsi_data["Close"] = np.linspace(100, 110, len(high_rsi_data))

        result = generate_rsi_signals(high_rsi_data, period=14, sell_threshold=70)

        # Should have at least one sell signal (-1)
        assert (
            result["signal"] == -1
        ).any(), "No sell signals generated for overbought condition"

    def test_returns_calculation(self, sample_price_data):
        """Test that returns are calculated correctly."""
        result = generate_rsi_signals(sample_price_data, period=14)

        # Check returns columns exist
        assert "returns" in result.columns
        assert "strategy_returns" in result.columns
        assert "cumulative_returns" in result.columns
        assert "strategy_cumulative_returns" in result.columns

        # Returns should be calculated correctly
        expected_returns = sample_price_data["Close"].pct_change()
        pd.testing.assert_series_equal(
            result["returns"], expected_returns, check_names=False
        )

        # Strategy returns should be signal * returns (shifted by 1)
        expected_strategy_returns = result["signal"].shift(1) * result["returns"]
        pd.testing.assert_series_equal(
            result["strategy_returns"], expected_strategy_returns, check_names=False
        )

    def test_parameter_validation(self, sample_price_data):
        """Test that invalid parameters raise appropriate errors."""
        # Test invalid period
        with pytest.raises(Exception):
            generate_rsi_signals(sample_price_data, period=0)

        # Test invalid thresholds
        with pytest.raises(Exception):
            generate_rsi_signals(
                sample_price_data, buy_threshold=80, sell_threshold=20
            )  # Inverted thresholds

    def test_empty_data_handling(self):
        """Test that empty data is handled correctly."""
        empty_data = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        with pytest.raises(Exception):
            generate_rsi_signals(empty_data)

    def test_missing_data_handling(self):
        """Test that missing data is handled correctly."""
        data_with_nans = pd.DataFrame(
            {
                "Open": [100, np.nan, 101],
                "High": [102, 103, np.nan],
                "Low": [98, 99, 100],
                "Close": [101, 102, 103],
                "Volume": [1000000, 1100000, 1200000],
            }
        )

        # Should handle NaN values gracefully
        result = generate_rsi_signals(data_with_nans, period=2)
        assert isinstance(result, pd.DataFrame)

    def test_optimized_settings_loading(self, sample_price_data):
        """Test loading of optimized settings."""
        mock_settings = {
            "optimal_period": 21,
            "buy_threshold": 25,
            "sell_threshold": 75,
        }

        with (
            patch("builtins.open", mock_open(read_data=json.dumps(mock_settings))),
            patch("pathlib.Path.exists", return_value=True),
        ):
            result = generate_rsi_signals(sample_price_data, ticker="AAPL")

            # Should use optimized settings
            assert "rsi" in result.columns

    def test_signal_consistency(self, sample_price_data):
        """Test that signals are consistent with RSI values."""
        result = generate_rsi_signals(
            sample_price_data, period=14, buy_threshold=30, sell_threshold=70
        )

        # Get non-NaN values
        mask = result["rsi"].notna() & result["signal"].notna()
        rsi_values = result.loc[mask, "rsi"]
        signals = result.loc[mask, "signal"]

        # Check that oversold RSI corresponds to buy signals
        oversold_mask = rsi_values < 30
        if oversold_mask.any():
            assert (
                signals[oversold_mask] == 1
            ).all(), "Oversold RSI should generate buy signals"

        # Check that overbought RSI corresponds to sell signals
        overbought_mask = rsi_values > 70
        if overbought_mask.any():
            assert (
                signals[overbought_mask] == -1
            ).all(), "Overbought RSI should generate sell signals"

    def test_different_periods(self, sample_price_data):
        """Test RSI calculation with different periods."""
        periods = [7, 14, 21]

        for period in periods:
            result = generate_rsi_signals(sample_price_data, period=period)

            # Check that RSI is calculated
            assert "rsi" in result.columns
            assert not result["rsi"].iloc[period:].isna().all()

    def test_threshold_adjustment(self, sample_price_data):
        """Test signal generation with different thresholds."""
        # Test with tighter thresholds
        result_tight = generate_rsi_signals(
            sample_price_data, buy_threshold=40, sell_threshold=60
        )

        # Test with wider thresholds
        result_wide = generate_rsi_signals(
            sample_price_data, buy_threshold=20, sell_threshold=80
        )

        # Should have different signal patterns
        assert not result_tight["signal"].equals(result_wide["signal"])

    def test_cumulative_returns_calculation(self, sample_price_data):
        """Test that cumulative returns are calculated correctly."""
        result = generate_rsi_signals(sample_price_data, period=14)

        # Check cumulative returns calculation
        expected_cumulative = (1 + result["returns"]).cumprod()
        pd.testing.assert_series_equal(
            result["cumulative_returns"], expected_cumulative, check_names=False
        )

        # Check strategy cumulative returns
        expected_strategy_cumulative = (1 + result["strategy_returns"]).cumprod()
        pd.testing.assert_series_equal(
            result["strategy_cumulative_returns"],
            expected_strategy_cumulative,
            check_names=False,
        )

    def test_error_handling(self, sample_price_data):
        """Test error handling in signal generation."""
        # Test with invalid column names
        invalid_data = sample_price_data.copy()
        invalid_data = invalid_data.rename(columns={"Close": "close"})  # Wrong case

        with pytest.raises(Exception):
            generate_rsi_signals(invalid_data)

    def test_performance_metrics(self, sample_price_data):
        """Test that the strategy generates meaningful performance metrics."""
        result = generate_rsi_signals(sample_price_data, period=14)

        # Check that we have strategy returns
        strategy_returns = result["strategy_returns"].dropna()
        assert len(strategy_returns) > 0

        # Check that cumulative returns are reasonable
        cumulative_returns = result["strategy_cumulative_returns"].dropna()
        assert len(cumulative_returns) > 0
        assert (
            cumulative_returns >= 0
        ).all()  # Cumulative returns should be non-negative
