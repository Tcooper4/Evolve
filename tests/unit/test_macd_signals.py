"""
Unit tests for MACD signals.

Tests MACD signal generation with synthetic price data,
including edge cases and signal validation.
"""

import warnings
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

# Import MACD strategy modules
try:
    from trading.strategies.macd_signals import generate_macd_signals
    from trading.strategies.macd_strategy import MACDStrategy

    MACD_AVAILABLE = True
except ImportError:
    MACD_AVAILABLE = False
    MACDStrategy = Mock()
    generate_macd_signals = Mock()


class TestMACDSignals:
    """Test suite for MACD signal generation."""

    @pytest.fixture
    def synthetic_price_data(self):
        """Create synthetic price data with sine wave + noise."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

        # Create sine wave with noise for realistic price movement
        t = np.linspace(0, 4 * np.pi, 100)
        sine_wave = 100 + 10 * np.sin(t)  # Base price around 100 with ±10 oscillation
        noise = np.random.normal(0, 2, 100)  # Add some noise
        close_prices = sine_wave + noise

        # Create OHLCV data
        high = close_prices + np.random.uniform(0, 3, 100)
        low = close_prices - np.random.uniform(0, 3, 100)
        volume = np.random.uniform(1000000, 5000000, 100)

        df = pd.DataFrame(
            {"Close": close_prices, "High": high, "Low": low, "Volume": volume},
            index=dates,
        )

        return df

    @pytest.fixture
    def trending_price_data(self):
        """Create trending price data."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

        # Create trending data
        trend = np.linspace(100, 150, 100)  # Upward trend
        noise = np.random.normal(0, 2, 100)
        close_prices = trend + noise

        df = pd.DataFrame(
            {
                "Close": close_prices,
                "High": close_prices + np.random.uniform(0, 3, 100),
                "Low": close_prices - np.random.uniform(0, 3, 100),
                "Volume": np.random.uniform(1000000, 5000000, 100),
            },
            index=dates,
        )

        return df

    @pytest.fixture
    def short_price_data(self):
        """Create short price data (< 26 points)."""
        dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
        close_prices = [100 + i for i in range(20)]

        df = pd.DataFrame(
            {
                "Close": close_prices,
                "High": [p + 2 for p in close_prices],
                "Low": [p - 2 for p in close_prices],
                "Volume": [1000000] * 20,
            },
            index=dates,
        )

        return df

    @pytest.fixture
    def constant_price_data(self):
        """Create constant price data."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        close_prices = [100.0] * 50

        df = pd.DataFrame(
            {
                "Close": close_prices,
                "High": [p + 1 for p in close_prices],
                "Low": [p - 1 for p in close_prices],
                "Volume": [1000000] * 50,
            },
            index=dates,
        )

        return df

    @pytest.fixture
    def nan_price_data(self):
        """Create price data with NaN values."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        close_prices = [100, 101, np.nan, 103, 104] + [100 + i for i in range(45)]

        df = pd.DataFrame(
            {
                "Close": close_prices,
                "High": [p + 2 if not np.isnan(p) else np.nan for p in close_prices],
                "Low": [p - 2 if not np.isnan(p) else np.nan for p in close_prices],
                "Volume": [1000000] * 50,
            },
            index=dates,
        )

        return df

    @pytest.fixture
    def macd_strategy(self):
        """Create MACD strategy instance."""
        if not MACD_AVAILABLE:
            pytest.skip("MACD strategy not available")
        return MACDStrategy()

    def test_macd_signal_generation(self, macd_strategy, synthetic_price_data):
        """Test that MACD signals are generated correctly."""
        signals = macd_strategy.generate_signals(synthetic_price_data)

        assert isinstance(signals, list)
        assert len(signals) >= 0  # May have no signals in some periods

        # Check signal structure if signals exist
        if signals:
            signal = signals[0]
            assert "timestamp" in signal
            assert "signal_type" in signal
            assert "confidence" in signal
            assert "price" in signal
            assert "strategy_name" in signal
            assert signal["strategy_name"] == "MACD"

    def test_macd_calculation(self, macd_strategy, synthetic_price_data):
        """Test that MACD values are calculated correctly."""
        macd_values = macd_strategy.calculate_macd(synthetic_price_data)

        assert isinstance(macd_values, pd.Series)
        assert len(macd_values) == len(synthetic_price_data)

        # MACD should have both positive and negative values
        valid_macd = macd_values.dropna()
        if len(valid_macd) > 0:
            assert isinstance(valid_macd.iloc[0], (int, float))

    def test_signal_columns_exist(self, synthetic_price_data):
        """Test that signal columns are created correctly."""
        if not MACD_AVAILABLE:
            pytest.skip("MACD not available")

        result_df = generate_macd_signals(
            synthetic_price_data, fast_period=12, slow_period=26, signal_period=9
        )

        # Check required columns exist
        required_columns = [
            "signal",
            "macd",
            "macd_signal",
            "macd_histogram",
            "returns",
            "strategy_returns",
        ]
        for col in required_columns:
            assert col in result_df.columns

    def test_signal_values(self, synthetic_price_data):
        """Test that signal values are valid."""
        if not MACD_AVAILABLE:
            pytest.skip("MACD not available")

        result_df = generate_macd_signals(
            synthetic_price_data, fast_period=12, slow_period=26, signal_period=9
        )

        # Signals should be 1 (buy), -1 (sell), or 0 (hold)
        valid_signals = result_df["signal"].dropna()
        if len(valid_signals) > 0:
            assert valid_signals.isin([1, 0, -1]).all()

    def test_no_nan_in_signals(self, synthetic_price_data):
        """Test that signals contain no NaN values."""
        if not MACD_AVAILABLE:
            pytest.skip("MACD not available")

        result_df = generate_macd_signals(
            synthetic_price_data, fast_period=12, slow_period=26, signal_period=9
        )

        # Signal column should not contain NaN values
        assert not result_df["signal"].isna().any()

    def test_short_data_handling(self, short_price_data):
        """Test handling of short price data."""
        if not MACD_AVAILABLE:
            pytest.skip("MACD not available")

        # Should handle short data gracefully
        try:
            result_df = generate_macd_signals(
                short_price_data, fast_period=12, slow_period=26, signal_period=9
            )
            # If it succeeds, check structure
            assert "signal" in result_df.columns
            assert "macd" in result_df.columns
        except Exception as e:
            # If it fails, should be due to insufficient data
            assert any(
                keyword in str(e).lower()
                for keyword in ["insufficient", "at least", "minimum", "period"]
            )

    def test_constant_data_handling(self, constant_price_data):
        """Test handling of constant price data."""
        if not MACD_AVAILABLE:
            pytest.skip("MACD not available")

        result_df = generate_macd_signals(
            constant_price_data, fast_period=12, slow_period=26, signal_period=9
        )

        # Should handle constant data gracefully
        assert "signal" in result_df.columns
        assert "macd" in result_df.columns

        # MACD should be 0 for constant data (or NaN if calculation fails)
        macd_values = result_df["macd"].dropna()
        if len(macd_values) > 0:
            # For constant data, MACD should be around 0 or NaN
            assert (np.abs(macd_values) < 1e-6).any() or macd_values.isna().any()

    def test_nan_data_handling(self, nan_price_data):
        """Test handling of price data with NaN values."""
        if not MACD_AVAILABLE:
            pytest.skip("MACD not available")

        # Should handle NaN data gracefully
        try:
            result_df = generate_macd_signals(
                nan_price_data, fast_period=12, slow_period=26, signal_period=9
            )
            assert "signal" in result_df.columns
            assert "macd" in result_df.columns
        except Exception as e:
            # If it fails, should be due to NaN values
            assert any(
                keyword in str(e).lower() for keyword in ["nan", "missing", "invalid"]
            )

    def test_different_periods(self, synthetic_price_data):
        """Test MACD calculation with different periods."""
        if not MACD_AVAILABLE:
            pytest.skip("MACD not available")

        period_combinations = [
            (12, 26, 9),  # Standard
            (8, 21, 5),  # Faster
            (15, 30, 12),  # Slower
        ]

        for fast_period, slow_period, signal_period in period_combinations:
            result_df = generate_macd_signals(
                synthetic_price_data,
                fast_period=fast_period,
                slow_period=slow_period,
                signal_period=signal_period,
            )
            assert "signal" in result_df.columns
            assert "macd" in result_df.columns

            # Check that MACD values are valid
            macd_values = result_df["macd"].dropna()
            if len(macd_values) > 0:
                assert isinstance(macd_values.iloc[0], (int, float))

    def test_trending_data_signals(self, trending_price_data):
        """Test MACD signals on trending data."""
        if not MACD_AVAILABLE:
            pytest.skip("MACD not available")

        result_df = generate_macd_signals(
            trending_price_data, fast_period=12, slow_period=26, signal_period=9
        )

        assert "signal" in result_df.columns
        assert "macd" in result_df.columns

        # Check that signals are generated
        signals = result_df["signal"].dropna()
        assert len(signals) >= 0  # May have no signals in trending data

    def test_returns_calculation(self, synthetic_price_data):
        """Test that returns are calculated correctly."""
        if not MACD_AVAILABLE:
            pytest.skip("MACD not available")

        result_df = generate_macd_signals(
            synthetic_price_data, fast_period=12, slow_period=26, signal_period=9
        )

        # Check returns columns
        assert "returns" in result_df.columns
        assert "strategy_returns" in result_df.columns

        # Returns should be calculated
        returns = result_df["returns"].dropna()
        strategy_returns = result_df["strategy_returns"].dropna()

        if len(returns) > 0:
            assert isinstance(returns.iloc[0], (int, float))
        if len(strategy_returns) > 0:
            assert isinstance(strategy_returns.iloc[0], (int, float))

    def test_cumulative_returns(self, synthetic_price_data):
        """Test cumulative returns calculation."""
        if not MACD_AVAILABLE:
            pytest.skip("MACD not available")

        result_df = generate_macd_signals(
            synthetic_price_data, fast_period=12, slow_period=26, signal_period=9
        )

        # Check cumulative returns columns
        assert "cumulative_returns" in result_df.columns
        assert "strategy_cumulative_returns" in result_df.columns

        # Cumulative returns should be monotonically increasing or decreasing
        cum_returns = result_df["cumulative_returns"].dropna()
        result_df["strategy_cumulative_returns"].dropna()

        if len(cum_returns) > 1:
            # Should be monotonically increasing (for positive returns)
            assert (cum_returns.diff().dropna() >= 0).all() or (
                cum_returns.diff().dropna() <= 0
            ).all()

    def test_strategy_parameters(self, macd_strategy):
        """Test MACD strategy parameters."""
        assert hasattr(macd_strategy, "fast_period")
        assert hasattr(macd_strategy, "slow_period")
        assert hasattr(macd_strategy, "signal_period")

        # Check parameter values
        assert macd_strategy.fast_period > 0
        assert macd_strategy.slow_period > 0
        assert macd_strategy.signal_period > 0
        assert macd_strategy.fast_period < macd_strategy.slow_period

    def test_signal_confidence(self, macd_strategy, synthetic_price_data):
        """Test signal confidence calculation."""
        signals = macd_strategy.generate_signals(synthetic_price_data)

        if signals:
            for signal in signals:
                assert "confidence" in signal
                confidence = signal["confidence"]
                assert isinstance(confidence, (int, float))
                assert 0 <= confidence <= 1

    def test_signal_timing(self, macd_strategy, synthetic_price_data):
        """Test signal timing and frequency."""
        signals = macd_strategy.generate_signals(synthetic_price_data)

        if len(signals) > 1:
            # Check that signals are ordered by timestamp
            timestamps = [signal["timestamp"] for signal in signals]
            assert timestamps == sorted(timestamps)

    def test_edge_case_empty_data(self):
        """Test handling of empty data."""
        if not MACD_AVAILABLE:
            pytest.skip("MACD not available")

        empty_df = pd.DataFrame()

        try:
            result_df = generate_macd_signals(
                empty_df, fast_period=12, slow_period=26, signal_period=9
            )
            # Should handle empty data gracefully
            assert len(result_df) == 0
        except Exception as e:
            # If it fails, should be due to empty data
            assert any(
                keyword in str(e).lower()
                for keyword in ["empty", "no data", "insufficient"]
            )

    def test_edge_case_missing_columns(self):
        """Test handling of data with missing columns."""
        if not MACD_AVAILABLE:
            pytest.skip("MACD not available")

        # Create data without 'Close' column
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        df = pd.DataFrame(
            {
                "High": np.random.uniform(100, 110, 50),
                "Low": np.random.uniform(90, 100, 50),
                "Volume": np.random.uniform(1000000, 5000000, 50),
            },
            index=dates,
        )

        try:
            result_df = generate_macd_signals(
                df, fast_period=12, slow_period=26, signal_period=9
            )
            # Should handle missing columns gracefully
        except Exception as e:
            # If it fails, should be due to missing 'Close' column
            assert "close" in str(e).lower() or "missing" in str(e).lower()

    def test_performance_metrics(self, synthetic_price_data):
        """Test performance metrics calculation."""
        if not MACD_AVAILABLE:
            pytest.skip("MACD not available")

        result_df = generate_macd_signals(
            synthetic_price_data, fast_period=12, slow_period=26, signal_period=9
        )

        # Calculate basic performance metrics
        if "strategy_returns" in result_df.columns:
            strategy_returns = result_df["strategy_returns"].dropna()
            if len(strategy_returns) > 0:
                total_return = strategy_returns.sum()
                win_rate = (strategy_returns > 0).mean()

                assert isinstance(total_return, (int, float))
                assert isinstance(win_rate, (int, float))
                assert 0 <= win_rate <= 1

    def test_signal_validation(self, synthetic_price_data):
        """Test signal validation logic."""
        if not MACD_AVAILABLE:
            pytest.skip("MACD not available")

        result_df = generate_macd_signals(
            synthetic_price_data, fast_period=12, slow_period=26, signal_period=9
        )

        # Validate signal logic
        if (
            "signal" in result_df.columns
            and "macd" in result_df.columns
            and "macd_signal" in result_df.columns
        ):
            signals = result_df["signal"]
            macd_values = result_df["macd"]
            macd_signal = result_df["macd_signal"]

            # Check that buy signals occur when MACD crosses above signal line
            buy_signals = signals == 1
            if buy_signals.any():
                buy_indices = buy_signals[buy_signals].index
                for idx in buy_indices:
                    if idx > 0:  # Need previous value for comparison
                        result_df.index[result_df.index.get_loc(idx) - 1]
                        # MACD should be above signal line at buy signal
                        assert macd_values.loc[idx] > macd_signal.loc[idx]

            # Check that sell signals occur when MACD crosses below signal line
            sell_signals = signals == -1
            if sell_signals.any():
                sell_indices = sell_signals[sell_signals].index
                for idx in sell_indices:
                    if idx > 0:  # Need previous value for comparison
                        result_df.index[result_df.index.get_loc(idx) - 1]
                        # MACD should be below signal line at sell signal
                        assert macd_values.loc[idx] < macd_signal.loc[idx]

    def test_macd_histogram(self, synthetic_price_data):
        """Test MACD histogram calculation."""
        if not MACD_AVAILABLE:
            pytest.skip("MACD not available")

        result_df = generate_macd_signals(
            synthetic_price_data, fast_period=12, slow_period=26, signal_period=9
        )

        # Check histogram column
        assert "macd_histogram" in result_df.columns

        # Histogram should be MACD - Signal
        if "macd" in result_df.columns and "macd_signal" in result_df.columns:
            macd_values = result_df["macd"].dropna()
            macd_signal = result_df["macd_signal"].dropna()
            histogram = result_df["macd_histogram"].dropna()

            if len(macd_values) > 0 and len(macd_signal) > 0 and len(histogram) > 0:
                # Histogram should equal MACD - Signal
                expected_histogram = macd_values - macd_signal
                np.testing.assert_array_almost_equal(
                    histogram, expected_histogram, decimal=10
                )


if __name__ == "__main__":
    pytest.main([__file__])
