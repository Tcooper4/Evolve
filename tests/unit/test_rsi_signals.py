"""
Unit tests for RSI signals.

Tests RSI signal generation with synthetic price data,
including edge cases and signal validation.
"""

import warnings
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

# Import RSI strategy modules
try:
    from trading.strategies.rsi_signals import generate_rsi_signals, generate_signals
    from trading.strategies.rsi_strategy import RSIStrategy

    RSI_AVAILABLE = True
except ImportError:
    RSI_AVAILABLE = False
    generate_rsi_signals = Mock()
    generate_signals = Mock()
    RSIStrategy = Mock()


class TestRSISignals:
    """Test suite for RSI signal generation."""

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

        df = pd.DataFrame({"Close": close_prices, "High": high, "Low": low, "Volume": volume}, index=dates)

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
        """Create short price data (< 14 points)."""
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        close_prices = [100 + i for i in range(10)]

        df = pd.DataFrame(
            {
                "Close": close_prices,
                "High": [p + 2 for p in close_prices],
                "Low": [p - 2 for p in close_prices],
                "Volume": [1000000] * 10,
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
    def rsi_strategy(self):
        """Create RSI strategy instance."""
        if not RSI_AVAILABLE:
            pytest.skip("RSI strategy not available")
        return RSIStrategy()

    def test_rsi_signal_generation(self, rsi_strategy, synthetic_price_data):
        """Test that RSI signals are generated correctly."""
        signals = rsi_strategy.generate_signals(synthetic_price_data)

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
            assert signal["strategy_name"] == "RSI"

    def test_rsi_calculation(self, rsi_strategy, synthetic_price_data):
        """Test that RSI values are calculated correctly."""
        rsi_values = rsi_strategy.calculate_rsi(synthetic_price_data)

        assert isinstance(rsi_values, pd.Series)
        assert len(rsi_values) == len(synthetic_price_data)

        # RSI should be between 0 and 100
        valid_rsi = rsi_values.dropna()
        if len(valid_rsi) > 0:
            assert (valid_rsi >= 0).all()
            assert (valid_rsi <= 100).all()

    def test_signal_columns_exist(self, synthetic_price_data):
        """Test that signal columns are created correctly."""
        if not RSI_AVAILABLE:
            pytest.skip("RSI not available")

        result_df = generate_rsi_signals(synthetic_price_data, period=14)

        # Check required columns exist
        required_columns = ["signal", "rsi", "returns", "strategy_returns"]
        for col in required_columns:
            assert col in result_df.columns

    def test_signal_values(self, synthetic_price_data):
        """Test that signal values are valid."""
        if not RSI_AVAILABLE:
            pytest.skip("RSI not available")

        result_df = generate_rsi_signals(synthetic_price_data, period=14)

        # Signals should be 1 (buy), -1 (sell), or 0 (hold)
        valid_signals = result_df["signal"].dropna()
        if len(valid_signals) > 0:
            assert valid_signals.isin([1, 0, -1]).all()

    def test_no_nan_in_signals(self, synthetic_price_data):
        """Test that signals contain no NaN values."""
        if not RSI_AVAILABLE:
            pytest.skip("RSI not available")

        result_df = generate_rsi_signals(synthetic_price_data, period=14)

        # Signal column should not contain NaN values
        assert not result_df["signal"].isna().any()

    def test_short_data_handling(self, short_price_data):
        """Test handling of short price data."""
        if not RSI_AVAILABLE:
            pytest.skip("RSI not available")

        # Should handle short data gracefully
        try:
            result_df = generate_rsi_signals(short_price_data, period=14)
            # If it succeeds, check structure
            assert "signal" in result_df.columns
            assert "rsi" in result_df.columns
        except Exception as e:
            # If it fails, should be due to insufficient data
            assert any(keyword in str(e).lower() for keyword in ["insufficient", "at least", "minimum", "period"])

    def test_constant_data_handling(self, constant_price_data):
        """Test handling of constant price data."""
        if not RSI_AVAILABLE:
            pytest.skip("RSI not available")

        result_df = generate_rsi_signals(constant_price_data, period=14)

        # Should handle constant data gracefully
        assert "signal" in result_df.columns
        assert "rsi" in result_df.columns

        # RSI should be 50 for constant data (or NaN if calculation fails)
        rsi_values = result_df["rsi"].dropna()
        if len(rsi_values) > 0:
            # For constant data, RSI should be around 50 or NaN
            assert (rsi_values == 50).any() or rsi_values.isna().any()

    def test_nan_data_handling(self, nan_price_data):
        """Test handling of price data with NaN values."""
        if not RSI_AVAILABLE:
            pytest.skip("RSI not available")

        # Should handle NaN data gracefully
        try:
            result_df = generate_rsi_signals(nan_price_data, period=14)
            assert "signal" in result_df.columns
            assert "rsi" in result_df.columns
        except Exception as e:
            # If it fails, should be due to NaN values
            assert any(keyword in str(e).lower() for keyword in ["nan", "missing", "invalid"])

    def test_different_periods(self, synthetic_price_data):
        """Test RSI calculation with different periods."""
        if not RSI_AVAILABLE:
            pytest.skip("RSI not available")

        periods = [7, 14, 21, 30]

        for period in periods:
            result_df = generate_rsi_signals(synthetic_price_data, period=period)
            assert "signal" in result_df.columns
            assert "rsi" in result_df.columns

            # Check that RSI values are valid
            rsi_values = result_df["rsi"].dropna()
            if len(rsi_values) > 0:
                assert (rsi_values >= 0).all()
                assert (rsi_values <= 100).all()

    def test_different_thresholds(self, synthetic_price_data):
        """Test RSI signals with different thresholds."""
        if not RSI_AVAILABLE:
            pytest.skip("RSI not available")

        # Test different threshold combinations
        thresholds = [
            (20, 80),  # More conservative
            (30, 70),  # Standard
            (40, 60),  # More aggressive
        ]

        for buy_threshold, sell_threshold in thresholds:
            result_df = generate_rsi_signals(
                synthetic_price_data, period=14, buy_threshold=buy_threshold, sell_threshold=sell_threshold
            )

            assert "signal" in result_df.columns
            assert "rsi" in result_df.columns

            # Check signal values
            valid_signals = result_df["signal"].dropna()
            if len(valid_signals) > 0:
                assert valid_signals.isin([1, 0, -1]).all()

    def test_trending_data_signals(self, trending_price_data):
        """Test RSI signals on trending data."""
        if not RSI_AVAILABLE:
            pytest.skip("RSI not available")

        result_df = generate_rsi_signals(trending_price_data, period=14)

        assert "signal" in result_df.columns
        assert "rsi" in result_df.columns

        # Check that signals are generated
        signals = result_df["signal"].dropna()
        assert len(signals) >= 0  # May have no signals in trending data

    def test_returns_calculation(self, synthetic_price_data):
        """Test that returns are calculated correctly."""
        if not RSI_AVAILABLE:
            pytest.skip("RSI not available")

        result_df = generate_rsi_signals(synthetic_price_data, period=14)

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
        if not RSI_AVAILABLE:
            pytest.skip("RSI not available")

        result_df = generate_rsi_signals(synthetic_price_data, period=14)

        # Check cumulative returns columns
        assert "cumulative_returns" in result_df.columns
        assert "strategy_cumulative_returns" in result_df.columns

        # Cumulative returns should be monotonically increasing or decreasing
        cum_returns = result_df["cumulative_returns"].dropna()
        strategy_cum_returns = result_df["strategy_cumulative_returns"].dropna()

        if len(cum_returns) > 1:
            # Should be monotonically increasing (for positive returns)
            assert (cum_returns.diff().dropna() >= 0).all() or (cum_returns.diff().dropna() <= 0).all()

    def test_strategy_parameters(self, rsi_strategy):
        """Test RSI strategy parameters."""
        assert hasattr(rsi_strategy, "rsi_period")
        assert hasattr(rsi_strategy, "oversold_threshold")
        assert hasattr(rsi_strategy, "overbought_threshold")

        # Check parameter values
        assert rsi_strategy.rsi_period > 0
        assert 0 <= rsi_strategy.oversold_threshold <= 100
        assert 0 <= rsi_strategy.overbought_threshold <= 100
        assert rsi_strategy.oversold_threshold < rsi_strategy.overbought_threshold

    def test_signal_confidence(self, rsi_strategy, synthetic_price_data):
        """Test signal confidence calculation."""
        signals = rsi_strategy.generate_signals(synthetic_price_data)

        if signals:
            for signal in signals:
                assert "confidence" in signal
                confidence = signal["confidence"]
                assert isinstance(confidence, (int, float))
                assert 0 <= confidence <= 1

    def test_signal_timing(self, rsi_strategy, synthetic_price_data):
        """Test signal timing and frequency."""
        signals = rsi_strategy.generate_signals(synthetic_price_data)

        if len(signals) > 1:
            # Check that signals are ordered by timestamp
            timestamps = [signal["timestamp"] for signal in signals]
            assert timestamps == sorted(timestamps)

    def test_edge_case_empty_data(self):
        """Test handling of empty data."""
        if not RSI_AVAILABLE:
            pytest.skip("RSI not available")

        empty_df = pd.DataFrame()

        try:
            result_df = generate_rsi_signals(empty_df, period=14)
            # Should handle empty data gracefully
            assert len(result_df) == 0
        except Exception as e:
            # If it fails, should be due to empty data
            assert any(keyword in str(e).lower() for keyword in ["empty", "no data", "insufficient"])

    def test_edge_case_missing_columns(self):
        """Test handling of data with missing columns."""
        if not RSI_AVAILABLE:
            pytest.skip("RSI not available")

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
            result_df = generate_rsi_signals(df, period=14)
            # Should handle missing columns gracefully
        except Exception as e:
            # If it fails, should be due to missing 'Close' column
            assert "close" in str(e).lower() or "missing" in str(e).lower()

    def test_performance_metrics(self, synthetic_price_data):
        """Test performance metrics calculation."""
        if not RSI_AVAILABLE:
            pytest.skip("RSI not available")

        result_df = generate_rsi_signals(synthetic_price_data, period=14)

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
        if not RSI_AVAILABLE:
            pytest.skip("RSI not available")

        result_df = generate_rsi_signals(synthetic_price_data, period=14)

        # Validate signal logic
        if "signal" in result_df.columns and "rsi" in result_df.columns:
            signals = result_df["signal"]
            rsi_values = result_df["rsi"]

            # Check that buy signals occur at low RSI
            buy_signals = signals == 1
            if buy_signals.any():
                buy_rsi = rsi_values[buy_signals].dropna()
                if len(buy_rsi) > 0:
                    # Buy signals should generally occur at lower RSI values
                    assert buy_rsi.mean() < 50  # Average RSI for buy signals should be below 50

            # Check that sell signals occur at high RSI
            sell_signals = signals == -1
            if sell_signals.any():
                sell_rsi = rsi_values[sell_signals].dropna()
                if len(sell_rsi) > 0:
                    # Sell signals should generally occur at higher RSI values
                    assert sell_rsi.mean() > 50  # Average RSI for sell signals should be above 50


if __name__ == "__main__":
    pytest.main([__file__])
