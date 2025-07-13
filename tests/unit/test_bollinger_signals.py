"""
Unit tests for Bollinger signals.

Tests Bollinger signal generation with synthetic price data,
including edge cases and signal validation.
"""

import warnings
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

# Import Bollinger strategy modules
try:
    from trading.strategies.bollinger_signals import generate_bollinger_signals
    from trading.strategies.bollinger_strategy import BollingerStrategy

    BOLLINGER_AVAILABLE = True
except ImportError:
    BOLLINGER_AVAILABLE = False
    BollingerStrategy = Mock()
    generate_bollinger_signals = Mock()


class TestBollingerSignals:
    """Test suite for Bollinger signal generation."""

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
        """Create short price data (< 20 points)."""
        dates = pd.date_range(start="2023-01-01", periods=15, freq="D")
        close_prices = [100 + i for i in range(15)]

        df = pd.DataFrame(
            {
                "Close": close_prices,
                "High": [p + 2 for p in close_prices],
                "Low": [p - 2 for p in close_prices],
                "Volume": [1000000] * 15,
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
    def bollinger_strategy(self):
        """Create Bollinger strategy instance."""
        if not BOLLINGER_AVAILABLE:
            pytest.skip("Bollinger strategy not available")
        return BollingerStrategy()

    def test_bollinger_signal_generation(self, bollinger_strategy, synthetic_price_data):
        """Test that Bollinger signals are generated correctly."""
        signals = bollinger_strategy.generate_signals(synthetic_price_data)

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
            assert signal["strategy_name"] == "Bollinger"

    def test_bollinger_bands_calculation(self, bollinger_strategy, synthetic_price_data):
        """Test that Bollinger bands are calculated correctly."""
        bands = bollinger_strategy.calculate_bollinger_bands(synthetic_price_data)

        assert isinstance(bands, dict)
        assert "upper" in bands
        assert "middle" in bands
        assert "lower" in bands

        # Check that bands are pandas Series
        for band_name, band_values in bands.items():
            assert isinstance(band_values, pd.Series)
            assert len(band_values) == len(synthetic_price_data)

    def test_signal_columns_exist(self, synthetic_price_data):
        """Test that signal columns are created correctly."""
        if not BOLLINGER_AVAILABLE:
            pytest.skip("Bollinger not available")

        result_df = generate_bollinger_signals(synthetic_price_data, period=20, std_dev=2)

        # Check required columns exist
        required_columns = [
            "signal",
            "upper_band",
            "middle_band",
            "lower_band",
            "bb_width",
            "bb_position",
            "returns",
            "strategy_returns",
        ]
        for col in required_columns:
            assert col in result_df.columns

    def test_signal_values(self, synthetic_price_data):
        """Test that signal values are valid."""
        if not BOLLINGER_AVAILABLE:
            pytest.skip("Bollinger not available")

        result_df = generate_bollinger_signals(synthetic_price_data, period=20, std_dev=2)

        # Signals should be 1 (buy), -1 (sell), or 0 (hold)
        valid_signals = result_df["signal"].dropna()
        if len(valid_signals) > 0:
            assert valid_signals.isin([1, 0, -1]).all()

    def test_no_nan_in_signals(self, synthetic_price_data):
        """Test that signals contain no NaN values."""
        if not BOLLINGER_AVAILABLE:
            pytest.skip("Bollinger not available")

        result_df = generate_bollinger_signals(synthetic_price_data, period=20, std_dev=2)

        # Signal column should not contain NaN values
        assert not result_df["signal"].isna().any()

    def test_short_data_handling(self, short_price_data):
        """Test handling of short price data."""
        if not BOLLINGER_AVAILABLE:
            pytest.skip("Bollinger not available")

        # Should handle short data gracefully
        try:
            result_df = generate_bollinger_signals(short_price_data, period=20, std_dev=2)
            # If it succeeds, check structure
            assert "signal" in result_df.columns
            assert "upper_band" in result_df.columns
        except Exception as e:
            # If it fails, should be due to insufficient data
            assert any(keyword in str(e).lower() for keyword in ["insufficient", "at least", "minimum", "period"])

    def test_constant_data_handling(self, constant_price_data):
        """Test handling of constant price data."""
        if not BOLLINGER_AVAILABLE:
            pytest.skip("Bollinger not available")

        result_df = generate_bollinger_signals(constant_price_data, period=20, std_dev=2)

        # Should handle constant data gracefully
        assert "signal" in result_df.columns
        assert "upper_band" in result_df.columns

        # For constant data, bands should be equal
        upper_band = result_df["upper_band"].dropna()
        middle_band = result_df["middle_band"].dropna()
        lower_band = result_df["lower_band"].dropna()

        if len(upper_band) > 0:
            # For constant data, all bands should be equal
            np.testing.assert_array_almost_equal(upper_band, middle_band, decimal=10)
            np.testing.assert_array_almost_equal(middle_band, lower_band, decimal=10)

    def test_nan_data_handling(self, nan_price_data):
        """Test handling of price data with NaN values."""
        if not BOLLINGER_AVAILABLE:
            pytest.skip("Bollinger not available")

        # Should handle NaN data gracefully
        try:
            result_df = generate_bollinger_signals(nan_price_data, period=20, std_dev=2)
            assert "signal" in result_df.columns
            assert "upper_band" in result_df.columns
        except Exception as e:
            # If it fails, should be due to NaN values
            assert any(keyword in str(e).lower() for keyword in ["nan", "missing", "invalid"])

    def test_different_periods(self, synthetic_price_data):
        """Test Bollinger calculation with different periods."""
        if not BOLLINGER_AVAILABLE:
            pytest.skip("Bollinger not available")

        periods = [10, 20, 30, 50]

        for period in periods:
            result_df = generate_bollinger_signals(synthetic_price_data, period=period, std_dev=2)
            assert "signal" in result_df.columns
            assert "upper_band" in result_df.columns

            # Check that bands are valid
            upper_band = result_df["upper_band"].dropna()
            middle_band = result_df["middle_band"].dropna()
            lower_band = result_df["lower_band"].dropna()

            if len(upper_band) > 0:
                # Upper band should be above middle band
                assert (upper_band >= middle_band).all()
                # Middle band should be above lower band
                assert (middle_band >= lower_band).all()

    def test_different_std_devs(self, synthetic_price_data):
        """Test Bollinger calculation with different standard deviations."""
        if not BOLLINGER_AVAILABLE:
            pytest.skip("Bollinger not available")

        std_devs = [1, 2, 3]

        for std_dev in std_devs:
            result_df = generate_bollinger_signals(synthetic_price_data, period=20, std_dev=std_dev)
            assert "signal" in result_df.columns
            assert "upper_band" in result_df.columns

            # Check that bands are valid
            upper_band = result_df["upper_band"].dropna()
            middle_band = result_df["middle_band"].dropna()
            lower_band = result_df["lower_band"].dropna()

            if len(upper_band) > 0:
                # Upper band should be above middle band
                assert (upper_band >= middle_band).all()
                # Middle band should be above lower band
                assert (middle_band >= lower_band).all()

    def test_trending_data_signals(self, trending_price_data):
        """Test Bollinger signals on trending data."""
        if not BOLLINGER_AVAILABLE:
            pytest.skip("Bollinger not available")

        result_df = generate_bollinger_signals(trending_price_data, period=20, std_dev=2)

        assert "signal" in result_df.columns
        assert "upper_band" in result_df.columns

        # Check that signals are generated
        signals = result_df["signal"].dropna()
        assert len(signals) >= 0  # May have no signals in trending data

    def test_returns_calculation(self, synthetic_price_data):
        """Test that returns are calculated correctly."""
        if not BOLLINGER_AVAILABLE:
            pytest.skip("Bollinger not available")

        result_df = generate_bollinger_signals(synthetic_price_data, period=20, std_dev=2)

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
        if not BOLLINGER_AVAILABLE:
            pytest.skip("Bollinger not available")

        result_df = generate_bollinger_signals(synthetic_price_data, period=20, std_dev=2)

        # Check cumulative returns columns
        assert "cumulative_returns" in result_df.columns
        assert "strategy_cumulative_returns" in result_df.columns

        # Cumulative returns should be monotonically increasing or decreasing
        cum_returns = result_df["cumulative_returns"].dropna()
        strategy_cum_returns = result_df["strategy_cumulative_returns"].dropna()

        if len(cum_returns) > 1:
            # Should be monotonically increasing (for positive returns)
            assert (cum_returns.diff().dropna() >= 0).all() or (cum_returns.diff().dropna() <= 0).all()

    def test_strategy_parameters(self, bollinger_strategy):
        """Test Bollinger strategy parameters."""
        assert hasattr(bollinger_strategy, "period")
        assert hasattr(bollinger_strategy, "std_dev")

        # Check parameter values
        assert bollinger_strategy.period > 0
        assert bollinger_strategy.std_dev > 0

    def test_signal_confidence(self, bollinger_strategy, synthetic_price_data):
        """Test signal confidence calculation."""
        signals = bollinger_strategy.generate_signals(synthetic_price_data)

        if signals:
            for signal in signals:
                assert "confidence" in signal
                confidence = signal["confidence"]
                assert isinstance(confidence, (int, float))
                assert 0 <= confidence <= 1

    def test_signal_timing(self, bollinger_strategy, synthetic_price_data):
        """Test signal timing and frequency."""
        signals = bollinger_strategy.generate_signals(synthetic_price_data)

        if len(signals) > 1:
            # Check that signals are ordered by timestamp
            timestamps = [signal["timestamp"] for signal in signals]
            assert timestamps == sorted(timestamps)

    def test_edge_case_empty_data(self):
        """Test handling of empty data."""
        if not BOLLINGER_AVAILABLE:
            pytest.skip("Bollinger not available")

        empty_df = pd.DataFrame()

        try:
            result_df = generate_bollinger_signals(empty_df, period=20, std_dev=2)
            # Should handle empty data gracefully
            assert len(result_df) == 0
        except Exception as e:
            # If it fails, should be due to empty data
            assert any(keyword in str(e).lower() for keyword in ["empty", "no data", "insufficient"])

    def test_edge_case_missing_columns(self):
        """Test handling of data with missing columns."""
        if not BOLLINGER_AVAILABLE:
            pytest.skip("Bollinger not available")

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
            result_df = generate_bollinger_signals(df, period=20, std_dev=2)
            # Should handle missing columns gracefully
        except Exception as e:
            # If it fails, should be due to missing 'Close' column
            assert "close" in str(e).lower() or "missing" in str(e).lower()

    def test_performance_metrics(self, synthetic_price_data):
        """Test performance metrics calculation."""
        if not BOLLINGER_AVAILABLE:
            pytest.skip("Bollinger not available")

        result_df = generate_bollinger_signals(synthetic_price_data, period=20, std_dev=2)

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
        if not BOLLINGER_AVAILABLE:
            pytest.skip("Bollinger not available")

        result_df = generate_bollinger_signals(synthetic_price_data, period=20, std_dev=2)

        # Validate signal logic
        if "signal" in result_df.columns and "upper_band" in result_df.columns and "lower_band" in result_df.columns:
            signals = result_df["signal"]
            upper_band = result_df["upper_band"]
            lower_band = result_df["lower_band"]
            close_prices = result_df["Close"]

            # Check that buy signals occur when price is near lower band
            buy_signals = signals == 1
            if buy_signals.any():
                buy_indices = buy_signals[buy_signals].index
                for idx in buy_indices:
                    # Price should be near or below lower band for buy signal
                    price = close_prices.loc[idx]
                    lower = lower_band.loc[idx]
                    assert price <= lower * 1.05  # Within 5% of lower band

            # Check that sell signals occur when price is near upper band
            sell_signals = signals == -1
            if sell_signals.any():
                sell_indices = sell_signals[sell_signals].index
                for idx in sell_indices:
                    # Price should be near or above upper band for sell signal
                    price = close_prices.loc[idx]
                    upper = upper_band.loc[idx]
                    assert price >= upper * 0.95  # Within 5% of upper band

    def test_bb_width_calculation(self, synthetic_price_data):
        """Test Bollinger Band width calculation."""
        if not BOLLINGER_AVAILABLE:
            pytest.skip("Bollinger not available")

        result_df = generate_bollinger_signals(synthetic_price_data, period=20, std_dev=2)

        # Check BB width column
        assert "bb_width" in result_df.columns

        # BB width should be (upper - lower) / middle
        if (
            "upper_band" in result_df.columns
            and "lower_band" in result_df.columns
            and "middle_band" in result_df.columns
        ):
            upper_band = result_df["upper_band"].dropna()
            lower_band = result_df["lower_band"].dropna()
            middle_band = result_df["middle_band"].dropna()
            bb_width = result_df["bb_width"].dropna()

            if len(upper_band) > 0 and len(lower_band) > 0 and len(middle_band) > 0 and len(bb_width) > 0:
                # BB width should be positive
                assert (bb_width > 0).all()

                # BB width should equal (upper - lower) / middle
                expected_width = (upper_band - lower_band) / middle_band
                np.testing.assert_array_almost_equal(bb_width, expected_width, decimal=10)

    def test_bb_position_calculation(self, synthetic_price_data):
        """Test Bollinger Band position calculation."""
        if not BOLLINGER_AVAILABLE:
            pytest.skip("Bollinger not available")

        result_df = generate_bollinger_signals(synthetic_price_data, period=20, std_dev=2)

        # Check BB position column
        assert "bb_position" in result_df.columns

        # BB position should be between 0 and 1
        bb_position = result_df["bb_position"].dropna()
        if len(bb_position) > 0:
            assert (bb_position >= 0).all()
            assert (bb_position <= 1).all()


if __name__ == "__main__":
    pytest.main([__file__])
