"""Tests for the MACD strategy using pandas-ta."""

import numpy as np
import pandas as pd
import pytest

from trading.strategies.macd_strategy import MACDStrategy


class TestMACDStrategy:
    """Test MACD strategy functionality."""

    @pytest.fixture
    def strategy(self):
        """Create a MACD strategy instance for testing."""
        return MACDStrategy(fast_period=12, slow_period=26, signal_period=9)

    @pytest.fixture
    def sample_data(self, sample_price_data):
        """Create sample price data for testing."""
        return sample_price_data

    def test_macd_calculation_with_pandas_ta(self, strategy, sample_data):
        """Test that MACD is calculated correctly using pandas-ta."""
        # Calculate MACD
        macd_result = strategy.calculate_macd(sample_data["Close"])

        # Check that MACD components exist
        assert "MACD_12_26_9" in macd_result.columns
        assert "MACDh_12_26_9" in macd_result.columns
        assert "MACDs_12_26_9" in macd_result.columns

        # Check data types and lengths
        assert isinstance(macd_result, pd.DataFrame)
        assert len(macd_result) == len(sample_data)

        # MACD line should not be all NaN
        macd_line = macd_result["MACD_12_26_9"].dropna()
        assert len(macd_line) > 0

    def test_signal_generation(self, strategy, sample_data):
        """Test that trading signals are generated correctly."""
        signals = strategy.generate_signals(sample_data)

        # Check signal properties
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data)

        # Signals should be 1 (buy), -1 (sell), or 0 (hold)
        valid_signals = signals.dropna()
        assert valid_signals.isin([1, 0, -1]).all()

    def test_bullish_crossover(self, strategy):
        """Test that bullish crossover triggers buy signal."""
        # Create data that will generate bullish crossover
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        # Create upward trend
        prices = np.linspace(100, 120, 50)
        data = pd.DataFrame({"Close": prices}, index=dates)

        signals = strategy.generate_signals(data)

        # Should have at least one buy signal
        assert (signals == 1).any(), "No buy signals generated for bullish crossover"

    def test_bearish_crossover(self, strategy):
        """Test that bearish crossover triggers sell signal."""
        # Create data that will generate bearish crossover
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        # Create downward trend
        prices = np.linspace(120, 100, 50)
        data = pd.DataFrame({"Close": prices}, index=dates)

        signals = strategy.generate_signals(data)

        # Should have at least one sell signal
        assert (signals == -1).any(), "No sell signals generated for bearish crossover"

    def test_parameter_validation(self):
        """Test that strategy parameters are validated."""
        # Test invalid fast period
        with pytest.raises(ValueError):
            MACDStrategy(fast_period=0, slow_period=26, signal_period=9)

        # Test invalid slow period
        with pytest.raises(ValueError):
            MACDStrategy(fast_period=12, slow_period=0, signal_period=9)

        # Test invalid signal period
        with pytest.raises(ValueError):
            MACDStrategy(fast_period=12, slow_period=26, signal_period=0)

        # Test fast period >= slow period
        with pytest.raises(ValueError):
            MACDStrategy(fast_period=30, slow_period=20, signal_period=9)

    def test_empty_data_handling(self, strategy):
        """Test that strategy handles empty data correctly."""
        empty_data = pd.DataFrame(columns=["Close"])

        with pytest.raises(ValueError):
            strategy.generate_signals(empty_data)

    def test_missing_data_handling(self, strategy):
        """Test that strategy handles missing data correctly."""
        data = pd.DataFrame({"Close": [100, np.nan, 101, 102]})

        with pytest.raises(ValueError):
            strategy.generate_signals(data)

    def test_signal_consistency(self, strategy, sample_data):
        """Test that signals are consistent with MACD values."""
        macd_result = strategy.calculate_macd(sample_data["Close"])
        signals = strategy.generate_signals(sample_data)

        # Get non-NaN values
        mask = (
            macd_result["MACD_12_26_9"].notna()
            & macd_result["MACDs_12_26_9"].notna()
            & signals.notna()
        )
        macd_line = macd_result.loc[mask, "MACD_12_26_9"]
        signal_line = macd_result.loc[mask, "MACDs_12_26_9"]
        signal_values = signals[mask]

        # Check bullish crossover (MACD line crosses above signal line)
        bullish_mask = (macd_line > signal_line) & (
            macd_line.shift(1) <= signal_line.shift(1)
        )
        if bullish_mask.any():
            assert (
                signal_values[bullish_mask] == 1
            ).all(), "Bullish crossover should generate buy signals"

        # Check bearish crossover (MACD line crosses below signal line)
        bearish_mask = (macd_line < signal_line) & (
            macd_line.shift(1) >= signal_line.shift(1)
        )
        if bearish_mask.any():
            assert (
                signal_values[bearish_mask] == -1
            ).all(), "Bearish crossover should generate sell signals"

    def test_strategy_configuration(self, strategy):
        """Test that strategy configuration is correct."""
        assert strategy.fast_period == 12
        assert strategy.slow_period == 26
        assert strategy.signal_period == 9
        assert strategy.name == "MACD"

    def test_different_parameters(self, sample_data):
        """Test MACD with different parameter combinations."""
        # Test with different fast/slow periods
        strategy1 = MACDStrategy(fast_period=8, slow_period=21, signal_period=5)
        strategy2 = MACDStrategy(fast_period=12, slow_period=26, signal_period=9)

        signals1 = strategy1.generate_signals(sample_data)
        signals2 = strategy2.generate_signals(sample_data)

        # Should have different signal patterns
        assert not signals1.equals(signals2)

    def test_macd_components(self, strategy, sample_data):
        """Test that all MACD components are calculated correctly."""
        macd_result = strategy.calculate_macd(sample_data["Close"])

        # Check all required columns exist
        required_columns = ["MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"]
        for col in required_columns:
            assert col in macd_result.columns

        # Check that MACD histogram is calculated correctly
        macd_line = macd_result["MACD_12_26_9"]
        signal_line = macd_result["MACDs_12_26_9"]
        histogram = macd_result["MACDh_12_26_9"]

        # Histogram should be MACD line - signal line
        expected_histogram = macd_line - signal_line
        pd.testing.assert_series_equal(histogram, expected_histogram, check_names=False)

    def test_performance_evaluation(self, strategy, sample_data):
        """Test that the strategy can be evaluated for performance."""
        signals = strategy.generate_signals(sample_data)
        performance = strategy.evaluate_performance(sample_data, signals)

        # Check performance metrics
        assert isinstance(performance, dict)
        required_metrics = [
            "returns",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "total_trades",
        ]

        for metric in required_metrics:
            assert metric in performance
            assert isinstance(performance[metric], (int, float))

    def test_signal_timing(self, strategy, sample_data):
        """Test that signals are generated at the correct times."""
        signals = strategy.generate_signals(sample_data)

        # Signals should be generated after enough data is available
        # MACD requires slow_period + signal_period data points
        min_required = strategy.slow_period + strategy.signal_period

        # First signals should be NaN due to insufficient data
        assert signals.iloc[:min_required].isna().all()

        # Should have valid signals after minimum required data
        assert not signals.iloc[min_required:].isna().all()

    def test_edge_cases(self, strategy):
        """Test edge cases for MACD calculation."""
        # Test with very short data
        short_data = pd.DataFrame({"Close": [100, 101, 102, 103, 104]})

        with pytest.raises(ValueError):
            strategy.generate_signals(short_data)

        # Test with constant price
        constant_data = pd.DataFrame({"Close": [100] * 50})

        signals = strategy.generate_signals(constant_data)
        assert isinstance(signals, pd.Series)

    def test_macd_convergence(self, strategy, sample_data):
        """Test that MACD values converge properly."""
        macd_result = strategy.calculate_macd(sample_data["Close"])

        # MACD line and signal line should have similar trends
        macd_line = macd_result["MACD_12_26_9"].dropna()
        signal_line = macd_result["MACDs_12_26_9"].dropna()

        # Both should have the same length
        assert len(macd_line) == len(signal_line)

        # Both should have finite values
        assert np.isfinite(macd_line).all()
        assert np.isfinite(signal_line).all()

    def test_signal_distribution(self, strategy, sample_data):
        """Test that signals have reasonable distribution."""
        signals = strategy.generate_signals(sample_data)
        valid_signals = signals.dropna()

        if len(valid_signals) > 0:
            # Should have a mix of buy, sell, and hold signals
            signal_counts = valid_signals.value_counts()

            # Should have at least some hold signals (0)
            assert 0 in signal_counts.index

            # Total signals should equal valid signal count
            assert signal_counts.sum() == len(valid_signals)
