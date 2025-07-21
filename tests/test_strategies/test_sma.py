"""Tests for the SMA strategy using pandas-ta."""

import numpy as np
import pandas as pd
import pytest

from trading.strategies.sma_strategy import SMAStrategy


class TestSMAStrategy:
    """Test SMA strategy functionality."""

    @pytest.fixture
    def strategy(self):
        """Create an SMA strategy instance for testing."""
        return SMAStrategy(short_window=20, long_window=50)

    @pytest.fixture
    def sample_data(self, sample_price_data):
        """Create sample price data for testing."""
        return sample_price_data

    def test_sma_calculation_with_pandas_ta(self, strategy, sample_data):
        """Test that SMA is calculated correctly using pandas-ta."""
        # Calculate SMAs
        sma_result = strategy.calculate_sma(sample_data["Close"])

        # Check that SMA columns exist
        assert f"SMA_{strategy.short_window}" in sma_result.columns
        assert f"SMA_{strategy.long_window}" in sma_result.columns

        # Check data types and lengths
        assert isinstance(sma_result, pd.DataFrame)
        assert len(sma_result) == len(sample_data)

        # SMAs should not be all NaN
        short_sma = sma_result[f"SMA_{strategy.short_window}"].dropna()
        long_sma = sma_result[f"SMA_{strategy.long_window}"].dropna()
        assert len(short_sma) > 0
        assert len(long_sma) > 0

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
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        # Create upward trend
        prices = np.linspace(100, 120, 60)
        data = pd.DataFrame({"Close": prices}, index=dates)

        signals = strategy.generate_signals(data)

        # Should have at least one buy signal
        assert (signals == 1).any(), "No buy signals generated for bullish crossover"

    def test_bearish_crossover(self, strategy):
        """Test that bearish crossover triggers sell signal."""
        # Create data that will generate bearish crossover
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        # Create downward trend
        prices = np.linspace(120, 100, 60)
        data = pd.DataFrame({"Close": prices}, index=dates)

        signals = strategy.generate_signals(data)

        # Should have at least one sell signal
        assert (signals == -1).any(), "No sell signals generated for bearish crossover"

    def test_parameter_validation(self):
        """Test that strategy parameters are validated."""
        # Test invalid short window
        with pytest.raises(ValueError):
            SMAStrategy(short_window=0, long_window=50)

        # Test invalid long window
        with pytest.raises(ValueError):
            SMAStrategy(short_window=20, long_window=0)

        # Test short window >= long window
        with pytest.raises(ValueError):
            SMAStrategy(short_window=50, long_window=20)

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
        """Test that signals are consistent with SMA values."""
        sma_result = strategy.calculate_sma(sample_data["Close"])
        signals = strategy.generate_signals(sample_data)

        # Get non-NaN values
        short_sma_col = f"SMA_{strategy.short_window}"
        long_sma_col = f"SMA_{strategy.long_window}"

        mask = (
            sma_result[short_sma_col].notna()
            & sma_result[long_sma_col].notna()
            & signals.notna()
        )
        short_sma = sma_result.loc[mask, short_sma_col]
        long_sma = sma_result.loc[mask, long_sma_col]
        signal_values = signals[mask]

        # Check bullish crossover (short SMA crosses above long SMA)
        bullish_mask = (short_sma > long_sma) & (
            short_sma.shift(1) <= long_sma.shift(1)
        )
        if bullish_mask.any():
            assert (
                signal_values[bullish_mask] == 1
            ).all(), "Bullish crossover should generate buy signals"

        # Check bearish crossover (short SMA crosses below long SMA)
        bearish_mask = (short_sma < long_sma) & (
            short_sma.shift(1) >= long_sma.shift(1)
        )
        if bearish_mask.any():
            assert (
                signal_values[bearish_mask] == -1
            ).all(), "Bearish crossover should generate sell signals"

    def test_strategy_configuration(self, strategy):
        """Test that strategy configuration is correct."""
        assert strategy.short_window == 20
        assert strategy.long_window == 50
        assert strategy.name == "SMA"

    def test_different_parameters(self, sample_data):
        """Test SMA with different parameter combinations."""
        # Test with different window sizes
        strategy1 = SMAStrategy(short_window=10, long_window=30)
        strategy2 = SMAStrategy(short_window=20, long_window=50)

        signals1 = strategy1.generate_signals(sample_data)
        signals2 = strategy2.generate_signals(sample_data)

        # Should have different signal patterns
        assert not signals1.equals(signals2)

    def test_sma_properties(self, strategy, sample_data):
        """Test that SMAs have correct mathematical properties."""
        sma_result = strategy.calculate_sma(sample_data["Close"])

        short_sma_col = f"SMA_{strategy.short_window}"
        long_sma_col = f"SMA_{strategy.long_window}"

        # Get non-NaN values
        mask = sma_result[short_sma_col].notna() & sma_result[long_sma_col].notna()
        short_sma = sma_result.loc[mask, short_sma_col]
        long_sma = sma_result.loc[mask, long_sma_col]
        sample_data.loc[mask, "Close"]

        # SMAs should be finite
        assert np.isfinite(short_sma).all()
        assert np.isfinite(long_sma).all()

        # SMAs should be positive
        assert (short_sma > 0).all()
        assert (long_sma > 0).all()

    def test_sma_smoothing(self, strategy, sample_data):
        """Test that SMAs provide smoothing of price data."""
        sma_result = strategy.calculate_sma(sample_data["Close"])

        short_sma_col = f"SMA_{strategy.short_window}"
        long_sma_col = f"SMA_{strategy.long_window}"

        # Get non-NaN values
        mask = sma_result[short_sma_col].notna() & sma_result[long_sma_col].notna()
        short_sma = sma_result.loc[mask, short_sma_col]
        long_sma = sma_result.loc[mask, long_sma_col]
        prices = sample_data.loc[mask, "Close"]

        # Calculate standard deviation of prices vs SMAs
        price_std = prices.std()
        short_sma_std = short_sma.std()
        long_sma_std = long_sma.std()

        # SMAs should be less volatile than prices
        assert short_sma_std < price_std
        assert long_sma_std < price_std
        assert long_sma_std < short_sma_std  # Longer SMA should be smoother

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
        # SMA requires long_window data points
        min_required = strategy.long_window

        # First signals should be NaN due to insufficient data
        assert signals.iloc[:min_required].isna().all()

        # Should have valid signals after minimum required data
        assert not signals.iloc[min_required:].isna().all()

    def test_edge_cases(self, strategy):
        """Test edge cases for SMA calculation."""
        # Test with very short data
        short_data = pd.DataFrame({"Close": [100, 101, 102, 103, 104]})

        with pytest.raises(ValueError):
            strategy.generate_signals(short_data)

        # Test with constant price
        constant_data = pd.DataFrame({"Close": [100] * 60})

        signals = strategy.generate_signals(constant_data)
        assert isinstance(signals, pd.Series)

    def test_trend_detection(self, strategy, sample_data):
        """Test that SMA strategy detects trends correctly."""
        sma_result = strategy.calculate_sma(sample_data["Close"])
        signals = strategy.generate_signals(sample_data)

        short_sma_col = f"SMA_{strategy.short_window}"
        long_sma_col = f"SMA_{strategy.long_window}"

        # Get non-NaN values
        mask = (
            sma_result[short_sma_col].notna()
            & sma_result[long_sma_col].notna()
            & signals.notna()
        )
        short_sma = sma_result.loc[mask, short_sma_col]
        long_sma = sma_result.loc[mask, long_sma_col]
        signal_values = signals[mask]

        # In uptrend, short SMA should be above long SMA
        uptrend_mask = short_sma > long_sma
        if uptrend_mask.any():
            # Should have more buy signals than sell signals in uptrend
            buy_signals = (signal_values[uptrend_mask] == 1).sum()
            sell_signals = (signal_values[uptrend_mask] == -1).sum()
            assert buy_signals >= sell_signals

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

    def test_parameter_sensitivity(self, strategy, sample_data):
        """Test how sensitive the strategy is to parameter changes."""
        # Test with very close windows
        close_strategy = SMAStrategy(short_window=20, long_window=25)
        close_signals = close_strategy.generate_signals(sample_data)

        # Test with far apart windows
        far_strategy = SMAStrategy(short_window=10, long_window=50)
        far_signals = far_strategy.generate_signals(sample_data)

        # Far apart windows should generate more signals
        close_valid = close_signals.dropna()
        far_valid = far_signals.dropna()

        if len(close_valid) > 0 and len(far_valid) > 0:
            # Far apart windows should be more sensitive to price changes
            assert len(far_valid) >= len(close_valid)
