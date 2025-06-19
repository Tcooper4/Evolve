"""Tests for the MACD strategy."""

import pytest
import pandas as pd
import numpy as np
from trading.strategies.macd_strategy import MACDStrategy

class TestMACDStrategy:
    @pytest.fixture
    def strategy(self):
        """Create a MACD strategy instance for testing."""
        return MACDStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9
        )

    @pytest.fixture
    def sample_data(self, sample_price_data):
        """Create sample price data for testing."""
        return sample_price_data

    def test_macd_calculation(self, strategy, sample_data):
        """Test that MACD is calculated correctly."""
        macd, signal, hist = strategy.calculate_macd(sample_data['close'])
        
        assert isinstance(macd, pd.Series)
        assert isinstance(signal, pd.Series)
        assert isinstance(hist, pd.Series)
        assert len(macd) == len(sample_data)
        assert not macd.isnull().any()
        assert not signal.isnull().any()
        assert not hist.isnull().any()

    def test_signal_generation(self, strategy, sample_data):
        """Test that trading signals are generated correctly."""
        signals = strategy.generate_signals(sample_data)
        
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data)
        assert signals.isin([1, 0, -1]).all()  # 1 for buy, -1 for sell, 0 for hold

    def test_bullish_crossover(self, strategy):
        """Test that bullish crossover triggers buy signal."""
        # Create data with bullish crossover
        data = pd.DataFrame({
            'close': [100] * 30 + [101] * 30  # Steady increase to trigger bullish crossover
        })
        signals = strategy.generate_signals(data)
        assert (signals == 1).any()  # Should have at least one buy signal

    def test_bearish_crossover(self, strategy):
        """Test that bearish crossover triggers sell signal."""
        # Create data with bearish crossover
        data = pd.DataFrame({
            'close': [100] * 30 + [99] * 30  # Steady decrease to trigger bearish crossover
        })
        signals = strategy.generate_signals(data)
        assert (signals == -1).any()  # Should have at least one sell signal

    def test_parameter_validation(self):
        """Test that strategy parameters are validated."""
        with pytest.raises(ValueError):
            MACDStrategy(fast_period=0)  # Invalid fast period
        with pytest.raises(ValueError):
            MACDStrategy(slow_period=10)  # Invalid slow period (must be > fast period)
        with pytest.raises(ValueError):
            MACDStrategy(signal_period=0)  # Invalid signal period

    def test_empty_data_handling(self, strategy):
        """Test that strategy handles empty data correctly."""
        empty_data = pd.DataFrame(columns=['close'])
        with pytest.raises(ValueError):
            strategy.generate_signals(empty_data)

    def test_missing_data_handling(self, strategy):
        """Test that strategy handles missing data correctly."""
        data = pd.DataFrame({
            'close': [100, np.nan, 101, 102]
        })
        with pytest.raises(ValueError):
            strategy.generate_signals(data)

    def test_signal_consistency(self, strategy, sample_data):
        """Test that signals are consistent with MACD values."""
        macd, signal, hist = strategy.calculate_macd(sample_data['close'])
        signals = strategy.generate_signals(sample_data)
        
        # Check that bullish crossover corresponds to buy signals
        bullish_mask = (hist > 0) & (hist.shift(1) <= 0)
        assert (signals[bullish_mask] == 1).all()
        
        # Check that bearish crossover corresponds to sell signals
        bearish_mask = (hist < 0) & (hist.shift(1) >= 0)
        assert (signals[bearish_mask] == -1).all()

    def test_strategy_configuration(self, strategy):
        """Test that strategy configuration is correct."""
        assert strategy.fast_period == 12
        assert strategy.slow_period == 26
        assert strategy.signal_period == 9
        assert strategy.name == 'MACD'

    def test_performance_metrics(self, strategy, sample_data):
        """Test that strategy calculates performance metrics correctly."""
        signals = strategy.generate_signals(sample_data)
        metrics = strategy.calculate_performance_metrics(sample_data, signals)
        
        assert 'returns' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert isinstance(metrics['returns'], float)
        assert isinstance(metrics['sharpe_ratio'], float)
        assert isinstance(metrics['max_drawdown'], float) 