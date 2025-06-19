"""Tests for the Simple Moving Average (SMA) strategy."""

import pytest
import pandas as pd
import numpy as np
from trading.strategies.sma_strategy import SMAStrategy

class TestSMAStrategy:
    @pytest.fixture
    def strategy(self):
        """Create an SMA strategy instance for testing."""
        return SMAStrategy(short_window=10, long_window=30)

    @pytest.fixture
    def sample_data(self, sample_price_data):
        """Create sample price data for testing."""
        return sample_price_data

    def test_sma_calculation(self, strategy, sample_data):
        """Test that SMAs are calculated correctly."""
        short_sma, long_sma = strategy.calculate_smas(sample_data['close'])
        
        assert isinstance(short_sma, pd.Series)
        assert isinstance(long_sma, pd.Series)
        assert len(short_sma) == len(sample_data)
        assert not short_sma.isnull().any()
        assert not long_sma.isnull().any()
        
        # Verify window lengths
        assert short_sma.iloc[0] == sample_data['close'].iloc[:10].mean()
        assert long_sma.iloc[0] == sample_data['close'].iloc[:30].mean()

    def test_signal_generation(self, strategy, sample_data):
        """Test that trading signals are generated correctly."""
        signals = strategy.generate_signals(sample_data)
        
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data)
        assert signals.isin([1, 0, -1]).all()  # 1 for buy, -1 for sell, 0 for hold

    def test_golden_cross(self, strategy):
        """Test that golden cross (short SMA crosses above long SMA) triggers buy signal."""
        # Create data with golden cross
        data = pd.DataFrame({
            'close': [100] * 30 + [110] * 10  # Short SMA will cross above long SMA
        })
        signals = strategy.generate_signals(data)
        assert (signals == 1).any()  # Should have at least one buy signal

    def test_death_cross(self, strategy):
        """Test that death cross (short SMA crosses below long SMA) triggers sell signal."""
        # Create data with death cross
        data = pd.DataFrame({
            'close': [100] * 30 + [90] * 10  # Short SMA will cross below long SMA
        })
        signals = strategy.generate_signals(data)
        assert (signals == -1).any()  # Should have at least one sell signal

    def test_parameter_validation(self):
        """Test that strategy parameters are validated."""
        with pytest.raises(ValueError):
            SMAStrategy(short_window=0, long_window=30)  # Invalid short window
        with pytest.raises(ValueError):
            SMAStrategy(short_window=10, long_window=0)  # Invalid long window
        with pytest.raises(ValueError):
            SMAStrategy(short_window=30, long_window=10)  # Short window > long window

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
        """Test that signals are consistent with SMA crossovers."""
        short_sma, long_sma = strategy.calculate_smas(sample_data['close'])
        signals = strategy.generate_signals(sample_data)
        
        # Check that golden cross corresponds to buy signals
        golden_cross = (short_sma > long_sma) & (short_sma.shift(1) <= long_sma.shift(1))
        assert (signals[golden_cross] == 1).all()
        
        # Check that death cross corresponds to sell signals
        death_cross = (short_sma < long_sma) & (short_sma.shift(1) >= long_sma.shift(1))
        assert (signals[death_cross] == -1).all()

    def test_strategy_configuration(self, strategy):
        """Test that strategy configuration is correct."""
        assert strategy.short_window == 10
        assert strategy.long_window == 30
        assert strategy.name == 'SMA'

    def test_sma_smoothing(self, strategy, sample_data):
        """Test that SMAs properly smooth price data."""
        short_sma, long_sma = strategy.calculate_smas(sample_data['close'])
        
        # Calculate price volatility
        price_volatility = sample_data['close'].pct_change().std()
        short_sma_volatility = short_sma.pct_change().std()
        long_sma_volatility = long_sma.pct_change().std()
        
        # Verify that SMAs reduce volatility
        assert short_sma_volatility < price_volatility
        assert long_sma_volatility < short_sma_volatility

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

    def test_trend_detection(self, strategy):
        """Test that strategy correctly identifies trends."""
        # Create data with clear uptrend
        uptrend_data = pd.DataFrame({
            'close': np.linspace(100, 200, 100)  # Linear uptrend
        })
        uptrend_signals = strategy.generate_signals(uptrend_data)
        assert (uptrend_signals == 1).any()  # Should generate buy signals
        
        # Create data with clear downtrend
        downtrend_data = pd.DataFrame({
            'close': np.linspace(200, 100, 100)  # Linear downtrend
        })
        downtrend_signals = strategy.generate_signals(downtrend_data)
        assert (downtrend_signals == -1).any()  # Should generate sell signals 