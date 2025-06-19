"""Tests for the RSI strategy."""

import pytest
import pandas as pd
import numpy as np
from trading.strategies.rsi_strategy import RSIStrategy

class TestRSIStrategy:
    @pytest.fixture
    def strategy(self):
        """Create an RSI strategy instance for testing."""
        return RSIStrategy(window=14, overbought=70, oversold=30)

    @pytest.fixture
    def sample_data(self, sample_price_data):
        """Create sample price data for testing."""
        return sample_price_data

    def test_rsi_calculation(self, strategy, sample_data):
        """Test that RSI is calculated correctly."""
        rsi = strategy.calculate_rsi(sample_data['close'])
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(sample_data)
        assert not rsi.isnull().any()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_signal_generation(self, strategy, sample_data):
        """Test that trading signals are generated correctly."""
        signals = strategy.generate_signals(sample_data)
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data)
        assert signals.isin([1, 0, -1]).all()  # 1 for buy, -1 for sell, 0 for hold

    def test_overbought_condition(self, strategy):
        """Test that overbought condition triggers sell signal."""
        # Create data with high RSI
        data = pd.DataFrame({
            'close': [100] * 15 + [101] * 15  # Steady increase to trigger overbought
        })
        signals = strategy.generate_signals(data)
        assert (signals == -1).any()  # Should have at least one sell signal

    def test_oversold_condition(self, strategy):
        """Test that oversold condition triggers buy signal."""
        # Create data with low RSI
        data = pd.DataFrame({
            'close': [100] * 15 + [99] * 15  # Steady decrease to trigger oversold
        })
        signals = strategy.generate_signals(data)
        assert (signals == 1).any()  # Should have at least one buy signal

    def test_parameter_validation(self):
        """Test that strategy parameters are validated."""
        with pytest.raises(ValueError):
            RSIStrategy(window=0)  # Invalid window
        with pytest.raises(ValueError):
            RSIStrategy(overbought=50)  # Invalid overbought threshold
        with pytest.raises(ValueError):
            RSIStrategy(oversold=50)  # Invalid oversold threshold

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
        """Test that signals are consistent with RSI values."""
        rsi = strategy.calculate_rsi(sample_data['close'])
        signals = strategy.generate_signals(sample_data)
        
        # Check that overbought RSI corresponds to sell signals
        overbought_mask = rsi > strategy.overbought
        assert (signals[overbought_mask] == -1).all()
        
        # Check that oversold RSI corresponds to buy signals
        oversold_mask = rsi < strategy.oversold
        assert (signals[oversold_mask] == 1).all()

    def test_strategy_configuration(self, strategy):
        """Test that strategy configuration is correct."""
        assert strategy.window == 14
        assert strategy.overbought == 70
        assert strategy.oversold == 30
        assert strategy.name == 'RSI'

    def test_strategy_performance(self, strategy, sample_data):
        """Test that the strategy can be evaluated for performance."""
        signals = strategy.generate_signals(sample_data)
        performance = strategy.evaluate_performance(sample_data, signals)
        
        assert isinstance(performance, dict)
        assert 'returns' in performance
        assert 'sharpe_ratio' in performance
        assert 'max_drawdown' in performance
        assert isinstance(performance['returns'], float)
        assert isinstance(performance['sharpe_ratio'], float)
        assert isinstance(performance['max_drawdown'], float) 