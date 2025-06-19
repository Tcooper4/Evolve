"""Tests for the Bollinger Bands strategy."""

import pytest
import pandas as pd
import numpy as np
from trading.strategies.bollinger_strategy import BollingerStrategy

class TestBollingerStrategy:
    @pytest.fixture
    def strategy(self):
        """Create a Bollinger Bands strategy instance for testing."""
        return BollingerStrategy(window=20, num_std=2)

    @pytest.fixture
    def sample_data(self, sample_price_data):
        """Create sample price data for testing."""
        return sample_price_data

    def test_bands_calculation(self, strategy, sample_data):
        """Test that Bollinger Bands are calculated correctly."""
        upper, middle, lower = strategy.calculate_bands(sample_data['close'])
        
        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)
        assert len(upper) == len(sample_data)
        assert not upper.isnull().any()
        assert not middle.isnull().any()
        assert not lower.isnull().any()
        
        # Verify band relationships
        assert (upper >= middle).all()
        assert (middle >= lower).all()

    def test_signal_generation(self, strategy, sample_data):
        """Test that trading signals are generated correctly."""
        signals = strategy.generate_signals(sample_data)
        
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data)
        assert signals.isin([1, 0, -1]).all()  # 1 for buy, -1 for sell, 0 for hold

    def test_upper_band_breakout(self, strategy):
        """Test that upper band breakout triggers sell signal."""
        # Create data with upper band breakout
        data = pd.DataFrame({
            'close': [100] * 20 + [120] * 5  # Sharp increase to trigger upper band breakout
        })
        signals = strategy.generate_signals(data)
        assert (signals == -1).any()  # Should have at least one sell signal

    def test_lower_band_breakout(self, strategy):
        """Test that lower band breakout triggers buy signal."""
        # Create data with lower band breakout
        data = pd.DataFrame({
            'close': [100] * 20 + [80] * 5  # Sharp decrease to trigger lower band breakout
        })
        signals = strategy.generate_signals(data)
        assert (signals == 1).any()  # Should have at least one buy signal

    def test_parameter_validation(self):
        """Test that strategy parameters are validated."""
        with pytest.raises(ValueError):
            BollingerStrategy(window=0)  # Invalid window
        with pytest.raises(ValueError):
            BollingerStrategy(num_std=0)  # Invalid standard deviation

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
        """Test that signals are consistent with band breakouts."""
        upper, middle, lower = strategy.calculate_bands(sample_data['close'])
        signals = strategy.generate_signals(sample_data)
        
        # Check that upper band breakout corresponds to sell signals
        upper_breakout = sample_data['close'] > upper
        assert (signals[upper_breakout] == -1).all()
        
        # Check that lower band breakout corresponds to buy signals
        lower_breakout = sample_data['close'] < lower
        assert (signals[lower_breakout] == 1).all()

    def test_strategy_configuration(self, strategy):
        """Test that strategy configuration is correct."""
        assert strategy.window == 20
        assert strategy.num_std == 2
        assert strategy.name == 'Bollinger'

    def test_band_width(self, strategy, sample_data):
        """Test that band width is calculated correctly."""
        upper, middle, lower = strategy.calculate_bands(sample_data['close'])
        band_width = (upper - lower) / middle
        
        assert isinstance(band_width, pd.Series)
        assert not band_width.isnull().any()
        assert (band_width >= 0).all()

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

    def test_volatility_adaptation(self, strategy):
        """Test that strategy adapts to changing volatility."""
        # Create data with changing volatility
        data = pd.DataFrame({
            'close': [100] * 20 + [120, 80, 130, 70, 140]  # High volatility period
        })
        
        # Calculate bands
        upper, middle, lower = strategy.calculate_bands(data['close'])
        
        # Verify band width increases with volatility
        band_width = (upper - lower) / middle
        assert band_width.iloc[-1] > band_width.iloc[0]  # Band width should increase 