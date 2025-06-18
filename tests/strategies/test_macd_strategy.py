"""
Test cases for MACD strategy.

This module tests:
- Strategy initialization
- MACD calculation
- Signal generation
- Parameter validation
- Performance metrics
- Edge cases
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
from trading.strategies.macd_strategy import MACDStrategy, MACDConfig

class TestMACDStrategy:
    @pytest.fixture
    def strategy_config(self) -> MACDConfig:
        """Get strategy configuration."""
        return MACDConfig(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            min_volume=1000.0,
            min_price=1.0
        )
        
    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample price data."""
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        prices = np.random.normal(100, 10, len(dates))
        volumes = np.random.normal(5000, 1000, len(dates))
        return pd.DataFrame({
            "close": prices,
            "volume": volumes
        }, index=dates)
        
    def test_strategy_initialization(self, strategy_config):
        """Test strategy initialization."""
        strategy = MACDStrategy(config=strategy_config)
        
        # Verify configuration
        assert strategy.config.fast_period == strategy_config.fast_period
        assert strategy.config.slow_period == strategy_config.slow_period
        assert strategy.config.signal_period == strategy_config.signal_period
        assert strategy.config.min_volume == strategy_config.min_volume
        assert strategy.config.min_price == strategy_config.min_price
        
    def test_macd_calculation(self, strategy_config, sample_data):
        """Test MACD calculation."""
        strategy = MACDStrategy(config=strategy_config)
        macd_line, signal_line, histogram = strategy.calculate_macd(sample_data)
        
        # Verify MACD components
        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal_line, pd.Series)
        assert isinstance(histogram, pd.Series)
        
        # Verify calculations
        assert not macd_line.isnull().any()
        assert not signal_line.isnull().any()
        assert not histogram.isnull().any()
        
    def test_signal_generation(self, strategy_config, sample_data):
        """Test signal generation."""
        strategy = MACDStrategy(config=strategy_config)
        signals = strategy.generate_signals(sample_data)
        
        # Verify signals
        assert isinstance(signals, pd.DataFrame)
        assert "signal" in signals.columns
        assert all(signal in [-1, 0, 1] for signal in signals["signal"])
        
        # Verify MACD components in signals
        assert "macd_line" in signals.columns
        assert "signal_line" in signals.columns
        assert "histogram" in signals.columns
        
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid periods
        with pytest.raises(ValueError):
            MACDConfig(fast_period=0)
            
        with pytest.raises(ValueError):
            MACDConfig(slow_period=0)
            
        with pytest.raises(ValueError):
            MACDConfig(signal_period=0)
            
    def test_performance_metrics(self, strategy_config, sample_data):
        """Test performance metrics calculation."""
        strategy = MACDStrategy(config=strategy_config)
        signals = strategy.generate_signals(sample_data)
        positions = strategy.calculate_positions(sample_data)
        
        # Verify positions
        assert isinstance(positions, pd.DataFrame)
        assert "position" in positions.columns
        assert all(position in [-1, 0, 1] for position in positions["position"])
        
    def test_edge_cases(self, strategy_config):
        """Test edge cases."""
        strategy = MACDStrategy(config=strategy_config)
        
        # Test empty data
        empty_data = pd.DataFrame(columns=["close", "volume"])
        with pytest.raises(ValueError):
            strategy.generate_signals(empty_data)
        
        # Test single row data
        single_row = pd.DataFrame({
            "close": [100],
            "volume": [5000]
        })
        signals = strategy.generate_signals(single_row)
        assert len(signals) == 1
        
        # Test all same values
        same_values = pd.DataFrame({
            "close": [100] * 100,
            "volume": [5000] * 100
        })
        signals = strategy.generate_signals(same_values)
        assert len(signals) == 100
        
    def test_parameter_updates(self, strategy_config, sample_data):
        """Test strategy parameter updates."""
        strategy = MACDStrategy(config=strategy_config)
        
        # Update parameters
        new_config = MACDConfig(
            fast_period=8,
            slow_period=21,
            signal_period=5,
            min_volume=2000.0,
            min_price=2.0
        )
        strategy.set_parameters(new_config.__dict__)
        
        # Verify updates
        assert strategy.config.fast_period == new_config.fast_period
        assert strategy.config.slow_period == new_config.slow_period
        assert strategy.config.signal_period == new_config.signal_period
        assert strategy.config.min_volume == new_config.min_volume
        assert strategy.config.min_price == new_config.min_price
        
        # Verify signals are reset
        assert strategy.signals is None
        assert strategy.positions is None 