"""
Test cases for Bollinger Bands strategy.

This module tests:
- Strategy initialization
- Band calculation
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
from trading.strategies.bollinger_strategy import BollingerStrategy, BollingerConfig
from utils.strategy_utils import calculate_returns

class TestBollingerStrategy:
    @pytest.fixture
    def strategy_config(self) -> BollingerConfig:
        """Get strategy configuration."""
        return BollingerConfig(
            window=20,
            num_std=2.0,
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
        strategy = BollingerStrategy(config=strategy_config)
        
        # Verify configuration
        assert strategy.config.window == strategy_config.window
        assert strategy.config.num_std == strategy_config.num_std
        assert strategy.config.min_volume == strategy_config.min_volume
        assert strategy.config.min_price == strategy_config.min_price
        
    def test_band_calculation(self, strategy_config, sample_data):
        """Test Bollinger Bands calculation."""
        strategy = BollingerStrategy(config=strategy_config)
        upper_band, middle_band, lower_band = strategy.calculate_bands(sample_data)
        
        # Verify bands
        assert isinstance(upper_band, pd.Series)
        assert isinstance(middle_band, pd.Series)
        assert isinstance(lower_band, pd.Series)
        
        # Verify calculations
        assert not upper_band.isnull().any()
        assert not middle_band.isnull().any()
        assert not lower_band.isnull().any()
        
        # Verify band relationships
        assert all(upper_band >= middle_band)
        assert all(lower_band <= middle_band)
        
    def test_signal_generation(self, strategy_config, sample_data):
        """Test signal generation."""
        strategy = BollingerStrategy(config=strategy_config)
        signals = strategy.generate_signals(sample_data)
        
        # Verify signals
        assert isinstance(signals, pd.DataFrame)
        assert "signal" in signals.columns
        assert all(signal in [-1, 0, 1] for signal in signals["signal"])
        
        # Verify bands in signals
        assert "upper_band" in signals.columns
        assert "middle_band" in signals.columns
        assert "lower_band" in signals.columns
        
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid window
        with pytest.raises(ValueError):
            BollingerConfig(window=0)
            
        # Test invalid standard deviation
        with pytest.raises(ValueError):
            BollingerConfig(num_std=0.0)
            
    def test_performance_metrics(self, strategy_config, sample_data):
        """Test performance metrics calculation."""
        strategy = BollingerStrategy(config=strategy_config)
        signals = strategy.generate_signals(sample_data)
        positions = strategy.calculate_positions(sample_data)
        
        # Verify positions
        assert isinstance(positions, pd.DataFrame)
        assert "position" in positions.columns
        assert all(position in [-1, 0, 1] for position in positions["position"])
        
    def test_edge_cases(self, strategy_config):
        """Test edge cases."""
        strategy = BollingerStrategy(config=strategy_config)
        
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
        strategy = BollingerStrategy(config=strategy_config)
        
        # Update parameters
        new_config = BollingerConfig(
            window=50,
            num_std=2.5,
            min_volume=2000.0,
            min_price=2.0
        )
        strategy.set_parameters(new_config.__dict__)
        
        # Verify updates
        assert strategy.config.window == new_config.window
        assert strategy.config.num_std == new_config.num_std
        assert strategy.config.min_volume == new_config.min_volume
        assert strategy.config.min_price == new_config.min_price
        
        # Verify signals are reset
        assert strategy.signals is None
        assert strategy.positions is None
        
    def test_band_width(self, strategy_config, sample_data):
        """Test band width calculation."""
        strategy = BollingerStrategy(config=strategy_config)
        upper_band, middle_band, lower_band = strategy.calculate_bands(sample_data)
        
        # Calculate band width
        band_width = (upper_band - lower_band) / middle_band
        
        # Verify band width
        assert not band_width.isnull().any()
        assert all(band_width >= 0)
        
    def test_signal_threshold(self, strategy_config, sample_data):
        """Test signal threshold application."""
        strategy = BollingerStrategy(config=strategy_config)
        signals = strategy.generate_signals(sample_data)
        
        # Verify threshold application
        assert all(abs(signal) <= 1 for signal in signals["signal"])
        
    def test_strategy_reset(self, strategy_config, sample_data):
        """Test strategy reset."""
        strategy = BollingerStrategy(config=strategy_config)
        
        # Generate signals
        signals = strategy.generate_signals(sample_data)
        
        # Reset strategy
        strategy.reset()
        
        # Verify reset
        assert strategy.signals is None
        assert strategy.positions is None
        
    def test_strategy_serialization(self, strategy_config):
        """Test strategy serialization."""
        strategy = BollingerStrategy(config=strategy_config)
        
        # Serialize
        config = strategy.get_config()
        
        # Verify serialization
        assert config["window"] == strategy_config.window
        assert config["num_std"] == strategy_config.num_std
        assert config["min_volume"] == strategy_config.min_volume
        assert config["min_price"] == strategy_config.min_price
        
    def test_band_breakout(self, strategy_config, sample_data):
        """Test band breakout detection."""
        strategy = BollingerStrategy(config=strategy_config)
        signals = strategy.generate_signals(sample_data)
        
        # Verify breakout signals
        assert "breakout" in signals.columns
        assert all(signal in [True, False] for signal in signals["breakout"]) 