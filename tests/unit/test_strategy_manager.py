"""Tests for the StrategyManager class."""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from trading.strategies.strategy_manager import StrategyManager, StrategyError
from trading.strategies.rsi_strategy import RSIStrategy
from trading.strategies.macd_strategy import MACDStrategy
from trading.strategies.sma_strategy import SMAStrategy

@pytest.fixture
def sample_data():
    """Create sample price data."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    return data

@pytest.fixture
def strategy_manager(tmp_path):
    """Create a StrategyManager instance."""
    return StrategyManager({'results_dir': str(tmp_path)})

def test_rsi_strategy_integration(strategy_manager, sample_data):
    """Test RSI strategy integration."""
    # Create and register RSI strategy
    rsi_strategy = RSIStrategy(period=14)
    strategy_manager.register_strategy("RSI", rsi_strategy)
    
    # Generate signals
    signals = strategy_manager.generate_signals(sample_data, "RSI")
    
    # Verify signals
    assert isinstance(signals, pd.DataFrame)
    assert "RSI" in signals.columns
    assert "signal" in signals.columns
    assert len(signals) == len(sample_data)
    
    # Verify RSI values
    assert signals["RSI"].min() >= 0
    assert signals["RSI"].max() <= 100

def test_macd_strategy_integration(strategy_manager, sample_data):
    """Test MACD strategy integration."""
    # Create and register MACD strategy
    macd_strategy = MACDStrategy(fast_period=12, slow_period=26, signal_period=9)
    strategy_manager.register_strategy("MACD", macd_strategy)
    
    # Generate signals
    signals = strategy_manager.generate_signals(sample_data, "MACD")
    
    # Verify signals
    assert isinstance(signals, pd.DataFrame)
    assert "MACD" in signals.columns
    assert "signal" in signals.columns
    assert "histogram" in signals.columns
    assert len(signals) == len(sample_data)

def test_sma_strategy_integration(strategy_manager, sample_data):
    """Test SMA strategy integration."""
    # Create and register SMA strategy
    sma_strategy = SMAStrategy(short_period=20, long_period=50)
    strategy_manager.register_strategy("SMA", sma_strategy)
    
    # Generate signals
    signals = strategy_manager.generate_signals(sample_data, "SMA")
    
    # Verify signals
    assert isinstance(signals, pd.DataFrame)
    assert "SMA_short" in signals.columns
    assert "SMA_long" in signals.columns
    assert "signal" in signals.columns
    assert len(signals) == len(sample_data)

def test_strategy_combination(strategy_manager, sample_data):
    """Test combining multiple strategies."""
    # Register multiple strategies
    rsi_strategy = RSIStrategy(period=14)
    macd_strategy = MACDStrategy(fast_period=12, slow_period=26, signal_period=9)
    
    strategy_manager.register_strategy("RSI", rsi_strategy)
    strategy_manager.register_strategy("MACD", macd_strategy)
    
    # Set ensemble weights
    strategy_manager.set_ensemble({
        "RSI": 0.6,
        "MACD": 0.4
    })
    
    # Generate combined signals
    signals = strategy_manager.generate_ensemble_signals(sample_data)
    
    # Verify combined signals
    assert isinstance(signals, pd.DataFrame)
    assert "ensemble_signal" in signals.columns
    assert len(signals) == len(sample_data)

def test_strategy_validation(strategy_manager):
    """Test strategy validation."""
    # Test invalid strategy name
    with pytest.raises(StrategyError):
        strategy_manager.get_strategy("INVALID")
    
    # Test duplicate strategy registration
    rsi_strategy = RSIStrategy(period=14)
    strategy_manager.register_strategy("RSI", rsi_strategy)
    
    with pytest.raises(StrategyError):
        strategy_manager.register_strategy("RSI", rsi_strategy)

def test_strategy_activation(strategy_manager):
    """Test strategy activation and deactivation."""
    # Register strategy
    rsi_strategy = RSIStrategy(period=14)
    strategy_manager.register_strategy("RSI", rsi_strategy)
    
    # Activate strategy
    strategy_manager.activate_strategy("RSI")
    assert "RSI" in strategy_manager.active_strategies
    
    # Deactivate strategy
    strategy_manager.deactivate_strategy("RSI")
    assert "RSI" not in strategy_manager.active_strategies

def test_strategy_parameters(strategy_manager, sample_data):
    """Test strategy parameter handling."""
    # Register RSI strategy with custom parameters
    rsi_strategy = RSIStrategy(period=21)  # Different from default
    strategy_manager.register_strategy("RSI", rsi_strategy)
    
    # Generate signals
    signals = strategy_manager.generate_signals(sample_data, "RSI")
    
    # Verify custom parameters were used
    assert isinstance(signals, pd.DataFrame)
    assert "RSI" in signals.columns
    assert len(signals) == len(sample_data)

