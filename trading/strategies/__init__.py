"""Trading strategies module."""

import logging
from typing import Dict, Any
import pandas as pd

from trading.strategies.strategy_manager import StrategyManager, Strategy, StrategyMetrics
from trading.strategies.rsi_signals import generate_rsi_signals, load_optimized_settings, generate_signals
from trading.strategies.bollinger_strategy import BollingerStrategy, BollingerConfig
from trading.strategies.macd_strategy import MACDStrategy, MACDConfig
from trading.strategies.sma_strategy import SMAStrategy, SMAConfig

def get_signals(strategy_name: str, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Get trading signals for a specific strategy.
    
    Args:
        strategy_name: Name of the strategy to use
        data: Price data DataFrame
        **kwargs: Strategy-specific parameters
        
    Returns:
        Dictionary containing signals and metadata
    """
    try:
        if strategy_name.lower() == 'rsi':
            return generate_signals(data, **kwargs)
        elif strategy_name.lower() == 'bollinger':
            strategy = BollingerStrategy()
            return strategy.generate_signals(data, **kwargs)
        elif strategy_name.lower() == 'macd':
            strategy = MACDStrategy()
            return strategy.generate_signals(data, **kwargs)
        elif strategy_name.lower() == 'sma':
            strategy = SMAStrategy()
            return strategy.generate_signals(data, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
    except Exception as e:
        logging.error(f"Error getting signals for {strategy_name}: {e}")
        raise RuntimeError(f"Signal generation failed for {strategy_name}: {e}")

__all__ = [
    'StrategyManager',
    'Strategy',
    'StrategyMetrics',
    'generate_rsi_signals',
    'load_optimized_settings',
    'generate_signals',
    'get_signals',
    'BollingerStrategy',
    'BollingerConfig',
    'MACDStrategy',
    'MACDConfig',
    'SMAStrategy',
    'SMAConfig'
] 