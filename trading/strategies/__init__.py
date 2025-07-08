"""Trading strategies module."""

import logging
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime

from trading.strategies.strategy_manager import StrategyManager, Strategy, StrategyMetrics
from trading.strategies.rsi_signals import generate_rsi_signals, load_optimized_settings, generate_signals
from trading.strategies.bollinger_strategy import BollingerStrategy, BollingerConfig
from trading.strategies.macd_strategy import MACDStrategy, MACDConfig
from trading.strategies.sma_strategy import SMAStrategy, SMAConfig
from trading.strategies.cci_strategy import CCIStrategy, CCIConfig, generate_cci_signals
from trading.strategies.atr_strategy import ATRStrategy, ATRConfig, generate_atr_signals

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
            return {'success': True, 'result': strategy.generate_signals(data, **kwargs), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        elif strategy_name.lower() == 'cci':
            return generate_cci_signals(data, **kwargs)
        elif strategy_name.lower() == 'atr':
            return generate_atr_signals(data, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
    except Exception as e:
        logging.error(f"Error getting signals for {strategy_name}: {e}")
        raise RuntimeError(f"Signal generation failed for {strategy_name}: {e}")

def get_available_strategies() -> List[str]:
    """Get list of available strategies.
    
    Returns:
        List[str]: Available strategy names
    """
    return ['rsi', 'bollinger', 'macd', 'sma', 'cci', 'atr']

def create_strategy(strategy_name: str, **kwargs) -> Any:
    """Create a strategy instance.
    
    Args:
        strategy_name: Name of the strategy
        **kwargs: Strategy parameters
        
    Returns:
        Strategy instance
    """
    strategy_map = {
        'rsi': lambda: None,  # RSI uses function-based approach
        'bollinger': lambda: BollingerStrategy(BollingerConfig(**kwargs)),
        'macd': lambda: MACDStrategy(MACDConfig(**kwargs)),
        'sma': lambda: SMAStrategy(SMAConfig(**kwargs)),
        'cci': lambda: CCIStrategy(CCIConfig(**kwargs)),
        'atr': lambda: ATRStrategy(ATRConfig(**kwargs))
    }
    
    if strategy_name.lower() not in strategy_map:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    return strategy_map[strategy_name.lower()]()

__all__ = [
    'StrategyManager',
    'Strategy',
    'StrategyMetrics',
    'generate_rsi_signals',
    'load_optimized_settings',
    'generate_signals',
    'get_signals',
    'get_available_strategies',
    'create_strategy',
    'BollingerStrategy',
    'BollingerConfig',
    'MACDStrategy',
    'MACDConfig',
    'SMAStrategy',
    'SMAConfig',
    'CCIStrategy',
    'CCIConfig',
    'generate_cci_signals',
    'ATRStrategy',
    'ATRConfig',
    'generate_atr_signals'
] 