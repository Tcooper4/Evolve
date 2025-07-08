"""
Commodity Channel Index (CCI) Strategy Implementation

This module implements a trading strategy based on the Commodity Channel Index (CCI),
which is a momentum oscillator used to identify cyclical trends in commodities.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class CCIConfig:
    """Configuration for CCI strategy."""
    period: int = 20
    constant: float = 0.015
    oversold_threshold: float = -100.0
    overbought_threshold: float = 100.0
    min_volume: float = 1000.0
    min_price: float = 1.0

class CCIStrategy:
    """Commodity Channel Index (CCI) trading strategy implementation."""
    
    def __init__(self, config: Optional[CCIConfig] = None):
        """
        Initialize the CCI strategy.
        
        Args:
            config: CCI strategy configuration
        """
        self.config = config or CCIConfig()
        self.logger = logging.getLogger(__name__)
        self.signals = None
        self.positions = None
        
    def calculate_cci(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate the Commodity Channel Index (CCI).
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            pd.Series: CCI values
        """
        if not all(col in data.columns for col in ['high', 'low', 'close']):
            raise ValueError("Data must contain 'high', 'low', 'close' columns")
        
        # Calculate Typical Price
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Calculate Simple Moving Average of Typical Price
        sma_tp = typical_price.rolling(window=self.config.period).mean()
        
        # Calculate Mean Deviation
        mean_deviation = typical_price.rolling(window=self.config.period).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        
        # Calculate CCI
        cci = (typical_price - sma_tp) / (self.config.constant * mean_deviation)
        
        return cci
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on CCI.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: Signals DataFrame
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        required_columns = ['high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        # Calculate CCI
        cci = self.calculate_cci(data)
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=data.index)
        signals['cci'] = cci
        signals['signal'] = 0
        
        # Generate signals based on CCI thresholds
        # Buy signal when CCI crosses above oversold threshold
        signals.loc[(cci > self.config.oversold_threshold) & 
                   (cci.shift(1) <= self.config.oversold_threshold), 'signal'] = 1
        
        # Sell signal when CCI crosses below overbought threshold
        signals.loc[(cci < self.config.overbought_threshold) & 
                   (cci.shift(1) >= self.config.overbought_threshold), 'signal'] = -1
        
        # Additional signals for extreme values
        signals.loc[cci < -200, 'signal'] = 1  # Strong buy signal
        signals.loc[cci > 200, 'signal'] = -1  # Strong sell signal
        
        # Filter signals based on volume and price
        volume_mask = data['volume'] >= self.config.min_volume
        price_mask = data['close'] >= self.config.min_price
        signals.loc[~(volume_mask & price_mask), 'signal'] = 0
        
        # Add signal strength based on CCI magnitude
        signals['signal_strength'] = np.abs(cci) / 100.0
        signals['signal_strength'] = signals['signal_strength'].clip(0, 1)
        
        self.signals = signals
        return signals
    
    def calculate_positions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trading positions based on signals.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: Positions DataFrame
        """
        if self.signals is None:
            self.generate_signals(data)
        
        positions = pd.DataFrame(index=data.index)
        positions['position'] = self.signals['signal'].cumsum()
        
        # Ensure positions are within bounds
        positions['position'] = positions['position'].clip(-1, 1)
        
        # Add position size based on signal strength
        positions['position_size'] = positions['position'] * self.signals['signal_strength']
        
        self.positions = positions
        return positions
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get strategy information.
        
        Returns:
            Dict: Strategy information
        """
        return {
            'name': 'CCI Strategy',
            'description': 'Commodity Channel Index momentum strategy',
            'parameters': {
                'period': self.config.period,
                'constant': self.config.constant,
                'oversold_threshold': self.config.oversold_threshold,
                'overbought_threshold': self.config.overbought_threshold
            },
            'signal_types': ['buy', 'sell', 'hold'],
            'indicators': ['cci']
        }
    
    def optimize_parameters(self, data: pd.DataFrame, 
                          optimization_metric: str = 'sharpe_ratio') -> CCIConfig:
        """
        Optimize strategy parameters using historical data.
        
        Args:
            data: Historical data for optimization
            optimization_metric: Metric to optimize ('sharpe_ratio', 'total_return', 'max_drawdown')
            
        Returns:
            CCIConfig: Optimized configuration
        """
        self.logger.info("Optimizing CCI strategy parameters...")
        
        # Parameter ranges to test
        periods = [10, 14, 20, 30]
        constants = [0.015, 0.02, 0.025]
        oversold_thresholds = [-100, -150, -200]
        overbought_thresholds = [100, 150, 200]
        
        best_config = None
        best_metric = float('-inf')
        
        for period in periods:
            for constant in constants:
                for oversold in oversold_thresholds:
                    for overbought in overbought_thresholds:
                        # Test configuration
                        test_config = CCIConfig(
                            period=period,
                            constant=constant,
                            oversold_threshold=oversold,
                            overbought_threshold=overbought
                        )
                        
                        # Create temporary strategy with test config
                        temp_strategy = CCIStrategy(test_config)
                        signals = temp_strategy.generate_signals(data)
                        
                        # Calculate performance metric
                        metric_value = self._calculate_performance_metric(signals, data, optimization_metric)
                        
                        if metric_value > best_metric:
                            best_metric = metric_value
                            best_config = test_config
        
        if best_config:
            self.config = best_config
            self.logger.info(f"Optimized parameters: {best_config}")
        
        return best_config or self.config
    
    def _calculate_performance_metric(self, signals: pd.DataFrame, data: pd.DataFrame, 
                                    metric: str) -> float:
        """
        Calculate performance metric for optimization.
        
        Args:
            signals: Trading signals
            data: Price data
            metric: Metric to calculate
            
        Returns:
            float: Performance metric value
        """
        try:
            # Calculate returns
            returns = data['close'].pct_change()
            strategy_returns = signals['signal'].shift(1) * returns
            
            if metric == 'sharpe_ratio':
                return strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
            elif metric == 'total_return':
                return strategy_returns.sum()
            elif metric == 'max_drawdown':
                cumulative_returns = (1 + strategy_returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                return drawdown.min()
            else:
                return 0.0
        except Exception as e:
            self.logger.error(f"Error calculating performance metric: {e}")
            return 0.0

def create_cci_strategy(config: Optional[CCIConfig] = None) -> CCIStrategy:
    """
    Create a CCI strategy instance.
    
    Args:
        config: Optional CCI configuration
        
    Returns:
        CCIStrategy: Configured CCI strategy
    """
    return CCIStrategy(config)

def generate_cci_signals(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """
    Generate CCI trading signals.
    
    Args:
        data: DataFrame with OHLCV data
        **kwargs: Strategy parameters
        
    Returns:
        Dict: Signals and metadata
    """
    try:
        config = CCIConfig(**kwargs)
        strategy = CCIStrategy(config)
        signals = strategy.generate_signals(data)
        
        return {
            'success': True,
            'signals': signals,
            'strategy_info': strategy.get_strategy_info(),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error generating CCI signals: {e}")
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        } 