"""MACD (Moving Average Convergence Divergence) trading strategy implementation."""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class MACDConfig:
    """Configuration for MACD strategy."""
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    min_volume: float = 1000.0
    min_price: float = 1.0

class MACDStrategy:
    """MACD trading strategy implementation."""
    
    def __init__(self, config: Optional[MACDConfig] = None):
        """Initialize the strategy with configuration."""
        self.config = config or MACDConfig()
        self.signals = None
        self.positions = None
        
    def calculate_macd(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD components for the given data."""
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
            
        # Calculate fast and slow EMAs
        fast_ema = data['close'].ewm(span=self.config.fast_period, adjust=False).mean()
        slow_ema = data['close'].ewm(span=self.config.slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=self.config.signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on MACD."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        required_columns = ['close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
            
        # Calculate MACD components
        macd_line, signal_line, histogram = self.calculate_macd(data)
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Generate signals based on MACD line crossing signal line
        signals.loc[macd_line > signal_line, 'signal'] = 1  # Buy signal
        signals.loc[macd_line < signal_line, 'signal'] = -1  # Sell signal
        
        # Add MACD components to signals
        signals['macd_line'] = macd_line
        signals['signal_line'] = signal_line
        signals['histogram'] = histogram
        
        # Filter signals based on volume and price
        volume_mask = data['volume'] >= self.config.min_volume
        price_mask = data['close'] >= self.config.min_price
        signals.loc[~(volume_mask & price_mask), 'signal'] = 0
        
        self.signals = signals
        return signals
        
    def calculate_positions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trading positions based on signals."""
        if self.signals is None:
            self.generate_signals(data)
            
        positions = pd.DataFrame(index=data.index)
        positions['position'] = self.signals['signal'].cumsum()
        
        # Ensure positions are within bounds
        positions['position'] = positions['position'].clip(-1, 1)
        
        self.positions = positions
        return positions
        
    def get_parameters(self) -> Dict:
        """Get strategy parameters."""
        return {
            'fast_period': self.config.fast_period,
            'slow_period': self.config.slow_period,
            'signal_period': self.config.signal_period,
            'min_volume': self.config.min_volume,
            'min_price': self.config.min_price
        }
        
    def set_parameters(self, params: Dict) -> None:
        """Set strategy parameters."""
        self.config = MACDConfig(**params)
        self.signals = None
        self.positions = None 