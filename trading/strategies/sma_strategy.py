"""Simple Moving Average (SMA) trading strategy implementation."""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings

# Import centralized technical indicators
from utils.technical_indicators import calculate_sma

@dataclass
class SMAConfig:
    """Configuration for SMA strategy."""
    short_window: int = 20
    long_window: int = 50
    min_volume: float = 1000.0
    min_price: float = 1.0
    confirmation_periods: int = 3  # Number of periods to confirm trend change

class SMAStrategy:
    """Simple Moving Average (SMA) trading strategy implementation."""
    
    def __init__(self, config: Optional[SMAConfig] = None):
        """Initialize the strategy with configuration."""
        self.config = config or SMAConfig()
        self.signals = None
        self.positions = None
        
    def calculate_sma(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate short and long SMAs for the given data."""
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
            
        # Use centralized SMA calculation
        short_sma = calculate_sma(data['close'], self.config.short_window)
        long_sma = calculate_sma(data['close'], self.config.long_window)
        
        return short_sma, long_sma
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on SMA crossover."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        required_columns = ['close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
            
        # Calculate SMAs
        short_sma, long_sma = self.calculate_sma(data)
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Generate signals based on SMA crossover with trend confirmation
        # Calculate the difference between short and long SMA
        sma_diff = short_sma - long_sma
        
        # Initialize signal arrays
        buy_signals = pd.Series(0, index=data.index)
        sell_signals = pd.Series(0, index=data.index)
        
        # Apply trend confirmation logic
        for i in range(self.config.confirmation_periods, len(data)):
            try:
                # Check if trend has been consistent for confirmation_periods
                start_idx = max(0, i - self.config.confirmation_periods)
                end_idx = min(len(sma_diff), i + 1)
                recent_diffs = sma_diff.iloc[start_idx:end_idx]
                
                # Buy signal: short SMA consistently above long SMA
                if len(recent_diffs) >= self.config.confirmation_periods and all(recent_diffs > 0):
                    buy_signals.iloc[i] = 1
                
                # Sell signal: short SMA consistently below long SMA
                if len(recent_diffs) >= self.config.confirmation_periods and all(recent_diffs < 0):
                    sell_signals.iloc[i] = 1
            except IndexError as e:
                # Log the error and continue with next iteration
                warnings.warn(f"Index error in SMA strategy at position {i}: {e}")
                continue
        
        # Combine signals (buy takes precedence over sell)
        signals['signal'] = buy_signals - sell_signals
        
        # Add SMAs to signals
        signals['short_sma'] = short_sma
        signals['long_sma'] = long_sma
        
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
            'short_window': self.config.short_window,
            'long_window': self.config.long_window,
            'min_volume': self.config.min_volume,
            'min_price': self.config.min_price
        }
        
    def set_parameters(self, params: Dict) -> Dict:
        """Set strategy parameters."""
        try:
            self.config = SMAConfig(**params)
            self.signals = None
            self.positions = None
            return {"status": "success", "parameters_updated": True, "config": self.get_parameters()}
        except Exception as e:
            return {"status": "error", "message": str(e)}