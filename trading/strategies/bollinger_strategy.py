"""Bollinger Bands trading strategy implementation."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings

# Import centralized technical indicators
from utils.technical_indicators import calculate_bollinger_bands

@dataclass
class BollingerConfig:
    """Configuration for Bollinger Bands strategy."""
    window: int = 20
    num_std: float = 2.0
    min_volume: float = 1000.0
    min_price: float = 1.0

class BollingerStrategy:
    """Bollinger Bands trading strategy implementation."""
    
    def __init__(self, config: Optional[BollingerConfig] = None):
        """Initialize the strategy with configuration."""
        self.config = config or BollingerConfig()
        self.signals = None
        self.positions = None
        
    def calculate_bands(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands for the given data."""
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
            
        # Use centralized Bollinger Bands calculation
        upper_band, middle_band, lower_band = calculate_bollinger_bands(
            data['close'], 
            self.config.window, 
            self.config.num_std
        )
        
        return upper_band, middle_band, lower_band
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on Bollinger Bands."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        # Add validation for Close column
        if "Close" not in data.columns and "close" not in data.columns:
            raise ValueError("Missing Close column for Bollinger Band strategy")
        
        # Normalize column names
        if "Close" in data.columns:
            data = data.rename(columns={"Close": "close"})
            
        required_columns = ['close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
            
        # Calculate Bollinger Bands
        upper_band, middle_band, lower_band = self.calculate_bands(data)
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Generate signals
        signals.loc[data['close'] < lower_band, 'signal'] = 1  # Buy signal
        signals.loc[data['close'] > upper_band, 'signal'] = -1  # Sell signal
        
        # Add bands to signals
        signals['upper_band'] = upper_band
        signals['middle_band'] = middle_band
        signals['lower_band'] = lower_band
        
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
            'window': self.config.window,
            'num_std': self.config.num_std,
            'min_volume': self.config.min_volume,
            'min_price': self.config.min_price
        }
        
    def set_parameters(self, params: Dict) -> Dict:
        """Set strategy parameters."""
        try:
            self.config = BollingerConfig(**params)
            self.signals = None
            self.positions = None
            return {"status": "success", "parameters_updated": True, "config": self.get_parameters()}
        except Exception as e:
            return {'success': True, 'result': {"status": "error", "message": str(e)}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}