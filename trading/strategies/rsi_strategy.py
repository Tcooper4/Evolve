"""
RSI Strategy Implementation

Simple RSI-based trading strategy using relative strength index signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import logging

from .rsi_utils import (
    calculate_rsi, 
    generate_rsi_signals_core, 
    validate_rsi_parameters,
    get_default_rsi_parameters
)

logger = logging.getLogger(__name__)

class RSIStrategy:
    """RSI-based trading strategy."""
    
    def __init__(self, rsi_period: int = 14, oversold_threshold: float = 30, 
                 overbought_threshold: float = 70):
        """Initialize RSI strategy.
        
        Args:
            rsi_period: Period for RSI calculation
            oversold_threshold: RSI threshold for oversold condition
            overbought_threshold: RSI threshold for overbought condition
        """
        # Validate parameters
        is_valid, error_msg = validate_rsi_parameters(rsi_period, oversold_threshold, overbought_threshold)
        if not is_valid:
            raise ValueError(f"Invalid RSI parameters: {error_msg}")
        
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        
    def calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """Calculate RSI for the given data using shared utilities.
        
        Args:
            data: Price data with 'Close' column
            
        Returns:
            RSI values
        """
        return calculate_rsi(data['Close'], self.rsi_period)
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
        """Generate trading signals based on RSI.
        
        Args:
            data: Price data
            **kwargs: Additional parameters
            
        Returns:
            List of trading signals
        """
        try:
            # Calculate RSI
            data = data.copy()
            data['RSI'] = self.calculate_rsi(data)
            
            signals = []
            
            for i in range(1, len(data)):
                current_rsi = data['RSI'].iloc[i]
                prev_rsi = data['RSI'].iloc[i-1]
                current_price = data['Close'].iloc[i]
                timestamp = data.index[i] if hasattr(data.index[i], 'to_pydatetime') else datetime.now()
                
                # Generate signals based on RSI thresholds
                if current_rsi < self.oversold_threshold and prev_rsi >= self.oversold_threshold:
                    # Oversold condition - buy signal
                    signals.append({
                        'timestamp': timestamp,
                        'signal_type': 'buy',
                        'confidence': 0.7,
                        'price': current_price,
                        'strategy_name': 'RSI',
                        'parameters': {
                            'rsi_value': current_rsi,
                            'rsi_period': self.rsi_period,
                            'oversold_threshold': self.oversold_threshold
                        },
                        'reasoning': f'RSI oversold ({current_rsi:.2f} < {self.oversold_threshold})'
                    })
                    
                elif current_rsi > self.overbought_threshold and prev_rsi <= self.overbought_threshold:
                    # Overbought condition - sell signal
                    signals.append({
                        'timestamp': timestamp,
                        'signal_type': 'sell',
                        'confidence': 0.7,
                        'price': current_price,
                        'strategy_name': 'RSI',
                        'parameters': {
                            'rsi_value': current_rsi,
                            'rsi_period': self.rsi_period,
                            'overbought_threshold': self.overbought_threshold
                        },
                        'reasoning': f'RSI overbought ({current_rsi:.2f} > {self.overbought_threshold})'
                    })
            
            logger.info(f"Generated {len(signals)} RSI signals")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating RSI signals: {e}")
            return []
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters.
        
        Returns:
            Strategy parameters
        """
        return {
            'rsi_period': self.rsi_period,
            'oversold_threshold': self.oversold_threshold,
            'overbought_threshold': self.overbought_threshold
        }
    
    def set_parameters(self, **kwargs):
        """Set strategy parameters.
        
        Args:
            **kwargs: Parameters to set
        """
        if 'rsi_period' in kwargs:
            self.rsi_period = kwargs['rsi_period']
        if 'oversold_threshold' in kwargs:
            self.oversold_threshold = kwargs['oversold_threshold']
        if 'overbought_threshold' in kwargs:
            self.overbought_threshold = kwargs['overbought_threshold'] 