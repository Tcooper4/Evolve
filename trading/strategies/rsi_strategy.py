"""
RSI Strategy Implementation

Simple RSI-based trading strategy using relative strength index signals.
Enhanced with RSI crossover logging and range filter parameters.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
import json

from .rsi_utils import (
    calculate_rsi, 
    generate_rsi_signals_core, 
    validate_rsi_parameters,
    get_default_rsi_parameters
)

logger = logging.getLogger(__name__)

class RSIStrategy:
    """RSI-based trading strategy with enhanced logging and range filters."""
    
    def __init__(self, rsi_period: int = 14, oversold_threshold: float = 30, 
                 overbought_threshold: float = 70, enable_range_filter: bool = True,
                 min_rsi_range: float = 10.0, max_rsi_range: float = 90.0,
                 log_crossovers: bool = True):
        """Initialize RSI strategy.
        
        Args:
            rsi_period: Period for RSI calculation
            oversold_threshold: RSI threshold for oversold condition
            overbought_threshold: RSI threshold for overbought condition
            enable_range_filter: Whether to enable RSI range filtering
            min_rsi_range: Minimum RSI value to consider valid
            max_rsi_range: Maximum RSI value to consider valid
            log_crossovers: Whether to log RSI crossover events
        """
        # Validate parameters
        is_valid, error_msg = validate_rsi_parameters(rsi_period, oversold_threshold, overbought_threshold)
        if not is_valid:
            raise ValueError(f"Invalid RSI parameters: {error_msg}")
        
        # Validate range filter parameters
        if min_rsi_range >= max_rsi_range:
            raise ValueError("min_rsi_range must be less than max_rsi_range")
        
        if min_rsi_range < 0 or max_rsi_range > 100:
            raise ValueError("RSI range must be between 0 and 100")
        
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.enable_range_filter = enable_range_filter
        self.min_rsi_range = min_rsi_range
        self.max_rsi_range = max_rsi_range
        self.log_crossovers = log_crossovers
        
        # Initialize crossover logging
        self.crossover_log: List[Dict[str, Any]] = []
        
    def calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """Calculate RSI for the given data using shared utilities.
        
        Args:
            data: Price data with 'Close' column
            
        Returns:
            RSI values
        """
        return calculate_rsi(data['Close'], self.rsi_period)
    
    def log_rsi_crossover(self, timestamp: datetime, rsi_value: float, 
                         crossover_type: str, price: float, 
                         additional_info: Dict[str, Any] = None):
        """Log RSI crossover events with detailed information.
        
        Args:
            timestamp: Time of the crossover
            rsi_value: RSI value at crossover
            crossover_type: Type of crossover ('oversold', 'overbought', 'range_exit')
            price: Price at crossover
            additional_info: Additional information to log
        """
        if not self.log_crossovers:
            return
        
        crossover_event = {
            'timestamp': timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp),
            'rsi_value': round(rsi_value, 4),
            'crossover_type': crossover_type,
            'price': round(price, 4),
            'strategy_params': {
                'rsi_period': self.rsi_period,
                'oversold_threshold': self.oversold_threshold,
                'overbought_threshold': self.overbought_threshold,
                'enable_range_filter': self.enable_range_filter,
                'min_rsi_range': self.min_rsi_range,
                'max_rsi_range': self.max_rsi_range
            },
            'additional_info': additional_info or {}
        }
        
        self.crossover_log.append(crossover_event)
        
        # Log to console with emoji indicators
        emoji_map = {
            'oversold': '📈',
            'overbought': '📉',
            'range_exit': '⚠️',
            'range_enter': '🔄'
        }
        
        emoji = emoji_map.get(crossover_type, '📊')
        logger.info(f"{emoji} RSI Crossover: {crossover_type.upper()} at {timestamp} | "
                   f"RSI: {rsi_value:.2f} | Price: ${price:.2f}")
    
    def is_rsi_in_valid_range(self, rsi_value: float) -> bool:
        """Check if RSI value is within the valid range.
        
        Args:
            rsi_value: RSI value to check
            
        Returns:
            True if RSI is in valid range, False otherwise
        """
        if not self.enable_range_filter:
            return True
        
        return self.min_rsi_range <= rsi_value <= self.max_rsi_range
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
        """Generate trading signals based on RSI with enhanced logging.
        
        Args:
            data: Price data
            **kwargs: Additional parameters
            
        Returns:
            List of trading signals
        """
        # Edge case fallback logic
        if data is None or data.empty or 'Close' not in data.columns:
            logger.warning("Invalid data provided to RSI strategy: data is None, empty, or missing 'Close' column")
            return []  # Return empty signals list
        
        try:
            # Calculate RSI
            data = data.copy()
            data['RSI'] = self.calculate_rsi(data)
            
            signals = []
            crossover_count = 0
            
            for i in range(1, len(data)):
                current_rsi = data['RSI'].iloc[i]
                prev_rsi = data['RSI'].iloc[i-1]
                current_price = data['Close'].iloc[i]
                timestamp = data.index[i] if hasattr(data.index[i], 'to_pydatetime') else datetime.now()
                
                # Skip if RSI is not in valid range
                if not self.is_rsi_in_valid_range(current_rsi):
                    if self.log_crossovers and self.is_rsi_in_valid_range(prev_rsi):
                        self.log_rsi_crossover(
                            timestamp=timestamp,
                            rsi_value=current_rsi,
                            crossover_type='range_exit',
                            price=current_price,
                            additional_info={'prev_rsi': round(prev_rsi, 4)}
                        )
                    continue
                
                # Check for range entry if previously outside range
                if self.log_crossovers and not self.is_rsi_in_valid_range(prev_rsi):
                    self.log_rsi_crossover(
                        timestamp=timestamp,
                        rsi_value=current_rsi,
                        crossover_type='range_enter',
                        price=current_price,
                        additional_info={'prev_rsi': round(prev_rsi, 4)}
                    )
                
                # Generate signals based on RSI thresholds
                if current_rsi < self.oversold_threshold and prev_rsi >= self.oversold_threshold:
                    # Oversold condition - buy signal
                    crossover_count += 1
                    self.log_rsi_crossover(
                        timestamp=timestamp,
                        rsi_value=current_rsi,
                        crossover_type='oversold',
                        price=current_price,
                        additional_info={'signal_type': 'buy', 'confidence': 0.7}
                    )
                    
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
                        }
                    })
                    
                elif current_rsi > self.overbought_threshold and prev_rsi <= self.overbought_threshold:
                    # Overbought condition - sell signal
                    crossover_count += 1
                    self.log_rsi_crossover(
                        timestamp=timestamp,
                        rsi_value=current_rsi,
                        crossover_type='overbought',
                        price=current_price,
                        additional_info={'signal_type': 'sell', 'confidence': 0.7}
                    )
                    
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
                        }
                    })
            
            logger.info(f"RSI Strategy generated {len(signals)} signals with {crossover_count} crossovers")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating RSI signals: {e}")
            return []  # Return empty signals list on error
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters including range filter settings.
        
        Returns:
            Strategy parameters
        """
        return {
            'rsi_period': self.rsi_period,
            'oversold_threshold': self.oversold_threshold,
            'overbought_threshold': self.overbought_threshold,
            'enable_range_filter': self.enable_range_filter,
            'min_rsi_range': self.min_rsi_range,
            'max_rsi_range': self.max_rsi_range,
            'log_crossovers': self.log_crossovers
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
        if 'enable_range_filter' in kwargs:
            self.enable_range_filter = kwargs['enable_range_filter']
        if 'min_rsi_range' in kwargs:
            self.min_rsi_range = kwargs['min_rsi_range']
        if 'max_rsi_range' in kwargs:
            self.max_rsi_range = kwargs['max_rsi_range']
        if 'log_crossovers' in kwargs:
            self.log_crossovers = kwargs['log_crossovers']
    
    def get_crossover_log(self) -> List[Dict[str, Any]]:
        """Get the crossover log.
        
        Returns:
            List of crossover events
        """
        return self.crossover_log.copy()
    
    def clear_crossover_log(self):
        """Clear the crossover log."""
        self.crossover_log.clear()
    
    def export_crossover_log(self, filepath: str):
        """Export crossover log to JSON file.
        
        Args:
            filepath: Path to save the log file
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.crossover_log, f, indent=2, default=str)
            logger.info(f"Exported crossover log to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export crossover log: {e}")
    
    def get_crossover_statistics(self) -> Dict[str, Any]:
        """Get statistics about RSI crossovers.
        
        Returns:
            Dictionary with crossover statistics
        """
        if not self.crossover_log:
            return {}
        
        crossover_types = [event['crossover_type'] for event in self.crossover_log]
        rsi_values = [event['rsi_value'] for event in self.crossover_log]
        
        stats = {
            'total_crossovers': len(self.crossover_log),
            'crossover_types': {
                crossover_type: crossover_types.count(crossover_type)
                for crossover_type in set(crossover_types)
            },
            'rsi_statistics': {
                'min': min(rsi_values),
                'max': max(rsi_values),
                'mean': np.mean(rsi_values),
                'std': np.std(rsi_values)
            },
            'date_range': {
                'start': min(event['timestamp'] for event in self.crossover_log),
                'end': max(event['timestamp'] for event in self.crossover_log)
            }
        }
        
        return stats 