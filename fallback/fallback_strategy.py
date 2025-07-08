"""
Fallback Strategy Module

This module provides fallback strategy implementations for when primary strategies are unavailable.
These are simplified implementations that ensure the system continues to function.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FallbackSignal:
    """Fallback trading signal."""
    timestamp: datetime
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float
    price: float
    strategy_name: str
    reasoning: str

class FallbackStrategy:
    """Base fallback strategy class."""
    
    def __init__(self, strategy_name: str = "FallbackStrategy"):
        """
        Initialize fallback strategy.
        
        Args:
            strategy_name: Name of the fallback strategy
        """
        self.strategy_name = strategy_name
        self.logger = logging.getLogger(__name__)
        self.logger.warning(f"Using fallback strategy: {strategy_name}")
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> List[FallbackSignal]:
        """
        Generate trading signals using the fallback strategy.
        
        Args:
            data: Price data
            **kwargs: Additional strategy parameters
            
        Returns:
            List[FallbackSignal]: Generated signals
        """
        self.logger.info(f"Generating signals with fallback strategy {self.strategy_name}")
        
        if data.empty:
            return []
        
        signals = []
        for i, row in data.iterrows():
            signal = self._generate_signal(row, i, data)
            if signal:
                signals.append(signal)
        
        return signals
    
    def _generate_signal(self, row: pd.Series, index: int, data: pd.DataFrame) -> Optional[FallbackSignal]:
        """
        Generate a single signal.
        
        Args:
            row: Current data row
            index: Row index
            data: Full dataset
            
        Returns:
            Optional[FallbackSignal]: Generated signal or None
        """
        # Simple fallback logic: buy on dips, sell on peaks
        if index < 20:  # Need some history
            return None
        
        recent_prices = data['Close'].iloc[index-20:index].values
        current_price = row['Close']
        
        # Calculate simple indicators
        sma_20 = np.mean(recent_prices)
        price_change = (current_price - recent_prices[0]) / recent_prices[0]
        
        # Generate signal based on price vs moving average
        if current_price < sma_20 * 0.95:  # 5% below SMA
            signal_type = 'buy'
            confidence = 0.7
            reasoning = f"Price {current_price:.2f} below SMA {sma_20:.2f}"
        elif current_price > sma_20 * 1.05:  # 5% above SMA
            signal_type = 'sell'
            confidence = 0.7
            reasoning = f"Price {current_price:.2f} above SMA {sma_20:.2f}"
        else:
            signal_type = 'hold'
            confidence = 0.5
            reasoning = f"Price {current_price:.2f} near SMA {sma_20:.2f}"
        
        return FallbackSignal(
            timestamp=row.get('timestamp', datetime.now()),
            signal_type=signal_type,
            confidence=confidence,
            price=current_price,
            strategy_name=self.strategy_name,
            reasoning=reasoning
        )

class FallbackRSIStrategy(FallbackStrategy):
    """Fallback RSI strategy implementation."""
    
    def __init__(self):
        """Initialize fallback RSI strategy."""
        super().__init__("FallbackRSI")
    
    def _generate_signal(self, row: pd.Series, index: int, data: pd.DataFrame) -> Optional[FallbackSignal]:
        """
        Generate RSI-based signal.
        
        Args:
            row: Current data row
            index: Row index
            data: Full dataset
            
        Returns:
            Optional[FallbackSignal]: Generated signal or None
        """
        if index < 14:  # Need at least 14 periods for RSI
            return None
        
        # Calculate simple RSI
        recent_prices = data['Close'].iloc[index-14:index+1].values
        price_changes = np.diff(recent_prices)
        
        gains = np.where(price_changes > 0, price_changes, 0)
        losses = np.where(price_changes < 0, -price_changes, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        current_price = row['Close']
        
        # Generate signal based on RSI
        if rsi < 30:  # Oversold
            signal_type = 'buy'
            confidence = 0.8
            reasoning = f"RSI {rsi:.1f} indicates oversold condition"
        elif rsi > 70:  # Overbought
            signal_type = 'sell'
            confidence = 0.8
            reasoning = f"RSI {rsi:.1f} indicates overbought condition"
        else:
            signal_type = 'hold'
            confidence = 0.6
            reasoning = f"RSI {rsi:.1f} in neutral range"
        
        return FallbackSignal(
            timestamp=row.get('timestamp', datetime.now()),
            signal_type=signal_type,
            confidence=confidence,
            price=current_price,
            strategy_name=self.strategy_name,
            reasoning=reasoning
        )

class FallbackMACDStrategy(FallbackStrategy):
    """Fallback MACD strategy implementation."""
    
    def __init__(self):
        """Initialize fallback MACD strategy."""
        super().__init__("FallbackMACD")
    
    def _generate_signal(self, row: pd.Series, index: int, data: pd.DataFrame) -> Optional[FallbackSignal]:
        """
        Generate MACD-based signal.
        
        Args:
            row: Current data row
            index: Row index
            data: Full dataset
            
        Returns:
            Optional[FallbackSignal]: Generated signal or None
        """
        if index < 26:  # Need at least 26 periods for MACD
            return None
        
        # Calculate simple MACD
        recent_prices = data['Close'].iloc[index-26:index+1].values
        
        # EMA 12
        ema_12 = self._calculate_ema(recent_prices, 12)
        
        # EMA 26
        ema_26 = self._calculate_ema(recent_prices, 26)
        
        # MACD line
        macd_line = ema_12 - ema_26
        
        # Signal line (EMA of MACD)
        if index >= 35:  # Need more history for signal line
            macd_values = []
            for i in range(index-9, index+1):
                prices = data['Close'].iloc[i-26:i+1].values
                ema_12_i = self._calculate_ema(prices, 12)
                ema_26_i = self._calculate_ema(prices, 26)
                macd_values.append(ema_12_i - ema_26_i)
            
            signal_line = self._calculate_ema(np.array(macd_values), 9)
            
            current_price = row['Close']
            
            # Generate signal based on MACD crossover
            if macd_line > signal_line and macd_line > 0:
                signal_type = 'buy'
                confidence = 0.75
                reasoning = f"MACD {macd_line:.3f} above signal {signal_line:.3f}"
            elif macd_line < signal_line and macd_line < 0:
                signal_type = 'sell'
                confidence = 0.75
                reasoning = f"MACD {macd_line:.3f} below signal {signal_line:.3f}"
            else:
                signal_type = 'hold'
                confidence = 0.6
                reasoning = f"MACD {macd_line:.3f} near signal {signal_line:.3f}"
            
            return FallbackSignal(
                timestamp=row.get('timestamp', datetime.now()),
                signal_type=signal_type,
                confidence=confidence,
                price=current_price,
                strategy_name=self.strategy_name,
                reasoning=reasoning
            )
        
        return None
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """
        Calculate Exponential Moving Average.
        
        Args:
            prices: Price array
            period: EMA period
            
        Returns:
            float: EMA value
        """
        if len(prices) < period:
            return prices[-1]
        
        alpha = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema

class FallbackBollingerStrategy(FallbackStrategy):
    """Fallback Bollinger Bands strategy implementation."""
    
    def __init__(self):
        """Initialize fallback Bollinger Bands strategy."""
        super().__init__("FallbackBollinger")
    
    def _generate_signal(self, row: pd.Series, index: int, data: pd.DataFrame) -> Optional[FallbackSignal]:
        """
        Generate Bollinger Bands-based signal.
        
        Args:
            row: Current data row
            index: Row index
            data: Full dataset
            
        Returns:
            Optional[FallbackSignal]: Generated signal or None
        """
        if index < 20:  # Need at least 20 periods for Bollinger Bands
            return None
        
        # Calculate Bollinger Bands
        recent_prices = data['Close'].iloc[index-20:index+1].values
        
        # Middle band (SMA)
        middle_band = np.mean(recent_prices)
        
        # Standard deviation
        std_dev = np.std(recent_prices)
        
        # Upper and lower bands
        upper_band = middle_band + (2 * std_dev)
        lower_band = middle_band - (2 * std_dev)
        
        current_price = row['Close']
        
        # Generate signal based on price position relative to bands
        if current_price <= lower_band:
            signal_type = 'buy'
            confidence = 0.8
            reasoning = f"Price {current_price:.2f} at lower band {lower_band:.2f}"
        elif current_price >= upper_band:
            signal_type = 'sell'
            confidence = 0.8
            reasoning = f"Price {current_price:.2f} at upper band {upper_band:.2f}"
        else:
            signal_type = 'hold'
            confidence = 0.6
            reasoning = f"Price {current_price:.2f} within bands"
        
        return FallbackSignal(
            timestamp=row.get('timestamp', datetime.now()),
            signal_type=signal_type,
            confidence=confidence,
            price=current_price,
            strategy_name=self.strategy_name,
            reasoning=reasoning
        )

class FallbackSMAStrategy(FallbackStrategy):
    """Fallback Simple Moving Average strategy implementation."""
    
    def __init__(self):
        """Initialize fallback SMA strategy."""
        super().__init__("FallbackSMA")
    
    def _generate_signal(self, row: pd.Series, index: int, data: pd.DataFrame) -> Optional[FallbackSignal]:
        """
        Generate SMA-based signal.
        
        Args:
            row: Current data row
            index: Row index
            data: Full dataset
            
        Returns:
            Optional[FallbackSignal]: Generated signal or None
        """
        if index < 50:  # Need at least 50 periods for SMA crossover
            return None
        
        # Calculate SMAs
        recent_prices = data['Close'].iloc[index-50:index+1].values
        
        sma_20 = np.mean(recent_prices[-20:])
        sma_50 = np.mean(recent_prices)
        
        current_price = row['Close']
        
        # Generate signal based on SMA crossover
        if sma_20 > sma_50 and current_price > sma_20:
            signal_type = 'buy'
            confidence = 0.75
            reasoning = f"SMA20 {sma_20:.2f} above SMA50 {sma_50:.2f}"
        elif sma_20 < sma_50 and current_price < sma_20:
            signal_type = 'sell'
            confidence = 0.75
            reasoning = f"SMA20 {sma_20:.2f} below SMA50 {sma_50:.2f}"
        else:
            signal_type = 'hold'
            confidence = 0.6
            reasoning = f"SMAs in neutral position"
        
        return FallbackSignal(
            timestamp=row.get('timestamp', datetime.now()),
            signal_type=signal_type,
            confidence=confidence,
            price=current_price,
            strategy_name=self.strategy_name,
            reasoning=reasoning
        )

def get_fallback_strategy(strategy_type: str) -> FallbackStrategy:
    """
    Get a fallback strategy instance.
    
    Args:
        strategy_type: Type of strategy needed
        
    Returns:
        FallbackStrategy: Appropriate fallback strategy
    """
    strategy_map = {
        'rsi': FallbackRSIStrategy,
        'macd': FallbackMACDStrategy,
        'bollinger': FallbackBollingerStrategy,
        'sma': FallbackSMAStrategy,
        'default': FallbackStrategy
    }
    
    strategy_class = strategy_map.get(strategy_type.lower(), FallbackStrategy)
    return strategy_class() 