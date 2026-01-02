"""
Strategy Implementations

This module contains individual strategy implementations for the multi-strategy
hybrid engine.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Signal types for strategy outputs."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


@dataclass
class StrategySignal:
    """Individual strategy signal."""

    strategy_name: str
    signal_type: SignalType
    confidence: float
    predicted_return: float
    position_size: float
    risk_score: float
    timestamp: datetime
    metadata: Dict[str, Any]


def momentum_strategy(
    data: pd.DataFrame, max_position_size: float = 1.0
) -> StrategySignal:
    """Momentum-based strategy.

    Args:
        data: Market data
        max_position_size: Maximum position size

    Returns:
        Strategy signal
    """
    try:
        # Calculate momentum indicators
        returns = data["Close"].pct_change()
        momentum_5 = returns.rolling(5).mean()
        momentum_20 = returns.rolling(20).mean()

        # Current momentum
        current_momentum = momentum_5.iloc[-1]
        long_momentum = momentum_20.iloc[-1]

        # Signal generation
        if current_momentum > 0 and long_momentum > 0:
            signal_type = SignalType.BUY
            confidence = min(0.9, abs(current_momentum) * 10)
            predicted_return = current_momentum * 252  # Annualized
        elif current_momentum < 0 and long_momentum < 0:
            signal_type = SignalType.SELL
            confidence = min(0.9, abs(current_momentum) * 10)
            predicted_return = current_momentum * 252
        else:
            signal_type = SignalType.HOLD
            confidence = 0.5
            predicted_return = 0.0

        # Risk score based on volatility
        volatility = returns.rolling(20).std().iloc[-1]
        risk_score = min(1.0, volatility * np.sqrt(252))

        # Position sizing
        position_size = calculate_position_size(
            confidence, risk_score, max_position_size
        )

        return StrategySignal(
            strategy_name="momentum",
            signal_type=signal_type,
            confidence=confidence,
            predicted_return=predicted_return,
            position_size=position_size,
            risk_score=risk_score,
            timestamp=datetime.now(),
            metadata={
                "current_momentum": current_momentum,
                "long_momentum": long_momentum,
                "volatility": volatility,
            },
        )

    except Exception as e:
        logger.error(f"Error in momentum strategy: {e}")
        return create_fallback_signal("momentum")


def mean_reversion_strategy(
    data: pd.DataFrame, max_position_size: float = 1.0
) -> StrategySignal:
    """Mean reversion strategy.

    Args:
        data: Market data
        max_position_size: Maximum position size

    Returns:
        Strategy signal
    """
    try:
        # Calculate Bollinger Bands
        sma = data["Close"].rolling(20).mean()
        std = data["Close"].rolling(20).std()
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)

        current_price = data["Close"].iloc[-1]
        current_sma = sma.iloc[-1]

        # Position within bands with safe division
        band_range = upper_band.iloc[-1] - lower_band.iloc[-1]
        if band_range > 1e-10:
            band_position = (current_price - lower_band.iloc[-1]) / band_range
        else:
            band_position = 0.5  # Neutral position if no range

        # Signal generation
        if band_position < 0.2:  # Near lower band
            signal_type = SignalType.BUY
            confidence = 0.8
            predicted_return = 0.05  # Expected 5% return
        elif band_position > 0.8:  # Near upper band
            signal_type = SignalType.SELL
            confidence = 0.8
            predicted_return = -0.05  # Expected -5% return
        else:
            signal_type = SignalType.HOLD
            confidence = 0.5
            predicted_return = 0.0

        # Risk score based on volatility
        volatility = data["Close"].pct_change().rolling(20).std().iloc[-1]
        risk_score = min(1.0, volatility * np.sqrt(252))

        # Position sizing
        position_size = calculate_position_size(
            confidence, risk_score, max_position_size
        )

        return StrategySignal(
            strategy_name="mean_reversion",
            signal_type=signal_type,
            confidence=confidence,
            predicted_return=predicted_return,
            position_size=position_size,
            risk_score=risk_score,
            timestamp=datetime.now(),
            metadata={
                "band_position": band_position,
                "current_sma": current_sma,
                "volatility": volatility,
            },
        )

    except Exception as e:
        logger.error(f"Error in mean reversion strategy: {e}")
        return create_fallback_signal("mean_reversion")


def volatility_breakout_strategy(
    data: pd.DataFrame, max_position_size: float = 1.0
) -> StrategySignal:
    """Volatility breakout strategy.

    Args:
        data: Market data
        max_position_size: Maximum position size

    Returns:
        Strategy signal
    """
    try:
        # Calculate volatility indicators
        returns = data["Close"].pct_change()
        current_volatility = returns.rolling(20).std().iloc[-1]
        avg_volatility = returns.rolling(60).std().iloc[-1]

        # Volatility ratio
        vol_ratio = current_volatility / avg_volatility if avg_volatility > 0 else 1.0

        # Price momentum
        price_momentum = data["Close"].pct_change(5).iloc[-1]

        # Signal generation
        if vol_ratio > 1.5 and price_momentum > 0:  # High volatility + upward momentum
            signal_type = SignalType.BUY
            confidence = min(0.9, vol_ratio * 0.3)
            predicted_return = price_momentum * 252
        elif (
            vol_ratio > 1.5 and price_momentum < 0
        ):  # High volatility + downward momentum
            signal_type = SignalType.SELL
            confidence = min(0.9, vol_ratio * 0.3)
            predicted_return = price_momentum * 252
        else:
            signal_type = SignalType.HOLD
            confidence = 0.5
            predicted_return = 0.0

        # Risk score based on volatility
        risk_score = min(1.0, current_volatility * np.sqrt(252))

        # Position sizing
        position_size = calculate_position_size(
            confidence, risk_score, max_position_size
        )

        return StrategySignal(
            strategy_name="volatility_breakout",
            signal_type=signal_type,
            confidence=confidence,
            predicted_return=predicted_return,
            position_size=position_size,
            risk_score=risk_score,
            timestamp=datetime.now(),
            metadata={
                "vol_ratio": vol_ratio,
                "current_volatility": current_volatility,
                "price_momentum": price_momentum,
            },
        )

    except Exception as e:
        logger.error(f"Error in volatility breakout strategy: {e}")
        return create_fallback_signal("volatility_breakout")


def trend_following_strategy(
    data: pd.DataFrame, max_position_size: float = 1.0
) -> StrategySignal:
    """Trend following strategy.

    Args:
        data: Market data
        max_position_size: Maximum position size

    Returns:
        Strategy signal
    """
    try:
        # Calculate moving averages
        sma_short = data["Close"].rolling(10).mean()
        sma_long = data["Close"].rolling(50).mean()

        data["Close"].iloc[-1]
        current_sma_short = sma_short.iloc[-1]
        current_sma_long = sma_long.iloc[-1]

        # Trend strength with safe division
        if current_sma_long > 1e-10:
            trend_strength = abs(current_sma_short - current_sma_long) / current_sma_long
        else:
            trend_strength = 0.0

        # Signal generation
        if current_sma_short > current_sma_long and trend_strength > 0.02:
            signal_type = SignalType.BUY
            confidence = min(0.9, trend_strength * 20)
            predicted_return = 0.08  # Expected 8% return
        elif current_sma_short < current_sma_long and trend_strength > 0.02:
            signal_type = SignalType.SELL
            confidence = min(0.9, trend_strength * 20)
            predicted_return = -0.08  # Expected -8% return
        else:
            signal_type = SignalType.HOLD
            confidence = 0.5
            predicted_return = 0.0

        # Risk score based on volatility
        volatility = data["Close"].pct_change().rolling(20).std().iloc[-1]
        risk_score = min(1.0, volatility * np.sqrt(252))

        # Position sizing
        position_size = calculate_position_size(
            confidence, risk_score, max_position_size
        )

        return StrategySignal(
            strategy_name="trend_following",
            signal_type=signal_type,
            confidence=confidence,
            predicted_return=predicted_return,
            position_size=position_size,
            risk_score=risk_score,
            timestamp=datetime.now(),
            metadata={
                "trend_strength": trend_strength,
                "sma_short": current_sma_short,
                "sma_long": current_sma_long,
            },
        )

    except Exception as e:
        logger.error(f"Error in trend following strategy: {e}")
        return create_fallback_signal("trend_following")


def volume_price_strategy(
    data: pd.DataFrame, max_position_size: float = 1.0
) -> StrategySignal:
    """Volume-price relationship strategy.

    Args:
        data: Market data
        max_position_size: Maximum position size

    Returns:
        Strategy signal
    """
    try:
        # Calculate volume indicators
        avg_volume = data["Volume"].rolling(20).mean()
        current_volume = data["Volume"].iloc[-1]
        volume_ratio = (
            current_volume / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1.0
        )

        # Price change
        price_change = data["Close"].pct_change().iloc[-1]

        # Signal generation
        if volume_ratio > 1.5 and price_change > 0:  # High volume + price increase
            signal_type = SignalType.BUY
            confidence = min(0.9, volume_ratio * 0.3)
            predicted_return = price_change * 252
        elif volume_ratio > 1.5 and price_change < 0:  # High volume + price decrease
            signal_type = SignalType.SELL
            confidence = min(0.9, volume_ratio * 0.3)
            predicted_return = price_change * 252
        else:
            signal_type = SignalType.HOLD
            confidence = 0.5
            predicted_return = 0.0

        # Risk score based on volatility
        volatility = data["Close"].pct_change().rolling(20).std().iloc[-1]
        risk_score = min(1.0, volatility * np.sqrt(252))

        # Position sizing
        position_size = calculate_position_size(
            confidence, risk_score, max_position_size
        )

        return StrategySignal(
            strategy_name="volume_price",
            signal_type=signal_type,
            confidence=confidence,
            predicted_return=predicted_return,
            position_size=position_size,
            risk_score=risk_score,
            timestamp=datetime.now(),
            metadata={
                "volume_ratio": volume_ratio,
                "price_change": price_change,
                "current_volume": current_volume,
            },
        )

    except Exception as e:
        logger.error(f"Error in volume-price strategy: {e}")
        return create_fallback_signal("volume_price")


def calculate_position_size(
    confidence: float, risk_score: float, max_position_size: float
) -> float:
    """Calculate position size based on confidence and risk.

    Args:
        confidence: Strategy confidence (0-1)
        risk_score: Risk score (0-1)
        max_position_size: Maximum position size

    Returns:
        Position size as fraction of portfolio
    """
    # Base position size from confidence
    base_size = confidence * max_position_size

    # Adjust for risk (reduce size for higher risk)
    risk_adjustment = 1.0 - (risk_score * 0.5)

    return base_size * risk_adjustment


def create_fallback_signal(strategy_name: str) -> StrategySignal:
    """Create a fallback signal when strategy fails.

    Args:
        strategy_name: Name of the strategy

    Returns:
        Fallback strategy signal
    """
    return StrategySignal(
        strategy_name=strategy_name,
        signal_type=SignalType.HOLD,
        confidence=0.3,
        predicted_return=0.0,
        position_size=0.0,
        risk_score=1.0,
        timestamp=datetime.now(),
        metadata={"error": "Strategy execution failed"},
    )
