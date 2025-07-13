"""
Ensemble Methods

This module contains methods for combining multiple strategy signals into
ensemble predictions.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

import numpy as np
from sklearn.ensemble import VotingRegressor

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


@dataclass
class HybridSignal:
    """Combined hybrid signal."""

    signal_type: SignalType
    confidence: float
    predicted_return: float
    position_size: float
    risk_score: float
    strategy_weights: Dict[str, float]
    individual_signals: List[StrategySignal]
    timestamp: datetime
    metadata: Dict[str, Any]


def combine_weighted_average(
    signals: List[StrategySignal], strategy_weights: Dict[str, float], confidence_threshold: float = 0.6
) -> HybridSignal:
    """Combine signals using weighted average method.

    Args:
        signals: List of strategy signals
        strategy_weights: Weights for each strategy
        confidence_threshold: Minimum confidence for signal generation

    Returns:
        Combined hybrid signal
    """
    try:
        if not signals:
            return create_fallback_hybrid_signal()

        # Calculate weighted average of predicted returns
        total_weight = 0
        weighted_return = 0
        weighted_confidence = 0
        weighted_risk = 0

        for signal in signals:
            weight = strategy_weights.get(signal.strategy_name, 1.0 / len(signals))
            total_weight += weight

            weighted_return += signal.predicted_return * weight
            weighted_confidence += signal.confidence * weight
            weighted_risk += signal.risk_score * weight

        # Normalize by total weight
        if total_weight > 0:
            avg_return = weighted_return / total_weight
            avg_confidence = weighted_confidence / total_weight
            avg_risk = weighted_risk / total_weight
        else:
            avg_return = 0.0
            avg_confidence = 0.5
            avg_risk = 1.0

        # Determine signal type based on average return and confidence
        if avg_confidence < confidence_threshold:
            signal_type = SignalType.HOLD
        elif avg_return > 0.05:  # 5% annual return threshold
            signal_type = SignalType.BUY if avg_return < 0.15 else SignalType.STRONG_BUY
        elif avg_return < -0.05:  # -5% annual return threshold
            signal_type = SignalType.SELL if avg_return > -0.15 else SignalType.STRONG_SELL
        else:
            signal_type = SignalType.HOLD

        # Calculate position size based on confidence and risk
        position_size = calculate_ensemble_position_size(avg_confidence, avg_risk)

        return HybridSignal(
            signal_type=signal_type,
            confidence=avg_confidence,
            predicted_return=avg_return,
            position_size=position_size,
            risk_score=avg_risk,
            strategy_weights=strategy_weights,
            individual_signals=signals,
            timestamp=datetime.now(),
            metadata={
                "method": "weighted_average",
                "total_weight": total_weight,
                "confidence_threshold": confidence_threshold,
            },
        )

    except Exception as e:
        logger.error(f"Error in weighted average combination: {e}")
        return create_fallback_hybrid_signal()


def combine_voting(
    signals: List[StrategySignal], strategy_weights: Dict[str, float], confidence_threshold: float = 0.6
) -> HybridSignal:
    """Combine signals using voting method.

    Args:
        signals: List of strategy signals
        strategy_weights: Weights for each strategy
        confidence_threshold: Minimum confidence for signal generation

    Returns:
        Combined hybrid signal
    """
    try:
        if not signals:
            return create_fallback_hybrid_signal()

        # Count votes for each signal type
        vote_counts = {
            SignalType.BUY: 0,
            SignalType.SELL: 0,
            SignalType.HOLD: 0,
            SignalType.STRONG_BUY: 0,
            SignalType.STRONG_SELL: 0,
        }

        weighted_confidence = 0
        weighted_return = 0
        weighted_risk = 0
        total_weight = 0

        for signal in signals:
            weight = strategy_weights.get(signal.strategy_name, 1.0 / len(signals))
            total_weight += weight

            # Add weighted vote
            vote_counts[signal.signal_type] += weight

            # Accumulate weighted metrics
            weighted_confidence += signal.confidence * weight
            weighted_return += signal.predicted_return * weight
            weighted_risk += signal.risk_score * weight

        # Determine winning signal type
        winning_signal = max(vote_counts.items(), key=lambda x: x[1])[0]

        # Calculate average metrics
        if total_weight > 0:
            avg_confidence = weighted_confidence / total_weight
            avg_return = weighted_return / total_weight
            avg_risk = weighted_risk / total_weight
        else:
            avg_confidence = 0.5
            avg_return = 0.0
            avg_risk = 1.0

        # Adjust confidence based on vote distribution
        vote_ratio = vote_counts[winning_signal] / total_weight
        adjusted_confidence = avg_confidence * vote_ratio

        # Apply confidence threshold
        if adjusted_confidence < confidence_threshold:
            final_signal_type = SignalType.HOLD
        else:
            final_signal_type = winning_signal

        # Calculate position size
        position_size = calculate_ensemble_position_size(adjusted_confidence, avg_risk)

        return HybridSignal(
            signal_type=final_signal_type,
            confidence=adjusted_confidence,
            predicted_return=avg_return,
            position_size=position_size,
            risk_score=avg_risk,
            strategy_weights=strategy_weights,
            individual_signals=signals,
            timestamp=datetime.now(),
            metadata={
                "method": "voting",
                "vote_counts": {k.value: v for k, v in vote_counts.items()},
                "winning_signal": winning_signal.value,
                "vote_ratio": vote_ratio,
            },
        )

    except Exception as e:
        logger.error(f"Error in voting combination: {e}")
        return create_fallback_hybrid_signal()


def combine_ensemble_model(
    signals: List[StrategySignal],
    ensemble_model: VotingRegressor,
    strategy_weights: Dict[str, float],
    confidence_threshold: float = 0.6,
) -> HybridSignal:
    """Combine signals using a trained ensemble model.

    Args:
        signals: List of strategy signals
        ensemble_model: Trained ensemble model
        strategy_weights: Weights for each strategy
        confidence_threshold: Minimum confidence for signal generation

    Returns:
        Combined hybrid signal
    """
    try:
        if not signals or ensemble_model is None:
            return create_fallback_hybrid_signal()

        # Prepare features for ensemble model
        features = []
        for signal in signals:
            signal_features = [
                signal.confidence,
                signal.predicted_return,
                signal.risk_score,
                1.0 if signal.signal_type == SignalType.BUY else 0.0,
                1.0 if signal.signal_type == SignalType.SELL else 0.0,
                1.0 if signal.signal_type == SignalType.HOLD else 0.0,
            ]
            features.append(signal_features)

        # Get ensemble prediction
        features_array = np.array(features).reshape(1, -1)
        predicted_return = ensemble_model.predict(features_array)[0]

        # Calculate average confidence and risk
        avg_confidence = np.mean([s.confidence for s in signals])
        avg_risk = np.mean([s.risk_score for s in signals])

        # Determine signal type
        if avg_confidence < confidence_threshold:
            signal_type = SignalType.HOLD
        elif predicted_return > 0.05:
            signal_type = SignalType.BUY if predicted_return < 0.15 else SignalType.STRONG_BUY
        elif predicted_return < -0.05:
            signal_type = SignalType.SELL if predicted_return > -0.15 else SignalType.STRONG_SELL
        else:
            signal_type = SignalType.HOLD

        # Calculate position size
        position_size = calculate_ensemble_position_size(avg_confidence, avg_risk)

        return HybridSignal(
            signal_type=signal_type,
            confidence=avg_confidence,
            predicted_return=predicted_return,
            position_size=position_size,
            risk_score=avg_risk,
            strategy_weights=strategy_weights,
            individual_signals=signals,
            timestamp=datetime.now(),
            metadata={
                "method": "ensemble_model",
                "model_type": type(ensemble_model).__name__,
                "predicted_return": predicted_return,
            },
        )

    except Exception as e:
        logger.error(f"Error in ensemble model combination: {e}")
        return create_fallback_hybrid_signal()


def calculate_ensemble_position_size(confidence: float, risk_score: float, max_position_size: float = 1.0) -> float:
    """Calculate position size for ensemble signal.

    Args:
        confidence: Ensemble confidence
        risk_score: Average risk score
        max_position_size: Maximum position size

    Returns:
        Position size as fraction of portfolio
    """
    # Base position size from confidence
    base_size = confidence * max_position_size

    # Adjust for risk (reduce size for higher risk)
    risk_adjustment = 1.0 - (risk_score * 0.5)

    return base_size * risk_adjustment


def create_fallback_hybrid_signal() -> HybridSignal:
    """Create a fallback hybrid signal when combination fails.

    Returns:
        Fallback hybrid signal
    """
    return HybridSignal(
        signal_type=SignalType.HOLD,
        confidence=0.3,
        predicted_return=0.0,
        position_size=0.0,
        risk_score=1.0,
        strategy_weights={},
        individual_signals=[],
        timestamp=datetime.now(),
        metadata={"error": "Signal combination failed"},
    )
