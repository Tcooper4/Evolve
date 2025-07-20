"""
Ensemble Methods

This module contains methods for combining multiple strategy signals into
ensemble predictions with dynamic weight tuning and time-decay weighting.
"""

import logging
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Try to import scikit-learn
try:
    from sklearn.ensemble import VotingRegressor
    from sklearn.metrics import mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError as e:
    print("⚠️ scikit-learn not available. Disabling ensemble voting capabilities.")
    print(f"   Missing: {e}")
    VotingRegressor = None
    mean_squared_error = None
    SKLEARN_AVAILABLE = False

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


@dataclass
class ModelPerformance:
    """Model performance tracking for dynamic weighting."""

    strategy_name: str
    mse_history: List[float]
    timestamps: List[datetime]
    recent_performance: float
    weight: float
    last_updated: datetime


class DynamicEnsembleVoter:
    """Ensemble voter with dynamic weight tuning and time-decay weighting."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the dynamic ensemble voter.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.epsilon = self.config.get("epsilon", 1e-6)  # Small value to prevent division by zero
        self.decay_factor = self.config.get("decay_factor", 0.95)  # Time decay factor
        self.lookback_window = self.config.get("lookback_window", 30)  # Days for performance tracking
        self.min_weight = self.config.get("min_weight", 0.01)  # Minimum weight for any model
        self.max_weight = self.config.get("max_weight", 0.5)  # Maximum weight for any model

        # Performance tracking
        self.model_performance: Dict[str, ModelPerformance] = {}
        self.audit_history: List[Dict[str, Any]] = []

        # Load existing performance data if available
        self._load_performance_data()

    def _load_performance_data(self):
        """Load performance data from disk."""
        try:
            performance_file = Path("data/ensemble_performance.json")
            if performance_file.exists():
                with open(performance_file, 'r') as f:
                    data = json.load(f)

                for strategy_name, perf_data in data.items():
                    self.model_performance[strategy_name] = ModelPerformance(
                        strategy_name=strategy_name,
                        mse_history=perf_data.get("mse_history", []),
                        timestamps=[datetime.fromisoformat(ts) for ts in perf_data.get("timestamps", [])],
                        recent_performance=perf_data.get("recent_performance", 1.0),
                        weight=perf_data.get("weight", 1.0),
                        last_updated=datetime.fromisoformat(perf_data.get("last_updated", datetime.now().isoformat()))
                    )
                logger.info(f"Loaded performance data for {len(self.model_performance)} models")
        except Exception as e:
            logger.warning(f"Could not load performance data: {e}")

    def _save_performance_data(self):
        """Save performance data to disk."""
        try:
            performance_file = Path("data/ensemble_performance.json")
            performance_file.parent.mkdir(parents=True, exist_ok=True)

            data = {}
            for strategy_name, perf in self.model_performance.items():
                data[strategy_name] = {
                    "mse_history": perf.mse_history,
                    "timestamps": [ts.isoformat() for ts in perf.timestamps],
                    "recent_performance": perf.recent_performance,
                    "weight": perf.weight,
                    "last_updated": perf.last_updated.isoformat()
                }

            with open(performance_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved performance data for {len(self.model_performance)} models")
        except Exception as e:
            logger.error(f"Could not save performance data: {e}")

    def update_model_performance(
        self,
        strategy_name: str,
        actual_returns: pd.Series,
        predicted_returns: pd.Series,
        timestamp: Optional[datetime] = None
    ):
        """Update model performance with new data.

        Args:
            strategy_name: Name of the strategy/model
            actual_returns: Actual returns series
            predicted_returns: Predicted returns series
            timestamp: Timestamp for the performance update
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Calculate MSE
        if not SKLEARN_AVAILABLE:
            mse = float('inf')
        elif len(actual_returns) > 0 and len(predicted_returns) > 0:
            # Align the series
            aligned_actual, aligned_pred = actual_returns.align(predicted_returns, join='inner')
            if len(aligned_actual) > 0:
                mse = mean_squared_error(aligned_actual, aligned_pred)
            else:
                mse = float('inf')
        else:
            mse = float('inf')

        # Initialize performance tracking if not exists
        if strategy_name not in self.model_performance:
            self.model_performance[strategy_name] = ModelPerformance(
                strategy_name=strategy_name,
                mse_history=[],
                timestamps=[],
                recent_performance=1.0,
                weight=1.0,
                last_updated=timestamp
            )

        # Update performance tracking
        perf = self.model_performance[strategy_name]
        perf.mse_history.append(mse)
        perf.timestamps.append(timestamp)
        perf.last_updated = timestamp

        # Keep only recent history
        cutoff_date = timestamp - timedelta(days=self.lookback_window)
        recent_indices = [i for i, ts in enumerate(perf.timestamps) if ts >= cutoff_date]
        perf.mse_history = [perf.mse_history[i] for i in recent_indices]
        perf.timestamps = [perf.timestamps[i] for i in recent_indices]

        # Calculate recent performance (inverse of average MSE)
        if len(perf.mse_history) > 0:
            avg_mse = np.mean(perf.mse_history)
            perf.recent_performance = 1.0 / (1.0 + avg_mse)  # Transform to [0, 1]
        else:
            perf.recent_performance = 1.0

        # Save updated performance data
        self._save_performance_data()

    def _calculate_dynamic_weight(self, perf: ModelPerformance) -> float:
        """Calculate dynamic weight based on recent performance."""
        # Apply time decay
        time_diff = datetime.now() - perf.last_updated
        decay = self.decay_factor ** (time_diff.days / 30.0)  # Monthly decay

        # Combine recent performance with time decay
        dynamic_weight = perf.recent_performance * decay

        # Apply min/max constraints
        dynamic_weight = max(self.min_weight, min(self.max_weight, dynamic_weight))

        return dynamic_weight

    def get_dynamic_weights(self, strategy_names: List[str]) -> Dict[str, float]:
        """Get dynamic weights for strategies."""
        weights = {}
        total_weight = 0.0

        for strategy_name in strategy_names:
            if strategy_name in self.model_performance:
                weight = self._calculate_dynamic_weight(self.model_performance[strategy_name])
            else:
                weight = 1.0  # Default weight for new strategies

            weights[strategy_name] = weight
            total_weight += weight

        # Normalize weights
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}

        return weights

    def combine_signals_dynamic(
        self,
        signals: List[StrategySignal],
        confidence_threshold: float = 0.6,
        use_dynamic_weights: bool = True
    ) -> HybridSignal:
        """Combine signals using dynamic weighting."""
        if not signals:
            return create_fallback_hybrid_signal()

        strategy_names = [signal.strategy_name for signal in signals]

        # Get weights
        if use_dynamic_weights:
            weights = self.get_dynamic_weights(strategy_names)
        else:
            # Equal weights
            weights = {name: 1.0 / len(strategy_names) for name in strategy_names}

        # Calculate weighted averages
        total_confidence = 0.0
        total_predicted_return = 0.0
        total_position_size = 0.0
        total_risk_score = 0.0
        total_weight = 0.0

        for signal in signals:
            weight = weights.get(signal.strategy_name, 0.0)
            total_confidence += signal.confidence * weight
            total_predicted_return += signal.predicted_return * weight
            total_position_size += signal.position_size * weight
            total_risk_score += signal.risk_score * weight
            total_weight += weight

        if total_weight > 0:
            avg_confidence = total_confidence / total_weight
            avg_predicted_return = total_predicted_return / total_weight
            avg_position_size = total_position_size / total_weight
            avg_risk_score = total_risk_score / total_weight
        else:
            avg_confidence = 0.5
            avg_predicted_return = 0.0
            avg_position_size = 0.0
            avg_risk_score = 0.5

        # Determine signal type based on predicted return and confidence
        if avg_confidence >= confidence_threshold:
            if avg_predicted_return > 0.01:
                signal_type = SignalType.BUY
            elif avg_predicted_return < -0.01:
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD
        else:
            signal_type = SignalType.HOLD

        # Record audit information
        self._record_audit(signals, weights, signal_type, avg_confidence)

        return HybridSignal(
            signal_type=signal_type,
            confidence=avg_confidence,
            predicted_return=avg_predicted_return,
            position_size=avg_position_size,
            risk_score=avg_risk_score,
            strategy_weights=weights,
            individual_signals=signals,
            timestamp=datetime.now(),
            metadata={"method": "dynamic_ensemble", "confidence_threshold": confidence_threshold}
        )

    def _record_audit(self, signals: List[StrategySignal], weights: Dict[str, float],
                     final_signal: SignalType, final_confidence: float):
        """Record audit information for transparency."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "num_signals": len(signals),
            "strategy_names": [s.strategy_name for s in signals],
            "weights": weights,
            "final_signal": final_signal.value,
            "final_confidence": final_confidence,
            "individual_confidences": [s.confidence for s in signals],
            "individual_returns": [s.predicted_return for s in signals]
        }
        self.audit_history.append(audit_entry)

        # Keep only recent audit history
        if len(self.audit_history) > 1000:
            self.audit_history = self.audit_history[-1000:]

    def get_ensemble_audit_report(self) -> Dict[str, Any]:
        """Get audit report for ensemble decisions."""
        if not self.audit_history:
            return {"error": "No audit history available"}

        recent_audits = self.audit_history[-100:]  # Last 100 decisions

        # Calculate statistics
        signal_counts = {}
        confidence_stats = {
            "mean": 0.0,
            "std": 0.0,
            "min": 1.0,
            "max": 0.0
        }

        for audit in recent_audits:
            signal_type = audit["final_signal"]
            signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1

            confidence = audit["final_confidence"]
            confidence_stats["mean"] += confidence
            confidence_stats["min"] = min(confidence_stats["min"], confidence)
            confidence_stats["max"] = max(confidence_stats["max"], confidence)

        if recent_audits:
            confidence_stats["mean"] /= len(recent_audits)
            confidence_stats["std"] = np.std([a["final_confidence"] for a in recent_audits])

        return {
            "total_decisions": len(self.audit_history),
            "recent_decisions": len(recent_audits),
            "signal_distribution": signal_counts,
            "confidence_statistics": confidence_stats,
            "recent_audits": recent_audits[-10:],  # Last 10 decisions
            "model_performance": {
                name: {
                    "recent_performance": perf.recent_performance,
                    "current_weight": perf.weight,
                    "mse_history_length": len(perf.mse_history)
                }
                for name, perf in self.model_performance.items()
            }
        }

    def save_state(self, filepath: str):
        """Save ensemble voter state to file."""
        try:
            state = {
                "config": self.config,
                "model_performance": {
                    name: {
                        "mse_history": perf.mse_history,
                        "timestamps": [ts.isoformat() for ts in perf.timestamps],
                        "recent_performance": perf.recent_performance,
                        "weight": perf.weight,
                        "last_updated": perf.last_updated.isoformat()
                    }
                    for name, perf in self.model_performance.items()
                },
                "audit_history": self.audit_history
            }

            with open(filepath, 'wb') as f:
                pickle.dump(state, f)

            logger.info(f"Saved ensemble voter state to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save ensemble voter state: {e}")

    def load_state(self, filepath: str):
        """Load ensemble voter state from file."""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)

            self.config = state.get("config", {})
            self.audit_history = state.get("audit_history", [])

            # Reconstruct model performance objects
            self.model_performance = {}
            for name, perf_data in state.get("model_performance", {}).items():
                self.model_performance[name] = ModelPerformance(
                    strategy_name=name,
                    mse_history=perf_data.get("mse_history", []),
                    timestamps=[datetime.fromisoformat(ts) for ts in perf_data.get("timestamps", [])],
                    recent_performance=perf_data.get("recent_performance", 1.0),
                    weight=perf_data.get("weight", 1.0),
                    last_updated=datetime.fromisoformat(perf_data.get("last_updated", datetime.now().isoformat()))
                )

            logger.info(f"Loaded ensemble voter state from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load ensemble voter state: {e}")


def combine_weighted_average(
    signals: List[StrategySignal],
    strategy_weights: Dict[str, float],
    confidence_threshold: float = 0.6,
) -> HybridSignal:
    """Combine signals using weighted average method."""
    if not signals:
        return create_fallback_hybrid_signal()

    # Calculate weighted averages
    total_confidence = 0.0
    total_predicted_return = 0.0
    total_position_size = 0.0
    total_risk_score = 0.0
    total_weight = 0.0

    for signal in signals:
        weight = strategy_weights.get(signal.strategy_name, 0.0)
        total_confidence += signal.confidence * weight
        total_predicted_return += signal.predicted_return * weight
        total_position_size += signal.position_size * weight
        total_risk_score += signal.risk_score * weight
        total_weight += weight

    if total_weight > 0:
        avg_confidence = total_confidence / total_weight
        avg_predicted_return = total_predicted_return / total_weight
        avg_position_size = total_position_size / total_weight
        avg_risk_score = total_risk_score / total_weight
    else:
        avg_confidence = 0.5
        avg_predicted_return = 0.0
        avg_position_size = 0.0
        avg_risk_score = 0.5

    # Determine signal type
    if avg_confidence >= confidence_threshold:
        if avg_predicted_return > 0.01:
            signal_type = SignalType.BUY
        elif avg_predicted_return < -0.01:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD
    else:
        signal_type = SignalType.HOLD

    return HybridSignal(
        signal_type=signal_type,
        confidence=avg_confidence,
        predicted_return=avg_predicted_return,
        position_size=avg_position_size,
        risk_score=avg_risk_score,
        strategy_weights=strategy_weights,
        individual_signals=signals,
        timestamp=datetime.now(),
        metadata={"method": "weighted_average", "confidence_threshold": confidence_threshold}
    )


def combine_voting(
    signals: List[StrategySignal],
    strategy_weights: Dict[str, float],
    confidence_threshold: float = 0.6,
) -> HybridSignal:
    """Combine signals using voting method."""
    if not signals:
        return create_fallback_hybrid_signal()

    # Collect votes
    votes = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
    total_weight = 0.0
    confidences = []

    for signal in signals:
        weight = strategy_weights.get(signal.strategy_name, 0.0)
        total_weight += weight

        # Determine vote based on predicted return
        if signal.predicted_return > 0.01:
            votes["buy"] += weight
        elif signal.predicted_return < -0.01:
            votes["sell"] += weight
        else:
            votes["hold"] += weight

        confidences.append(signal.confidence * weight)

    # Determine winning vote
    if total_weight > 0:
        winning_vote = max(votes.items(), key=lambda x: x[1])[0]
        avg_confidence = sum(confidences) / total_weight
    else:
        winning_vote = "hold"
        avg_confidence = 0.5

    # Map vote to signal type
    signal_type_map = {
        "buy": SignalType.BUY,
        "sell": SignalType.SELL,
        "hold": SignalType.HOLD
    }

    signal_type = signal_type_map.get(winning_vote, SignalType.HOLD)

    # Calculate weighted averages for other metrics
    total_predicted_return = 0.0
    total_position_size = 0.0
    total_risk_score = 0.0

    for signal in signals:
        weight = strategy_weights.get(signal.strategy_name, 0.0)
        total_predicted_return += signal.predicted_return * weight
        total_position_size += signal.position_size * weight
        total_risk_score += signal.risk_score * weight

    if total_weight > 0:
        avg_predicted_return = total_predicted_return / total_weight
        avg_position_size = total_position_size / total_weight
        avg_risk_score = total_risk_score / total_weight
    else:
        avg_predicted_return = 0.0
        avg_position_size = 0.0
        avg_risk_score = 0.5

    return HybridSignal(
        signal_type=signal_type,
        confidence=avg_confidence,
        predicted_return=avg_predicted_return,
        position_size=avg_position_size,
        risk_score=avg_risk_score,
        strategy_weights=strategy_weights,
        individual_signals=signals,
        timestamp=datetime.now(),
        metadata={"method": "voting", "confidence_threshold": confidence_threshold}
    )


def combine_ensemble_model(
    signals: List[StrategySignal],
    ensemble_model: VotingRegressor,
    strategy_weights: Dict[str, float],
    confidence_threshold: float = 0.6,
) -> HybridSignal:
    """Combine signals using scikit-learn ensemble model."""
    if not SKLEARN_AVAILABLE or ensemble_model is None:
        logger.warning("Scikit-learn not available, falling back to weighted average")
        return combine_weighted_average(signals, strategy_weights, confidence_threshold)

    if not signals:
        return create_fallback_hybrid_signal()

    try:
        # Prepare features for ensemble model
        features = []
        for signal in signals:
            feature_vector = [
                signal.confidence,
                signal.predicted_return,
                signal.position_size,
                signal.risk_score
            ]
            features.append(feature_vector)

        # Make ensemble prediction
        if len(features) > 0:
            ensemble_prediction = ensemble_model.predict([np.mean(features, axis=0)])[0]
        else:
            ensemble_prediction = 0.0

        # Calculate weighted averages for other metrics
        total_confidence = 0.0
        total_position_size = 0.0
        total_risk_score = 0.0
        total_weight = 0.0

        for signal in signals:
            weight = strategy_weights.get(signal.strategy_name, 0.0)
            total_confidence += signal.confidence * weight
            total_position_size += signal.position_size * weight
            total_risk_score += signal.risk_score * weight
            total_weight += weight

        if total_weight > 0:
            avg_confidence = total_confidence / total_weight
            avg_position_size = total_position_size / total_weight
            avg_risk_score = total_risk_score / total_weight
        else:
            avg_confidence = 0.5
            avg_position_size = 0.0
            avg_risk_score = 0.5

        # Determine signal type based on ensemble prediction
        if avg_confidence >= confidence_threshold:
            if ensemble_prediction > 0.01:
                signal_type = SignalType.BUY
            elif ensemble_prediction < -0.01:
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD
        else:
            signal_type = SignalType.HOLD

        return HybridSignal(
            signal_type=signal_type,
            confidence=avg_confidence,
            predicted_return=ensemble_prediction,
            position_size=avg_position_size,
            risk_score=avg_risk_score,
            strategy_weights=strategy_weights,
            individual_signals=signals,
            timestamp=datetime.now(),
            metadata={"method": "ensemble_model", "confidence_threshold": confidence_threshold}
        )

    except Exception as e:
        logger.error(f"Error in ensemble model combination: {e}")
        return combine_weighted_average(signals, strategy_weights, confidence_threshold)


def calculate_ensemble_position_size(
    confidence: float, risk_score: float, max_position_size: float = 1.0
) -> float:
    """Calculate position size based on confidence and risk."""
    # Base position size on confidence
    base_size = confidence * max_position_size

    # Adjust for risk (higher risk = smaller position)
    risk_adjustment = 1.0 - (risk_score * 0.5)  # Reduce by up to 50% for high risk
    risk_adjustment = max(0.1, risk_adjustment)  # Minimum 10% position

    final_size = base_size * risk_adjustment
    return min(max_position_size, max(0.0, final_size))


def create_fallback_hybrid_signal() -> HybridSignal:
    """Create a fallback hybrid signal when no signals are available."""
    return HybridSignal(
        signal_type=SignalType.HOLD,
        confidence=0.5,
        predicted_return=0.0,
        position_size=0.0,
        risk_score=0.5,
        strategy_weights={},
        individual_signals=[],
        timestamp=datetime.now(),
        metadata={"method": "fallback", "reason": "no_signals_available"}
    ) 