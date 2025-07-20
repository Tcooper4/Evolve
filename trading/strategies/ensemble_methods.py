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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Try to import scikit-learn
try:
    from sklearn.ensemble import VotingRegressor
    from sklearn.metrics import mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError as e:
    print("âš ï¸ scikit-learn not available. Disabling ensemble voting capabilities.")
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
        
        if recent_indices:
            perf.mse_history = [perf.mse_history[i] for i in recent_indices]
            perf.timestamps = [perf.timestamps[i] for i in recent_indices]
            
        # Calculate recent performance (inverse of average MSE)
        if perf.mse_history:
            avg_mse = np.mean(perf.mse_history)
            perf.recent_performance = 1.0 / (avg_mse + self.epsilon)
        else:
            perf.recent_performance = 1.0
            
        # Update weight using dynamic formula
        perf.weight = self._calculate_dynamic_weight(perf)
        
        logger.debug(f"Updated performance for {strategy_name}: MSE={mse:.6f}, Weight={perf.weight:.4f}")
        
    def _calculate_dynamic_weight(self, perf: ModelPerformance) -> float:
        """Calculate dynamic weight based on performance and time decay.
        
        Args:
            perf: Model performance data
            
        Returns:
            Calculated weight
        """
        # Base weight from MSE
        base_weight = 1.0 / (perf.recent_performance + self.epsilon)
        
        # Apply time decay
        if perf.timestamps:
            days_since_update = (datetime.now() - perf.last_updated).days
            time_decay = self.decay_factor ** days_since_update
            base_weight *= time_decay
            
        # Apply min/max constraints
        weight = np.clip(base_weight, self.min_weight, self.max_weight)
        
        return weight
        
    def get_dynamic_weights(self, strategy_names: List[str]) -> Dict[str, float]:
        """Get dynamic weights for the given strategies.
        
        Args:
            strategy_names: List of strategy names
            
        Returns:
            Dictionary mapping strategy names to weights
        """
        weights = {}
        total_weight = 0.0
        
        for strategy_name in strategy_names:
            if strategy_name in self.model_performance:
                weight = self.model_performance[strategy_name].weight
            else:
                # Default weight for new strategies
                weight = 1.0
                
            weights[strategy_name] = weight
            total_weight += weight
            
        # Normalize weights
        if total_weight > 0:
            for strategy_name in weights:
                weights[strategy_name] /= total_weight
                
        return weights
        
    def combine_signals_dynamic(
        self,
        signals: List[StrategySignal],
        confidence_threshold: float = 0.6,
        use_dynamic_weights: bool = True
    ) -> HybridSignal:
        """Combine signals using dynamic weighting.
        
        Args:
            signals: List of strategy signals
            confidence_threshold: Minimum confidence for signal generation
            use_dynamic_weights: Whether to use dynamic weights
            
        Returns:
            Combined hybrid signal
        """
        try:
            if not signals:
                return create_fallback_hybrid_signal()
                
            # Get strategy names
            strategy_names = [signal.strategy_name for signal in signals]
            
            # Get weights
            if use_dynamic_weights:
                strategy_weights = self.get_dynamic_weights(strategy_names)
            else:
                # Equal weights
                strategy_weights = {name: 1.0 / len(strategy_names) for name in strategy_names}
                
            # Calculate weighted metrics
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
                
            # Determine signal type
            if avg_confidence < confidence_threshold:
                signal_type = SignalType.HOLD
            elif avg_return > 0.05:
                signal_type = SignalType.BUY if avg_return < 0.15 else SignalType.STRONG_BUY
            elif avg_return < -0.05:
                signal_type = SignalType.SELL if avg_return > -0.15 else SignalType.STRONG_SELL
            else:
                signal_type = SignalType.HOLD
                
            # Calculate position size
            position_size = calculate_ensemble_position_size(avg_confidence, avg_risk)
            
            # Record audit information
            self._record_audit(signals, strategy_weights, signal_type, avg_confidence)
            
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
                    "method": "dynamic_weighting",
                    "total_weight": total_weight,
                    "confidence_threshold": confidence_threshold,
                    "use_dynamic_weights": use_dynamic_weights,
                },
            )
            
        except Exception as e:
            logger.error(f"Error in dynamic signal combination: {e}")
            return create_fallback_hybrid_signal()
            
    def _record_audit(self, signals: List[StrategySignal], weights: Dict[str, float], 
                     final_signal: SignalType, final_confidence: float):
        """Record audit information for ensemble analysis."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "signals": [
                {
                    "strategy": signal.strategy_name,
                    "signal": signal.signal_type.value,
                    "confidence": signal.confidence,
                    "weight": weights.get(signal.strategy_name, 0.0)
                }
                for signal in signals
            ],
            "final_signal": final_signal.value,
            "final_confidence": final_confidence,
            "weights": weights
        }
        
        self.audit_history.append(audit_entry)
        
        # Keep only recent audit history
        if len(self.audit_history) > 1000:
            self.audit_history = self.audit_history[-1000:]
            
    def get_ensemble_audit_report(self) -> Dict[str, Any]:
        """Generate ensemble audit report showing model performance.
        
        Returns:
            Dictionary with audit report data
        """
        if not self.audit_history:
            return {"message": "No audit history available"}
            
        # Count wins for each model
        model_wins = {}
        total_decisions = len(self.audit_history)
        
        for audit in self.audit_history:
            # Find the model with highest weighted confidence
            best_model = None
            best_score = -1
            
            for signal_info in audit["signals"]:
                score = signal_info["confidence"] * signal_info["weight"]
                if score > best_score:
                    best_score = score
                    best_model = signal_info["strategy"]
                    
            if best_model:
                model_wins[best_model] = model_wins.get(best_model, 0) + 1
                
        # Calculate win rates
        win_rates = {}
        for model, wins in model_wins.items():
            win_rates[model] = wins / total_decisions
            
        # Get recent performance
        recent_performance = {}
        for strategy_name, perf in self.model_performance.items():
            recent_performance[strategy_name] = {
                "recent_mse": np.mean(perf.mse_history[-10:]) if perf.mse_history else float('inf'),
                "current_weight": perf.weight,
                "last_updated": perf.last_updated.isoformat()
            }
            
        return {
            "total_decisions": total_decisions,
            "win_rates": win_rates,
            "recent_performance": recent_performance,
            "audit_summary": {
                "models_tracked": len(self.model_performance),
                "decisions_analyzed": total_decisions,
                "report_generated": datetime.now().isoformat()
            }
        }
        
    def save_state(self, filepath: str):
        """Save ensemble voter state to file."""
        try:
            state = {
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
                "audit_history": self.audit_history,
                "config": self.config
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
                
            logger.info(f"Saved ensemble voter state to {filepath}")
        except Exception as e:
            logger.error(f"Could not save ensemble voter state: {e}")
            
    def load_state(self, filepath: str):
        """Load ensemble voter state from file."""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
                
            # Restore model performance
            self.model_performance = {}
            for name, perf_data in state["model_performance"].items():
                self.model_performance[name] = ModelPerformance(
                    strategy_name=name,
                    mse_history=perf_data["mse_history"],
                    timestamps=[datetime.fromisoformat(ts) for ts in perf_data["timestamps"]],
                    recent_performance=perf_data["recent_performance"],
                    weight=perf_data["weight"],
                    last_updated=datetime.fromisoformat(perf_data["last_updated"])
                )
                
            # Restore audit history
            self.audit_history = state.get("audit_history", [])
            
            # Restore config
            if "config" in state:
                self.config.update(state["config"])
                
            logger.info(f"Loaded ensemble voter state from {filepath}")
        except Exception as e:
            logger.error(f"Could not load ensemble voter state: {e}")


# Global instance for backward compatibility
_dynamic_voter = DynamicEnsembleVoter()


def combine_weighted_average(
    signals: List[StrategySignal],
    strategy_weights: Dict[str, float],
    confidence_threshold: float = 0.6,
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
            signal_type = (
                SignalType.SELL if avg_return > -0.15 else SignalType.STRONG_SELL
            )
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
    signals: List[StrategySignal],
    strategy_weights: Dict[str, float],
    confidence_threshold: float = 0.6,
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
            final_signal = SignalType.HOLD
        else:
            final_signal = winning_signal

        # Calculate position size
        position_size = calculate_ensemble_position_size(adjusted_confidence, avg_risk)

        return HybridSignal(
            signal_type=final_signal,
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
                "vote_ratio": vote_ratio,
                "confidence_threshold": confidence_threshold,
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
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn not available. Cannot use ensemble model.")
        return create_fallback_hybrid_signal()
    """Combine signals using ensemble model.

    Args:
        signals: List of strategy signals
        ensemble_model: Trained ensemble model
        strategy_weights: Weights for each strategy
        confidence_threshold: Minimum confidence for signal generation

    Returns:
        Combined hybrid signal
    """
    try:
        if not signals:
            return create_fallback_hybrid_signal()

        # Extract features from signals
        features = []
        for signal in signals:
            feature_vector = [
                signal.confidence,
                signal.predicted_return,
                signal.risk_score,
                signal.position_size,
            ]
            features.append(feature_vector)

        # Make ensemble prediction
        if len(features) > 0:
            features_array = np.array(features).reshape(1, -1)
            ensemble_prediction = ensemble_model.predict(features_array)[0]
        else:
            ensemble_prediction = 0.0

        # Calculate weighted metrics
        total_weight = 0
        weighted_confidence = 0
        weighted_risk = 0

        for signal in signals:
            weight = strategy_weights.get(signal.strategy_name, 1.0 / len(signals))
            total_weight += weight
            weighted_confidence += signal.confidence * weight
            weighted_risk += signal.risk_score * weight

        # Determine signal type based on ensemble prediction
        if weighted_confidence < confidence_threshold:
            signal_type = SignalType.HOLD
        elif ensemble_prediction > 0.05:
            signal_type = SignalType.BUY if ensemble_prediction < 0.15 else SignalType.STRONG_BUY
        elif ensemble_prediction < -0.05:
            signal_type = SignalType.SELL if ensemble_prediction > -0.15 else SignalType.STRONG_SELL
        else:
            signal_type = SignalType.HOLD

        # Calculate position size
        position_size = calculate_ensemble_position_size(weighted_confidence, weighted_risk)

        return HybridSignal(
            signal_type=signal_type,
            confidence=weighted_confidence,
            predicted_return=ensemble_prediction,
            position_size=position_size,
            risk_score=weighted_risk,
            strategy_weights=strategy_weights,
            individual_signals=signals,
            timestamp=datetime.now(),
            metadata={
                "method": "ensemble_model",
                "ensemble_prediction": ensemble_prediction,
                "confidence_threshold": confidence_threshold,
            },
        )

    except Exception as e:
        logger.error(f"Error in ensemble model combination: {e}")
        return create_fallback_hybrid_signal()


def calculate_ensemble_position_size(
    confidence: float, risk_score: float, max_position_size: float = 1.0
) -> float:
    """Calculate position size for ensemble signal.

    Args:
        confidence: Signal confidence
        risk_score: Risk score
        max_position_size: Maximum position size

    Returns:
        Calculated position size
    """
    # Base position size from confidence
    base_size = confidence * max_position_size

    # Adjust for risk
    risk_adjustment = 1.0 / (1.0 + risk_score)
    adjusted_size = base_size * risk_adjustment

    # Ensure within bounds
    return np.clip(adjusted_size, 0.0, max_position_size)


def create_fallback_hybrid_signal() -> HybridSignal:
    """Create a fallback hybrid signal when no signals are available.

    Returns:
        Fallback hybrid signal
    """
    return HybridSignal(
        signal_type=SignalType.HOLD,
        confidence=0.5,
        predicted_return=0.0,
        position_size=0.0,
        risk_score=1.0,
        strategy_weights={},
        individual_signals=[],
        timestamp=datetime.now(),
        metadata={"method": "fallback", "reason": "no_signals_available"},
    )


# Convenience functions for backward compatibility
def update_model_performance(strategy_name: str, actual_returns: pd.Series, 
                           predicted_returns: pd.Series, timestamp: Optional[datetime] = None):
    """Update model performance (convenience function)."""
    _dynamic_voter.update_model_performance(strategy_name, actual_returns, predicted_returns, timestamp)


def combine_signals_dynamic(signals: List[StrategySignal], confidence_threshold: float = 0.6,
                          use_dynamic_weights: bool = True) -> HybridSignal:
    """Combine signals using dynamic weighting (convenience function)."""
    return _dynamic_voter.combine_signals_dynamic(signals, confidence_threshold, use_dynamic_weights)


def get_ensemble_audit_report() -> Dict[str, Any]:
    """Get ensemble audit report (convenience function)."""
    return _dynamic_voter.get_ensemble_audit_report()
