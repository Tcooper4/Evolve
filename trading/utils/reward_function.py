"""
RewardFunction: Multi-objective reward calculation for model and strategy evaluation.
Optimizes for return, Sharpe, and consistency (win rate over drawdown).
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class RewardFunction:
    """
    Multi-objective reward function for evaluating models and strategies.
    Supports weighted aggregation of return, Sharpe, and consistency (win rate/drawdown).
    Now includes adaptive weighting based on market conditions and prediction variance.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        adaptive_mode: bool = True,
        lookback_period: int = 30,
    ):
        """
        Args:
            weights: Dict of weights for each objective (return, sharpe, consistency)
            adaptive_mode: Whether to use adaptive weighting based on market conditions
            lookback_period: Number of periods to consider for adaptive weighting
        """
        self.base_weights = weights or {
            "return": 0.4,
            "sharpe": 0.4,
            "consistency": 0.2,
        }
        self.weights = self.base_weights.copy()
        self.adaptive_mode = adaptive_mode
        self.lookback_period = lookback_period
        self.performance_history = []
        self.market_regime_history = []
        self.last_adaptation = None
        self.adaptation_interval = timedelta(hours=1)  # Re-adapt every hour

    def compute(
        self,
        metrics: Dict[str, Any],
        prediction_variance: Optional[float] = None,
        market_volatility: Optional[float] = None,
    ) -> float:
        """
        Compute the overall reward score from metrics with adaptive weighting.
        Args:
            metrics: Dict with keys 'total_return', 'sharpe_ratio', 'win_rate', 'max_drawdown'
            prediction_variance: Variance of model predictions (higher = less confident)
            market_volatility: Current market volatility level
        Returns:
            Aggregated reward score (float)
        """
        # Update adaptive weights if needed
        if self.adaptive_mode:
            self._update_adaptive_weights(prediction_variance, market_volatility)

        objectives = self.compute_objectives(metrics, prediction_variance)
        return self.aggregate(objectives)

    def compute_objectives(
        self, metrics: Dict[str, Any], prediction_variance: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Compute individual objective scores (normalized where possible).
        Args:
            metrics: Dict with keys 'total_return', 'sharpe_ratio', 'win_rate', 'max_drawdown'
            prediction_variance: Variance of model predictions for confidence adjustment
        Returns:
            Dict of objective scores
        """
        total_return = metrics.get("total_return", 0.0)
        sharpe = metrics.get("sharpe_ratio", 0.0)
        win_rate = metrics.get("win_rate", 0.0)
        max_drawdown = (
            abs(metrics.get("max_drawdown", 1e-6)) or 1e-6
        )  # Avoid div by zero

        # Consistency: win rate over drawdown (higher is better)
        consistency = win_rate / max_drawdown if max_drawdown > 0 else 0.0

        # Apply prediction variance penalty if available
        if prediction_variance is not None:
            confidence_penalty = min(1.0, prediction_variance / 0.1)  # Normalize to 0-1
            total_return *= 1 - confidence_penalty * 0.2  # Reduce return by up to 20%
            sharpe *= 1 - confidence_penalty * 0.15  # Reduce Sharpe by up to 15%

        return {"return": total_return, "sharpe": sharpe, "consistency": consistency}

    def _update_adaptive_weights(
        self,
        prediction_variance: Optional[float] = None,
        market_volatility: Optional[float] = None,
    ) -> None:
        """
        Update weights adaptively based on market conditions and prediction confidence.
        """
        now = datetime.now()
        if (
            self.last_adaptation
            and now - self.last_adaptation < self.adaptation_interval
        ):
            return  # Too soon to adapt again

        # Store current performance for history
        if self.performance_history:
            recent_performance = np.mean(
                [p["sharpe"] for p in self.performance_history[-self.lookback_period:]]
            )
        else:
            recent_performance = 0.0  # Default value when no history available

        # Determine market regime
        regime = self._classify_market_regime(market_volatility, prediction_variance)
        self.market_regime_history.append(
            {
                "timestamp": now,
                "regime": regime,
                "volatility": market_volatility,
                "prediction_variance": prediction_variance,
            }
        )

        # Adaptive weight adjustments
        if regime == "high_volatility":
            # In high volatility, prioritize consistency and Sharpe over raw returns
            self.weights = {
                "return": self.base_weights["return"] * 0.7,
                "sharpe": self.base_weights["sharpe"] * 1.3,
                "consistency": self.base_weights["consistency"] * 1.5,
            }
        elif regime == "low_confidence":
            # When predictions are uncertain, reduce return weight and increase consistency
            self.weights = {
                "return": self.base_weights["return"] * 0.8,
                "sharpe": self.base_weights["sharpe"] * 1.1,
                "consistency": self.base_weights["consistency"] * 1.4,
            }
        elif regime == "stable":
            # In stable conditions, use base weights
            self.weights = self.base_weights.copy()
        else:  # unknown regime
            self.weights = self.base_weights.copy()

        # Normalize weights to sum to 1.0
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

        self.last_adaptation = now
        logger.info(f"Adaptive weights updated for regime '{regime}': {self.weights}")

    def _classify_market_regime(
        self, volatility: Optional[float], prediction_variance: Optional[float]
    ) -> str:
        """
        Classify current market regime based on volatility and prediction confidence.
        """
        if volatility is None and prediction_variance is None:
            return "stable"

        high_vol = volatility and volatility > 0.25  # 25% annualized volatility
        low_conf = (
            prediction_variance and prediction_variance > 0.05
        )  # 5% prediction variance

        if high_vol and low_conf:
            return "crisis"
        elif high_vol:
            return "high_volatility"
        elif low_conf:
            return "low_confidence"
        else:
            return "stable"

    def aggregate(self, objectives: Dict[str, float]) -> float:
        """
        Aggregate objectives into a single reward score using weights.
        Args:
            objectives: Dict of objective scores
        Returns:
            Weighted sum (float)
        """
        return sum(
            self.weights.get(k, 0.0) * objectives.get(k, 0.0) for k in self.weights
        )

    def multi_objective_vector(self, metrics: Dict[str, Any]) -> List[float]:
        """
        Return the vector of objectives for multi-objective optimization algorithms.
        Args:
            metrics: Dict with keys 'total_return', 'sharpe_ratio', 'win_rate', 'max_drawdown'
        Returns:
            List of objective values [return, sharpe, consistency]
        """
        obj = self.compute_objectives(metrics)
        return [obj["return"], obj["sharpe"], obj["consistency"]]

    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        Set new weights for the objectives.
        Args:
            weights: Dict of weights for each objective
        """
        self.base_weights = weights.copy()
        self.weights = weights.copy()

    def get_adaptive_weights(self) -> Dict[str, float]:
        """
        Get current adaptive weights.
        Returns:
            Current weight configuration
        """
        return self.weights.copy()

    def add_performance_record(self, metrics: Dict[str, Any]) -> None:
        """
        Add performance record for adaptive weighting history.
        Args:
            metrics: Performance metrics to store
        """
        self.performance_history.append({"timestamp": datetime.now(), **metrics})

        # Keep only recent history
        if len(self.performance_history) > self.lookback_period * 2:
            self.performance_history = self.performance_history[-self.lookback_period:]

    def get_regime_summary(self) -> Dict[str, Any]:
        """
        Get summary of recent market regimes and weight adaptations.
        Returns:
            Dict with regime statistics and weight history
        """
        if not self.market_regime_history:
            return {"regimes": [], "weight_history": []}

        recent_regimes = self.market_regime_history[-self.lookback_period:]
        regime_counts = {}
        for record in recent_regimes:
            regime = record["regime"]
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        return {
            "regimes": regime_counts,
            "weight_history": self.market_regime_history[-10:],  # Last 10 adaptations
            "current_weights": self.weights,
            "base_weights": self.base_weights,
        }
