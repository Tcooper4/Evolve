"""
Weighted Ensemble Strategy Implementation.

This module provides a weighted ensemble strategy that combines multiple
trading strategies using different combination methods.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for Weighted Ensemble Strategy."""

    strategy_weights: Dict[str, float]
    combination_method: str = "weighted_average"  # "weighted_average" or "voting"
    confidence_threshold: float = 0.6
    consensus_threshold: float = 0.5  # Minimum agreement for consensus signals
    position_size_multiplier: float = 1.0
    risk_adjustment: bool = True
    dynamic_weighting: bool = False
    rebalance_frequency: str = "daily"  # "daily", "weekly", "monthly"


class WeightedEnsembleStrategy:
    """Weighted ensemble strategy that combines multiple trading strategies."""

    def __init__(self, config: Optional[EnsembleConfig] = None):
        """Initialize the ensemble strategy."""
        self.config = config or EnsembleConfig(strategy_weights={})
        self.signals = None
        self.positions = None
        self.performance_history = []

    def validate_weights(self, weights: Dict[str, float]) -> bool:
        """Validate that weights sum to approximately 1.0."""
        total_weight = sum(weights.values())
        return abs(total_weight - 1.0) < 1e-6

    def normalize_weights(
        self, weights: Dict[str, float], epsilon: float = 1e-8
    ) -> Dict[str, float]:
        """Normalize weights to sum to 1.0."""
        total_weight = sum(weights.values())
        if total_weight < epsilon:
            # If all weights are zero, distribute equally
            n_strategies = len(weights)
            return {name: 1.0 / n_strategies for name in weights.keys()}

        return {name: weight / total_weight for name, weight in weights.items()}

    def combine_signals(
        self, strategy_signals: Dict[str, pd.DataFrame], method: Optional[str] = None
    ) -> pd.DataFrame:
        """Combine signals from multiple strategies."""
        if not strategy_signals:
            raise ValueError("No strategy signals provided")

        # Use provided method or default from config
        combination_method = method or self.config.combination_method

        # Get common index
        indices = [df.index for df in strategy_signals.values()]
        common_index = (
            indices[0].intersection(*indices[1:]) if len(indices) > 1 else indices[0]
        )

        if len(common_index) == 0:
            raise ValueError("No common timestamps found across strategies")

        # Align all signals to common index
        aligned_signals = {}
        for strategy_name, signals in strategy_signals.items():
            aligned_signals[strategy_name] = signals.loc[common_index]

        # Combine signals based on method
        if combination_method == "weighted_average":
            combined = self._combine_weighted_average(
                aligned_signals, self.config.strategy_weights, common_index
            )
        elif combination_method == "voting":
            combined = self._combine_voting(
                aligned_signals, self.config.strategy_weights, common_index
            )
        else:
            raise ValueError(f"Unknown combination method: {combination_method}")

        # Apply confidence threshold
        if self.config.confidence_threshold > 0:
            mask = combined["confidence"] >= self.config.confidence_threshold
            combined.loc[~mask, "signal"] = 0.0

        # Apply consensus threshold for voting method
        if combination_method == "voting" and self.config.consensus_threshold > 0:
            mask = combined["consensus"] >= self.config.consensus_threshold
            combined.loc[~mask, "signal"] = 0.0

        self.signals = combined
        return combined

    def _combine_weighted_average(
        self,
        strategy_signals: Dict[str, pd.DataFrame],
        weights: Dict[str, float],
        index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Combine signals using weighted average method."""
        combined = pd.DataFrame(index=index)
        combined["signal"] = 0.0
        combined["confidence"] = 0.0
        combined["weighted_score"] = 0.0

        for strategy_name, weight in weights.items():
            if strategy_name in strategy_signals:
                signals = strategy_signals[strategy_name]

                # Extract signal values
                if "signal" in signals.columns:
                    signal_values = signals["signal"].fillna(0)
                else:
                    signal_values = self._extract_signal_from_dataframe(signals)

                # Extract confidence values
                if "confidence" in signals.columns:
                    confidence_values = signals["confidence"].fillna(0.5)
                else:
                    confidence_values = pd.Series(0.5, index=signals.index)

                # Apply weights
                combined["signal"] += signal_values * weight
                combined["confidence"] += confidence_values * weight
                combined["weighted_score"] += signal_values * confidence_values * weight

        return combined

    def _combine_voting(
        self,
        strategy_signals: Dict[str, pd.DataFrame],
        weights: Dict[str, float],
        index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Combine signals using voting method."""
        combined = pd.DataFrame(index=index)
        combined["signal"] = 0.0
        combined["confidence"] = 0.0
        combined["weighted_score"] = 0.0
        combined["consensus"] = 0.0

        # Collect votes for each timestamp
        for timestamp in index:
            votes = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
            total_weight = 0.0
            confidences = []

            for strategy_name, weight in weights.items():
                if strategy_name in strategy_signals:
                    signals = strategy_signals[strategy_name].loc[timestamp]

                    # Determine vote based on signal
                    if "signal" in signals:
                        signal_value = signals["signal"]
                    else:
                        signal_value = self._extract_signal_from_series(signals)

                    # Classify vote
                    if signal_value > 0.1:
                        votes["buy"] += weight
                    elif signal_value < -0.1:
                        votes["sell"] += weight
                    else:
                        votes["hold"] += weight

                    # Collect confidence
                    if "confidence" in signals:
                        confidences.append(signals["confidence"] * weight)
                    else:
                        confidences.append(0.5 * weight)

                    total_weight += weight

            # Determine winning vote
            if total_weight > 0:
                winning_vote = max(votes.items(), key=lambda x: x[1])[0]

                if winning_vote == "buy":
                    combined.loc[timestamp, "signal"] = 1.0
                elif winning_vote == "sell":
                    combined.loc[timestamp, "signal"] = -1.0
                else:
                    combined.loc[timestamp, "signal"] = 0.0

                # Calculate weighted confidence
                combined.loc[timestamp, "confidence"] = sum(confidences) / total_weight

                # Calculate consensus (proportion of weight for winning vote)
                combined.loc[timestamp, "consensus"] = (
                    votes[winning_vote] / total_weight
                )

        return combined

    def _extract_signal_from_dataframe(self, df: pd.DataFrame) -> pd.Series:
        """Extract signal values from DataFrame when 'signal' column is missing."""
        # Try common signal column names
        signal_columns = ["signal", "Signal", "SIGNAL", "position", "Position"]
        for col in signal_columns:
            if col in df.columns:
                return df[col].fillna(0)

        # If no signal column found, try to infer from price-based columns
        if (
            "close" in df.columns
            and "upper_band" in df.columns
            and "lower_band" in df.columns
        ):
            # Bollinger Bands logic
            signals = pd.Series(0, index=df.index)
            signals.loc[df["close"] < df["lower_band"]] = 1  # Buy signal
            signals.loc[df["close"] > df["upper_band"]] = -1  # Sell signal
            return signals

        # Default to zero signals
        return pd.Series(0, index=df.index)

    def _extract_signal_from_series(self, series: pd.Series) -> float:
        """Extract signal value from Series when 'signal' key is missing."""
        if "signal" in series:
            return series["signal"]

        # Try to infer from other values
        if "close" in series and "upper_band" in series and "lower_band" in series:
            close = series["close"]
            upper = series["upper_band"]
            lower = series["lower_band"]

            if close < lower:
                return 1.0  # Buy signal
            elif close > upper:
                return -1.0  # Sell signal

        return 0.0

    def calculate_positions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trading positions based on ensemble signals."""
        if self.signals is None:
            raise ValueError("Signals must be generated before calculating positions")

        positions = pd.DataFrame(index=self.signals.index)
        positions["position"] = self.signals["signal"].cumsum()

        # Apply position size multiplier
        positions["position"] *= self.config.position_size_multiplier

        # Ensure positions are within bounds
        positions["position"] = positions["position"].clip(-1, 1)

        # Add confidence and consensus information
        positions["confidence"] = self.signals["confidence"]
        positions["consensus"] = self.signals["consensus"]

        self.positions = positions
        return positions

    def update_weights(self, new_weights: Dict[str, float]) -> Dict:
        """Update strategy weights."""
        try:
            # Validate and normalize new weights
            if not self.validate_weights(new_weights):
                new_weights = self.normalize_weights(new_weights)
                logger.info("Weights normalized to sum to 1.0")

            self.config.strategy_weights = new_weights
            self.signals = None  # Reset signals to force regeneration
            self.positions = None

            return {
                "success": True,
                "result": {
                    "status": "weights_updated",
                    "new_weights": new_weights,
                    "total_weight": sum(new_weights.values()),
                },
                "message": "Strategy weights updated successfully",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error updating weights: {e}")
            return {
                "success": False,
                "result": {"status": "error", "message": str(e)},
                "message": "Failed to update strategy weights",
                "timestamp": datetime.now().isoformat(),
            }

    def get_performance_metrics(self) -> Dict:
        """Get ensemble performance metrics."""
        if self.signals is None:
            return {"error": "No signals available"}

        metrics = {
            "total_signals": len(self.signals),
            "buy_signals": (self.signals["signal"] > 0).sum(),
            "sell_signals": (self.signals["signal"] < 0).sum(),
            "hold_signals": (self.signals["signal"] == 0).sum(),
            "avg_confidence": self.signals["confidence"].mean(),
            "avg_consensus": self.signals["consensus"].mean(),
            "strategy_weights": self.config.strategy_weights,
            "combination_method": self.config.combination_method,
        }

        return {
            "success": True,
            "result": metrics,
            "message": "Performance metrics retrieved successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def get_parameters(self) -> Dict:
        """Get ensemble strategy parameters."""
        return {
            "strategy_weights": self.config.strategy_weights,
            "combination_method": self.config.combination_method,
            "confidence_threshold": self.config.confidence_threshold,
            "consensus_threshold": self.config.consensus_threshold,
            "position_size_multiplier": self.config.position_size_multiplier,
            "risk_adjustment": self.config.risk_adjustment,
            "dynamic_weighting": self.config.dynamic_weighting,
            "rebalance_frequency": self.config.rebalance_frequency,
        }

    def set_parameters(self, params: Dict) -> Dict:
        """Set ensemble strategy parameters."""
        try:
            # Update config with new parameters
            for key, value in params.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

            # Reset signals and positions
            self.signals = None
            self.positions = None

            return {
                "success": True,
                "result": {"status": "parameters_updated"},
                "message": "Strategy parameters updated successfully",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error setting parameters: {e}")
            return {
                "success": False,
                "result": {"status": "error", "message": str(e)},
                "message": "Failed to update strategy parameters",
                "timestamp": datetime.now().isoformat(),
            }


def create_ensemble_strategy(
    strategy_weights: Dict[str, float],
    combination_method: str = "weighted_average",
    **kwargs,
) -> WeightedEnsembleStrategy:
    """Create an ensemble strategy with specified weights and method."""
    config = EnsembleConfig(
        strategy_weights=strategy_weights,
        combination_method=combination_method,
        **kwargs,
    )
    return WeightedEnsembleStrategy(config)


def create_rsi_macd_bollinger_ensemble() -> WeightedEnsembleStrategy:
    """Create a balanced ensemble with RSI, MACD, and Bollinger Bands."""
    weights = {"rsi": 0.4, "macd": 0.35, "bollinger": 0.25}
    return create_ensemble_strategy(weights, "weighted_average")


def create_balanced_ensemble() -> WeightedEnsembleStrategy:
    """Create a balanced ensemble with equal weights."""
    weights = {"rsi": 0.33, "macd": 0.33, "bollinger": 0.34}
    return create_ensemble_strategy(weights, "voting")


def create_conservative_ensemble() -> WeightedEnsembleStrategy:
    """Create a conservative ensemble with higher confidence thresholds."""
    weights = {"rsi": 0.5, "macd": 0.3, "bollinger": 0.2}
    return create_ensemble_strategy(
        weights, "weighted_average", confidence_threshold=0.7, consensus_threshold=0.6
    )
