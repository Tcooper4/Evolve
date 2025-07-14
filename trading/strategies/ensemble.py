"""
Weighted Ensemble Strategy

This module provides a WeightedEnsembleStrategy class that combines multiple
strategy outputs using configurable weights to produce a final buy/sell signal.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from trading.strategies.ensemble_methods import (
    HybridSignal,
    SignalType,
    StrategySignal,
    combine_weighted_average,
    combine_voting,
)

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
    """
    Weighted Ensemble Strategy that combines multiple strategy outputs.
    
    This class takes multiple strategy signal DataFrames and combines them
    using configurable weights to produce a final buy/sell signal.
    """

    def __init__(self, config: Optional[EnsembleConfig] = None):
        """Initialize the ensemble strategy with configuration."""
        self.config = config or EnsembleConfig(
            strategy_weights={"rsi": 0.4, "macd": 0.4, "bollinger": 0.2}
        )
        self.signals = None
        self.positions = None
        self.performance_history = []
        self.last_rebalance = None
        self.last_successful_signals = None # Added for fallback

    def validate_weights(self, weights: Dict[str, float]) -> bool:
        """Validate that strategy weights sum to approximately 1.0."""
        total_weight = sum(weights.values())
        return abs(total_weight - 1.0) < 0.01

    def normalize_weights(self, weights: Dict[str, float], epsilon: float = 1e-8) -> Dict[str, float]:
        """Normalize weights to sum to 1.0, with epsilon to prevent NaNs."""
        total_weight = sum(weights.values())
        if abs(total_weight) < epsilon:
            # Equal weighting if all weights are zero or near zero
            num_strategies = len(weights)
            return {strategy: 1.0 / num_strategies for strategy in weights}
        return {strategy: weight / (total_weight + epsilon) for strategy, weight in weights.items()}

    def combine_signals(
        self, 
        strategy_signals: Dict[str, pd.DataFrame],
        method: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Combine multiple strategy signals into a single ensemble signal.
        
        Args:
            strategy_signals: Dictionary mapping strategy names to signal DataFrames
            method: Combination method ("weighted_average" or "voting")
            
        Returns:
            Combined signal DataFrame
        """
        if not strategy_signals:
            raise ValueError("No strategy signals provided")

        method = method or self.config.combination_method
        
        # Normalize weights
        weights = self.normalize_weights(self.config.strategy_weights)
        
        # Ensure all strategies have signals
        available_strategies = list(strategy_signals.keys())
        missing_strategies = [s for s in weights.keys() if s not in available_strategies]
        if missing_strategies:
            logger.warning(f"Missing signals for strategies: {missing_strategies}")
            # Remove missing strategies from weights
            weights = {k: v for k, v in weights.items() if k in available_strategies}
            weights = self.normalize_weights(weights)

        # Fallback: If all models return None, return last successful forecast
        all_none = all(
            df is None or df.isnull().all().all() for df in strategy_signals.values()
        )
        if all_none:
            if hasattr(self, 'last_successful_signals') and self.last_successful_signals is not None:
                logger.warning("All models returned None. Using last successful forecast as fallback.")
                return self.last_successful_signals.copy()
            else:
                raise ValueError("All models returned None and no fallback available.")

        # Get common index
        common_index = strategy_signals[available_strategies[0]].index
        for strategy in available_strategies[1:]:
            common_index = common_index.intersection(strategy_signals[strategy].index)

        if len(common_index) == 0:
            raise ValueError("No common timestamps found across strategy signals")

        # Initialize combined signals DataFrame
        combined_signals = pd.DataFrame(index=common_index)
        combined_signals["signal"] = 0.0
        combined_signals["confidence"] = 0.0
        combined_signals["weighted_score"] = 0.0
        combined_signals["consensus"] = 0.0

        if method == "weighted_average":
            combined_signals = self._combine_weighted_average(
                strategy_signals, weights, common_index
            )
        elif method == "voting":
            combined_signals = self._combine_voting(
                strategy_signals, weights, common_index
            )
        else:
            raise ValueError(f"Unknown combination method: {method}")

        # Clip confidence scores to [0.0, 1.0]
        combined_signals["confidence"] = combined_signals["confidence"].clip(0.0, 1.0)

        # Apply confidence threshold
        low_confidence_mask = combined_signals["confidence"] < self.config.confidence_threshold
        combined_signals.loc[low_confidence_mask, "signal"] = 0

        # Apply consensus threshold for stronger signals
        consensus_mask = combined_signals["consensus"] >= self.config.consensus_threshold
        combined_signals["strong_signal"] = combined_signals["signal"] * consensus_mask

        self.signals = combined_signals
        # Save last successful signals for fallback
        self.last_successful_signals = combined_signals.copy()
        return combined_signals

    def _combine_weighted_average(
        self, 
        strategy_signals: Dict[str, pd.DataFrame], 
        weights: Dict[str, float],
        index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Combine signals using weighted average method."""
        combined = pd.DataFrame(index=index)
        combined["signal"] = 0.0
        combined["confidence"] = 0.0
        combined["weighted_score"] = 0.0
        combined["consensus"] = 0.0

        # Calculate weighted average of signals
        total_weight = 0.0
        signal_scores = []

        for strategy_name, weight in weights.items():
            if strategy_name in strategy_signals:
                signals = strategy_signals[strategy_name].loc[index]
                
                # Extract signal values (assuming 'signal' column exists)
                if "signal" in signals.columns:
                    signal_values = signals["signal"].fillna(0)
                else:
                    # If no signal column, try to infer from other columns
                    signal_values = self._extract_signal_from_dataframe(signals)
                
                # Extract confidence values
                if "confidence" in signals.columns:
                    confidence_values = signals["confidence"].fillna(0.5)
                else:
                    confidence_values = pd.Series(0.5, index=index)

                # Weight the signals
                weighted_signals = signal_values * weight
                weighted_confidence = confidence_values * weight

                combined["signal"] += weighted_signals
                combined["confidence"] += weighted_confidence
                combined["weighted_score"] += signal_values * weight

                signal_scores.append(signal_values)
                total_weight += weight

        # Normalize by total weight
        if total_weight > 0:
            combined["signal"] /= total_weight
            combined["confidence"] /= total_weight
            combined["weighted_score"] /= total_weight

        # Calculate consensus (agreement among strategies)
        if signal_scores:
            signal_array = np.array(signal_scores)
            # Count strategies that agree on direction
            positive_agreement = (signal_array > 0).sum(axis=0) / len(signal_scores)
            negative_agreement = (signal_array < 0).sum(axis=0) / len(signal_scores)
            combined["consensus"] = np.maximum(positive_agreement, negative_agreement)

        return combined

    def _combine_voting(
        self, 
        strategy_signals: Dict[str, pd.DataFrame], 
        weights: Dict[str, float],
        index: pd.DatetimeIndex
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
                combined.loc[timestamp, "consensus"] = votes[winning_vote] / total_weight

        return combined

    def _extract_signal_from_dataframe(self, df: pd.DataFrame) -> pd.Series:
        """Extract signal values from DataFrame when 'signal' column is missing."""
        # Try common signal column names
        signal_columns = ["signal", "Signal", "SIGNAL", "position", "Position"]
        for col in signal_columns:
            if col in df.columns:
                return df[col].fillna(0)
        
        # If no signal column found, try to infer from price-based columns
        if "close" in df.columns and "upper_band" in df.columns and "lower_band" in df.columns:
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
                    "total_weight": sum(new_weights.values())
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
                "result": {
                    "status": "parameters_updated",
                    "new_parameters": self.get_parameters()
                },
                "message": "Parameters updated successfully",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error setting parameters: {e}")
            return {
                "success": False,
                "result": {"status": "error", "message": str(e)},
                "message": "Failed to update parameters",
                "timestamp": datetime.now().isoformat(),
            }


def create_ensemble_strategy(
    strategy_weights: Dict[str, float],
    combination_method: str = "weighted_average",
    **kwargs
) -> WeightedEnsembleStrategy:
    """
    Factory function to create a WeightedEnsembleStrategy.
    
    Args:
        strategy_weights: Dictionary mapping strategy names to weights
        combination_method: Method to combine signals ("weighted_average" or "voting")
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured WeightedEnsembleStrategy instance
    """
    config = EnsembleConfig(
        strategy_weights=strategy_weights,
        combination_method=combination_method,
        **kwargs
    )
    return WeightedEnsembleStrategy(config)


# Example usage and common ensemble configurations
def create_rsi_macd_bollinger_ensemble() -> WeightedEnsembleStrategy:
    """Create a common ensemble with RSI, MACD, and Bollinger Bands."""
    weights = {"rsi": 0.4, "macd": 0.4, "bollinger": 0.2}
    return create_ensemble_strategy(weights, "weighted_average")


def create_balanced_ensemble() -> WeightedEnsembleStrategy:
    """Create a balanced ensemble with equal weights."""
    weights = {"rsi": 0.33, "macd": 0.33, "bollinger": 0.34}
    return create_ensemble_strategy(weights, "voting")


def create_conservative_ensemble() -> WeightedEnsembleStrategy:
    """Create a conservative ensemble with higher confidence thresholds."""
    weights = {"rsi": 0.3, "macd": 0.3, "bollinger": 0.4}
    return create_ensemble_strategy(
        weights, 
        "weighted_average",
        confidence_threshold=0.7,
        consensus_threshold=0.6
    ) 