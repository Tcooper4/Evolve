"""
Performance weights management for trading models.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class PerformanceWeightsManager:
    """Manages model weights with decay and reinforcement mechanisms."""

    def __init__(self, base_decay_rate: float = 0.95, reinforcement_rate: float = 1.05):
        """Initialize the weights manager.

        Args:
            base_decay_rate: Base decay rate for poor performance (0.95 = 5% decay)
            reinforcement_rate: Reinforcement rate for good performance (1.05 = 5% boost)
        """
        self.base_decay_rate = base_decay_rate
        self.reinforcement_rate = reinforcement_rate
        self.weights_history = {}  # ticker -> list of weight records
        self.performance_history = {}  # ticker -> model -> list of performance records

    def update_weights_with_sharpe(
        self, ticker: str, model_performance: Dict[str, float], window_size: int = 20
    ) -> Dict[str, Any]:
        """Update model weights based on rolling Sharpe ratio performance.

        Args:
            ticker: Trading symbol
            model_performance: Dictionary of model performance metrics
            window_size: Number of periods for rolling Sharpe calculation

        Returns:
            Dictionary with updated weights and adjustment details
        """
        try:
            # Initialize history if needed
            if ticker not in self.weights_history:
                self.weights_history[ticker] = []
            if ticker not in self.performance_history:
                self.performance_history[ticker] = {}

            # Get current weights
            current_weights = self.get_current_weights(ticker)

            # Calculate rolling Sharpe ratios
            sharpe_ratios = {}
            weight_adjustments = {}

            for model, performance in model_performance.items():
                # Get historical performance for this model
                if model not in self.performance_history[ticker]:
                    self.performance_history[ticker][model] = []

                # Add current performance to history
                self.performance_history[ticker][model].append(
                    {"timestamp": datetime.now(), "performance": performance}
                )

                # Keep only recent history
                if len(self.performance_history[ticker][model]) > window_size * 2:
                    self.performance_history[ticker][model] = self.performance_history[
                        ticker
                    ][model][-window_size:]

                # Calculate rolling Sharpe ratio
                if len(self.performance_history[ticker][model]) >= window_size:
                    recent_performance = [
                        record["performance"]
                        for record in self.performance_history[ticker][model][
                            -window_size:
                        ]
                    ]
                    sharpe_ratio = self._calculate_sharpe_ratio(recent_performance)
                    sharpe_ratios[model] = sharpe_ratio

                    # Determine weight adjustment based on Sharpe ratio
                    adjustment = self._calculate_weight_adjustment(sharpe_ratio)
                    weight_adjustments[model] = adjustment
                else:
                    # Not enough data, use neutral adjustment
                    weight_adjustments[model] = 1.0

            # Apply weight adjustments
            new_weights = {}
            total_weight = 0

            for model, current_weight in current_weights.items():
                adjustment = weight_adjustments.get(model, 1.0)
                new_weight = current_weight * adjustment
                new_weights[model] = new_weight
                total_weight += new_weight

            # Normalize weights to sum to 1.0
            if total_weight > 0:
                for model in new_weights:
                    new_weights[model] /= total_weight
            else:
                # Fallback to equal weights
                model_count = len(new_weights)
                for model in new_weights:
                    new_weights[model] = 1.0 / model_count

            # Store weight update
            weight_record = {
                "timestamp": datetime.now().isoformat(),
                "weights": new_weights.copy(),
                "adjustments": weight_adjustments,
                "sharpe_ratios": sharpe_ratios,
                "reason": "sharpe_based_adjustment",
            }

            self.weights_history[ticker].append(weight_record)

            # Keep only recent history
            if len(self.weights_history[ticker]) > 100:
                self.weights_history[ticker] = self.weights_history[ticker][-100:]

            # Save to file
            self._save_weights(ticker, new_weights)

            return {
                "success": True,
                "ticker": ticker,
                "new_weights": new_weights,
                "adjustments": weight_adjustments,
                "sharpe_ratios": sharpe_ratios,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error updating weights with Sharpe ratio: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _calculate_sharpe_ratio(
        self, returns: List[float], risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio from a list of returns.

        Args:
            returns: List of return values
            risk_free_rate: Risk-free rate (annualized)

        Returns:
            Sharpe ratio
        """
        try:
            if len(returns) < 2:
                return 0.0

            returns_array = np.array(returns)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)

            if std_return == 0:
                return 0.0

            # Annualize (assuming daily returns)
            annualized_return = mean_return * 252
            annualized_volatility = std_return * np.sqrt(252)

            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
            return sharpe_ratio

        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    def _calculate_weight_adjustment(self, sharpe_ratio: float) -> float:
        """Calculate weight adjustment based on Sharpe ratio.

        Args:
            sharpe_ratio: Sharpe ratio value

        Returns:
            Weight adjustment multiplier
        """
        try:
            if sharpe_ratio > 1.0:
                # Good performance - reinforce
                return self.reinforcement_rate
            elif sharpe_ratio > 0.5:
                # Moderate performance - slight reinforcement
                return 1.0 + (self.reinforcement_rate - 1.0) * 0.5
            elif sharpe_ratio > 0.0:
                # Poor performance - slight decay
                return 1.0 - (1.0 - self.base_decay_rate) * 0.5
            else:
                # Very poor performance - full decay
                return self.base_decay_rate

        except Exception as e:
            logger.error(f"Error calculating weight adjustment: {e}")
            return 1.0

    def get_current_weights(self, ticker: str) -> Dict[str, float]:
        """Get current weights for a ticker.

        Args:
            ticker: Trading symbol

        Returns:
            Dictionary of current model weights
        """
        try:
            # Try to load from file first
            weights_file = f"memory/{ticker}_weights.json"
            if os.path.exists(weights_file):
                with open(weights_file, "r") as f:
                    weights_data = json.load(f)
                    return weights_data.get("weights", self._get_default_weights())
            else:
                return self._get_default_weights()

        except Exception as e:
            logger.error(f"Error loading current weights: {e}")
            return self._get_default_weights()

    def _get_default_weights(self) -> Dict[str, float]:
        """Get default model weights.

        Returns:
            Dictionary of default weights
        """
        return {
            "lstm": 0.3,
            "xgboost": 0.25,
            "prophet": 0.2,
            "ensemble": 0.15,
            "tcn": 0.1,
        }

    def _save_weights(self, ticker: str, weights: Dict[str, float]) -> bool:
        """Save weights to file.

        Args:
            ticker: Trading symbol
            weights: Dictionary of weights to save

        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs("memory", exist_ok=True)

            weights_file = f"memory/{ticker}_weights.json"
            weights_data = {
                "ticker": ticker,
                "weights": weights,
                "timestamp": datetime.now().isoformat(),
                "strategy": "sharpe_based",
            }

            with open(weights_file, "w") as f:
                json.dump(weights_data, f, indent=2)

            return True

        except Exception as e:
            logger.error(f"Error saving weights: {e}")
            return False

    def get_weight_history(self, ticker: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get weight adjustment history for a ticker.

        Args:
            ticker: Trading symbol
            limit: Maximum number of records to return

        Returns:
            List of weight history records
        """
        try:
            if ticker in self.weights_history:
                return self.weights_history[ticker][-limit:]
            else:
                return []

        except Exception as e:
            logger.error(f"Error getting weight history: {e}")
            return []

    def reset_weights(self, ticker: str) -> Dict[str, Any]:
        """Reset weights to default values.

        Args:
            ticker: Trading symbol

        Returns:
            Dictionary with reset status
        """
        try:
            default_weights = self._get_default_weights()
            self._save_weights(ticker, default_weights)

            # Clear history
            if ticker in self.weights_history:
                self.weights_history[ticker].clear()

            return {
                "success": True,
                "ticker": ticker,
                "weights": default_weights,
                "message": "Weights reset to default values",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error resetting weights: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }


# Global instance
weights_manager = PerformanceWeightsManager()


def export_weights_to_file(ticker: str, strategy: str = "balanced") -> Dict[str, float]:
    """
    Export model weights to file.

    Args:
        ticker: Trading symbol
        strategy: Strategy type

    Returns:
        Dictionary of model weights
    """
    try:
        # Default weights for demonstration
        default_weights = {
            "lstm": 0.3,
            "xgboost": 0.25,
            "prophet": 0.2,
            "ensemble": 0.15,
            "tcn": 0.1,
        }

        try:
            os.makedirs("memory", exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create memory directory: {e}")

        # Save weights to file
        weights_file = f"memory/{ticker}_weights.json"
        weights_data = {
            "ticker": ticker,
            "strategy": strategy,
            "weights": default_weights,
            "timestamp": datetime.now().isoformat(),
        }

        with open(weights_file, "w") as f:
            json.dump(weights_data, f, indent=2)

        return default_weights

    except Exception as e:
        logger.error(f"Error exporting weights: {e}")
        return {"lstm": 1.0}  # Fallback to single model


def get_latest_weights(ticker: str = "AAPL") -> Dict[str, float]:
    """
    Get the latest performance weights for a ticker.

    Args:
        ticker: Trading symbol

    Returns:
        Dictionary of model weights
    """
    try:
        weights_file = f"memory/{ticker}_weights.json"
        if os.path.exists(weights_file):
            with open(weights_file, "r") as f:
                weights_data = json.load(f)
                return {
                    "success": True,
                    "result": weights_data.get("weights", {}),
                    "message": "Operation completed successfully",
                    "timestamp": datetime.now().isoformat(),
                }
        else:
            # Return default weights if file doesn't exist
            return {
                "lstm": 0.3,
                "xgboost": 0.25,
                "prophet": 0.2,
                "ensemble": 0.15,
                "tcn": 0.1,
            }
    except Exception as e:
        logger.error(f"Error loading weights: {e}")
        return {"lstm": 1.0}  # Fallback to single model
