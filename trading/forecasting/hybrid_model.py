import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HybridModel:
    """
    Hybrid ensemble model that tracks model performance, auto-updates weights, and persists state.
    Uses risk-aware weighting based on Sharpe ratio, drawdown, or MSE with user-selectable metrics.
    """

    def __init__(
        self,
        model_dict: Dict[str, Any],
        weight_file: str = "hybrid_weights.json",
        perf_file: str = "hybrid_performance.json",
    ):
        """
        Args:
            model_dict: Dictionary of model_name: model_instance
            weight_file: Path to save/load ensemble weights
            perf_file: Path to save/load model performance
        """
        self.models = model_dict
        self.weight_file = weight_file
        self.perf_file = perf_file
        self.weights = {name: 1.0 / len(model_dict) for name in model_dict}
        self.performance = {
            name: [] for name in model_dict
        }  # List of recent performance metrics
        self.scoring_config = {
            "method": "risk_aware",  # "risk_aware", "weighted_average", "ahp", "composite"
            "weighting_metric": "sharpe",  # "sharpe", "drawdown", "mse"
            "metrics": {
                "sharpe_ratio": {"weight": 0.4, "direction": "maximize"},
                "win_rate": {"weight": 0.3, "direction": "maximize"},
                "max_drawdown": {"weight": 0.2, "direction": "minimize"},
                "mse": {"weight": 0.1, "direction": "minimize"},
                "total_return": {"weight": 0.0, "direction": "maximize"},
            },
            "min_performance_threshold": 0.1,  # Minimum performance to avoid zero weights
            "recency_weight": 0.7,  # Weight for recent vs historical performance
            "risk_free_rate": 0.02,  # Risk-free rate for Sharpe calculations
            "sharpe_floor": 0.0,  # Minimum Sharpe ratio to avoid negative weights
            "drawdown_ceiling": -0.5,  # Maximum drawdown threshold
            "mse_ceiling": 1000.0,  # Maximum MSE threshold
        }
        self.load_state()

    def fit(self, data: pd.DataFrame, window: int = 50):
        """Fit all models and update performance tracking."""
        for name, model in self.models.items():
            try:
                model.fit(data)
                preds = model.predict(data)
                actual = data["close"].values[-len(preds) :]

                # Calculate comprehensive performance metrics
                performance_metrics = self._calculate_performance_metrics(
                    actual, preds, data
                )

                self.performance[name].append(
                    {"timestamp": datetime.now().isoformat(), **performance_metrics}
                )
                # Keep only trailing window
                self.performance[name] = self.performance[name][-window:]
            except Exception as e:
                logger.warning(f"Model {name} failed to fit or predict: {e}")
                self.performance[name].append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "sharpe_ratio": -1.0,
                        "win_rate": 0.0,
                        "max_drawdown": -1.0,
                        "mse": float("inf"),
                        "total_return": -10,
                    }
                )
        self.save_state()
        self.update_weights()

    def _calculate_performance_metrics(
        self, actual: np.ndarray, preds: np.ndarray, data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics for model evaluation."""
        try:
            # Ensure arrays are the same length
            min_len = min(len(actual), len(preds))
            actual = actual[-min_len:]
            preds = preds[-min_len:]

            # Calculate returns
            actual_returns = np.diff(actual) / actual[:-1]
            pred_returns = np.diff(preds) / preds[:-1]

            # Sharpe Ratio
            sharpe_ratio = self._calculate_sharpe_ratio(actual_returns, pred_returns)

            # Win Rate
            win_rate = self._calculate_win_rate(actual_returns, pred_returns)

            # Maximum Drawdown
            max_drawdown = self._calculate_max_drawdown(actual_returns, pred_returns)

            # MSE (keeping for backward compatibility)
            mse = float(np.mean((actual - preds) ** 2))
            # Total Return
            total_return = self._calculate_total_return(actual_returns, pred_returns)

            return {
                "sharpe_ratio": sharpe_ratio,
                "win_rate": win_rate,
                "max_drawdown": max_drawdown,
                "mse": mse,
                "total_return": total_return,
            }

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {
                "sharpe_ratio": -1.0,
                "win_rate": 0.0,
                "max_drawdown": -1.0,
                "mse": float("inf"),
                "total_return": -10,
            }

    def _calculate_sharpe_ratio(
        self, actual_returns: np.ndarray, pred_returns: np.ndarray
    ) -> float:
        """Calculate Sharpe ratio based on prediction accuracy."""
        try:
            # Calculate strategy returns (assuming we follow predictions)
            strategy_returns = actual_returns * np.sign(pred_returns)

            # Remove NaN values
            strategy_returns = strategy_returns[~np.isnan(strategy_returns)]

            if len(strategy_returns) == 0:
                return -1.0
            # Calculate Sharpe ratio (annualized)
            mean_return = np.mean(strategy_returns)
            std_return = np.std(strategy_returns)

            if std_return == 0:
                return 0.0
            # Annualize (assuming daily data)
            sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
            return float(sharpe_ratio)

        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return -1.0

    def _calculate_win_rate(
        self, actual_returns: np.ndarray, pred_returns: np.ndarray
    ) -> float:
        """Calculate win rate based on prediction accuracy."""
        try:
            # Calculate strategy returns
            strategy_returns = actual_returns * np.sign(pred_returns)

            # Remove NaN values
            strategy_returns = strategy_returns[~np.isnan(strategy_returns)]

            if len(strategy_returns) == 0:
                return 0.0
            # Count positive returns (wins)
            wins = np.sum(strategy_returns > 0)
            total_trades = len(strategy_returns)

            win_rate = wins / total_trades if total_trades > 0 else 0.0
            return float(win_rate)

        except Exception as e:
            logger.error(f"Error calculating win rate: {e}")
            return 0.0

    def _calculate_max_drawdown(
        self, actual_returns: np.ndarray, pred_returns: np.ndarray
    ) -> float:
        """Calculate maximum drawdown based on prediction accuracy."""
        try:
            # Calculate strategy returns
            strategy_returns = actual_returns * np.sign(pred_returns)

            # Remove NaN values
            strategy_returns = strategy_returns[~np.isnan(strategy_returns)]

            if len(strategy_returns) == 0:
                return -1.0
            # Calculate cumulative returns
            cumulative_returns = np.cumprod(1 + strategy_returns)

            # Calculate running maximum
            running_max = np.maximum.accumulate(cumulative_returns)

            # Calculate drawdown
            drawdown = (cumulative_returns - running_max) / running_max

            # Get maximum drawdown
            max_drawdown = np.min(drawdown)

            return float(max_drawdown)

        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return -1.0

    def _calculate_total_return(
        self, actual_returns: np.ndarray, pred_returns: np.ndarray
    ) -> float:
        """Calculate total return based on prediction accuracy."""
        try:
            # Calculate strategy returns
            strategy_returns = actual_returns * np.sign(pred_returns)

            # Remove NaN values
            strategy_returns = strategy_returns[~np.isnan(strategy_returns)]

            if len(strategy_returns) == 0:
                return -1.0
            # Calculate total return
            total_return = np.prod(1 + strategy_returns) - 1

            return float(total_return)

        except Exception as e:
            logger.error(f"Error calculating total return: {e}")
            return -1.0

    def update_weights(self):
        """Auto-update ensemble weights based on risk-aware performance metrics."""
        try:
            if self.scoring_config["method"] == "risk_aware":
                self.weights = self._calculate_risk_aware_weights()
            elif self.scoring_config["method"] == "weighted_average":
                self.weights = self._calculate_weighted_average_weights()
            elif self.scoring_config["method"] == "ahp":
                self.weights = self._calculate_ahp_weights()
            elif self.scoring_config["method"] == "composite":
                self.weights = self._calculate_composite_weights()
            else:
                logger.warning(
                    f"Unknown scoring method: {self.scoring_config['method']}, using risk-aware"
                )
                self.weights = self._calculate_risk_aware_weights()

            self.save_state()

        except Exception as e:
            logger.error(f"Error updating weights: {e}")
            # Fallback to equal weights
            self.weights = {name: 1.0 / len(self.models) for name in self.models}

    def _calculate_risk_aware_weights(self) -> Dict[str, float]:
        """
        Calculate weights using risk-aware metrics (Sharpe, Drawdown, or MSE).
        weight = metric_value / total_metric_value
        """
        model_scores = {}
        weighting_metric = self.scoring_config["weighting_metric"]

        for name, perf_list in self.performance.items():
            if not perf_list:
                model_scores[name] = 0.0
                continue

            # Calculate average metric over recent performance
            recent_perf = perf_list[-10:]  # Last 10 performance records

            if weighting_metric == "sharpe":
                # Use Sharpe ratio: weight = Sharpe / total_Sharpe
                sharpe_values = [
                    p.get("sharpe_ratio", -1.0)
                    for p in recent_perf
                    if p.get("sharpe_ratio") is not None
                ]
                if sharpe_values:
                    avg_sharpe = np.mean(sharpe_values)
                    # Apply floor to avoid negative weights
                    score = max(self.scoring_config["sharpe_floor"], avg_sharpe)
                else:
                    score = self.scoring_config["sharpe_floor"]

            elif weighting_metric == "drawdown":
                # Use inverse of drawdown: weight = (1 + drawdown) / total
                # Note: drawdown is negative, so we add 1 to make it positive
                drawdown_values = [
                    p.get("max_drawdown", -1.0)
                    for p in recent_perf
                    if p.get("max_drawdown") is not None
                ]
                if drawdown_values:
                    avg_drawdown = np.mean(drawdown_values)
                    # Apply ceiling and convert to positive score
                    capped_drawdown = max(
                        self.scoring_config["drawdown_ceiling"], avg_drawdown
                    )
                    score = 1.0 + capped_drawdown  # This makes it positive
                else:
                    score = 1.0 + self.scoring_config["drawdown_ceiling"]

            elif weighting_metric == "mse":
                # Use inverse of MSE: weight = (1/MSE) / total(1/MSE)
                mse_values = [
                    p.get("mse", float("inf"))
                    for p in recent_perf
                    if p.get("mse") is not None
                ]
                if mse_values:
                    avg_mse = np.mean(mse_values)
                    # Apply ceiling and convert to inverse
                    capped_mse = min(self.scoring_config["mse_ceiling"], avg_mse)
                    score = 1.0 / (1.0 + capped_mse)  # Add 1 to avoid division by zero
                else:
                    score = 1.0 / (1.0 + self.scoring_config["mse_ceiling"])

            else:
                logger.warning(
                    f"Unknown weighting metric: {weighting_metric}, using Sharpe"
                )
                # Fallback to Sharpe ratio
                sharpe_values = [
                    p.get("sharpe_ratio", -1.0)
                    for p in recent_perf
                    if p.get("sharpe_ratio") is not None
                ]
                if sharpe_values:
                    avg_sharpe = np.mean(sharpe_values)
                    score = max(self.scoring_config["sharpe_floor"], avg_sharpe)
                else:
                    score = self.scoring_config["sharpe_floor"]

            # Apply minimum performance threshold
            if score < self.scoring_config["min_performance_threshold"]:
                score = self.scoring_config["min_performance_threshold"]

            model_scores[name] = score

        # Normalize weights
        total_score = sum(model_scores.values())
        if total_score > 0:
            weights = {
                name: score / total_score for name, score in model_scores.items()
            }
        else:
            # Equal weights if no positive scores
            weights = {name: 1.0 / len(self.models) for name in self.models}

        return weights

    def _calculate_weighted_average_weights(self) -> Dict[str, float]:
        """Calculate weights using weighted average of performance metrics."""
        model_scores = {}

        for name, perf_list in self.performance.items():
            if not perf_list:
                model_scores[name] = 0.0
                continue

            # Calculate average metrics over recent performance
            recent_perf = perf_list[-10:]  # Last 10 performance records

            avg_metrics = {}
            for metric in [
                "sharpe_ratio",
                "win_rate",
                "max_drawdown",
                "mse",
                "total_return",
            ]:
                values = [
                    p.get(metric, 0.0) for p in recent_perf if p.get(metric) is not None
                ]
                if values:
                    avg_metrics[metric] = np.mean(values)
                else:
                    avg_metrics[metric] = 0.0
            # Calculate composite score
            score = 0.0
            for metric, config in self.scoring_config["metrics"].items():
                if metric in avg_metrics:
                    value = avg_metrics[metric]

                    # Normalize and apply direction
                    if config["direction"] == "maximize":
                        normalized_value = max(0, min(1, value))  # Clamp to [0, 1]
                    else:  # minimize
                        if metric == "max_drawdown":
                            normalized_value = max(
                                0, min(1, 1 + value)
                            )  # Drawdown is negative
                        else:
                            normalized_value = max(
                                0, min(1, 1 / (1 + value))
                            )  # Inverse for minimization

                    score += config["weight"] * normalized_value

            # Apply minimum performance threshold
            if score < self.scoring_config["min_performance_threshold"]:
                score = self.scoring_config["min_performance_threshold"]

            model_scores[name] = score

        # Normalize weights
        total_score = sum(model_scores.values())
        if total_score > 0:
            weights = {
                name: score / total_score for name, score in model_scores.items()
            }
        else:
            # Equal weights if no positive scores
            weights = {name: 1.0 / len(self.models) for name in self.models}

        return weights

    def _calculate_ahp_weights(self) -> Dict[str, float]:
        """Calculate weights using Analytic Hierarchy Process (AHP)."""
        # This is a simplified AHP implementation
        # In a full implementation, you would use pairwise comparisons

        # For now, we'll use a simplified version based on performance ranking
        model_rankings = {}

        for name, perf_list in self.performance.items():
            if not perf_list:
                model_rankings[name] = 0.0
                continue

            # Calculate average Sharpe ratio as primary ranking metric
            recent_perf = perf_list[-10:]
            sharpe_values = [
                p.get("sharpe_ratio", -1.0)
                for p in recent_perf
                if p.get("sharpe_ratio") is not None
            ]

            if sharpe_values:
                avg_sharpe = np.mean(sharpe_values)
                model_rankings[name] = max(0, avg_sharpe)  # Ensure non-negative
            else:
                model_rankings[name] = 0.0
        # Convert rankings to weights using exponential weighting
        max_ranking = max(model_rankings.values()) if model_rankings else 1.0
        if max_ranking > 0:
            weights = {
                name: np.exp(ranking / max_ranking)
                for name, ranking in model_rankings.items()
            }
        else:
            weights = {name: 1.0 for name in model_rankings.keys()}

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        else:
            weights = {name: 1.0 / len(self.models) for name in self.models}

        return weights

    def _calculate_composite_weights(self) -> Dict[str, float]:
        """Calculate weights using a composite scoring system."""
        model_scores = {}

        for name, perf_list in self.performance.items():
            if not perf_list:
                model_scores[name] = 0.0
                continue

            # Get recent performance
            recent_perf = perf_list[-5:]  # Last 5 performance records

            # Calculate trend-adjusted scores
            sharpe_trend = self._calculate_trend(
                [p.get("sharpe_ratio", -1.0) for p in recent_perf]
            )
            win_rate_trend = self._calculate_trend(
                [p.get("win_rate", 0.0) for p in recent_perf]
            )

            # Base score from latest performance
            latest = recent_perf[-1]
            base_score = (
                0.4 * max(0, latest.get("sharpe_ratio", -1.0))
                + 0.3 * latest.get("win_rate", 0.0)
                + 0.2
                * max(0, 1 + latest.get("max_drawdown", -10))  # Drawdown is negative
                + 0.1 * max(0, 1 / (1 + latest.get("mse", float("inf"))))
            )

            # Trend adjustment
            trend_adjustment = (sharpe_trend + win_rate_trend) / 2

            # Final score
            final_score = base_score * (1.0 + 0.2 * trend_adjustment)  # 20% adjustment

            model_scores[name] = max(
                self.scoring_config["min_performance_threshold"], final_score
            )

        # Normalize weights
        total_score = sum(model_scores.values())
        if total_score > 0:
            weights = {
                name: score / total_score for name, score in model_scores.items()
            }
        else:
            weights = {name: 1.0 / len(self.models) for name in self.models}

        return weights

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend of a series of values."""
        if len(values) < 2:
            return 0.0
        try:
            # Simple linear trend calculation
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]

            # Normalize trend to [-1, 1] range
            max_change = (
                max(values) - min(values) if max(values) != min(values) else 1.0
            )
            normalized_trend = slope / max_change if max_change > 0 else 0.0
            return np.clip(normalized_trend, -1.0, 1.0)
        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return 0.0

    def set_scoring_config(self, config: Dict[str, Any]):
        """Update the scoring configuration."""
        self.scoring_config.update(config)
        logger.info(f"Updated scoring config: {self.scoring_config}")

    def set_weighting_metric(self, metric: str):
        """
        Set the primary weighting metric for risk-aware weighting.

        Args:
            metric: One of "sharpe", "drawdown", or "mse"
        """
        valid_metrics = ["sharpe", "drawdown", "mse"]
        if metric not in valid_metrics:
            logger.warning(
                f"Invalid weighting metric: {metric}. Must be one of {valid_metrics}"
            )
            return

        self.scoring_config["weighting_metric"] = metric
        self.scoring_config["method"] = "risk_aware"
        logger.info(f"Set weighting metric to: {metric}")

        # Update weights immediately
        self.update_weights()

    def get_weighting_metric_info(self) -> Dict[str, Any]:
        """Get information about the current weighting metric and available options."""
        return {
            "current_metric": self.scoring_config["weighting_metric"],
            "current_method": self.scoring_config["method"],
            "available_metrics": {
                "sharpe": {
                    "description": "Sharpe ratio weighting (weight = Sharpe / total_Sharpe)",
                    "direction": "maximize",
                    "floor": self.scoring_config["sharpe_floor"],
                },
                "drawdown": {
                    "description": "Inverse drawdown weighting (weight = (1 + drawdown) / total)",
                    "direction": "minimize",
                    "ceiling": self.scoring_config["drawdown_ceiling"],
                },
                "mse": {
                    "description": "Inverse MSE weighting (weight = (1/MSE) / total(1/MSE))",
                    "direction": "minimize",
                    "ceiling": self.scoring_config["mse_ceiling"],
                },
            },
            "available_methods": {
                "risk_aware": "Risk-aware weighting using single metric",
                "weighted_average": "Weighted average of multiple metrics",
                "ahp": "Analytic Hierarchy Process",
                "composite": "Composite scoring with trend adjustment",
            },
        }

    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Provide a summary of model performance for analysis."""
        summary = {}

        for name, perf_list in self.performance.items():
            if not perf_list:
                summary[name] = {"status": "no_data"}
                continue

            recent_perf = perf_list[-10:]  # Last 10 records

            # Calculate averages
            avg_metrics = {}
            for metric in [
                "sharpe_ratio",
                "win_rate",
                "max_drawdown",
                "mse",
                "total_return",
            ]:
                values = [
                    p.get(metric, 0.0) for p in recent_perf if p.get(metric) is not None
                ]
                if values:
                    avg_metrics[metric] = np.mean(values)
                else:
                    avg_metrics[metric] = 0.0
            summary[name] = {
                "status": "active",
                "current_weight": self.weights.get(name, 0.0),
                "avg_metrics": avg_metrics,
                "performance_count": len(perf_list),
            }

        return summary

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Weighted ensemble prediction with fallback for None/mismatched forecasts."""
        valid_preds = []

        for name, model in self.models.items():
            try:
                pred = model.predict(data)

                # Validate prediction
                if pred is None:
                    logger.warning(f"Model {name} returned None prediction, skipping")
                    continue

                # Convert to numpy array if needed
                if not isinstance(pred, np.ndarray):
                    pred = np.array(pred)

                # Check for valid values
                if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                    logger.warning(
                        f"Model {name} returned invalid values (NaN/Inf), skipping"
                    )
                    continue

                # Check for reasonable prediction range
                if np.any(pred < -1e6) or np.any(pred > 1e6):
                    logger.warning(f"Model {name} returned extreme values, skipping")
                    continue

                valid_preds.append((name, pred))

            except Exception as e:
                logger.warning(f"Model {name} failed to predict: {e}")
                continue

        # Handle case where no valid predictions
        if not valid_preds:
            logger.error("No valid predictions from any model, returning fallback")
            # Return simple moving average as fallback
            if len(data) > 0 and "close" in data.columns:
                fallback_pred = np.full(len(data), data["close"].mean())
                return fallback_pred
            else:
                return np.array([])

        # Align predictions to same length
        min_len = min(len(p) for _, p in valid_preds)
        if min_len == 0:
            logger.error("All predictions have zero length")
            return np.array([])

        # Truncate all predictions to minimum length
        aligned_preds = []
        for name, pred in valid_preds:
            if len(pred) > min_len:
                # Take the last min_len values
                aligned_pred = pred[-min_len:]
            else:
                aligned_pred = pred
            aligned_preds.append((name, aligned_pred))

        # Calculate weighted ensemble
        weighted = np.zeros(min_len)
        total_weight = 0.0
        for name, pred in aligned_preds:
            weight = self.weights.get(name, 0.0)
            if weight > 0:
                weighted += weight * pred
                total_weight += weight

        # Normalize by total weight
        if total_weight > 0:
            weighted = weighted / total_weight
        else:
            # Equal weighting if no valid weights
            weighted = np.mean([pred for _, pred in aligned_preds], axis=0)

        return weighted

    def save_state(self):
        """Save weights and performance to disk (JSON and joblib)."""
        try:
            with open(self.weight_file, "w") as f:
                json.dump(self.weights, f, indent=2)
            with open(self.perf_file, "w") as f:
                json.dump(self.performance, f, indent=2)
            joblib.dump(self.weights, self.weight_file + ".joblib")
            joblib.dump(self.performance, self.perf_file + ".joblib")
        except Exception as e:
            logger.error(f"Failed to save hybrid model state: {e}")

    def load_state(self):
        """Load weights and performance from disk if available."""
        try:
            if os.path.exists(self.weight_file):
                with open(self.weight_file, "r") as f:
                    self.weights = json.load(f)
            if os.path.exists(self.perf_file):
                with open(self.perf_file, "r") as f:
                    self.performance = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load hybrid model state: {e}")
        # Try joblib as fallback
        try:
            if os.path.exists(self.weight_file + ".joblib"):
                self.weights = joblib.load(self.weight_file + ".joblib")
            if os.path.exists(self.perf_file + ".joblib"):
                self.performance = joblib.load(self.perf_file + ".joblib")
        except Exception as e:
            logger.warning(f"Failed to load hybrid model state from joblib: {e}")
