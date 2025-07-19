"""
Model Scorer

Provides scoring functions for model evaluation and selection.
Now includes rolling score decay for shorter test windows using exponential weighting.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ModelScorer:
    """Scorer for model evaluation and selection with rolling score decay."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the model scorer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.default_metrics = ["mse", "mae", "rmse", "sharpe", "return"]

        # Rolling score decay configuration
        self.enable_rolling_decay = self.config.get("enable_rolling_decay", True)
        self.decay_span = self.config.get("decay_span", 5)  # ewm span for decay
        self.min_window_size = self.config.get("min_window_size", 10)
        self.score_history = {}  # model_name -> list of (timestamp, scores) tuples

    def model_score(
        self,
        y_true: Union[np.ndarray, pd.Series, List],
        y_pred: Union[np.ndarray, pd.Series, List],
        metrics: Optional[List[str]] = None,
        returns: Optional[Union[np.ndarray, pd.Series, List]] = None,
        model_name: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Calculate model scores for given metrics with optional rolling decay.

        Args:
            y_true: True values
            y_pred: Predicted values
            metrics: List of metrics to calculate
            returns: Returns series for financial metrics
            model_name: Name of the model for tracking history
            timestamp: Timestamp for the score calculation

        Returns:
            Dictionary of metric scores (with decay applied if enabled)
        """
        # Convert inputs to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")

        if len(y_true) == 0:
            logger.warning("Empty data provided for scoring")
            return {}

        metrics = metrics or self.default_metrics
        scores = {}

        # Calculate each metric
        for metric in metrics:
            try:
                if metric == "mse":
                    scores[metric] = self._calculate_mse(y_true, y_pred)
                elif metric == "mae":
                    scores[metric] = self._calculate_mae(y_true, y_pred)
                elif metric == "rmse":
                    scores[metric] = self._calculate_rmse(y_true, y_pred)
                elif metric == "sharpe":
                    if returns is not None:
                        scores[metric] = self._calculate_sharpe(returns)
                    else:
                        logger.warning("Returns not provided for Sharpe calculation")
                elif metric == "return":
                    if returns is not None:
                        scores[metric] = self._calculate_total_return(returns)
                    else:
                        logger.warning("Returns not provided for return calculation")
                else:
                    logger.warning(f"Unknown metric: {metric}")

            except Exception as e:
                logger.error(f"Error calculating {metric}: {e}")
                scores[metric] = np.nan

        # Apply rolling decay if enabled and model name provided
        if self.enable_rolling_decay and model_name and timestamp:
            scores = self._apply_rolling_decay(model_name, scores, timestamp)

        return scores

    def _apply_rolling_decay(
        self,
        model_name: str,
        current_scores: Dict[str, float],
        timestamp: datetime
    ) -> Dict[str, float]:
        """
        Apply rolling score decay using exponential weighting.

        Args:
            model_name: Name of the model
            current_scores: Current metric scores
            timestamp: Current timestamp

        Returns:
            Decay-adjusted scores
        """
        try:
            if model_name not in self.score_history:
                self.score_history[model_name] = []

            # Add current scores to history
            self.score_history[model_name].append((timestamp, current_scores))

            # Keep only recent history to prevent memory bloat
            max_history = 100
            if len(self.score_history[model_name]) > max_history:
                self.score_history[model_name] = self.score_history[model_name][-max_history:]

            # Check if we have enough history for decay
            if len(self.score_history[model_name]) < self.min_window_size:
                return current_scores

            # Create DataFrame for rolling calculations
            history_data = []
            for ts, scores in self.score_history[model_name]:
                row = {"timestamp": ts}
                row.update(scores)
                history_data.append(row)

            df = pd.DataFrame(history_data)

            # Apply exponential weighted moving average for each metric
            decayed_scores = {}
            for metric in current_scores.keys():
                if metric in df.columns and not df[metric].isna().all():
                    # Calculate ewm with the specified span
                    ewm_values = df[metric].ewm(span=self.decay_span).mean()
                    # Use the latest decayed value
                    decayed_scores[metric] = ewm_values.iloc[-1]
                else:
                    # Use current score if no history available
                    decayed_scores[metric] = current_scores[metric]

            logger.debug(f"Applied rolling decay to {model_name}: {len(decayed_scores)} metrics")
            return decayed_scores

        except Exception as e:
            logger.error(f"Error applying rolling decay to {model_name}: {e}")
            return current_scores

    def get_score_trend(
        self,
        model_name: str,
        metric: str,
        window: int = 20
    ) -> Dict[str, Any]:
        """
        Get score trend for a specific model and metric.

        Args:
            model_name: Name of the model
            metric: Metric to analyze
            window: Window size for trend calculation

        Returns:
            Dictionary with trend information
        """
        try:
            if model_name not in self.score_history:
                return {"error": "No history available for model"}

            # Extract metric values
            metric_values = []
            timestamps = []

            for ts, scores in self.score_history[model_name]:
                if metric in scores and not np.isnan(scores[metric]):
                    metric_values.append(scores[metric])
                    timestamps.append(ts)

            if len(metric_values) < 2:
                return {"error": "Insufficient data for trend analysis"}

            # Calculate trend using linear regression
            x = np.arange(len(metric_values))
            y = np.array(metric_values)

            # Simple linear trend
            slope = np.polyfit(x, y, 1)[0]

            # Calculate rolling mean for smoothing
            if len(metric_values) >= window:
                rolling_mean = pd.Series(metric_values).rolling(window=window).mean()
                recent_trend = rolling_mean.iloc[-1] - rolling_mean.iloc[-window//2]
            else:
                recent_trend = slope

            return {
                "model_name": model_name,
                "metric": metric,
                "current_value": metric_values[-1],
                "trend_slope": slope,
                "recent_trend": recent_trend,
                "data_points": len(metric_values),
                "timestamps": timestamps,
                "values": metric_values
            }

        except Exception as e:
            logger.error(f"Error calculating score trend: {e}")
            return {"error": str(e)}

    def clear_history(self, model_name: Optional[str] = None):
        """
        Clear score history for a model or all models.

        Args:
            model_name: Specific model to clear, or None for all models
        """
        if model_name:
            if model_name in self.score_history:
                del self.score_history[model_name]
                logger.info(f"Cleared score history for {model_name}")
        else:
            self.score_history.clear()
            logger.info("Cleared all score history")

    def _calculate_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        return np.mean((y_true - y_pred) ** 2)

    def _calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return np.mean(np.abs(y_true - y_pred))

    def _calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return np.sqrt(self._calculate_mse(y_true, y_pred))

    def _calculate_sharpe(self, returns: Union[np.ndarray, pd.Series]) -> float:
        """Calculate Sharpe ratio."""
        returns = np.array(returns)
        if len(returns) == 0:
            return np.nan

        # Remove NaN values
        returns = returns[~np.isnan(returns)]
        if len(returns) == 0:
            return np.nan

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return np.nan

        return mean_return / std_return

    def _calculate_total_return(self, returns: Union[np.ndarray, pd.Series]) -> float:
        """Calculate total return."""
        returns = np.array(returns)
        if len(returns) == 0:
            return np.nan

        # Remove NaN values
        returns = returns[~np.isnan(returns)]
        if len(returns) == 0:
            return np.nan

        # Calculate cumulative return
        cumulative_return = np.prod(1 + returns) - 1
        return cumulative_return

    def compare_models(
        self,
        model_scores: Dict[str, Dict[str, float]],
        metric: str = "mse"
    ) -> List[Tuple[str, float]]:
        """
        Compare models by a specific metric.

        Args:
            model_scores: Dictionary mapping model names to score dictionaries
            metric: Metric to compare by

        Returns:
            List of (model_name, score) tuples sorted by score
        """
        comparison = []

        for model_name, scores in model_scores.items():
            if metric in scores and not np.isnan(scores[metric]):
                comparison.append((model_name, scores[metric]))

        # Sort based on metric type
        if metric in ["mse", "mae", "rmse"]:
            # Lower is better for error metrics
            comparison.sort(key=lambda x: x[1])
        else:
            # Higher is better for other metrics
            comparison.sort(key=lambda x: x[1], reverse=True)

        return comparison

    def get_best_model(
        self,
        model_scores: Dict[str, Dict[str, float]],
        metric: str = "mse"
    ) -> Optional[str]:
        """
        Get the best model by a specific metric.

        Args:
            model_scores: Dictionary mapping model names to score dictionaries
            metric: Metric to optimize

        Returns:
            Name of the best model, or None if no valid scores
        """
        comparison = self.compare_models(model_scores, metric)
        return comparison[0][0] if comparison else None

    def calculate_ensemble_score(
        self,
        individual_scores: List[Dict[str, float]],
        weights: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Calculate ensemble score from individual model scores.

        Args:
            individual_scores: List of score dictionaries for each model
            weights: Optional weights for each model

        Returns:
            Ensemble score dictionary
        """
        if not individual_scores:
            return {}

        # Use equal weights if not provided
        if weights is None:
            weights = [1.0 / len(individual_scores)] * len(individual_scores)

        if len(weights) != len(individual_scores):
            raise ValueError("Number of weights must match number of models")

        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        # Get all unique metrics
        all_metrics = set()
        for scores in individual_scores:
            all_metrics.update(scores.keys())

        ensemble_scores = {}

        for metric in all_metrics:
            metric_scores = []
            metric_weights = []

            for i, scores in enumerate(individual_scores):
                if metric in scores and not np.isnan(scores[metric]):
                    metric_scores.append(scores[metric])
                    metric_weights.append(weights[i])

            if metric_scores:
                # Calculate weighted average
                ensemble_scores[metric] = np.average(metric_scores, weights=metric_weights)
            else:
                ensemble_scores[metric] = np.nan

        return ensemble_scores

    def validate_scores(self, scores: Dict[str, float]) -> bool:
        """
        Validate that scores are reasonable.

        Args:
            scores: Dictionary of metric scores

        Returns:
            True if scores are valid
        """
        if not scores:
            return False

        for metric, score in scores.items():
            if np.isnan(score) or np.isinf(score):
                logger.warning(f"Invalid score for {metric}: {score}")
                return False

            # Check for reasonable ranges
            if metric in ["mse", "mae", "rmse"] and score < 0:
                logger.warning(f"Negative error metric {metric}: {score}")
                return False

        return True 