"""
Hybrid Model Selector

Allows metric selection for model evaluation and selection in hybrid models.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class HybridModelSelector:
    """Hybrid model selector with configurable evaluation metrics.

    Tie-breaking: In the event of a tie (multiple models with the same best score),
    the model with the lexicographically smallest (alphabetically first) name is selected.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the hybrid model selector.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.default_metric = self.config.get("default_metric", "mse")
        self.score_strategy = self.config.get("score_strategy", "highest_sharpe")
        self.available_metrics = ["mse", "sharpe", "return"]

    def select_best_model(
        self,
        model_scores: Dict[str, Dict[str, float]],
        metric: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> str:
        """
        Select best model based on specified metric and strategy.

        Args:
            model_scores: Dictionary mapping model names to score dictionaries
            metric: Evaluation metric ('mse', 'sharpe', 'return')
            strategy: Selection strategy ('highest_sharpe', 'weighted_return', 'lowest_mse')

        Returns:
            Name of the best model
        """
        metric = metric or self.default_metric
        strategy = strategy or self.score_strategy

        if metric not in self.available_metrics:
            logger.warning(f"Unknown metric: {metric}, using {self.default_metric}")
            metric = self.default_metric

        logger.info(
            f"Selecting best model using metric: {metric}, strategy: {strategy}"
        )

        # Extract scores for the specified metric
        metric_scores = {}
        for model_name, scores in model_scores.items():
            if metric in scores:
                metric_scores[model_name] = scores[metric]
            else:
                logger.warning(f"Metric {metric} not found for model {model_name}")

        if not metric_scores:
            logger.error(f"No valid scores found for metric {metric}")
            if not model_scores:
                raise IndexError("No model scores provided")
            return list(model_scores.keys())[0]

        # If a specific metric is provided, use metric-based selection
        # unless a specific strategy is also provided
        if metric != self.default_metric and strategy == self.score_strategy:
            return self._select_by_metric(metric_scores, metric)

        # Apply selection strategy
        if strategy == "highest_sharpe":
            return self._select_highest_sharpe(model_scores)
        elif strategy == "weighted_return":
            return self._select_weighted_return(model_scores)
        elif strategy == "lowest_mse":
            return self._select_lowest_mse(model_scores)
        else:
            return self._select_by_metric(metric_scores, metric)

    def _select_highest_sharpe(self, model_scores: Dict[str, Dict[str, float]]) -> str:
        """Select model with highest Sharpe ratio."""
        sharpe_scores = {}
        for model_name, scores in model_scores.items():
            if "sharpe" in scores:
                sharpe_scores[model_name] = scores["sharpe"]

        if not sharpe_scores:
            logger.warning("No Sharpe ratios found, using first available model")
            return list(model_scores.keys())[0]

        best_model = max(sharpe_scores, key=sharpe_scores.get)
        logger.info(
            f"Selected model {best_model} with Sharpe ratio {sharpe_scores[best_model]:.4f}"
        )
        return best_model

    def _select_weighted_return(self, model_scores: Dict[str, Dict[str, float]]) -> str:
        """Select model using weighted return metric."""
        weighted_scores = {}
        for model_name, scores in model_scores.items():
            if "return" in scores and "sharpe" in scores:
                # Weight return by Sharpe ratio
                weighted_scores[model_name] = scores["return"] * scores["sharpe"]
            elif "return" in scores:
                weighted_scores[model_name] = scores["return"]

        if not weighted_scores:
            logger.warning(
                "No weighted return scores found, using first available model"
            )
            return list(model_scores.keys())[0]

        best_model = max(weighted_scores, key=weighted_scores.get)
        logger.info(
            f"Selected model {best_model} with weighted return {weighted_scores[best_model]:.4f}"
        )
        return best_model

    def _select_lowest_mse(self, model_scores: Dict[str, Dict[str, float]]) -> str:
        """
        Select model with lowest MSE.
        Tie-breaking: If multiple models have the same lowest MSE, the alphabetically first model is chosen.
        """
        mse_scores = {k: v["mse"] for k, v in model_scores.items() if "mse" in v}
        if not mse_scores:
            logger.warning("No MSE scores found, using first available model")
            return list(model_scores.keys())[0]
        min_value = min(mse_scores.values())
        candidates = [k for k, v in mse_scores.items() if v == min_value]
        best_model = sorted(candidates)[0]
        logger.info(
            f"Selected model {best_model} with MSE {mse_scores[best_model]:.4f}"
        )
        return best_model

    def _select_by_metric(self, metric_scores: Dict[str, float], metric: str) -> str:
        """
        Select model by specific metric.
        Tie-breaking: If multiple models have the same best score, the alphabetically first model is chosen.
        """
        # Sort keys to break ties deterministically
        if metric == "mse":
            min_value = min(metric_scores.values())
            candidates = [k for k, v in metric_scores.items() if v == min_value]
            best_model = sorted(candidates)[0]
        else:  # sharpe, return
            max_value = max(metric_scores.values())
            candidates = [k for k, v in metric_scores.items() if v == max_value]
            best_model = sorted(candidates)[0]
        logger.info(
            f"Selected model {best_model} with {metric} {metric_scores[best_model]:.4f}"
        )
        return best_model

    def get_model_ranking(
        self, model_scores: Dict[str, Dict[str, float]], metric: str = "mse"
    ) -> List[Tuple[str, float]]:
        """
        Get ranked list of models by metric.

        Args:
            model_scores: Dictionary mapping model names to score dictionaries
            metric: Metric to rank by

        Returns:
            List of (model_name, score) tuples sorted by score
        """
        metric_scores = {}
        for model_name, scores in model_scores.items():
            if metric in scores:
                metric_scores[model_name] = scores[metric]

        if metric == "mse":
            # Lower is better for MSE
            return sorted(metric_scores.items(), key=lambda x: x[1])
        else:
            # Higher is better for other metrics
            return sorted(metric_scores.items(), key=lambda x: x[1], reverse=True)

    def validate_scores(self, model_scores: Dict[str, Dict[str, float]]) -> bool:
        """
        Validate that model scores contain required metrics.

        Args:
            model_scores: Dictionary mapping model names to score dictionaries

        Returns:
            True if scores are valid
        """
        if not model_scores:
            return False

        for model_name, scores in model_scores.items():
            if not isinstance(scores, dict):
                logger.error(f"Invalid scores format for model {model_name}")
                return False

            # Check if at least one metric is present
            if not any(metric in scores for metric in self.available_metrics):
                logger.warning(f"No valid metrics found for model {model_name}")

        return True
