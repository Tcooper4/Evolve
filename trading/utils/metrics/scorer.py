"""
Model Scorer

Provides scoring functions for model evaluation and selection.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple

logger = logging.getLogger(__name__)


class ModelScorer:
    """Scorer for model evaluation and selection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the model scorer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.default_metrics = ["mse", "mae", "rmse", "sharpe", "return"]
        
    def model_score(
        self,
        y_true: Union[np.ndarray, pd.Series, List],
        y_pred: Union[np.ndarray, pd.Series, List],
        metrics: Optional[List[str]] = None,
        returns: Optional[Union[np.ndarray, pd.Series, List]] = None
    ) -> Dict[str, float]:
        """
        Calculate model scores for given metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metrics: List of metrics to calculate
            returns: Returns series for financial metrics
            
        Returns:
            Dictionary of metric scores
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
                
        return scores
        
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