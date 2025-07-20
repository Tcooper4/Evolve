"""
Hybrid Engine

This module handles hybrid model logic for combining multiple forecast results.
It provides various combination strategies and ensemble methods for improved
forecast accuracy.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class HybridEngine:
    """
    Hybrid engine for combining multiple forecast results.
    
    Features:
    - Multiple combination strategies
    - Dynamic weighting based on model performance
    - Confidence interval combination
    - Outlier detection and handling
    - Ensemble optimization
    """

    def __init__(
        self,
        combination_method: str = "weighted_average",
        outlier_detection: bool = True,
        confidence_weighting: bool = True,
        performance_history_window: int = 100,
    ):
        """
        Initialize the hybrid engine.

        Args:
            combination_method: Method for combining forecasts
            outlier_detection: Whether to detect and handle outliers
            confidence_weighting: Whether to weight by confidence scores
            performance_history_window: Window for performance history
        """
        self.combination_method = combination_method
        self.outlier_detection = outlier_detection
        self.confidence_weighting = confidence_weighting
        self.performance_history_window = performance_history_window
        
        # Performance tracking
        self.model_performance: Dict[str, List[float]] = {}
        self.combination_history: List[Dict[str, Any]] = []
        
        # Available combination methods
        self.combination_methods = {
            "weighted_average": self._weighted_average_combination,
            "median": self._median_combination,
            "trimmed_mean": self._trimmed_mean_combination,
            "bayesian": self._bayesian_combination,
            "stacking": self._stacking_combination,
            "voting": self._voting_combination,
        }
        
        logger.info(f"HybridEngine initialized with method: {combination_method}")

    def combine_forecasts(
        self, forecast_results: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine multiple forecast results using the specified method.

        Args:
            forecast_results: List of forecast results from different models
            context: Context information for combination

        Returns:
            Combined forecast result
        """
        if not forecast_results:
            return self._get_empty_result()
        
        if len(forecast_results) == 1:
            return forecast_results[0]
        
        try:
            logger.info(f"Combining {len(forecast_results)} forecasts using {self.combination_method}")
            
            # Validate and preprocess forecasts
            valid_forecasts = self._validate_forecasts(forecast_results)
            if not valid_forecasts:
                return self._get_empty_result()
            
            # Detect and handle outliers if enabled
            if self.outlier_detection:
                valid_forecasts = self._handle_outliers(valid_forecasts)
            
            # Get combination method
            combination_func = self.combination_methods.get(
                self.combination_method, self._weighted_average_combination
            )
            
            # Combine forecasts
            combined_result = combination_func(valid_forecasts, context)
            
            # Add metadata
            combined_result.update({
                "combination_method": self.combination_method,
                "models_combined": len(valid_forecasts),
                "combination_timestamp": datetime.now().isoformat(),
                "outliers_removed": len(forecast_results) - len(valid_forecasts),
            })
            
            # Update performance tracking
            self._update_combination_history(valid_forecasts, combined_result)
            
            logger.info("Forecast combination completed successfully")
            return combined_result
            
        except Exception as e:
            logger.error(f"Forecast combination failed: {e}")
            return self._get_fallback_result(forecast_results)

    def _validate_forecasts(self, forecast_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate forecast results and extract forecasts.

        Args:
            forecast_results: List of forecast results

        Returns:
            List of valid forecast results
        """
        valid_forecasts = []
        
        for result in forecast_results:
            try:
                # Check for required fields
                if "forecast" not in result:
                    logger.warning(f"Forecast result missing 'forecast' field: {result.get('model', 'unknown')}")
                    continue
                
                forecast = result["forecast"]
                if not isinstance(forecast, (list, np.ndarray)):
                    logger.warning(f"Invalid forecast type: {type(forecast)}")
                    continue
                
                # Convert to numpy array
                forecast_array = np.array(forecast)
                if len(forecast_array) == 0:
                    logger.warning(f"Empty forecast array: {result.get('model', 'unknown')}")
                    continue
                
                # Check for NaN or infinite values
                if np.any(np.isnan(forecast_array)) or np.any(np.isinf(forecast_array)):
                    logger.warning(f"Forecast contains NaN or infinite values: {result.get('model', 'unknown')}")
                    continue
                
                # Add to valid forecasts
                valid_result = result.copy()
                valid_result["forecast_array"] = forecast_array
                valid_forecasts.append(valid_result)
                
            except Exception as e:
                logger.error(f"Error validating forecast: {e}")
                continue
        
        return valid_forecasts

    def _handle_outliers(self, forecasts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect and handle outliers in forecasts.

        Args:
            forecasts: List of forecast results

        Returns:
            List of forecasts with outliers handled
        """
        if len(forecasts) < 3:
            return forecasts  # Need at least 3 forecasts for outlier detection
        
        try:
            # Extract forecast arrays
            forecast_arrays = [f["forecast_array"] for f in forecasts]
            forecast_matrix = np.column_stack(forecast_arrays)
            
            # Calculate z-scores for each time step
            z_scores = np.abs(stats.zscore(forecast_matrix, axis=1))
            
            # Identify outliers (z-score > 2.5)
            outlier_mask = z_scores > 2.5
            
            # Remove forecasts with too many outliers
            valid_forecasts = []
            for i, forecast in enumerate(forecasts):
                outlier_count = np.sum(outlier_mask[:, i])
                outlier_ratio = outlier_count / len(forecast_arrays[0])
                
                if outlier_ratio < 0.3:  # Less than 30% outliers
                    valid_forecasts.append(forecast)
                else:
                    logger.warning(f"Removing forecast with {outlier_ratio:.2%} outliers: {forecast.get('model', 'unknown')}")
            
            return valid_forecasts if valid_forecasts else forecasts
            
        except Exception as e:
            logger.error(f"Outlier detection failed: {e}")
            return forecasts

    def _weighted_average_combination(
        self, forecasts: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine forecasts using weighted average.

        Args:
            forecasts: List of forecast results
            context: Context information

        Returns:
            Combined forecast result
        """
        # Calculate weights
        weights = self._calculate_weights(forecasts, context)
        
        # Weighted average
        forecast_arrays = [f["forecast_array"] for f in forecasts]
        combined_forecast = np.zeros_like(forecast_arrays[0])
        
        for forecast_array, weight in zip(forecast_arrays, weights):
            combined_forecast += weight * forecast_array
        
        # Calculate combined confidence
        confidences = [f.get("confidence", 0.5) for f in forecasts]
        combined_confidence = np.average(confidences, weights=weights)
        
        return {
            "forecast": combined_forecast,
            "confidence": combined_confidence,
            "model": "Hybrid_Weighted",
            "weights": weights,
            "models_used": [f.get("model", "unknown") for f in forecasts],
        }

    def _median_combination(
        self, forecasts: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine forecasts using median.

        Args:
            forecasts: List of forecast results
            context: Context information

        Returns:
            Combined forecast result
        """
        forecast_arrays = [f["forecast_array"] for f in forecasts]
        forecast_matrix = np.column_stack(forecast_arrays)
        
        # Calculate median
        combined_forecast = np.median(forecast_matrix, axis=1)
        
        # Calculate confidence based on agreement
        agreement = self._calculate_agreement(forecast_matrix)
        
        return {
            "forecast": combined_forecast,
            "confidence": agreement,
            "model": "Hybrid_Median",
            "models_used": [f.get("model", "unknown") for f in forecasts],
        }

    def _trimmed_mean_combination(
        self, forecasts: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine forecasts using trimmed mean.

        Args:
            forecasts: List of forecast results
            context: Context information

        Returns:
            Combined forecast result
        """
        forecast_arrays = [f["forecast_array"] for f in forecasts]
        forecast_matrix = np.column_stack(forecast_arrays)
        
        # Calculate trimmed mean (remove 10% from each end)
        combined_forecast = stats.trim_mean(forecast_matrix, 0.1, axis=1)
        
        # Calculate confidence
        confidences = [f.get("confidence", 0.5) for f in forecasts]
        combined_confidence = np.mean(confidences)
        
        return {
            "forecast": combined_forecast,
            "confidence": combined_confidence,
            "model": "Hybrid_TrimmedMean",
            "models_used": [f.get("model", "unknown") for f in forecasts],
        }

    def _bayesian_combination(
        self, forecasts: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine forecasts using Bayesian approach.

        Args:
            forecasts: List of forecast results
            context: Context information

        Returns:
            Combined forecast result
        """
        # This is a simplified Bayesian combination
        # In practice, you might want to use more sophisticated Bayesian methods
        
        forecast_arrays = [f["forecast_array"] for f in forecasts]
        confidences = [f.get("confidence", 0.5) for f in forecasts]
        
        # Use confidence as precision (inverse variance)
        precisions = np.array(confidences)
        precisions = precisions / np.sum(precisions)  # Normalize
        
        # Bayesian weighted average
        combined_forecast = np.zeros_like(forecast_arrays[0])
        for forecast_array, precision in zip(forecast_arrays, precisions):
            combined_forecast += precision * forecast_array
        
        # Calculate posterior precision
        posterior_precision = np.sum(precisions)
        combined_confidence = posterior_precision / len(precisions)
        
        return {
            "forecast": combined_forecast,
            "confidence": combined_confidence,
            "model": "Hybrid_Bayesian",
            "models_used": [f.get("model", "unknown") for f in forecasts],
        }

    def _stacking_combination(
        self, forecasts: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine forecasts using stacking approach.

        Args:
            forecasts: List of forecast results
            context: Context information

        Returns:
            Combined forecast result
        """
        # This is a simplified stacking approach
        # In practice, you might want to use cross-validation for meta-learner training
        
        forecast_arrays = [f["forecast_array"] for f in forecasts]
        forecast_matrix = np.column_stack(forecast_arrays)
        
        # Simple linear combination as meta-learner
        # In practice, you could train a more sophisticated meta-learner
        meta_weights = np.ones(len(forecasts)) / len(forecasts)
        
        combined_forecast = np.dot(forecast_matrix, meta_weights)
        
        # Calculate confidence
        confidences = [f.get("confidence", 0.5) for f in forecasts]
        combined_confidence = np.mean(confidences)
        
        return {
            "forecast": combined_forecast,
            "confidence": combined_confidence,
            "model": "Hybrid_Stacking",
            "models_used": [f.get("model", "unknown") for f in forecasts],
        }

    def _voting_combination(
        self, forecasts: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine forecasts using voting approach.

        Args:
            forecasts: List of forecast results
            context: Context information

        Returns:
            Combined forecast result
        """
        forecast_arrays = [f["forecast_array"] for f in forecasts]
        forecast_matrix = np.column_stack(forecast_arrays)
        
        # Calculate direction votes for each time step
        directions = np.diff(forecast_matrix, axis=0)
        votes = np.sign(directions)
        
        # Majority vote for direction
        majority_votes = np.sign(np.sum(votes, axis=1))
        
        # Reconstruct forecast using majority direction
        combined_forecast = np.zeros_like(forecast_arrays[0])
        combined_forecast[0] = np.mean([f[0] for f in forecast_arrays])  # First value
        
        for i in range(1, len(combined_forecast)):
            if majority_votes[i-1] > 0:
                # Upward trend
                combined_forecast[i] = combined_forecast[i-1] * 1.01
            elif majority_votes[i-1] < 0:
                # Downward trend
                combined_forecast[i] = combined_forecast[i-1] * 0.99
            else:
                # No clear direction
                combined_forecast[i] = combined_forecast[i-1]
        
        # Calculate confidence based on agreement
        agreement = self._calculate_agreement(forecast_matrix)
        
        return {
            "forecast": combined_forecast,
            "confidence": agreement,
            "model": "Hybrid_Voting",
            "models_used": [f.get("model", "unknown") for f in forecasts],
        }

    def _calculate_weights(
        self, forecasts: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> List[float]:
        """
        Calculate weights for forecast combination.

        Args:
            forecasts: List of forecast results
            context: Context information

        Returns:
            List of weights
        """
        weights = []
        
        for forecast in forecasts:
            weight = 1.0
            
            # Weight by confidence if enabled
            if self.confidence_weighting:
                confidence = forecast.get("confidence", 0.5)
                weight *= confidence
            
            # Weight by historical performance
            model_name = forecast.get("model", "unknown")
            if model_name in self.model_performance:
                performance = np.mean(self.model_performance[model_name][-10:])  # Last 10 performances
                weight *= performance
            
            # Weight by context relevance
            context_weight = self._calculate_context_weight(forecast, context)
            weight *= context_weight
            
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        
        return weights

    def _calculate_context_weight(
        self, forecast: Dict[str, Any], context: Dict[str, Any]
    ) -> float:
        """
        Calculate context-based weight for a forecast.

        Args:
            forecast: Forecast result
            context: Context information

        Returns:
            Context weight
        """
        weight = 1.0
        
        # Adjust weight based on market conditions
        market_volatility = context.get("market_volatility", "medium")
        model_name = forecast.get("model", "").lower()
        
        if market_volatility == "high":
            if "arima" in model_name or "xgboost" in model_name:
                weight *= 1.2  # Boost models good for high volatility
        elif market_volatility == "low":
            if "prophet" in model_name or "lstm" in model_name:
                weight *= 1.2  # Boost models good for low volatility
        
        # Adjust weight based on data characteristics
        data_length = context.get("data_length", 100)
        if data_length < 50:
            if "arima" in model_name:
                weight *= 1.1  # ARIMA good for short series
        elif data_length > 200:
            if "transformer" in model_name:
                weight *= 1.1  # Transformer good for long series
        
        return weight

    def _calculate_agreement(self, forecast_matrix: np.ndarray) -> float:
        """
        Calculate agreement between forecasts.

        Args:
            forecast_matrix: Matrix of forecasts

        Returns:
            Agreement score (0-1)
        """
        try:
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(forecast_matrix.T)
            
            # Calculate average correlation (excluding diagonal)
            n_models = corr_matrix.shape[0]
            total_corr = 0
            count = 0
            
            for i in range(n_models):
                for j in range(i+1, n_models):
                    total_corr += corr_matrix[i, j]
                    count += 1
            
            if count > 0:
                avg_correlation = total_corr / count
                # Convert correlation to agreement score (0-1)
                agreement = (avg_correlation + 1) / 2
                return max(0.0, min(1.0, agreement))
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating agreement: {e}")
            return 0.5

    def _update_combination_history(
        self, forecasts: List[Dict[str, Any]], result: Dict[str, Any]
    ) -> None:
        """
        Update combination history for performance tracking.

        Args:
            forecasts: List of forecasts used
            result: Combined result
        """
        self.combination_history.append({
            "timestamp": datetime.now().isoformat(),
            "method": self.combination_method,
            "models_count": len(forecasts),
            "confidence": result.get("confidence", 0.0),
            "models_used": [f.get("model", "unknown") for f in forecasts],
        })
        
        # Keep only recent history
        if len(self.combination_history) > self.performance_history_window:
            self.combination_history = self.combination_history[-self.performance_history_window:]

    def _get_empty_result(self) -> Dict[str, Any]:
        """Get empty result when no forecasts are available."""
        return {
            "forecast": np.array([]),
            "confidence": 0.0,
            "model": "Hybrid_Empty",
            "error": "No valid forecasts available",
        }

    def _get_fallback_result(self, forecasts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get fallback result when combination fails.

        Args:
            forecasts: List of original forecasts

        Returns:
            Fallback result
        """
        if not forecasts:
            return self._get_empty_result()
        
        # Use simple average as fallback
        forecast_arrays = []
        for forecast in forecasts:
            if "forecast" in forecast:
                try:
                    forecast_arrays.append(np.array(forecast["forecast"]))
                except:
                    continue
        
        if not forecast_arrays:
            return self._get_empty_result()
        
        # Simple average
        combined_forecast = np.mean(forecast_arrays, axis=0)
        
        return {
            "forecast": combined_forecast,
            "confidence": 0.5,
            "model": "Hybrid_Fallback",
            "error": "Combination failed, using simple average",
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of hybrid combinations."""
        if not self.combination_history:
            return {"total_combinations": 0, "average_confidence": 0.0}
        
        total_combinations = len(self.combination_history)
        average_confidence = np.mean([h["confidence"] for h in self.combination_history])
        
        # Method usage statistics
        method_usage = {}
        for history in self.combination_history:
            method = history["method"]
            method_usage[method] = method_usage.get(method, 0) + 1
        
        return {
            "total_combinations": total_combinations,
            "average_confidence": average_confidence,
            "method_usage": method_usage,
            "recent_confidence": np.mean([h["confidence"] for h in self.combination_history[-10:]]),
        }

    def set_combination_method(self, method: str) -> None:
        """
        Set the combination method.

        Args:
            method: Combination method name
        """
        if method in self.combination_methods:
            self.combination_method = method
            logger.info(f"Combination method set to: {method}")
        else:
            logger.warning(f"Unknown combination method: {method}. Using weighted_average")
            self.combination_method = "weighted_average"
