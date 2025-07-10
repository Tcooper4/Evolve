"""
Model evaluation utilities for the trading system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Model evaluation utility class."""
    
    def __init__(self):
        self.evaluation_results = {}
    
    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate regression model performance."""
        try:
            # Remove NaN values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) == 0:
                return {}
            
            metrics = {
                'mse': mean_squared_error(y_true_clean, y_pred_clean),
                'rmse': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
                'mae': mean_absolute_error(y_true_clean, y_pred_clean),
                'r2': r2_score(y_true_clean, y_pred_clean),
                'mape': np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
            }
            
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating regression model: {e}")
            return {}
    
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluate classification model performance."""
        try:
            # Remove NaN values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) == 0:
                return {}
            
            # Basic classification metrics
            accuracy = np.mean(y_true_clean == y_pred_clean)
            
            # Calculate precision, recall, F1 for each class
            unique_classes = np.unique(y_true_clean)
            precision = {}
            recall = {}
            f1 = {}
            
            for cls in unique_classes:
                tp = np.sum((y_true_clean == cls) & (y_pred_clean == cls))
                fp = np.sum((y_true_clean != cls) & (y_pred_clean == cls))
                fn = np.sum((y_true_clean == cls) & (y_pred_clean != cls))
                
                precision[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1[cls] = 2 * (precision[cls] * recall[cls]) / (precision[cls] + recall[cls]) if (precision[cls] + recall[cls]) > 0 else 0
            
            metrics = {
                'accuracy': accuracy,
                'precision': np.mean(list(precision.values())),
                'recall': np.mean(list(recall.values())),
                'f1_score': np.mean(list(f1.values()))
            }
            
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating classification model: {e}")
            return {}
    
    def evaluate_forecast(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         direction_accuracy: bool = True) -> Dict[str, float]:
        """Evaluate forecasting model performance."""
        try:
            # Basic regression metrics
            metrics = self.evaluate_regression(y_true, y_pred)
            
            if direction_accuracy:
                # Calculate direction accuracy
                true_direction = np.diff(y_true) > 0
                pred_direction = np.diff(y_pred) > 0
                
                # Align arrays
                min_len = min(len(true_direction), len(pred_direction))
                true_direction = true_direction[:min_len]
                pred_direction = pred_direction[:min_len]
                
                direction_accuracy = np.mean(true_direction == pred_direction)
                metrics['direction_accuracy'] = direction_accuracy
            
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating forecast model: {e}")
            return {}
    
    def cross_validate(self, model, X: np.ndarray, y: np.ndarray, 
                      cv: int = 5, scoring: str = 'neg_mean_squared_error') -> Dict[str, float]:
        """Perform cross-validation."""
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            
            return {
                'cv_mean': scores.mean(),
                'cv_std': scores.std(),
                'cv_scores': scores.tolist()
            }
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return {}

class ModelValidator:
    """Model validation utility class."""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_model_inputs(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Validate model input data."""
        try:
            # Check for NaN values
            if np.isnan(X).any() or np.isnan(y).any():
                logger.warning("NaN values found in input data")
                return False
            
            # Check for infinite values
            if np.isinf(X).any() or np.isinf(y).any():
                logger.warning("Infinite values found in input data")
                return False
            
            # Check shapes
            if len(X) != len(y):
                logger.error("X and y have different lengths")
                return False
            
            # Check for sufficient data
            if len(X) < 10:
                logger.warning("Insufficient data for reliable model training")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating model inputs: {e}")
            return False
    
    def validate_model_outputs(self, y_pred: np.ndarray, y_true: np.ndarray) -> bool:
        """Validate model output predictions."""
        try:
            # Check for NaN values
            if np.isnan(y_pred).any():
                logger.warning("NaN values found in predictions")
                return False
            
            # Check for infinite values
            if np.isinf(y_pred).any():
                logger.warning("Infinite values found in predictions")
                return False
            
            # Check prediction range
            if np.all(y_pred == y_pred[0]):
                logger.warning("All predictions are identical")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating model outputs: {e}")
            return False
    
    def validate_model_performance(self, metrics: Dict[str, float], 
                                 thresholds: Dict[str, float]) -> Dict[str, bool]:
        """Validate model performance against thresholds."""
        try:
            validation_results = {}
            
            for metric, threshold in thresholds.items():
                if metric in metrics:
                    validation_results[metric] = metrics[metric] >= threshold
                else:
                    validation_results[metric] = False
                    logger.warning(f"Metric {metric} not found in results")
            
            self.validation_results = validation_results
            return validation_results
        except Exception as e:
            logger.error(f"Error validating model performance: {e}")
            return {}

class ModelMonitor:
    """Model monitoring utility class."""
    
    def __init__(self):
        self.monitoring_data = {}
        self.drift_threshold = 0.1
    
    def detect_data_drift(self, reference_data: np.ndarray, 
                         current_data: np.ndarray) -> Dict[str, float]:
        """Detect data drift between reference and current data."""
        try:
            drift_metrics = {}
            
            # Statistical drift detection
            ref_mean = np.mean(reference_data, axis=0)
            ref_std = np.std(reference_data, axis=0)
            curr_mean = np.mean(current_data, axis=0)
            curr_std = np.std(current_data, axis=0)
            
            # Mean drift
            mean_drift = np.abs(curr_mean - ref_mean) / (ref_std + 1e-8)
            drift_metrics['mean_drift'] = np.mean(mean_drift)
            
            # Standard deviation drift
            std_drift = np.abs(curr_std - ref_std) / (ref_std + 1e-8)
            drift_metrics['std_drift'] = np.mean(std_drift)
            
            # Distribution drift (KS test approximation)
            drift_metrics['distribution_drift'] = np.mean(mean_drift + std_drift)
            
            return drift_metrics
        except Exception as e:
            logger.error(f"Error detecting data drift: {e}")
            return {}
    
    def detect_performance_drift(self, reference_metrics: Dict[str, float],
                               current_metrics: Dict[str, float]) -> Dict[str, float]:
        """Detect performance drift between reference and current metrics."""
        try:
            drift_metrics = {}
            
            for metric in reference_metrics:
                if metric in current_metrics:
                    ref_val = reference_metrics[metric]
                    curr_val = current_metrics[metric]
                    
                    if ref_val != 0:
                        drift = abs(curr_val - ref_val) / abs(ref_val)
                        drift_metrics[f'{metric}_drift'] = drift
                    else:
                        drift_metrics[f'{metric}_drift'] = 0
            
            return drift_metrics
        except Exception as e:
            logger.error(f"Error detecting performance drift: {e}")
            return {}
    
    def check_model_health(self, drift_metrics: Dict[str, float]) -> Dict[str, bool]:
        """Check model health based on drift metrics."""
        try:
            health_status = {}
            
            for metric, value in drift_metrics.items():
                health_status[metric] = value <= self.drift_threshold
            
            return health_status
        except Exception as e:
            logger.error(f"Error checking model health: {e}")
            return {} 