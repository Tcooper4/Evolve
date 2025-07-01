"""Utility modules for the trading package."""

import logging as std_logging
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from .logging import LogManager, ModelLogger, DataLogger, PerformanceLogger

utils_logger = std_logging.getLogger(__name__)

def check_model_performance(metrics: Dict[str, Any]) -> bool:
    """Check if model performance meets minimum criteria."""
    try:
        # Basic performance checks
        if 'accuracy' in metrics and metrics['accuracy'] < 0.6:
            return False
        if 'sharpe_ratio' in metrics and metrics['sharpe_ratio'] < 0.5:
            return False
        if 'max_drawdown' in metrics and metrics['max_drawdown'] < -0.2:
            return False
        return True
    except Exception as e:
        utils_logger.error(f"Error checking model performance: {e}")
        return False

def detect_model_drift(model_id: str) -> bool:
    """Detect if model has drifted."""
    try:
        # Simple drift detection - in practice, you'd use statistical tests
        metrics_path = f"metrics/{model_id}_metrics.json"
        if not os.path.exists(metrics_path):
            return False
            
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
        # Check if recent performance has degraded
        recent_accuracy = metrics.get('recent_accuracy', 1.0)
        if recent_accuracy < 0.5:
            return True
            
        return False
    except Exception as e:
        utils_logger.error(f"Error detecting model drift: {e}")
        return False

def validate_update_result(model_id: str, result: Dict[str, Any]) -> bool:
    """Validate model update result."""
    try:
        # Basic validation
        required_keys = ['accuracy', 'sharpe_ratio', 'timestamp']
        for key in required_keys:
            if key not in result:
                return False
                
        # Check if performance improved
        if result.get('accuracy', 0) < 0.5:
            return False
            
        return True
    except Exception as e:
        utils_logger.error(f"Error validating update result: {e}")
        return False

def calculate_reweighting_factors(models: List[str]) -> Dict[str, float]:
    """Calculate ensemble reweighting factors."""
    try:
        factors = {}
        total_performance = 0.0
        
        for model_id in models:
            metrics = get_model_metrics(model_id)
            performance = metrics.get('sharpe_ratio', 0.0)
            factors[model_id] = max(performance, 0.0)
            total_performance += factors[model_id]
            
        # Normalize weights
        if total_performance > 0:
            for model_id in factors:
                factors[model_id] /= total_performance
        else:
            # Equal weights if no performance data
            equal_weight = 1.0 / len(models)
            for model_id in models:
                factors[model_id] = equal_weight
                
        return factors
    except Exception as e:
        utils_logger.error(f"Error calculating reweighting factors: {e}")
        return {model_id: 1.0/len(models) for model_id in models}

def get_model_metrics(model_id: str) -> Dict[str, Any]:
    """Get metrics for a specific model."""
    try:
        metrics_path = f"metrics/{model_id}_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        utils_logger.error(f"Error getting model metrics: {e}")
        return {}

def check_update_frequency(model_id: str) -> bool:
    """Check if model needs updating based on frequency."""
    try:
        metrics = get_model_metrics(model_id)
        last_update = metrics.get('last_update')
        if not last_update:
            return True
            
        last_update_dt = datetime.fromisoformat(last_update)
        days_since_update = (datetime.now() - last_update_dt).days
        
        # Update if more than 7 days old
        return days_since_update > 7
    except Exception as e:
        utils_logger.error(f"Error checking update frequency: {e}")
        return True

def get_ensemble_weights() -> Dict[str, float]:
    """Get current ensemble weights."""
    try:
        weights_path = "models/ensemble_weights.json"
        if os.path.exists(weights_path):
            with open(weights_path, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        utils_logger.error(f"Error getting ensemble weights: {e}")
        return {}

def save_ensemble_weights(weights: Dict[str, float]) -> None:
    """Save ensemble weights."""
    try:
        weights_path = "models/ensemble_weights.json"
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        with open(weights_path, 'w') as f:
            json.dump(weights, f, indent=2)
    except Exception as e:
        utils_logger.error(f"Error saving ensemble weights: {e}")

def check_data_quality(data: pd.DataFrame) -> bool:
    """Check data quality for model training."""
    try:
        # Basic quality checks
        if data.empty:
            return False
            
        # Check for missing values
        if data.isnull().sum().sum() > len(data) * 0.1:
            return False
            
        # Check for sufficient data
        if len(data) < 100:
            return False
            
        return True
    except Exception as e:
        utils_logger.error(f"Error checking data quality: {e}")
        return False

__all__ = [
    'LogManager',
    'ModelLogger',
    'DataLogger',
    'PerformanceLogger',
    'check_model_performance',
    'detect_model_drift',
    'validate_update_result',
    'calculate_reweighting_factors',
    'get_model_metrics',
    'check_update_frequency',
    'get_ensemble_weights',
    'save_ensemble_weights',
    'check_data_quality'
] 