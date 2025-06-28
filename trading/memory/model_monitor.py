"""Model monitoring utilities for detecting drift and model performance issues."""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ModelMonitor:
    """Model monitoring class for tracking model performance and detecting issues."""
    
    def __init__(self):
        """Initialize the model monitor."""
        self.logger = logging.getLogger(__name__)
        self.trust_levels = {
            "lstm": 0.85,
            "xgboost": 0.78,
            "prophet": 0.72,
            "ensemble": 0.91,
            "tcn": 0.68,
            "transformer": 0.82
        }
    
    def get_model_trust_levels(self) -> Dict[str, float]:
        """Get trust levels for different models.
        
        Returns:
            Dictionary mapping model names to trust levels (0-1)
        """
        try:
            self.logger.info(f"Model trust levels retrieved: {self.trust_levels}")
            return self.trust_levels
        except Exception as e:
            self.logger.error(f"Error getting model trust levels: {str(e)}")
            return {
                "lstm": 0.5,
                "xgboost": 0.5,
                "prophet": 0.5,
                "ensemble": 0.5,
                "tcn": 0.5,
                "transformer": 0.5
            }
    
    def update_trust_level(self, model_name: str, new_trust: float):
        """Update trust level for a specific model.
        
        Args:
            model_name: Name of the model
            new_trust: New trust level (0-1)
        """
        try:
            self.trust_levels[model_name] = max(0.0, min(1.0, new_trust))
            self.logger.info(f"Updated trust level for {model_name}: {new_trust:.3f}")
        except Exception as e:
            self.logger.error(f"Error updating trust level: {str(e)}")
    
    def detect_drift(self, current_data: pd.DataFrame, historical_data: pd.DataFrame, 
                    threshold: float = 0.1, method: str = "ks_test") -> Dict[str, Any]:
        """Detect data drift between current and historical data."""
        return detect_drift(current_data, historical_data, threshold, method)
    
    def generate_strategy_priority(self, performance_metrics: Dict[str, float], 
                                 market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategy priority based on performance and market conditions."""
        return generate_strategy_priority(performance_metrics, market_conditions)


def detect_drift(
    current_data: pd.DataFrame,
    historical_data: pd.DataFrame,
    threshold: float = 0.1,
    method: str = "ks_test"
) -> Dict[str, Any]:
    """
    Detect data drift between current and historical data.
    
    Args:
        current_data: Current data distribution
        historical_data: Historical data distribution
        threshold: Drift detection threshold
        method: Drift detection method ('ks_test', 'chi_square', 'wasserstein')
        
    Returns:
        Dictionary with drift detection results
    """
    try:
        # Basic drift detection using statistical tests
        drift_results = {
            "drift_detected": False,
            "drift_score": 0.0,
            "method": method,
            "timestamp": datetime.now().isoformat(),
            "details": {}
        }
        
        # For now, return a simple implementation
        # In a real implementation, you would use proper statistical tests
        
        if method == "ks_test":
            # Kolmogorov-Smirnov test for distribution differences
            drift_score = _calculate_ks_drift(current_data, historical_data)
        elif method == "chi_square":
            # Chi-square test for categorical variables
            drift_score = _calculate_chi_square_drift(current_data, historical_data)
        elif method == "wasserstein":
            # Wasserstein distance for continuous variables
            drift_score = _calculate_wasserstein_drift(current_data, historical_data)
        else:
            drift_score = 0.0
            
        drift_results["drift_score"] = drift_score
        drift_results["drift_detected"] = drift_score > threshold
        
        logger.info(f"Drift detection completed: score={drift_score:.4f}, detected={drift_results['drift_detected']}")
        
        return drift_results
        
    except Exception as e:
        logger.error(f"Error in drift detection: {str(e)}")
        return {
            "drift_detected": False,
            "drift_score": 0.0,
            "method": method,
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

def _calculate_ks_drift(current_data: pd.DataFrame, historical_data: pd.DataFrame) -> float:
    """Calculate drift using Kolmogorov-Smirnov test."""
    try:
        # Simple implementation - in practice, you'd use scipy.stats.ks_2samp
        current_mean = current_data.mean().mean()
        historical_mean = historical_data.mean().mean()
        
        # Calculate relative difference
        drift_score = abs(current_mean - historical_mean) / (abs(historical_mean) + 1e-8)
        return min(drift_score, 1.0)  # Cap at 1.0
        
    except Exception as e:
        logger.error(f"Error in KS drift calculation: {str(e)}")
        return 0.0

def _calculate_chi_square_drift(current_data: pd.DataFrame, historical_data: pd.DataFrame) -> float:
    """Calculate drift using Chi-square test."""
    try:
        # Simple implementation for categorical variables
        current_std = current_data.std().mean()
        historical_std = historical_data.std().mean()
        
        drift_score = abs(current_std - historical_std) / (abs(historical_std) + 1e-8)
        return min(drift_score, 1.0)
        
    except Exception as e:
        logger.error(f"Error in Chi-square drift calculation: {str(e)}")
        return 0.0

def _calculate_wasserstein_drift(current_data: pd.DataFrame, historical_data: pd.DataFrame) -> float:
    """Calculate drift using Wasserstein distance."""
    try:
        # Simple implementation using mean difference
        current_median = current_data.median().mean()
        historical_median = historical_data.median().mean()
        
        drift_score = abs(current_median - historical_median) / (abs(historical_median) + 1e-8)
        return min(drift_score, 1.0)
        
    except Exception as e:
        logger.error(f"Error in Wasserstein drift calculation: {str(e)}")
        return 0.0

def generate_strategy_priority(
    performance_metrics: Dict[str, float],
    market_conditions: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate strategy priority based on performance and market conditions.
    
    Args:
        performance_metrics: Dictionary of performance metrics
        market_conditions: Dictionary of market conditions
        
    Returns:
        Dictionary with strategy priority information
    """
    try:
        priority_results = {
            "priority_score": 0.0,
            "recommended_action": "hold",
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat(),
            "reasoning": []
        }
        
        # Calculate priority score based on performance
        sharpe_ratio = performance_metrics.get("sharpe_ratio", 0.0)
        win_rate = performance_metrics.get("win_rate", 0.5)
        max_drawdown = performance_metrics.get("max_drawdown", 0.0)
        
        # Simple priority calculation
        priority_score = (sharpe_ratio * 0.4 + win_rate * 0.3 - max_drawdown * 0.3)
        priority_score = max(0.0, min(1.0, priority_score))
        
        priority_results["priority_score"] = priority_score
        
        # Determine recommended action
        if priority_score > 0.7:
            priority_results["recommended_action"] = "increase"
            priority_results["confidence"] = priority_score
        elif priority_score < 0.3:
            priority_results["recommended_action"] = "decrease"
            priority_results["confidence"] = 1.0 - priority_score
        else:
            priority_results["recommended_action"] = "hold"
            priority_results["confidence"] = 0.5
            
        # Add reasoning
        if sharpe_ratio > 1.0:
            priority_results["reasoning"].append("High Sharpe ratio indicates good risk-adjusted returns")
        if win_rate > 0.6:
            priority_results["reasoning"].append("High win rate suggests consistent performance")
        if max_drawdown > 0.2:
            priority_results["reasoning"].append("High drawdown indicates elevated risk")
            
        logger.info(f"Strategy priority generated: score={priority_score:.3f}, action={priority_results['recommended_action']}")
        
        return priority_results
        
    except Exception as e:
        logger.error(f"Error in strategy priority generation: {str(e)}")
        return {
            "priority_score": 0.5,
            "recommended_action": "hold",
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

def get_model_trust_levels() -> Dict[str, float]:
    """
    Get trust levels for different models.
    
    Returns:
        Dictionary mapping model names to trust levels (0-1)
    """
    try:
        # Default trust levels for demonstration
        trust_levels = {
            "lstm": 0.85,
            "xgboost": 0.78,
            "prophet": 0.72,
            "ensemble": 0.91,
            "tcn": 0.68,
            "transformer": 0.82
        }
        
        logger.info(f"Model trust levels retrieved: {trust_levels}")
        return trust_levels
        
    except Exception as e:
        logger.error(f"Error getting model trust levels: {str(e)}")
        return {
            "lstm": 0.5,
            "xgboost": 0.5,
            "prophet": 0.5,
            "ensemble": 0.5,
            "tcn": 0.5,
            "transformer": 0.5
        } 