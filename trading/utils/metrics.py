"""Utility functions for calculating performance metrics."""

import numpy as np
import pandas as pd
from typing import Dict, Union, Tuple, Any, Optional

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate forecast performance metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    # Calculate basic metrics
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    
    # Calculate directional accuracy
    direction_true = np.sign(np.diff(y_true))
    direction_pred = np.sign(np.diff(y_pred))
    directional_accuracy = np.mean(direction_true == direction_pred)
    
    # Calculate R-squared
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'directional_accuracy': directional_accuracy,
        'r2': r2
    }

def calculate_trading_metrics(returns: np.ndarray, 
                            benchmark_returns: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Calculate trading performance metrics.
    
    Args:
        returns: Strategy returns
        benchmark_returns: Optional benchmark returns
        
    Returns:
        Dictionary of metrics
    """
    # Calculate basic metrics
    total_return = np.prod(1 + returns) - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = np.std(returns) * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
    
    # Calculate drawdown
    cumulative_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    # Calculate win rate
    win_rate = np.mean(returns > 0)
    
    # Calculate alpha and beta if benchmark provided
    if benchmark_returns is not None:
        beta = np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
        alpha = annualized_return - beta * np.mean(benchmark_returns) * 252
    else:
        alpha = None
        beta = None
    
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate
    }
    
    if alpha is not None:
        metrics['alpha'] = alpha
        metrics['beta'] = beta
    
    return metrics

def calculate_regime_metrics(returns: np.ndarray, 
                           regime_labels: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Calculate performance metrics for each market regime.
    
    Args:
        returns: Strategy returns
        regime_labels: Array of regime labels (e.g., 'bull', 'bear', 'neutral')
        
    Returns:
        Dictionary of metrics for each regime
    """
    regime_metrics = {}
    
    for regime in np.unique(regime_labels):
        regime_returns = returns[regime_labels == regime]
        regime_metrics[regime] = calculate_trading_metrics(regime_returns)
    
    return regime_metrics

def calculate_model_confidence(predictions: np.ndarray, 
                             confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> float:
    """Calculate model confidence score.
    
    Args:
        predictions: Model predictions
        confidence_intervals: Optional tuple of (lower, upper) confidence bounds
        
    Returns:
        Confidence score between 0 and 1
    """
    if confidence_intervals is not None:
        lower, upper = confidence_intervals
        interval_width = upper - lower
        confidence = 1 - np.mean(interval_width / np.abs(predictions))
    else:
        # Use prediction stability as proxy for confidence
        rolling_std = pd.Series(predictions).rolling(window=5).std()
        confidence = 1 - np.mean(rolling_std / np.abs(predictions))
    
    return np.clip(confidence, 0, 1) 