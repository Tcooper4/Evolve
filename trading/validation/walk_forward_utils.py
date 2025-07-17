Forward Validation Utilities

This module provides utility functions for walk-forward validation,
wrapping the existing WalkForwardAgent for easier use.
"

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from trading.agents.walk_forward_agent import WalkForwardAgent, WalkForwardResult
from trading.agents.base_agent_interface import AgentConfig

logger = logging.getLogger(__name__)


def walk_forward_validate(
    data: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    model_factory: Callable,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    train_window_days: int = 252
    test_window_days: int =63    step_size_days: int = 21
    **kwargs
) -> Dict[str, Any]:
    ""
    Walk-forward validation utility function.
    
    Args:
        data: Input data DataFrame
        target_column: Name of the target column
        feature_columns: List of feature column names
        model_factory: Function to create model instances
        start_date: Start date for validation (optional)
        end_date: End date for validation (optional)
        train_window_days: Training window size in days
        test_window_days: Test window size in days
        step_size_days: Step size between windows in days
        **kwargs: Additional configuration parameters
        
    Returns:
        Dictionary containing validation results and performance metrics    try:
        # Create agent configuration
        config = AgentConfig(
            name="WalkForwardValidator",
            enabled=True,
            priority=1           max_concurrent_runs=1,
            timeout_seconds=300
            retry_attempts=3,
            custom_config={
                train_window_days": train_window_days,
               test_window_days": test_window_days,
               step_size_days": step_size_days,
                **kwargs
            }
        )
        
        # Create walk-forward agent
        agent = WalkForwardAgent(config)
        
        # Run validation
        results = agent.run_walk_forward_validation(
            data=data,
            target_column=target_column,
            feature_columns=feature_columns,
            model_factory=model_factory,
            start_date=start_date,
            end_date=end_date
        )
        
        # Calculate summary statistics
        performance_summary = _calculate_performance_summary(results)
        
        return {
           success True,
            results": [result.__dict__ for result in results],
          performance_summary: performance_summary,
          total_windows": len(results),
            train_window_days": train_window_days,
           test_window_days": test_window_days,
           step_size_days: step_size_days
        }
        
    except Exception as e:
        logger.error(f"Walk-forward validation failed: {str(e)}")
        return {
            successFalse,
           errortr(e),
         results:    performance_summary:[object Object]
        }


def _calculate_performance_summary(results: List[WalkForwardResult]) -> Dict[str, Any]:
    ""Calculatesummary statistics from walk-forward results."""
    if not results:
        return {}
    
    # Extract performance metrics
    sharpe_ratios = [r.model_performance.get(sharpe_ratio", 0) for r in results]
    max_drawdowns = [r.model_performance.get(max_drawdown", 0) for r in results]
    mse_scores = [r.model_performance.get(mse) for r in results]
    win_rates = [r.model_performance.get(win_rate", 0) for r in results]
    
    # Calculate statistics
    summary = [object Object]       mean_sharpe": pd.Series(sharpe_ratios).mean(),
        std_sharpe": pd.Series(sharpe_ratios).std(),
      mean_drawdown:pd.Series(max_drawdowns).mean(),
        max_drawdown:pd.Series(max_drawdowns).max(),
      mean_mse: pd.Series(mse_scores).mean(),
      mean_win_rate": pd.Series(win_rates).mean(),
      total_windows": len(results),
        performance_trend": _calculate_performance_trend(results)
    }
    
    return summary


def _calculate_performance_trend(results: List[WalkForwardResult]) -> Dict[str, float]:
    """Calculate performance trend over time."""
    if len(results) <2
        return {"trend": insufficient_data"}
    
    # Use first and last few windows to calculate trend
    early_periods = min(3, len(results) // 3)
    late_periods = min(3, len(results) //3
    
    early_sharpe = pd.Series([r.model_performance.get(sharpe_ratio", 0) 
                             for r in results[:early_periods]]).mean()
    late_sharpe = pd.Series([r.model_performance.get(sharpe_ratio", 0) 
                            for r in results[-late_periods:]]).mean()
    
    trend = late_sharpe - early_sharpe
    
    return {
     early_sharpe": early_sharpe,
    late_sharpe": late_sharpe,
      trend": trend,
        trend_direction": "improving" if trend > 0lse "declining" if trend < 0 elsestable"
    }


def get_walk_forward_agent() -> WalkForwardAgent:
    " a configured walk-forward agent instance."""
    config = AgentConfig(
        name="WalkForwardValidator,      enabled=True,
        priority=1    max_concurrent_runs=1,
        timeout_seconds=30     retry_attempts=3,
        custom_config={}
    )
    return WalkForwardAgent(config) 