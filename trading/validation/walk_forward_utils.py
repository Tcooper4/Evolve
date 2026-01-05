# Walk Forward Validation Utilities

import logging
from typing import Any, Dict, List

import pandas as pd

from trading.agents.base_agent_interface import AgentConfig
from trading.agents.walk_forward_agent import WalkForwardAgent, WalkForwardResult

logger = logging.getLogger(__name__)


def walk_forward_validate(data, test_window_days: int = 63, step_size_days: int = 21):
    """Walk forward validation utilities."""
    # Implementation here


def _calculate_performance_summary(results: List[WalkForwardResult]) -> Dict[str, Any]:
    """Calculatesummary statistics from walk-forward results."""
    if not results:
        return {}

    # Extract performance metrics
    sharpe_ratios = [r.model_performance.get("sharpe_ratio", 0) for r in results]
    max_drawdowns = [r.model_performance.get("max_drawdown", 0) for r in results]
    mse_scores = [r.model_performance.get(mse) for r in results]
    win_rates = [r.model_performance.get("win_rate", 0) for r in results]

    # Calculate statistics
    summary = {
        "mean_sharpe": pd.Series(sharpe_ratios).mean(),
        "mean_drawdown": pd.Series(max_drawdowns).mean(),
        "mean_mse": pd.Series(mse_scores).mean(),
        "mean_win_rate": pd.Series(win_rates).mean(),
    }

    return summary


def _calculate_performance_trend(results: List[WalkForwardResult]) -> Dict[str, float]:
    """Calculate performance trend over time."""
    if len(results) < 2:
        return {"trend": "insufficient_data"}

    # Use first and last few windows to calculate trend
    early_periods = min(3, len(results) // 3)
    late_periods = min(3, len(results) // 3)

    early_sharpe = pd.Series(
        [r.model_performance.get("sharpe_ratio", 0) for r in results[:early_periods]]
    ).mean()
    late_sharpe = pd.Series(
        [r.model_performance.get("sharpe_ratio", 0) for r in results[-late_periods:]]
    ).mean()

    trend = late_sharpe - early_sharpe

    return {"early_sharpe": early_sharpe, "late_sharpe": late_sharpe, "trend": trend}


def get_walk_forward_agent() -> WalkForwardAgent:
    """Get a configured walk-forward agent instance."""
    config = AgentConfig(
        name="WalkForwardValidator",
        enabled=True,
        priority=1,
        max_concurrent_runs=1,
        timeout_seconds=30,
        retry_attempts=3,
        custom_config={},
    )
    return WalkForwardAgent(config)


class WalkForwardValidator:
    """Wrapper class for walk-forward validation functionality."""
    
    def __init__(self):
        """Initialize the walk-forward validator."""
        self.agent = get_walk_forward_agent()
        self.logger = logging.getLogger(__name__)
    
    def validate(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        model_factory: Any,
        test_window_days: int = 63,
        step_size_days: int = 21,
    ) -> Dict[str, Any]:
        """
        Run walk-forward validation.
        
        Args:
            data: Input data with datetime index
            target_column: Target variable column
            feature_columns: Feature columns
            model_factory: Function to create model instances
            test_window_days: Test window size in days
            step_size_days: Step size for walk-forward in days
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Use the agent to run validation
            import asyncio
            result = asyncio.run(
                self.agent.execute(
                    action="run_walk_forward_validation",
                    data=data,
                    target_column=target_column,
                    feature_columns=feature_columns,
                    model_factory=model_factory,
                )
            )
            
            if result.success:
                return {
                    "success": True,
                    "results": result.data.get("walk_forward_results", []),
                    "performance_summary": result.data.get("performance_summary", {}),
                }
            else:
                return {
                    "success": False,
                    "error": result.error_message,
                }
        except Exception as e:
            self.logger.error(f"Error in walk-forward validation: {e}")
            return {
                "success": False,
                "error": str(e),
            }