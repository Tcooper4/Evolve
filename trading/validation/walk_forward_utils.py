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
