# Walk Forward Validation Utilities

import logging
from typing import Any, Dict, List

import pandas as pd

from dataclasses import dataclass

from trading.agents.base_agent_interface import AgentConfig

try:
    from trading.agents.walk_forward_agent import WalkForwardAgent, WalkForwardResult
except ImportError:
    WalkForwardAgent = None

    @dataclass
    class WalkForwardResult:
        """Stub when WalkForwardAgent is not available (rationalized to _dead_code)."""
        model_performance: Dict[str, Any]

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


def get_walk_forward_agent():
    """Get a configured walk-forward agent instance. Returns None if agent was rationalized."""
    if WalkForwardAgent is None:
        return None
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
            if self.agent is None:
                return {"success": False, "error": "Walk-forward validation is not available (agent rationalized)."}
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

    def walk_forward_test(
        self,
        strategy: Any,
        data: pd.DataFrame,
        train_window: int = 252,
        test_window: int = 63,
        step_size: int = 21,
        num_iterations: int = 5,
        progress_callback: Any = None,
    ) -> Dict[str, Any]:
        """
        Run walk-forward backtest: roll train/test windows, collect returns, aggregate metrics.
        Compatible with Strategy Testing page (strategy, data, train_window, test_window, step_size, num_iterations).

        Returns:
            Dict with avg_return, consistency_score, win_rate, num_iterations, returns.
        """
        try:
            if data is None or len(data) < train_window + test_window:
                return {
                    "avg_return": 0.0,
                    "consistency_score": 0.0,
                    "win_rate": 0.0,
                    "error": "Insufficient data for walk-forward",
                }
            returns_list = []
            wins = 0
            total = 0
            n = len(data)
            for i in range(num_iterations):
                start = i * step_size
                train_end = start + train_window
                test_end = min(train_end + test_window, n)
                if test_end > n or train_end > n - 1:
                    break
                train_df = data.iloc[start:train_end]
                test_df = data.iloc[train_end:test_end]
                if progress_callback:
                    progress_callback(i + 1, num_iterations)
                try:
                    if hasattr(strategy, "backtest"):
                        result = strategy.backtest(train_df, test_df)
                    elif hasattr(strategy, "run_backtest"):
                        result = strategy.run_backtest(train_df, test_df)
                    else:
                        total_return = 0.0
                        if len(test_df) >= 2 and "close" in test_df.columns:
                            total_return = (test_df["close"].iloc[-1] / test_df["close"].iloc[0] - 1.0)
                        returns_list.append(total_return)
                        total += 1
                        wins += 1 if total_return > 0 else 0
                        continue
                    tr = result.get("total_return", result.get("returns", 0.0))
                    if isinstance(tr, (list, pd.Series)):
                        tr = float(tr[-1]) if len(tr) else 0.0
                    returns_list.append(float(tr))
                    total += 1
                    wins += 1 if float(tr) > 0 else 0
                except Exception as e:
                    self.logger.debug("Walk-forward window %s failed: %s", i, e)
            if progress_callback and total == num_iterations:
                progress_callback(num_iterations, num_iterations)
            avg_return = float(pd.Series(returns_list).mean()) if returns_list else 0.0
            consistency = float(pd.Series(returns_list).std()) if len(returns_list) > 1 else 0.0
            consistency_score = (1.0 / (1.0 + consistency)) if consistency >= 0 else 0.0
            win_rate = (wins / total) if total else 0.0
            return {
                "avg_return": avg_return,
                "consistency_score": consistency_score,
                "win_rate": win_rate,
                "num_iterations": total,
                "num_windows": total,
                "returns": returns_list,
            }
        except Exception as e:
            self.logger.error(f"Walk-forward test failed: {e}")
            return {
                "avg_return": 0.0,
                "consistency_score": 0.0,
                "win_rate": 0.0,
                "error": str(e),
            }