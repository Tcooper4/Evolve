"""
Performance Analysis for Backtesting

This module contains the PerformanceAnalyzer class for aggregating and reporting
backtest performance metrics, attribution, and summary statistics.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """Aggregates and reports backtest performance metrics and attribution."""

    def __init__(self):
        self.metrics_history: List[Dict[str, Any]] = []

    def compute_metrics(
        self, df: pd.DataFrame, trade_log: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compute performance metrics from price and trade data."""
        metrics = {}
        try:
            metrics["total_return"] = (
                (df["equity_curve"].iloc[-1] / df["equity_curve"].iloc[0]) - 1
                if "equity_curve" in df
                else np.nan
            )

            # Use log returns for accurate compounding
            if "returns" in df:
                log_returns = np.log1p(df["returns"])
                metrics["annualized_return"] = log_returns.mean() * 252
                metrics["volatility"] = log_returns.std() * np.sqrt(252)
                metrics["sharpe_ratio"] = (
                    metrics["annualized_return"] / metrics["volatility"]
                    if metrics["volatility"] > 0
                    else np.nan
                )
                
                # Warn users when Sharpe ratio is low
                if metrics["sharpe_ratio"] < 1:
                    logger.warning("⚠️ Warning: Strategy Sharpe ratio below 1.0")
            else:
                metrics["annualized_return"] = np.nan
                metrics["volatility"] = np.nan
                metrics["sharpe_ratio"] = np.nan

            metrics["max_drawdown"] = (
                self._max_drawdown(df["equity_curve"])
                if "equity_curve" in df
                else np.nan
            )
            metrics["num_trades"] = len(trade_log) if trade_log is not None else 0
            metrics["win_rate"] = (
                (trade_log["pnl"] > 0).mean() if "pnl" in trade_log else np.nan
            )
            
            # Warn users when win rate is low
            if metrics["win_rate"] < 0.5:
                logger.warning("⚠️ Warning: Strategy win rate below 50%")
            metrics["avg_win"] = (
                trade_log[trade_log["pnl"] > 0]["pnl"].mean()
                if "pnl" in trade_log
                else np.nan
            )
            metrics["avg_loss"] = (
                trade_log[trade_log["pnl"] < 0]["pnl"].mean()
                if "pnl" in trade_log
                else np.nan
            )
            metrics["profit_factor"] = (
                abs(metrics["avg_win"] / metrics["avg_loss"])
                if metrics["avg_loss"]
                else np.nan
            )
        except Exception as e:
            logger.warning(f"Performance metric calculation failed: {e}")
        return metrics

    def _max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown from equity curve."""
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown.min()

    def combine_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine multiple metrics dictionaries into a summary."""
        if not metrics_list:
            return {}
        summary = {}
        try:
            for key in metrics_list[0].keys():
                values = [
                    m[key] for m in metrics_list if key in m and m[key] is not None
                ]
                if values:
                    summary[key] = np.mean(values)
                else:
                    summary[key] = np.nan
        except Exception as e:
            logger.warning(f"Failed to combine metrics: {e}")
        return summary

    def add_metrics(self, metrics: Dict[str, Any]) -> None:
        """Add a metrics dictionary to the history."""
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]

    def get_history(self) -> List[Dict[str, Any]]:
        """Get the history of all computed metrics."""
        return self.metrics_history.copy()

    def clear_history(self) -> None:
        """Clear the metrics history."""
        self.metrics_history.clear()

    def get_fallback_metrics(self) -> Dict[str, Any]:
        """Return a fallback metrics dictionary in case of failure."""
        return {
            "total_return": np.nan,
            "annualized_return": np.nan,
            "volatility": np.nan,
            "sharpe_ratio": np.nan,
            "max_drawdown": np.nan,
            "num_trades": 0,
            "win_rate": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "profit_factor": np.nan,
        }

    def run_monte_carlo_simulation(
        self,
        returns: pd.Series,
        n_paths: int = 1000,
        noise_std: float = 0.0,
        output_csv: str = None,
        output_plot: str = None,
        seed: int = 42,
        confidence: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation by resampling returns and injecting noise.
        Args:
            returns: Series of returns (can be daily, etc.)
            n_paths: Number of simulation paths
            noise_std: Stddev of Gaussian noise to inject (as fraction, e.g. 0.01)
            output_csv: Optional path to save simulated equity paths
            output_plot: Optional path to save plot of paths and confidence bounds
            seed: Random seed
            confidence: Confidence interval (e.g. 0.95 for 95%%)
        Returns:
            Dict with mean, worst-case, and confidence bounds
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        np.random.seed(seed)
        returns = returns.dropna().values
        n = len(returns)
        paths = np.zeros((n_paths, n))
        for i in range(n_paths):
            sampled = np.random.choice(returns, size=n, replace=True)
            if noise_std > 0:
                sampled = sampled + np.random.normal(0, noise_std, size=n)
            paths[i] = sampled
        # Simulate equity curves
        equity_curves = np.cumprod(1 + paths, axis=1)
        final_equity = equity_curves[:, -1]
        mean_return = np.mean(final_equity - 1)
        worst_case = np.min(final_equity - 1)
        lower = np.percentile(final_equity - 1, (1 - confidence) / 2 * 100)
        upper = np.percentile(final_equity - 1, (1 + confidence) / 2 * 100)
        result = {
            'mean_return': mean_return,
            'worst_case': worst_case,
            f'lower_{int(confidence*100)}': lower,
            f'upper_{int(confidence*100)}': upper,
        }
        if output_csv:
            pd.DataFrame(equity_curves).to_csv(output_csv, index=False)
        if output_plot:
            plt.figure(figsize=(12, 6))
            plt.plot(equity_curves.T, color='gray', alpha=0.01)
            plt.plot(np.mean(equity_curves, axis=0), color='blue', label='Mean Path')
            plt.title('Monte Carlo Simulated Equity Paths')
            plt.xlabel('Time Step')
            plt.ylabel('Equity')
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_plot)
            plt.close()
        return result
