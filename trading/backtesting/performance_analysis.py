"""
Enhanced Performance Analysis for Backtesting

This module contains the PerformanceAnalyzer class for aggregating and reporting
backtest performance metrics, attribution, and summary statistics with comprehensive
cost modeling including commission, slippage, and cash drag adjustments.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CostParameters:
    """Configuration for trading costs."""

    commission_rate: float = 0.001  # 0.1% commission per trade
    slippage_rate: float = 0.002  # 0.2% slippage per trade
    spread_rate: float = 0.0005  # 0.05% bid-ask spread
    cash_drag_rate: float = 0.02  # 2% annual cash drag (opportunity cost)
    min_commission: float = 1.0  # Minimum commission per trade
    max_commission: float = 1000.0  # Maximum commission per trade
    enable_cost_adjustment: bool = True


class PerformanceAnalyzer:
    """Enhanced performance analyzer with comprehensive cost modeling."""

    def __init__(self, cost_params: Optional[CostParameters] = None):
        self.metrics_history: List[Dict[str, Any]] = []
        self.cost_params = cost_params or CostParameters()
        self.logger = logging.getLogger(self.__class__.__name__)

    def compute_metrics(
        self,
        df: pd.DataFrame,
        trade_log: pd.DataFrame,
        cost_params: Optional[CostParameters] = None,
    ) -> Dict[str, Any]:
        """
        Compute comprehensive performance metrics including cost-adjusted returns.

        Args:
            df: DataFrame with price and equity data
            trade_log: DataFrame with trade information
            cost_params: Optional cost parameters to override defaults

        Returns:
            Dictionary with comprehensive performance metrics
        """
        if cost_params is not None:
            self.cost_params = cost_params

        metrics = {}
        try:
            # Basic return calculations
            metrics.update(self._calculate_basic_returns(df))

            # Cost-adjusted metrics
            if self.cost_params.enable_cost_adjustment:
                metrics.update(self._calculate_cost_adjusted_metrics(df, trade_log))

            # Risk metrics
            metrics.update(self._calculate_risk_metrics(df))

            # Trade analysis
            metrics.update(self._calculate_trade_metrics(trade_log))

            # Cost breakdown
            if self.cost_params.enable_cost_adjustment:
                metrics.update(self._calculate_cost_breakdown(trade_log))

            # Cash efficiency metrics
            metrics.update(self._calculate_cash_efficiency_metrics(df, trade_log))

        except Exception as e:
            self.logger.warning(f"Performance metric calculation failed: {e}")
            metrics = self.get_fallback_metrics()

        return metrics

    def _calculate_basic_returns(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate basic return metrics."""
        metrics = {}

        if "equity_curve" in df.columns:
            metrics["total_return"] = (
                df["equity_curve"].iloc[-1] / df["equity_curve"].iloc[0]
            ) - 1
        else:
            metrics["total_return"] = np.nan

        # Use log returns for accurate compounding
        if "returns" in df.columns:
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
                self.logger.warning("âš ï¸ Warning: Strategy Sharpe ratio below 1.0")
        else:
            metrics["annualized_return"] = np.nan
            metrics["volatility"] = np.nan
            metrics["sharpe_ratio"] = np.nan

        metrics["max_drawdown"] = (
            self._max_drawdown(df["equity_curve"])
            if "equity_curve" in df.columns
            else np.nan
        )

        return metrics

    def _calculate_cost_adjusted_metrics(
        self, df: pd.DataFrame, trade_log: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate cost-adjusted performance metrics."""
        metrics = {}

        if trade_log.empty:
            return {
                "cost_adjusted_return": metrics.get("total_return", np.nan),
                "cost_adjusted_sharpe": metrics.get("sharpe_ratio", np.nan),
                "total_trading_costs": 0.0,
                "cost_per_trade": 0.0,
                "cost_impact": 0.0,
            }

        # Calculate total trading costs
        total_costs = self._calculate_total_trading_costs(trade_log)

        # Calculate cost-adjusted equity curve
        if "equity_curve" in df.columns:
            cost_adjusted_equity = df["equity_curve"].copy()

            # Subtract cumulative costs from equity curve
            cumulative_costs = 0.0
            for _, trade in trade_log.iterrows():
                trade_cost = self._calculate_trade_cost(trade)
                cumulative_costs += trade_cost
                # Find the index after this trade and adjust equity
                trade_idx = df.index.get_loc(trade.get("timestamp", df.index[0]))
                if trade_idx < len(cost_adjusted_equity):
                    cost_adjusted_equity.iloc[trade_idx:] -= trade_cost

            # Calculate cost-adjusted returns
            cost_adjusted_return = (
                cost_adjusted_equity.iloc[-1] / cost_adjusted_equity.iloc[0]
            ) - 1

            # Calculate cost-adjusted Sharpe ratio
            cost_adjusted_returns = cost_adjusted_equity.pct_change().dropna()
            cost_adjusted_volatility = cost_adjusted_returns.std() * np.sqrt(252)
            cost_adjusted_sharpe = (
                cost_adjusted_returns.mean() * 252 / cost_adjusted_volatility
                if cost_adjusted_volatility > 0
                else np.nan
            )

            metrics.update(
                {
                    "cost_adjusted_return": cost_adjusted_return,
                    "cost_adjusted_sharpe": cost_adjusted_sharpe,
                    "total_trading_costs": total_costs,
                    "cost_per_trade": (
                        total_costs / len(trade_log) if len(trade_log) > 0 else 0.0
                    ),
                    "cost_impact": (
                        metrics.get("total_return", 0) - cost_adjusted_return
                    )
                    * 100,
                }
            )

        return metrics

    def _calculate_total_trading_costs(self, trade_log: pd.DataFrame) -> float:
        """Calculate total trading costs from trade log."""
        total_costs = 0.0

        for _, trade in trade_log.iterrows():
            trade_cost = self._calculate_trade_cost(trade)
            total_costs += trade_cost

        return total_costs

    def _calculate_trade_cost(self, trade: pd.Series) -> float:
        """Calculate cost for a single trade."""
        try:
            # Extract trade information
            price = trade.get("price", 0)
            quantity = trade.get("quantity", 0)
            trade_value = abs(price * quantity)

            if trade_value == 0:
                return 0.0

            # Commission
            commission = max(
                self.cost_params.min_commission,
                min(
                    trade_value * self.cost_params.commission_rate,
                    self.cost_params.max_commission,
                ),
            )

            # Slippage
            slippage = trade_value * self.cost_params.slippage_rate

            # Spread
            spread = trade_value * self.cost_params.spread_rate

            # Total cost
            total_cost = commission + slippage + spread

            return total_cost

        except Exception as e:
            self.logger.warning(f"Error calculating trade cost: {e}")
            return 0.0

    def _calculate_risk_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        metrics = {}

        if "returns" in df.columns:
            returns = df["returns"].dropna()

            # Value at Risk (VaR)
            metrics["var_95"] = np.percentile(returns, 5)
            metrics["var_99"] = np.percentile(returns, 1)

            # Conditional Value at Risk (CVaR)
            metrics["cvar_95"] = returns[returns <= metrics["var_95"]].mean()
            metrics["cvar_99"] = returns[returns <= metrics["var_99"]].mean()

            # Downside deviation
            downside_returns = returns[returns < 0]
            metrics["downside_deviation"] = (
                downside_returns.std() if len(downside_returns) > 0 else 0
            )

            # Sortino ratio
            risk_free_return = 0.02 / 252  # Daily risk-free rate
            excess_return = returns.mean() - risk_free_return
            metrics["sortino_ratio"] = (
                excess_return / metrics["downside_deviation"]
                if metrics["downside_deviation"] > 0
                else np.nan
            )

            # Calmar ratio
            metrics["calmar_ratio"] = (
                metrics.get("annualized_return", 0)
                / abs(metrics.get("max_drawdown", 1))
                if metrics.get("max_drawdown", 0) != 0
                else np.nan
            )

        return metrics

    def _calculate_trade_metrics(self, trade_log: pd.DataFrame) -> Dict[str, float]:
        """Calculate trade-specific metrics."""
        metrics = {}

        if trade_log.empty:
            return {
                "num_trades": 0,
                "win_rate": np.nan,
                "avg_win": np.nan,
                "avg_loss": np.nan,
                "profit_factor": np.nan,
                "avg_trade_duration": np.nan,
                "largest_win": np.nan,
                "largest_loss": np.nan,
            }

        metrics["num_trades"] = len(trade_log)

        if "pnl" in trade_log.columns:
            pnl_series = trade_log["pnl"].dropna()

            if len(pnl_series) > 0:
                metrics["win_rate"] = (pnl_series > 0).mean()
                metrics["avg_win"] = (
                    pnl_series[pnl_series > 0].mean()
                    if (pnl_series > 0).any()
                    else np.nan
                )
                metrics["avg_loss"] = (
                    pnl_series[pnl_series < 0].mean()
                    if (pnl_series < 0).any()
                    else np.nan
                )
                metrics["largest_win"] = (
                    pnl_series.max() if len(pnl_series) > 0 else np.nan
                )
                metrics["largest_loss"] = (
                    pnl_series.min() if len(pnl_series) > 0 else np.nan
                )

                # Profit factor
                total_wins = pnl_series[pnl_series > 0].sum()
                total_losses = abs(pnl_series[pnl_series < 0].sum())
                metrics["profit_factor"] = (
                    total_wins / total_losses if total_losses > 0 else np.nan
                )

                # Warn users when win rate is low
                if metrics["win_rate"] < 0.5:
                    self.logger.warning("âš ï¸ Warning: Strategy win rate below 50%")
            else:
                metrics.update(
                    {
                        "win_rate": np.nan,
                        "avg_win": np.nan,
                        "avg_loss": np.nan,
                        "profit_factor": np.nan,
                        "largest_win": np.nan,
                        "largest_loss": np.nan,
                    }
                )

        # Calculate average trade duration if timestamps are available
        if "timestamp" in trade_log.columns and "exit_time" in trade_log.columns:
            try:
                durations = []
                for _, trade in trade_log.iterrows():
                    if pd.notna(trade["timestamp"]) and pd.notna(trade["exit_time"]):
                        duration = (
                            trade["exit_time"] - trade["timestamp"]
                        ).total_seconds() / 3600  # hours
                        durations.append(duration)

                metrics["avg_trade_duration"] = (
                    np.mean(durations) if durations else np.nan
                )
            except Exception as e:
                self.logger.warning(f"Error calculating trade duration: {e}")
                metrics["avg_trade_duration"] = np.nan
        else:
            metrics["avg_trade_duration"] = np.nan

        return metrics

    def _calculate_cost_breakdown(self, trade_log: pd.DataFrame) -> Dict[str, float]:
        """Calculate detailed cost breakdown."""
        if trade_log.empty:
            return {
                "total_commission": 0.0,
                "total_slippage": 0.0,
                "total_spread": 0.0,
                "cost_percentage": 0.0,
                "commission_percentage": 0.0,
                "slippage_percentage": 0.0,
                "spread_percentage": 0.0,
            }

        total_commission = 0.0
        total_slippage = 0.0
        total_spread = 0.0
        total_trade_value = 0.0

        for _, trade in trade_log.iterrows():
            price = trade.get("price", 0)
            quantity = trade.get("quantity", 0)
            trade_value = abs(price * quantity)

            if trade_value > 0:
                # Commission
                commission = max(
                    self.cost_params.min_commission,
                    min(
                        trade_value * self.cost_params.commission_rate,
                        self.cost_params.max_commission,
                    ),
                )

                # Slippage
                slippage = trade_value * self.cost_params.slippage_rate

                # Spread
                spread = trade_value * self.cost_params.spread_rate

                total_commission += commission
                total_slippage += slippage
                total_spread += spread
                total_trade_value += trade_value

        total_costs = total_commission + total_slippage + total_spread

        return {
            "total_commission": total_commission,
            "total_slippage": total_slippage,
            "total_spread": total_spread,
            "cost_percentage": (
                (total_costs / total_trade_value * 100)
                if total_trade_value > 0
                else 0.0
            ),
            "commission_percentage": (
                (total_commission / total_trade_value * 100)
                if total_trade_value > 0
                else 0.0
            ),
            "slippage_percentage": (
                (total_slippage / total_trade_value * 100)
                if total_trade_value > 0
                else 0.0
            ),
            "spread_percentage": (
                (total_spread / total_trade_value * 100)
                if total_trade_value > 0
                else 0.0
            ),
        }

    def _calculate_cash_efficiency_metrics(
        self, df: pd.DataFrame, trade_log: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate cash efficiency and utilization metrics."""
        metrics = {}

        if "equity_curve" in df.columns and "cash" in df.columns:
            # Cash utilization
            avg_cash_utilization = (df["equity_curve"] - df["cash"]) / df[
                "equity_curve"
            ]
            metrics["avg_cash_utilization"] = avg_cash_utilization.mean()
            metrics["min_cash_utilization"] = avg_cash_utilization.min()
            metrics["max_cash_utilization"] = avg_cash_utilization.max()

            # Cash drag (opportunity cost)
            avg_cash = df["cash"].mean()
            initial_cash = df["cash"].iloc[0]
            cash_drag_cost = avg_cash * self.cost_params.cash_drag_rate / 252 * len(df)
            metrics["cash_drag_cost"] = cash_drag_cost
            metrics["cash_drag_percentage"] = (
                (cash_drag_cost / initial_cash * 100) if initial_cash > 0 else 0.0
            )

            # Turnover ratio
            if not trade_log.empty and "quantity" in trade_log.columns:
                total_volume = trade_log["quantity"].abs().sum()
                avg_portfolio_value = df["equity_curve"].mean()
                metrics["turnover_ratio"] = (
                    total_volume / avg_portfolio_value
                    if avg_portfolio_value > 0
                    else 0.0
                )
            else:
                metrics["turnover_ratio"] = np.nan
        else:
            metrics.update(
                {
                    "avg_cash_utilization": np.nan,
                    "min_cash_utilization": np.nan,
                    "max_cash_utilization": np.nan,
                    "cash_drag_cost": 0.0,
                    "cash_drag_percentage": 0.0,
                    "turnover_ratio": np.nan,
                }
            )

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
            self.logger.warning(f"Failed to combine metrics: {e}")
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
            "cost_adjusted_return": np.nan,
            "cost_adjusted_sharpe": np.nan,
            "total_trading_costs": 0.0,
            "cost_per_trade": 0.0,
            "cost_impact": 0.0,
            "num_trades": 0,
            "win_rate": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "profit_factor": np.nan,
            "var_95": np.nan,
            "var_99": np.nan,
            "cvar_95": np.nan,
            "cvar_99": np.nan,
            "sortino_ratio": np.nan,
            "calmar_ratio": np.nan,
            "total_commission": 0.0,
            "total_slippage": 0.0,
            "total_spread": 0.0,
            "cost_percentage": 0.0,
            "avg_cash_utilization": np.nan,
            "cash_drag_cost": 0.0,
            "turnover_ratio": np.nan,
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
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

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
            "mean_return": mean_return,
            "worst_case": worst_case,
            f"lower_{int(confidence * 100)}": lower,
            f"upper_{int(confidence * 100)}": upper,
        }
        if output_csv:
            pd.DataFrame(equity_curves).to_csv(output_csv, index=False)
        if output_plot:
            plt.figure(figsize=(12, 6))
            plt.plot(equity_curves.T, color="gray", alpha=0.01)
            plt.plot(np.mean(equity_curves, axis=0), color="blue", label="Mean Path")
            plt.title("Monte Carlo Simulated Equity Paths")
            plt.xlabel("Time Step")
            plt.ylabel("Equity")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_plot)
            plt.close()
        return result
