"""RSI Strategy Optimizer - Non-legacy implementation."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trading.optimization.base_optimizer import BaseOptimizer
from trading.strategies.rsi_signals import generate_rsi_signals
from trading.utils.safe_math import safe_drawdown, safe_divide

logger = logging.getLogger(__name__)


@dataclass
class RSIParameters:
    """Container for RSI parameters."""

    period: int
    overbought: float
    oversold: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence_score: Optional[float] = None


@dataclass
class OptimizationResult:
    """Container for optimization results."""

    parameters: RSIParameters
    returns: pd.Series
    metrics: Dict[str, float]
    signals: pd.Series
    equity_curve: pd.Series
    drawdown: pd.Series


@dataclass
class RSIOptimizationResult:
    period: int
    overbought: float
    oversold: float
    sharpe_ratio: float
    win_rate: float
    metrics: Dict[str, float]


class RSIOptimizer(BaseOptimizer):
    """Modular RSI parameter optimizer (non-legacy)."""

    def __init__(self, data: pd.DataFrame, verbose: bool = False):
        super().__init__(data, strategy_type="RSI", verbose=verbose)
        self.data = data
        self.verbose = verbose
        logger.info("Initialized RSIOptimizer")

    def optimize(
        self,
        period_range: Tuple[int, int] = (10, 30),
        overbought_range: Tuple[float, float] = (70, 90),
        oversold_range: Tuple[float, float] = (10, 30),
        n_trials: int = 20,
    ) -> RSIOptimizationResult:
        """Grid search for best RSI parameters."""
        best_score = float("-inf")
        best_result = None
        results = []
        periods = np.linspace(period_range[0], period_range[1], num=5, dtype=int)
        overboughts = np.linspace(overbought_range[0], overbought_range[1], num=5)
        oversolds = np.linspace(oversold_range[0], oversold_range[1], num=5)
        for period in periods:
            for overbought in overboughts:
                for oversold in oversolds:
                    if overbought <= oversold:
                        continue
                    try:
                        df = self.data.copy()
                        # Ensure 'Close' column for RSI signals
                        if "Close" not in df.columns and "close" in df.columns:
                            df["Close"] = df["close"]
                        signals = generate_rsi_signals(
                            df,
                            period=period,
                            buy_threshold=oversold,
                            sell_threshold=overbought,
                        )
                        metrics = self.calculate_metrics(
                            signals["strategy_returns"].dropna(),
                            signals["signal"],
                            signals["strategy_cumulative_returns"],
                        )
                        sharpe = metrics.get("sharpe_ratio", 0)
                        win_rate = metrics.get("win_rate", 0)
                        result = RSIOptimizationResult(
                            period=period,
                            overbought=overbought,
                            oversold=oversold,
                            sharpe_ratio=sharpe,
                            win_rate=win_rate,
                            metrics=metrics,
                        )
                        results.append(result)
                        if sharpe > best_score:
                            best_score = sharpe
                            best_result = result
                        logger.debug(
                            f"Tested period={period}, overbought={overbought}, oversold={oversold}, Sharpe={sharpe:.3f}, WinRate={win_rate:.3f}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed for period={period}, overbought={overbought}, oversold={oversold}: {e}"
                        )
        logger.info(
            f"Best Sharpe: {best_result.sharpe_ratio:.3f} (period={best_result.period}, overbought={best_result.overbought}, oversold={best_result.oversold})"
        )
        return best_result

    def optimize_rsi_parameters(
        self, objective: str = "sharpe", n_top: int = 3, **kwargs
    ) -> List[OptimizationResult]:
        """Optimize RSI parameters.

        Args:
            objective: Optimization objective ('sharpe', 'returns', 'win_rate')
            n_top: Number of top results to return
            **kwargs: Additional optimization parameters

        Returns:
            List of optimization results
        """
        try:
            # Define parameter space for RSI
            param_space = {
                "period": {"start": 10, "end": 30},
                "overbought": {"start": 70, "end": 90},
                "oversold": {"start": 10, "end": 30},
            }

            # Create objective function
            def objective_function(params):
                return self._evaluate_rsi_params(params, objective)

            # Run optimization
            optimization_result = self.strategy_optimizer.optimize(
                strategy_class=None,  # We're using a custom objective function
                data=self.data,
                initial_params=param_space,
            )

            # Convert results to our format
            results = []
            for i, (params, score) in enumerate(
                zip(
                    optimization_result.get("all_params", []),
                    optimization_result.get("all_scores", []),
                )
            ):
                if i >= n_top:
                    break

                # Create RSI parameters
                rsi_params = RSIParameters(
                    period=params.get("period", 14),
                    overbought=params.get("overbought", 70),
                    oversold=params.get("oversold", 30),
                )

                # Generate signals and calculate metrics
                signals_df = generate_rsi_signals(
                    self.data,
                    period=rsi_params.period,
                    buy_threshold=rsi_params.oversold,
                    sell_threshold=rsi_params.overbought,
                )

                # Calculate metrics
                metrics = self._calculate_metrics(signals_df)

                # Create result
                result = OptimizationResult(
                    parameters=rsi_params,
                    returns=signals_df["strategy_returns"],
                    metrics=metrics,
                    signals=signals_df["signal"],
                    equity_curve=signals_df["strategy_cumulative_returns"],
                    drawdown=self._calculate_drawdown(
                        signals_df["strategy_cumulative_returns"]
                    ),
                )

                results.append(result)

            logger.info(f"RSI optimization completed with {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error in RSI optimization: {e}")
            return []

    def _evaluate_rsi_params(self, params: Dict[str, Any], objective: str) -> float:
        """Evaluate RSI parameters.

        Args:
            params: RSI parameters
            objective: Evaluation objective

        Returns:
            Evaluation score
        """
        try:
            # Generate signals
            signals_df = generate_rsi_signals(
                self.data,
                period=params.get("period", 14),
                buy_threshold=params.get("oversold", 30),
                sell_threshold=params.get("overbought", 70),
            )

            # Calculate metrics
            metrics = self._calculate_metrics(signals_df)

            # Return appropriate metric
            if objective == "sharpe":
                return metrics.get("sharpe_ratio", 0)
            elif objective == "returns":
                return metrics.get("total_return", 0)
            elif objective == "win_rate":
                return metrics.get("win_rate", 0)
            else:
                return metrics.get("sharpe_ratio", 0)

        except Exception as e:
            logger.error(f"Error evaluating RSI parameters: {e}")
            return 0

    def _calculate_metrics(self, signals_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics.

        Args:
            signals_df: DataFrame with signals and returns

        Returns:
            Dictionary of metrics
        """
        try:
            returns = signals_df["strategy_returns"].dropna()
            equity_curve = signals_df["strategy_cumulative_returns"]

            # Basic metrics
            total_return = equity_curve.iloc[-1] - 1 if len(equity_curve) > 0 else 0
            annual_return = (
                (1 + total_return) ** (252 / len(returns)) - 1
                if len(returns) > 0
                else 0
            )
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
            sharpe_ratio = safe_divide(annual_return, volatility, default=0.0)

            # Win rate
            winning_trades = returns[returns > 0]
            total_trades = returns[returns != 0]
            win_rate = (
                len(winning_trades) / len(total_trades) if len(total_trades) > 0 else 0
            )

            # Drawdown
            max_drawdown = self._calculate_drawdown(equity_curve).min()

            return {
                "total_return": total_return,
                "annual_return": annual_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "win_rate": win_rate,
                "max_drawdown": max_drawdown,
            }

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {
                "total_return": 0,
                "annual_return": 0,
                "volatility": 0,
                "sharpe_ratio": 0,
                "win_rate": 0,
                "max_drawdown": 0,
            }

    def _calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series.

        Args:
            equity_curve: Equity curve series

        Returns:
            Drawdown series
        """
        drawdown = safe_drawdown(equity_curve)
        return drawdown

    def plot_equity_curve(
        self, result: OptimizationResult, title: str = "Equity Curve"
    ) -> go.Figure:
        """Plot equity curve.

        Args:
            result: Optimization result
            title: Plot title

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=result.equity_curve.index,
                y=result.equity_curve.values,
                mode="lines",
                name="Strategy",
                line=dict(color="blue"),
            )
        )

        fig.update_layout(
            title=title, xaxis_title="Date", yaxis_title="Equity", showlegend=True
        )

        return fig

    def plot_drawdown(
        self, result: OptimizationResult, title: str = "Drawdown"
    ) -> go.Figure:
        """Plot drawdown.

        Args:
            result: Optimization result
            title: Plot title

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=result.drawdown.index,
                y=result.drawdown.values * 100,  # Convert to percentage
                mode="lines",
                name="Drawdown",
                line=dict(color="red"),
                fill="tonexty",
            )
        )

        fig.update_layout(
            title=title, xaxis_title="Date", yaxis_title="Drawdown (%)", showlegend=True
        )

        return fig

    def plot_signals(
        self, result: OptimizationResult, title: str = "RSI Signals"
    ) -> go.Figure:
        """Plot RSI signals.

        Args:
            result: Optimization result
            title: Plot title

        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Price", "RSI Signals"),
            vertical_spacing=0.1,
        )

        # Price plot
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data["close"],
                mode="lines",
                name="Price",
                line=dict(color="black"),
            ),
            row=1,
            col=1,
        )

        # Signals plot
        fig.add_trace(
            go.Scatter(
                x=result.signals.index,
                y=result.signals.values,
                mode="markers",
                name="Signals",
                marker=dict(
                    color=[
                        "red" if s == -1 else "green" if s == 1 else "gray"
                        for s in result.signals.values
                    ],
                    size=8,
                ),
            ),
            row=2,
            col=1,
        )

        fig.update_layout(title=title, showlegend=True, height=600)

        return fig
