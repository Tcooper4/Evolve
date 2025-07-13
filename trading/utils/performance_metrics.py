"""
Performance metrics utilities for the trading system.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Performance metrics calculation utility class."""

    def __init__(self):
        self.metrics = {}

    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate returns from price series."""
        return prices.pct_change().dropna()

    def calculate_cumulative_returns(self, returns: pd.Series) -> pd.Series:
        """Calculate cumulative returns."""
        return (1 + returns).cumprod() - 1

    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        try:
            excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
            return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        except:
            return 0.0

    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio."""
        try:
            excess_returns = returns - risk_free_rate / 252
            downside_returns = excess_returns[excess_returns < 0]
            downside_std = np.sqrt(np.mean(downside_returns**2))
            return np.sqrt(252) * excess_returns.mean() / downside_std
        except:
            return 0.0

    def calculate_max_drawdown(self, cumulative_returns: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """Calculate maximum drawdown and its period."""
        try:
            rolling_max = cumulative_returns.expanding().max()
            drawdown = cumulative_returns - rolling_max
            max_dd = drawdown.min()

            # Find the peak and trough dates
            peak_idx = rolling_max.idxmax()
            trough_idx = drawdown.idxmin()

            return max_dd, peak_idx, trough_idx
        except:
            return 0.0, None, None

    def calculate_calmar_ratio(self, returns: pd.Series, cumulative_returns: pd.Series) -> float:
        """Calculate Calmar ratio."""
        try:
            max_dd, _, _ = self.calculate_max_drawdown(cumulative_returns)
            annual_return = returns.mean() * 252
            return annual_return / abs(max_dd) if max_dd != 0 else 0
        except:
            return 0.0

    def calculate_win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate."""
        try:
            return (returns > 0).mean()
        except:
            return 0.0

    def calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor."""
        try:
            gross_profit = returns[returns > 0].sum()
            gross_loss = abs(returns[returns < 0].sum())
            return gross_profit / gross_loss if gross_loss != 0 else float("inf")
        except:
            return 0.0

    def calculate_volatility(self, returns: pd.Series, annualize: bool = True) -> float:
        """Calculate volatility."""
        try:
            vol = returns.std()
            return vol * np.sqrt(252) if annualize else vol
        except:
            return 0.0

    def calculate_beta(self, returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta relative to market."""
        try:
            covariance = np.cov(returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            return covariance / market_variance if market_variance != 0 else 0
        except:
            return 0.0

    def calculate_alpha(self, returns: pd.Series, market_returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate alpha."""
        try:
            beta = self.calculate_beta(returns, market_returns)
            excess_returns = returns - risk_free_rate / 252
            excess_market_returns = market_returns - risk_free_rate / 252
            return excess_returns.mean() - beta * excess_market_returns.mean()
        except:
            return 0.0

    def calculate_all_metrics(
        self, returns: pd.Series, market_returns: Optional[pd.Series] = None, risk_free_rate: float = 0.02
    ) -> Dict[str, float]:
        """Calculate all performance metrics."""
        try:
            cumulative_returns = self.calculate_cumulative_returns(returns)

            metrics = {
                "total_return": cumulative_returns.iloc[-1],
                "annual_return": returns.mean() * 252,
                "volatility": self.calculate_volatility(returns),
                "sharpe_ratio": self.calculate_sharpe_ratio(returns, risk_free_rate),
                "sortino_ratio": self.calculate_sortino_ratio(returns, risk_free_rate),
                "max_drawdown": self.calculate_max_drawdown(cumulative_returns)[0],
                "calmar_ratio": self.calculate_calmar_ratio(returns, cumulative_returns),
                "win_rate": self.calculate_win_rate(returns),
                "profit_factor": self.calculate_profit_factor(returns),
                "skewness": returns.skew(),
                "kurtosis": returns.kurtosis(),
            }

            if market_returns is not None:
                metrics["beta"] = self.calculate_beta(returns, market_returns)
                metrics["alpha"] = self.calculate_alpha(returns, market_returns, risk_free_rate)

            self.metrics = metrics
            return metrics

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}


class RiskMetrics:
    """Risk metrics calculation utility class."""

    def __init__(self):
        self.risk_metrics = {}

    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk."""
        try:
            return np.percentile(returns, confidence_level * 100)
        except:
            return 0.0

    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        try:
            var = self.calculate_var(returns, confidence_level)
            return returns[returns <= var].mean()
        except:
            return 0.0

    def calculate_ulcer_index(self, cumulative_returns: pd.Series) -> float:
        """Calculate Ulcer Index."""
        try:
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            return np.sqrt(np.mean(drawdown**2))
        except:
            return 0.0

    def calculate_gain_to_pain_ratio(self, returns: pd.Series) -> float:
        """Calculate Gain to Pain ratio."""
        try:
            gains = returns[returns > 0].sum()
            losses = abs(returns[returns < 0].sum())
            return gains / losses if losses != 0 else float("inf")
        except:
            return 0.0

    def calculate_all_risk_metrics(self, returns: pd.Series, cumulative_returns: pd.Series) -> Dict[str, float]:
        """Calculate all risk metrics."""
        try:
            risk_metrics = {
                "var_95": self.calculate_var(returns, 0.05),
                "var_99": self.calculate_var(returns, 0.01),
                "cvar_95": self.calculate_cvar(returns, 0.05),
                "cvar_99": self.calculate_cvar(returns, 0.01),
                "ulcer_index": self.calculate_ulcer_index(cumulative_returns),
                "gain_to_pain_ratio": self.calculate_gain_to_pain_ratio(returns),
                "downside_deviation": returns[returns < 0].std(),
                "upside_deviation": returns[returns > 0].std(),
            }

            self.risk_metrics = risk_metrics
            return risk_metrics

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}


class TradingMetrics:
    """Trading-specific metrics calculation utility class."""

    def __init__(self):
        self.trading_metrics = {}

    def calculate_trade_metrics(self, trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate trading-specific metrics."""
        try:
            if trades.empty:
                return {}

            # Basic trade metrics
            total_trades = len(trades)
            winning_trades = len(trades[trades["pnl"] > 0])
            losing_trades = len(trades[trades["pnl"] < 0])

            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            loss_rate = losing_trades / total_trades if total_trades > 0 else 0

            # PnL metrics
            total_pnl = trades["pnl"].sum()
            avg_win = trades[trades["pnl"] > 0]["pnl"].mean() if winning_trades > 0 else 0
            avg_loss = trades[trades["pnl"] < 0]["pnl"].mean() if losing_trades > 0 else 0

            # Risk metrics
            max_win = trades["pnl"].max()
            max_loss = trades["pnl"].min()

            # Duration metrics
            if "duration" in trades.columns:
                avg_duration = trades["duration"].mean()
                max_duration = trades["duration"].max()
                min_duration = trades["duration"].min()
            else:
                avg_duration = max_duration = min_duration = 0

            trading_metrics = {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "loss_rate": loss_rate,
                "total_pnl": total_pnl,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "max_win": max_win,
                "max_loss": max_loss,
                "profit_factor": abs(avg_win * winning_trades / (avg_loss * losing_trades))
                if avg_loss != 0
                else float("inf"),
                "avg_duration": avg_duration,
                "max_duration": max_duration,
                "min_duration": min_duration,
            }

            self.trading_metrics = trading_metrics
            return trading_metrics

        except Exception as e:
            logger.error(f"Error calculating trading metrics: {e}")
            return {}


# Convenience functions


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio."""
    metrics = PerformanceMetrics()
    return metrics.calculate_sharpe_ratio(returns, risk_free_rate)


def calculate_max_drawdown(cumulative_returns: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """Calculate maximum drawdown."""
    metrics = PerformanceMetrics()
    return metrics.calculate_max_drawdown(cumulative_returns)


def calculate_win_rate(returns: pd.Series) -> float:
    """Calculate win rate."""
    metrics = PerformanceMetrics()
    return metrics.calculate_win_rate(returns)


def calculate_profit_factor(returns: pd.Series) -> float:
    """Calculate profit factor."""
    metrics = PerformanceMetrics()
    return metrics.calculate_profit_factor(returns)
