"""
Backtest Utilities

Utility functions for backtesting with safety checks and guard clauses.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from trading.utils.safe_math import safe_drawdown, safe_divide

logger = logging.getLogger(__name__)


@dataclass
class BacktestReport:
    """Backtest report with results and metadata."""

    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profitable_trades: int
    avg_trade_return: float
    equity_curve: pd.Series
    trade_history: pd.DataFrame
    metadata: Dict[str, Any]


class BacktestUtils:
    """Utility functions for backtesting with safety checks."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize backtest utilities.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.required_columns = self.config.get(
            "required_columns", ["Buy", "Sell", "Close"]
        )
        self.min_trades = self.config.get("min_trades", 1)
        self.max_drawdown_threshold = self.config.get("max_drawdown_threshold", 0.5)

    def validate_backtest_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate backtest data for required columns and data quality.

        Args:
            df: DataFrame with backtest data

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check if DataFrame is empty
        if df is None or df.empty:
            errors.append("DataFrame is empty or None")
            return False, errors

        # Check for required columns
        missing_columns = [
            col for col in self.required_columns if col not in df.columns
        ]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")

        # Check for sufficient data
        if len(df) < 10:
            errors.append("Insufficient data for backtesting (minimum 10 rows)")

        # Check for numeric data in key columns
        if "Close" in df.columns:
            if not pd.api.types.is_numeric_dtype(df["Close"]):
                errors.append("Close column must be numeric")

        # Check for datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            errors.append("DataFrame must have datetime index")

        return len(errors) == 0, errors

    def generate_backtest_report(self, df: pd.DataFrame) -> Optional[BacktestReport]:
        """
        Generate backtest report with guard clauses.

        Args:
            df: DataFrame with backtest data

        Returns:
            BacktestReport or None if validation fails
        """
        # Guard clause: Check for required columns
        if "Buy" not in df.columns:
            logger.error("Missing 'Buy' column in backtest data")
            return self._generate_empty_report("Missing 'Buy' column")

        # Validate data
        is_valid, errors = self.validate_backtest_data(df)
        if not is_valid:
            logger.error(f"Backtest data validation failed: {errors}")
            return self._generate_empty_report(
                f"Validation failed: {', '.join(errors)}"
            )

        try:
            # Calculate basic metrics
            total_return = self._calculate_total_return(df)
            sharpe_ratio = self._calculate_sharpe_ratio(df)
            max_drawdown = self._calculate_max_drawdown(df)

            # Calculate trade statistics
            trade_stats = self._calculate_trade_statistics(df)

            # Generate equity curve
            equity_curve = self._generate_equity_curve(df)

            # Create trade history
            trade_history = self._create_trade_history(df)

            return BacktestReport(
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=trade_stats["win_rate"],
                total_trades=trade_stats["total_trades"],
                profitable_trades=trade_stats["profitable_trades"],
                avg_trade_return=trade_stats["avg_trade_return"],
                equity_curve=equity_curve,
                trade_history=trade_history,
                metadata={
                    "data_points": len(df),
                    "date_range": f"{df.index[0]} to {df.index[-1]}",
                    "validation_passed": True,
                },
            )

        except Exception as e:
            logger.error(f"Error generating backtest report: {e}")
            return self._generate_empty_report(f"Report generation failed: {str(e)}")

    def _generate_empty_report(self, reason: str) -> BacktestReport:
        """Generate empty report when validation fails.

        Args:
            reason: Reason for empty report

        Returns:
            Empty BacktestReport
        """
        return BacktestReport(
            total_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            total_trades=0,
            profitable_trades=0,
            avg_trade_return=0.0,
            equity_curve=pd.Series(),
            trade_history=pd.DataFrame(),
            metadata={"error": reason, "validation_passed": False},
        )

    def _calculate_total_return(self, df: pd.DataFrame) -> float:
        """Calculate total return from backtest data."""
        if "Close" not in df.columns:
            return 0.0

        initial_price = df["Close"].iloc[0]
        final_price = df["Close"].iloc[-1]

        if initial_price == 0:
            return 0.0

        return (final_price - initial_price) / initial_price

    def _calculate_sharpe_ratio(self, df: pd.DataFrame) -> float:
        """Calculate Sharpe ratio from backtest data."""
        if "Close" not in df.columns:
            return 0.0

        returns = df["Close"].pct_change().dropna()

        if len(returns) == 0:
            return 0.0

        mean_return = returns.mean()
        std_return = returns.std()

        if std_return == 0:
            return 0.0

        return mean_return / std_return

    def _calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        """Calculate maximum drawdown from backtest data."""
        if "Close" not in df.columns:
            return 0.0

        prices = df["Close"]
        drawdown = safe_drawdown(prices)

        return abs(drawdown.min())

    def _calculate_trade_statistics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate trade statistics from backtest data."""
        if "Buy" not in df.columns or "Sell" not in df.columns:
            return {
                "win_rate": 0.0,
                "total_trades": 0,
                "profitable_trades": 0,
                "avg_trade_return": 0.0,
            }

        # Find buy and sell signals
        buy_signals = df["Buy"] == 1
        sell_signals = df["Sell"] == 1

        total_trades = min(buy_signals.sum(), sell_signals.sum())

        if total_trades == 0:
            return {
                "win_rate": 0.0,
                "total_trades": 0,
                "profitable_trades": 0,
                "avg_trade_return": 0.0,
            }

        # Calculate trade returns
        trade_returns = []
        buy_indices = df[buy_signals].index
        sell_indices = df[sell_signals].index

        for i in range(min(len(buy_indices), len(sell_indices))):
            buy_price = df.loc[buy_indices[i], "Close"]
            sell_price = df.loc[sell_indices[i], "Close"]

            if buy_price > 0:
                trade_return = (sell_price - buy_price) / buy_price
                trade_returns.append(trade_return)

        if not trade_returns:
            return {
                "win_rate": 0.0,
                "total_trades": 0,
                "profitable_trades": 0,
                "avg_trade_return": 0.0,
            }

        profitable_trades = sum(1 for ret in trade_returns if ret > 0)
        win_rate = safe_divide(profitable_trades, len(trade_returns), default=0.0)
        avg_trade_return = np.mean(trade_returns)

        return {
            "win_rate": win_rate,
            "total_trades": len(trade_returns),
            "profitable_trades": profitable_trades,
            "avg_trade_return": avg_trade_return,
        }

    def _generate_equity_curve(self, df: pd.DataFrame) -> pd.Series:
        """Generate equity curve from backtest data."""
        if "Close" not in df.columns:
            return pd.Series()

        # Simple equity curve based on price changes
        initial_price = df["Close"].iloc[0]
        equity_curve = df["Close"] / initial_price

        return equity_curve

    def _create_trade_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create trade history DataFrame."""
        if "Buy" not in df.columns or "Sell" not in df.columns:
            return pd.DataFrame()

        # Find trade entry and exit points
        buy_signals = df["Buy"] == 1
        sell_signals = df["Sell"] == 1

        trades = []
        buy_indices = df[buy_signals].index
        sell_indices = df[sell_signals].index

        for i in range(min(len(buy_indices), len(sell_indices))):
            trade = {
                "entry_date": buy_indices[i],
                "exit_date": sell_indices[i],
                "entry_price": df.loc[buy_indices[i], "Close"],
                "exit_price": df.loc[sell_indices[i], "Close"],
                "return": (
                    df.loc[sell_indices[i], "Close"] - df.loc[buy_indices[i], "Close"]
                )
                / df.loc[buy_indices[i], "Close"],
            }
            trades.append(trade)

        return pd.DataFrame(trades)

    def validate_strategy_signals(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate strategy signals in the DataFrame.

        Args:
            df: DataFrame with strategy signals

        Returns:
            Tuple of (is_valid, warning_messages)
        """
        warnings = []

        # Check for signal conflicts
        if "Buy" in df.columns and "Sell" in df.columns:
            conflicts = (df["Buy"] == 1) & (df["Sell"] == 1)
            if conflicts.any():
                warnings.append(f"Found {conflicts.sum()} conflicting buy/sell signals")

        # Check for signal frequency
        if "Buy" in df.columns:
            buy_frequency = df["Buy"].sum() / len(df)
            if buy_frequency > 0.5:
                warnings.append(f"High buy signal frequency: {buy_frequency:.2%}")

        if "Sell" in df.columns:
            sell_frequency = df["Sell"].sum() / len(df)
            if sell_frequency > 0.5:
                warnings.append(f"High sell signal frequency: {sell_frequency:.2%}")

        # Check for signal clustering
        if "Buy" in df.columns:
            buy_clusters = self._check_signal_clustering(df["Buy"])
            if buy_clusters:
                warnings.append(f"Buy signals are clustered: {buy_clusters}")

        return len(warnings) == 0, warnings

    def _check_signal_clustering(self, signals: pd.Series) -> Optional[str]:
        """Check if signals are clustered together."""
        signal_indices = signals[signals == 1].index

        if len(signal_indices) < 2:
            return None

        # Check for consecutive signals
        consecutive_count = 0
        max_consecutive = 0

        for i in range(1, len(signal_indices)):
            if (signal_indices[i] - signal_indices[i - 1]).days <= 1:
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)
            else:
                consecutive_count = 0

        if max_consecutive > 3:
            return f"Found {max_consecutive} consecutive signals"

        return None
