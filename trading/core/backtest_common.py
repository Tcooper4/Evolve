"""
Backtest Common Utilities

This module provides common utilities and functions used across backtesting modules,
extracting repeated logic and adding frequency parameter support for correct metric scaling.
"""

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Frequency(Enum):
    """Trading frequency enumeration."""

    TICK = "tick"
    MINUTE_1 = "1min"
    MINUTE_5 = "5min"
    MINUTE_15 = "15min"
    MINUTE_30 = "30min"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1m"


class BacktestCommon:
    """
    Common utilities for backtesting operations.

    Features:
    - Frequency-aware metric scaling
    - Common data validation and preprocessing
    - Performance calculation utilities
    - Risk metric computations
    - Trade analysis functions
    """

    def __init__(self):
        """Initialize backtest common utilities."""
        self.frequency_scaling = {
            Frequency.TICK: 1,
            Frequency.MINUTE_1: 1,
            Frequency.MINUTE_5: 5,
            Frequency.MINUTE_15: 15,
            Frequency.MINUTE_30: 30,
            Frequency.HOUR_1: 60,
            Frequency.HOUR_4: 240,
            Frequency.DAILY: 1440,  # 24 * 60
            Frequency.WEEKLY: 10080,  # 7 * 24 * 60
            Frequency.MONTHLY: 43200,  # 30 * 24 * 60
        }

    def validate_data(
        self,
        data: pd.DataFrame,
        required_columns: List[str] = None,
        min_length: int = 10,
    ) -> Tuple[bool, str]:
        """
        Validate backtest data.

        Args:
            data: DataFrame to validate
            required_columns: List of required columns
            min_length: Minimum number of rows required

        Returns:
            Tuple of (is_valid, error_message)
        """
        if data is None or data.empty:
            return False, "Data is None or empty"

        if len(data) < min_length:
            return False, f"Data has {len(data)} rows, minimum {min_length} required"

        if required_columns:
            missing_cols = [col for col in required_columns if col not in data.columns]
            if missing_cols:
                return False, f"Missing required columns: {missing_cols}"

        # Check for infinite values
        if np.isinf(data.select_dtypes(include=[np.number])).any().any():
            return False, "Data contains infinite values"

        # Check for excessive NaN values
        nan_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if nan_ratio > 0.1:  # More than 10% NaN
            return False, f"Data contains {nan_ratio:.1%} NaN values"

        return True, "Data is valid"

    def preprocess_data(
        self,
        data: pd.DataFrame,
        frequency: Frequency = Frequency.DAILY,
        fill_method: str = "ffill",
    ) -> pd.DataFrame:
        """
        Preprocess data for backtesting.

        Args:
            data: Raw data DataFrame
            frequency: Trading frequency
            fill_method: Method to fill missing values

        Returns:
            Preprocessed DataFrame
        """
        processed_data = data.copy()

        # Ensure datetime index
        if not isinstance(processed_data.index, pd.DatetimeIndex):
            if "date" in processed_data.columns:
                processed_data.set_index("date", inplace=True)
            elif "timestamp" in processed_data.columns:
                processed_data.set_index("timestamp", inplace=True)
            else:
                processed_data.index = pd.date_range(
                    start=datetime.now() - timedelta(days=len(processed_data)),
                    periods=len(processed_data),
                    freq="D",
                )

        # Sort by index
        processed_data.sort_index(inplace=True)

        # Fill missing values
        if fill_method == "ffill":
            processed_data.ffill(inplace=True)
        elif fill_method == "bfill":
            processed_data.bfill(inplace=True)
        elif fill_method == "interpolate":
            processed_data.interpolate(method="linear", inplace=True)

        # Remove any remaining NaN values
        processed_data.dropna(inplace=True)

        # Resample to specified frequency if needed
        if frequency != Frequency.TICK:
            freq_str = self._get_freq_string(frequency)
            processed_data = (
                processed_data.resample(freq_str)
                .agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                )
                .dropna()
            )

        return processed_data

    def _get_freq_string(self, frequency: Frequency) -> str:
        """Convert frequency enum to pandas frequency string."""
        freq_mapping = {
            Frequency.MINUTE_1: "1T",
            Frequency.MINUTE_5: "5T",
            Frequency.MINUTE_15: "15T",
            Frequency.MINUTE_30: "30T",
            Frequency.HOUR_1: "1H",
            Frequency.HOUR_4: "4H",
            Frequency.DAILY: "D",
            Frequency.WEEKLY: "W",
            Frequency.MONTHLY: "M",
        }
        return freq_mapping.get(frequency, "D")

    def calculate_returns(self, prices: pd.Series, method: str = "log") -> pd.Series:
        """
        Calculate returns from price series.

        Args:
            prices: Price series
            method: Return calculation method ('log' or 'simple')

        Returns:
            Returns series
        """
        if method == "log":
            returns = np.log(prices / prices.shift(1))
        else:  # simple
            returns = (prices - prices.shift(1)) / prices.shift(1)

        return returns.dropna()

    def calculate_volatility(
        self,
        returns: pd.Series,
        window: int = 252,
        frequency: Frequency = Frequency.DAILY,
    ) -> pd.Series:
        """
        Calculate rolling volatility with frequency scaling.

        Args:
            returns: Returns series
            window: Rolling window size
            frequency: Trading frequency for scaling

        Returns:
            Volatility series
        """
        # Scale window based on frequency
        scaled_window = self._scale_window(window, frequency)

        # Calculate rolling volatility
        volatility = returns.rolling(window=scaled_window).std()

        # Annualize if needed
        if frequency != Frequency.DAILY:
            annualization_factor = self._get_annualization_factor(frequency)
            volatility = volatility * np.sqrt(annualization_factor)

        return volatility

    def _scale_window(self, window: int, frequency: Frequency) -> int:
        """Scale window size based on frequency."""
        if frequency == Frequency.DAILY:
            return window

        # Convert daily window to frequency-specific window
        daily_minutes = 1440  # 24 * 60
        freq_minutes = self.frequency_scaling[frequency]

        scaled_window = int(window * daily_minutes / freq_minutes)
        return max(1, scaled_window)

    def _get_annualization_factor(self, frequency: Frequency) -> float:
        """Get annualization factor for different frequencies."""
        annualization_factors = {
            Frequency.TICK: 525600,  # 365 * 24 * 60
            Frequency.MINUTE_1: 525600,
            Frequency.MINUTE_5: 105120,  # 525600 / 5
            Frequency.MINUTE_15: 35040,  # 525600 / 15
            Frequency.MINUTE_30: 17520,  # 525600 / 30
            Frequency.HOUR_1: 8760,  # 365 * 24
            Frequency.HOUR_4: 2190,  # 8760 / 4
            Frequency.DAILY: 252,  # Trading days per year
            Frequency.WEEKLY: 52,  # Weeks per year
            Frequency.MONTHLY: 12,  # Months per year
        }
        return annualization_factors.get(frequency, 252)

    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        frequency: Frequency = Frequency.DAILY,
    ) -> float:
        """
        Calculate Sharpe ratio with frequency scaling.

        Args:
            returns: Returns series
            risk_free_rate: Annual risk-free rate
            frequency: Trading frequency

        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0

        # Calculate excess returns
        excess_returns = returns - risk_free_rate / self._get_annualization_factor(
            frequency
        )

        # Calculate Sharpe ratio
        mean_excess_return = excess_returns.mean()
        std_excess_return = excess_returns.std()

        if std_excess_return == 0:
            return 0.0

        sharpe_ratio = mean_excess_return / std_excess_return

        # Annualize
        annualization_factor = self._get_annualization_factor(frequency)
        return sharpe_ratio * np.sqrt(annualization_factor)

    def calculate_max_drawdown(
        self, prices: pd.Series
    ) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        Calculate maximum drawdown.

        Args:
            prices: Price series

        Returns:
            Tuple of (max_drawdown, peak_date, trough_date)
        """
        if len(prices) < 2:
            return 0.0, None, None

        # Calculate cumulative returns
        cumulative_returns = (prices / prices.iloc[0]) - 1

        # Calculate running maximum
        running_max = cumulative_returns.expanding().max()

        # Calculate drawdown
        drawdown = cumulative_returns - running_max

        # Find maximum drawdown
        max_drawdown = drawdown.min()
        trough_idx = drawdown.idxmin()
        peak_idx = running_max.loc[:trough_idx].idxmax()

        return max_drawdown, peak_idx, trough_idx

    def calculate_win_rate(self, trades: pd.DataFrame) -> float:
        """
        Calculate win rate from trades DataFrame.

        Args:
            trades: DataFrame with trade information

        Returns:
            Win rate (0-1)
        """
        if len(trades) == 0:
            return 0.0

        if "pnl" in trades.columns:
            winning_trades = (trades["pnl"] > 0).sum()
        elif "return" in trades.columns:
            winning_trades = (trades["return"] > 0).sum()
        else:
            return 0.0

        return winning_trades / len(trades)

    def calculate_profit_factor(self, trades: pd.DataFrame) -> float:
        """
        Calculate profit factor from trades DataFrame.

        Args:
            trades: DataFrame with trade information

        Returns:
            Profit factor
        """
        if len(trades) == 0:
            return 0.0

        if "pnl" in trades.columns:
            gross_profit = trades[trades["pnl"] > 0]["pnl"].sum()
            gross_loss = abs(trades[trades["pnl"] < 0]["pnl"].sum())
        elif "return" in trades.columns:
            gross_profit = trades[trades["return"] > 0]["return"].sum()
            gross_loss = abs(trades[trades["return"] < 0]["return"].sum())
        else:
            return 0.0

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def calculate_calmar_ratio(
        self,
        returns: pd.Series,
        prices: pd.Series,
        frequency: Frequency = Frequency.DAILY,
    ) -> float:
        """
        Calculate Calmar ratio (annual return / max drawdown).

        Args:
            returns: Returns series
            prices: Price series
            frequency: Trading frequency

        Returns:
            Calmar ratio
        """
        if len(returns) < 2:
            return 0.0

        # Calculate annual return
        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
        years = len(returns) / self._get_annualization_factor(frequency)
        annual_return = (1 + total_return) ** (1 / years) - 1

        # Calculate max drawdown
        max_dd, _, _ = self.calculate_max_drawdown(prices)

        if max_dd == 0:
            return 0.0

        return annual_return / abs(max_dd)

    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        frequency: Frequency = Frequency.DAILY,
    ) -> float:
        """
        Calculate Sortino ratio (downside deviation).

        Args:
            returns: Returns series
            risk_free_rate: Annual risk-free rate
            frequency: Trading frequency

        Returns:
            Sortino ratio
        """
        if len(returns) < 2:
            return 0.0

        # Calculate excess returns
        excess_returns = returns - risk_free_rate / self._get_annualization_factor(
            frequency
        )

        # Calculate downside deviation
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return 0.0

        downside_deviation = downside_returns.std()

        if downside_deviation == 0:
            return 0.0

        # Calculate Sortino ratio
        mean_excess_return = excess_returns.mean()
        sortino_ratio = mean_excess_return / downside_deviation

        # Annualize
        annualization_factor = self._get_annualization_factor(frequency)
        return sortino_ratio * np.sqrt(annualization_factor)

    def calculate_metrics_summary(
        self,
        returns: pd.Series,
        prices: pd.Series,
        trades: pd.DataFrame = None,
        risk_free_rate: float = 0.02,
        frequency: Frequency = Frequency.DAILY,
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics summary.

        Args:
            returns: Returns series
            prices: Price series
            trades: Trades DataFrame (optional)
            risk_free_rate: Annual risk-free rate
            frequency: Trading frequency

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Basic metrics
        metrics["total_return"] = (prices.iloc[-1] / prices.iloc[0]) - 1
        metrics["annual_return"] = self._calculate_annual_return(returns, frequency)
        metrics["volatility"] = returns.std() * np.sqrt(
            self._get_annualization_factor(frequency)
        )

        # Risk-adjusted metrics
        metrics["sharpe_ratio"] = self.calculate_sharpe_ratio(
            returns, risk_free_rate, frequency
        )
        metrics["sortino_ratio"] = self.calculate_sortino_ratio(
            returns, risk_free_rate, frequency
        )
        metrics["calmar_ratio"] = self.calculate_calmar_ratio(
            returns, prices, frequency
        )

        # Drawdown metrics
        max_dd, peak_date, trough_date = self.calculate_max_drawdown(prices)
        metrics["max_drawdown"] = max_dd
        metrics["peak_date"] = peak_date
        metrics["trough_date"] = trough_date

        # Trade metrics (if available)
        if trades is not None and len(trades) > 0:
            metrics["win_rate"] = self.calculate_win_rate(trades)
            metrics["profit_factor"] = self.calculate_profit_factor(trades)
            metrics["total_trades"] = len(trades)

        return metrics

    def _calculate_annual_return(
        self, returns: pd.Series, frequency: Frequency
    ) -> float:
        """Calculate annualized return."""
        total_return = (1 + returns).prod() - 1
        years = len(returns) / self._get_annualization_factor(frequency)
        return (1 + total_return) ** (1 / years) - 1

    def generate_backtest_report(
        self,
        metrics: Dict[str, float],
        trades: pd.DataFrame = None,
        frequency: Frequency = Frequency.DAILY,
    ) -> str:
        """
        Generate text report from backtest metrics.

        Args:
            metrics: Dictionary of metrics
            trades: Trades DataFrame (optional)
            frequency: Trading frequency

        Returns:
            Formatted report string
        """
        report = f"""
# Backtest Report
## Performance Summary
- **Total Return**: {metrics.get('total_return', 0):.2%}
- **Annual Return**: {metrics.get('annual_return', 0):.2%}
- **Volatility**: {metrics.get('volatility', 0):.2%}
- **Sharpe Ratio**: {metrics.get('sharpe_ratio', 0):.2f}
- **Sortino Ratio**: {metrics.get('sortino_ratio', 0):.2f}
- **Calmar Ratio**: {metrics.get('calmar_ratio', 0):.2f}
- **Max Drawdown**: {metrics.get('max_drawdown', 0):.2%}

## Risk Metrics
- **Peak Date**: {metrics.get('peak_date', 'N/A')}
- **Trough Date**: {metrics.get('trough_date', 'N/A')}
- **Frequency**: {frequency.value}
"""

        if trades is not None and len(trades) > 0:
            report += f"""
## Trade Analysis
- **Total Trades**: {metrics.get('total_trades', 0)}
- **Win Rate**: {metrics.get('win_rate', 0):.2%}
- **Profit Factor**: {metrics.get('profit_factor', 0):.2f}
"""

        return report


# Convenience functions for backward compatibility
def validate_backtest_data(
    data: pd.DataFrame, required_columns: List[str] = None
) -> Tuple[bool, str]:
    """Validate backtest data."""
    common = BacktestCommon()
    return common.validate_data(data, required_columns)


def calculate_backtest_metrics(
    returns: pd.Series, prices: pd.Series, frequency: Frequency = Frequency.DAILY
) -> Dict[str, float]:
    """Calculate backtest metrics."""
    common = BacktestCommon()
    return common.calculate_metrics_summary(returns, prices, frequency=frequency)


def scale_metrics_for_frequency(
    metrics: Dict[str, float], frequency: Frequency
) -> Dict[str, float]:
    """Scale metrics for different frequencies."""
    common = BacktestCommon()

    # Scale volatility and Sharpe ratio
    if "volatility" in metrics:
        annualization_factor = common._get_annualization_factor(frequency)
        metrics["volatility"] = metrics["volatility"] * np.sqrt(annualization_factor)

    if "sharpe_ratio" in metrics:
        annualization_factor = common._get_annualization_factor(frequency)
        metrics["sharpe_ratio"] = metrics["sharpe_ratio"] * np.sqrt(
            annualization_factor
        )

    return metrics
