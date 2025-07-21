"""
Edge Case Handler for Backtesting

This module handles edge cases in backtesting such as empty signals,
missing data, and broken charts to ensure robust visualization and reporting.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


class EdgeCaseHandler:
    """Handles edge cases in backtesting visualization and reporting."""

    def __init__(self):
        """Initialize the edge case handler."""
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def validate_signals(self, signals: pd.Series) -> Dict[str, Any]:
        """Validate trading signals and handle edge cases.

        Args:
            signals: Trading signals to validate

        Returns:
            Dictionary with validation status and recommendations
        """
        try:
            if signals is None or signals.empty:
                return {
                    "status": "warning",
                    "message": "No trading signals found for the selected period",
                    "recommendation": "Try adjusting the date range or strategy parameters",
                    "has_signals": False,
                    "signal_count": 0,
                }

            # Check for valid signal values
            valid_signals = [-1, 0, 1]
            invalid_signals = signals[~signals.isin(valid_signals)]

            if not invalid_signals.empty:
                return {
                    "status": "error",
                    "message": f"Invalid signal values found: {invalid_signals.unique()}",
                    "recommendation": "Check strategy implementation for signal generation",
                    "has_signals": False,
                    "signal_count": 0,
                }

            # Count signal types
            buy_signals = (signals == 1).sum()
            sell_signals = (signals == -1).sum()
            hold_signals = (signals == 0).sum()

            # Check for signal imbalance
            total_signals = buy_signals + sell_signals
            if total_signals == 0:
                return {
                    "status": "warning",
                    "message": "No buy or sell signals generated",
                    "recommendation": "Strategy may be too conservative or market conditions unsuitable",
                    "has_signals": False,
                    "signal_count": 0,
                    "buy_signals": buy_signals,
                    "sell_signals": sell_signals,
                    "hold_signals": hold_signals,
                }

            # Check for signal clustering
            signal_ratio = buy_signals / total_signals if total_signals > 0 else 0
            if signal_ratio > 0.8 or signal_ratio < 0.2:
                self.warnings.append(
                    f"Signal imbalance detected: {signal_ratio:.1%} buy signals"
                )

            return {
                "status": "success",
                "message": "Signals validated successfully",
                "has_signals": True,
                "signal_count": total_signals,
                "buy_signals": buy_signals,
                "sell_signals": sell_signals,
                "hold_signals": hold_signals,
                "signal_ratio": signal_ratio,
            }

        except Exception as e:
            logger.error(f"Error validating signals: {e}")
            return {
                "status": "error",
                "message": f"Error validating signals: {str(e)}",
                "has_signals": False,
                "signal_count": 0,
            }

    def validate_trade_data(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate trade data and handle edge cases.

        Args:
            trades: List of trade dictionaries

        Returns:
            Dictionary with validation status and trade summary
        """
        try:
            if not trades:
                return {
                    "status": "warning",
                    "message": "No trades executed during backtest period",
                    "recommendation": "Check strategy parameters or market conditions",
                    "has_trades": False,
                    "trade_count": 0,
                }

            # Validate trade structure
            required_fields = ["timestamp", "asset", "quantity", "price", "type"]
            invalid_trades = []

            for i, trade in enumerate(trades):
                missing_fields = [
                    field for field in required_fields if field not in trade
                ]
                if missing_fields:
                    invalid_trades.append(f"Trade {i}: Missing fields {missing_fields}")

            if invalid_trades:
                return {
                    "status": "error",
                    "message": f'Invalid trade data: {"; ".join(invalid_trades)}',
                    "has_trades": False,
                    "trade_count": 0,
                }

            # Analyze trade distribution
            trade_types = [trade.get("type", "unknown") for trade in trades]
            buy_trades = trade_types.count("buy")
            sell_trades = trade_types.count("sell")

            # Check for trade imbalance
            total_trades = buy_trades + sell_trades
            if total_trades == 0:
                return {
                    "status": "warning",
                    "message": "No buy or sell trades found",
                    "has_trades": False,
                    "trade_count": 0,
                }

            return {
                "status": "success",
                "message": "Trade data validated successfully",
                "has_trades": True,
                "trade_count": len(trades),
                "buy_trades": buy_trades,
                "sell_trades": sell_trades,
                "total_trades": total_trades,
            }

        except Exception as e:
            logger.error(f"Error validating trade data: {e}")
            return {
                "status": "error",
                "message": f"Error validating trade data: {str(e)}",
                "has_trades": False,
                "trade_count": 0,
            }

    def create_fallback_chart(
        self, data: pd.DataFrame, chart_type: str = "equity"
    ) -> Dict[str, Any]:
        """Create a fallback chart when no signals or trades are available.

        Args:
            data: Price data DataFrame
            chart_type: Type of chart to create

        Returns:
            Dictionary with fallback chart data
        """
        try:
            if data.empty:
                return {
                    "status": "error",
                    "message": "No data available for chart",
                    "chart_data": None,
                }

            if chart_type == "equity":
                # Create simple price chart
                chart_data = {
                    "x": data.index.tolist(),
                    "y": (
                        data["close"].tolist()
                        if "close" in data.columns
                        else data.iloc[:, 0].tolist()
                    ),
                    "type": "scatter",
                    "mode": "lines",
                    "name": "Price",
                }

                return {
                    "status": "success",
                    "message": "Fallback price chart created",
                    "chart_data": chart_data,
                    "chart_type": "price_only",
                }

            elif chart_type == "performance":
                # Create performance metrics placeholder
                chart_data = {
                    "metrics": {
                        "total_return": 0.0,
                        "sharpe_ratio": 0.0,
                        "max_drawdown": 0.0,
                        "win_rate": 0.0,
                    },
                    "message": "No performance data available",
                }

                return {
                    "status": "warning",
                    "message": "No performance data available",
                    "chart_data": chart_data,
                    "chart_type": "metrics_placeholder",
                }

            else:
                return {
                    "status": "error",
                    "message": f"Unknown chart type: {chart_type}",
                    "chart_data": None,
                }

        except Exception as e:
            logger.error(f"Error creating fallback chart: {e}")
            return {
                "status": "error",
                "message": f"Error creating fallback chart: {str(e)}",
                "chart_data": None,
            }

    def generate_edge_case_report(self) -> Dict[str, Any]:
        """Generate a report of all edge cases encountered.

        Returns:
            Dictionary with edge case summary
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "warnings": self.warnings.copy(),
            "errors": self.errors.copy(),
            "warning_count": len(self.warnings),
            "error_count": len(self.errors),
            "summary": {
                "total_issues": len(self.warnings) + len(self.errors),
                "has_warnings": len(self.warnings) > 0,
                "has_errors": len(self.errors) > 0,
            },
        }

    def clear_history(self) -> None:
        """Clear warning and error history."""
        self.warnings.clear()
        self.errors.clear()

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
        logger.warning(warning)

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        logger.error(error)
