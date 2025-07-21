"""Risk Calculator Module.

This module contains risk calculation logic extracted from execution_agent.py.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from trading.portfolio.portfolio_manager import Position

from .risk_controls import RiskControls, RiskThresholdType

logger = logging.getLogger(__name__)


class RiskCalculator:
    """Risk calculation utilities."""

    def __init__(self):
        """Initialize risk calculator."""
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate_atr(
        self, symbol: str, period: int, market_data: Dict[str, Any]
    ) -> float:
        """Calculate Average True Range (ATR)."""
        try:
            if symbol not in market_data or "ohlc" not in market_data[symbol]:
                return 0.0

            ohlc_data = market_data[symbol]["ohlc"]
            if len(ohlc_data) < period + 1:
                return 0.0

            # Calculate True Range
            high = ohlc_data["High"].values
            low = ohlc_data["Low"].values
            close = ohlc_data["Close"].values

            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))

            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            atr = np.mean(true_range[-period:])

            return float(atr)

        except Exception as e:
            self.logger.error(f"Error calculating ATR for {symbol}: {e}")
            return 0.0

    def calculate_portfolio_correlation(self, market_data: Dict[str, Any]) -> float:
        """Calculate portfolio correlation."""
        try:
            # Get portfolio positions and their returns
            positions = market_data.get("portfolio_positions", {})
            if not positions:
                return 0.0

            returns_data = {}
            for symbol, position in positions.items():
                if symbol in market_data and "returns" in market_data[symbol]:
                    returns_data[symbol] = market_data[symbol]["returns"]

            if len(returns_data) < 2:
                return 0.0

            # Calculate correlation matrix
            returns_df = pd.DataFrame(returns_data)
            correlation_matrix = returns_df.corr()

            # Calculate average correlation (excluding diagonal)
            correlations = []
            for i in range(len(correlation_matrix)):
                for j in range(i + 1, len(correlation_matrix)):
                    corr = correlation_matrix.iloc[i, j]
                    if not np.isnan(corr):
                        correlations.append(corr)

            return float(np.mean(correlations)) if correlations else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating portfolio correlation: {e}")
            return 0.0

    def calculate_portfolio_risk_exposure(self, market_data: Dict[str, Any]) -> float:
        """Calculate portfolio risk exposure."""
        try:
            positions = market_data.get("portfolio_positions", {})
            if not positions:
                return 0.0

            total_exposure = 0.0
            for symbol, position in positions.items():
                current_price = market_data.get(symbol, {}).get("current_price", 0)
                if current_price > 0:
                    exposure = abs(position.size * current_price)
                    total_exposure += exposure

            portfolio_value = market_data.get("portfolio_value", 1.0)
            return total_exposure / portfolio_value if portfolio_value > 0 else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk exposure: {e}")
            return 0.0

    def calculate_position_risk_metrics(
        self, position: Position, current_price: float, market_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate position risk metrics."""
        try:
            metrics = {}

            # Calculate unrealized P&L
            if position.entry_price > 0:
                if position.direction.value == "long":
                    pnl = (current_price - position.entry_price) * position.size
                else:
                    pnl = (position.entry_price - current_price) * position.size
                metrics["unrealized_pnl"] = pnl
                metrics["unrealized_pnl_pct"] = pnl / (
                    position.entry_price * position.size
                )

            # Calculate drawdown
            if position.entry_price > 0:
                if position.direction.value == "long":
                    drawdown = (
                        position.entry_price - current_price
                    ) / position.entry_price
                else:
                    drawdown = (
                        current_price - position.entry_price
                    ) / position.entry_price
                metrics["drawdown"] = drawdown

            # Calculate volatility (if market data available)
            symbol = position.symbol
            if symbol in market_data and "returns" in market_data[symbol]:
                returns = market_data[symbol]["returns"]
                if len(returns) > 0:
                    metrics["volatility"] = float(np.std(returns))

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating position risk metrics: {e}")
            return {}

    def calculate_portfolio_risk_metrics(
        self, market_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate portfolio risk metrics."""
        try:
            metrics = {}

            # Calculate total portfolio value
            portfolio_value = market_data.get("portfolio_value", 0.0)
            metrics["portfolio_value"] = portfolio_value

            # Calculate total unrealized P&L
            total_pnl = 0.0
            positions = market_data.get("portfolio_positions", {})
            for position in positions.values():
                current_price = market_data.get(position.symbol, {}).get(
                    "current_price", 0
                )
                if current_price > 0:
                    if position.direction.value == "long":
                        pnl = (current_price - position.entry_price) * position.size
                    else:
                        pnl = (position.entry_price - current_price) * position.size
                    total_pnl += pnl

            metrics["total_unrealized_pnl"] = total_pnl
            metrics["total_unrealized_pnl_pct"] = (
                total_pnl / portfolio_value if portfolio_value > 0 else 0.0
            )

            # Calculate portfolio volatility
            portfolio_returns = market_data.get("portfolio_returns", [])
            if portfolio_returns:
                metrics["portfolio_volatility"] = float(np.std(portfolio_returns))

            # Calculate correlation
            metrics["portfolio_correlation"] = self.calculate_portfolio_correlation(
                market_data
            )

            # Calculate risk exposure
            metrics["risk_exposure"] = self.calculate_portfolio_risk_exposure(
                market_data
            )

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk metrics: {e}")
            return {}

    def calculate_stop_loss_price(
        self,
        position: Position,
        risk_controls: RiskControls,
        market_data: Dict[str, Any],
    ) -> float:
        """Calculate stop loss price."""
        try:
            threshold = risk_controls.stop_loss

            if threshold.threshold_type == RiskThresholdType.PERCENTAGE:
                if position.direction.value == "long":
                    return position.entry_price * (1 - threshold.value)
                else:
                    return position.entry_price * (1 + threshold.value)

            elif threshold.threshold_type == RiskThresholdType.ATR_BASED:
                atr = self.calculate_atr(
                    position.symbol, threshold.atr_period, market_data
                )
                multiplier = threshold.atr_multiplier or 2.0

                if position.direction.value == "long":
                    return position.entry_price - (atr * multiplier)
                else:
                    return position.entry_price + (atr * multiplier)

            elif threshold.threshold_type == RiskThresholdType.FIXED:
                if position.direction.value == "long":
                    return position.entry_price - threshold.value
                else:
                    return position.entry_price + threshold.value

            return 0.0

        except Exception as e:
            self.logger.error(f"Error calculating stop loss price: {e}")
            return 0.0

    def calculate_take_profit_price(
        self,
        position: Position,
        risk_controls: RiskControls,
        market_data: Dict[str, Any],
    ) -> float:
        """Calculate take profit price."""
        try:
            threshold = risk_controls.take_profit

            if threshold.threshold_type == RiskThresholdType.PERCENTAGE:
                if position.direction.value == "long":
                    return position.entry_price * (1 + threshold.value)
                else:
                    return position.entry_price * (1 - threshold.value)

            elif threshold.threshold_type == RiskThresholdType.ATR_BASED:
                atr = self.calculate_atr(
                    position.symbol, threshold.atr_period, market_data
                )
                multiplier = threshold.atr_multiplier or 3.0

                if position.direction.value == "long":
                    return position.entry_price + (atr * multiplier)
                else:
                    return position.entry_price - (atr * multiplier)

            elif threshold.threshold_type == RiskThresholdType.FIXED:
                if position.direction.value == "long":
                    return position.entry_price + threshold.value
                else:
                    return position.entry_price - threshold.value

            return 0.0

        except Exception as e:
            self.logger.error(f"Error calculating take profit price: {e}")
            return 0.0
