"""
Fallback Portfolio Manager Implementation

Provides fallback functionality for portfolio management when
the primary portfolio manager is unavailable.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class FallbackPortfolioManager:
    """
    Fallback implementation of the Portfolio Manager.

    Provides basic portfolio operations and risk metrics when the
    primary portfolio manager is unavailable.
    """

    def __init__(self) -> None:
        """
        Initialize the fallback portfolio manager.

        Sets up basic logging and initializes mock portfolio data for
        fallback operations.
        """
        self._status = "fallback"
        self._mock_portfolio = self._initialize_mock_portfolio()
        logger.info("FallbackPortfolioManager initialized")

    def _initialize_mock_portfolio(self) -> Dict[str, Any]:
        """
        Initialize mock portfolio data for fallback operations.

        Returns:
            Dict[str, Any]: Mock portfolio data
        """
        return {
            "cash": 50000.0,
            "total_value": 150000.0,
            "positions": [
                {
                    "symbol": "AAPL",
                    "quantity": 100,
                    "avg_price": 150.25,
                    "current_price": 155.50,
                    "market_value": 15550.0,
                    "unrealized_pnl": 525.0,
                    "unrealized_pnl_pct": 3.49,
                },
                {
                    "symbol": "TSLA",
                    "quantity": 50,
                    "avg_price": 240.00,
                    "current_price": 245.50,
                    "market_value": 12275.0,
                    "unrealized_pnl": 275.0,
                    "unrealized_pnl_pct": 2.29,
                },
                {
                    "symbol": "GOOGL",
                    "quantity": 25,
                    "avg_price": 2700.00,
                    "current_price": 2750.00,
                    "market_value": 68750.0,
                    "unrealized_pnl": 1250.0,
                    "unrealized_pnl_pct": 1.85,
                },
            ],
            "total_unrealized_pnl": 2050.0,
            "total_unrealized_pnl_pct": 1.37,
        }

    def get_position_summary(self) -> Dict[str, Any]:
        """
        Get portfolio position summary (fallback implementation).

        Returns:
            Dict[str, Any]: Portfolio position summary
        """
        try:
            logger.debug("Getting portfolio position summary from fallback manager")

            summary = self._mock_portfolio.copy()
            summary["timestamp"] = datetime.now().isoformat()
            summary["fallback_mode"] = True

            return summary

        except Exception as e:
            logger.error(f"Error getting position summary: {e}")
            return {"positions": [], "total_value": 0, "cash": 100000, "fallback_mode": True, "error": str(e)}

    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get portfolio risk metrics (fallback implementation).

        Returns:
            Dict[str, Any]: Portfolio risk metrics
        """
        try:
            logger.debug("Getting risk metrics from fallback manager")

            # Calculate basic risk metrics
            total_value = self._mock_portfolio["total_value"]
            cash = self._mock_portfolio["cash"]

            # Mock risk metrics
            risk_metrics = {
                "volatility": 0.15,  # 15% annual volatility
                "var_95": total_value * 0.02,  # 2% VaR at 95% confidence
                "var_99": total_value * 0.03,  # 3% VaR at 99% confidence
                "beta": 1.05,  # Slightly higher than market
                "sharpe_ratio": 1.2,
                "sortino_ratio": 1.5,
                "max_drawdown": 0.08,  # 8% max drawdown
                "correlation": 0.75,  # Market correlation
                "concentration": 0.45,  # Top 3 positions concentration
                "timestamp": datetime.now().isoformat(),
                "fallback_mode": True,
            }

            return risk_metrics

        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            return {
                "volatility": 0.0,
                "var_95": 0.0,
                "var_99": 0.0,
                "beta": 0.0,
                "fallback_mode": True,
                "error": str(e),
            }

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get position for a specific symbol (fallback implementation).

        Args:
            symbol: The stock symbol

        Returns:
            Optional[Dict[str, Any]]: Position data or None if not found
        """
        try:
            for position in self._mock_portfolio["positions"]:
                if position["symbol"] == symbol:
                    position_copy = position.copy()
                    position_copy["timestamp"] = datetime.now().isoformat()
                    position_copy["fallback_mode"] = True
                    return position_copy
            return None

        except Exception as e:
            logger.error(f"Error getting position for {symbol}: {e}")
            return None

    def add_position(self, symbol: str, quantity: int, price: float) -> bool:
        """
        Add a new position (fallback implementation).

        Args:
            symbol: The stock symbol
            quantity: Number of shares
            price: Price per share

        Returns:
            bool: True if position was added successfully
        """
        try:
            logger.info(f"Adding position: {symbol} {quantity} shares at ${price}")

            # Check if position already exists
            existing_position = self.get_position(symbol)

            if existing_position:
                # Update existing position
                new_quantity = existing_position["quantity"] + quantity
                new_avg_price = (
                    existing_position["quantity"] * existing_position["avg_price"] + quantity * price
                ) / new_quantity

                # Update the position in mock portfolio
                for i, pos in enumerate(self._mock_portfolio["positions"]):
                    if pos["symbol"] == symbol:
                        self._mock_portfolio["positions"][i].update(
                            {"quantity": new_quantity, "avg_price": new_avg_price}
                        )
                        break
            else:
                # Add new position
                new_position = {
                    "symbol": symbol,
                    "quantity": quantity,
                    "avg_price": price,
                    "current_price": price,
                    "market_value": quantity * price,
                    "unrealized_pnl": 0.0,
                    "unrealized_pnl_pct": 0.0,
                }
                self._mock_portfolio["positions"].append(new_position)

            # Update cash
            self._mock_portfolio["cash"] -= quantity * price

            logger.info(f"Successfully added position for {symbol}")
            return True

        except Exception as e:
            logger.error(f"Error adding position for {symbol}: {e}")
            return False

    def remove_position(self, symbol: str, quantity: int) -> bool:
        """
        Remove shares from a position (fallback implementation).

        Args:
            symbol: The stock symbol
            quantity: Number of shares to remove

        Returns:
            bool: True if position was updated successfully
        """
        try:
            logger.info(f"Removing {quantity} shares of {symbol}")

            existing_position = self.get_position(symbol)
            if not existing_position:
                logger.warning(f"No position found for {symbol}")
                return False

            if existing_position["quantity"] < quantity:
                logger.warning(f"Insufficient shares for {symbol}")
                return False

            # Update position
            new_quantity = existing_position["quantity"] - quantity
            current_price = existing_position["current_price"]

            # Update the position in mock portfolio
            for i, pos in enumerate(self._mock_portfolio["positions"]):
                if pos["symbol"] == symbol:
                    if new_quantity == 0:
                        # Remove position entirely
                        self._mock_portfolio["positions"].pop(i)
                    else:
                        # Update quantity
                        self._mock_portfolio["positions"][i]["quantity"] = new_quantity
                    break

            # Update cash
            self._mock_portfolio["cash"] += quantity * current_price

            logger.info(f"Successfully removed {quantity} shares of {symbol}")
            return True

        except Exception as e:
            logger.error(f"Error removing position for {symbol}: {e}")
            return False

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get the health status of the fallback portfolio manager.

        Returns:
            Dict[str, Any]: Health status information
        """
        try:
            return {
                "status": self._status,
                "total_positions": len(self._mock_portfolio["positions"]),
                "total_value": self._mock_portfolio["total_value"],
                "fallback_mode": True,
                "message": "Using fallback portfolio manager",
            }
        except Exception as e:
            logger.error(f"Error getting fallback portfolio manager health: {e}")
            return {"status": "error", "total_positions": 0, "total_value": 0, "fallback_mode": True, "error": str(e)}
