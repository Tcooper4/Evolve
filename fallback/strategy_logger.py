"""
Fallback Strategy Logger Implementation

Provides fallback functionality for strategy decision logging when
the primary strategy logger is unavailable.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FallbackStrategyLogger:
    """
    Fallback implementation of the Strategy Logger.

    Provides basic decision logging functionality when the primary
    strategy logger is unavailable.
    """

    def __init__(self) -> None:
        """
        Initialize the fallback strategy logger.

        Sets up basic logging and initializes internal storage for
        fallback operations.
        """
        self._decisions: List[Dict[str, Any]] = []
        self._status = "fallback"
        logger.info("FallbackStrategyLogger initialized")

    def get_recent_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent strategy decisions (fallback implementation).

        Args:
            limit: Maximum number of decisions to return

        Returns:
            List[Dict[str, Any]]: List of recent decisions
        """
        try:
            logger.debug(f"Getting recent decisions (limit: {limit})")

            # Return mock decisions for fallback mode
            if not self._decisions:
                self._decisions = self._generate_mock_decisions()

            return self._decisions[:limit]

        except Exception as e:
            logger.error(f"Error getting recent decisions: {e}")
            return []

    def log_decision(self, decision: Dict[str, Any]) -> None:
        """
        Log a strategy decision (fallback implementation).

        Args:
            decision: The decision data to log
        """
        try:
            decision["timestamp"] = datetime.now().isoformat()
            decision["fallback_mode"] = True
            decision["logger"] = "fallback_strategy_logger"

            self._decisions.append(decision)

            # Keep only recent decisions to prevent memory issues
            if len(self._decisions) > 100:
                self._decisions = self._decisions[-50:]

            logger.debug(f"Logged fallback decision: {decision.get('action', 'unknown')}")

        except Exception as e:
            logger.error(f"Error logging decision: {e}")

    def _generate_mock_decisions(self) -> List[Dict[str, Any]]:
        """
        Generate mock decisions for demonstration purposes.

        Returns:
            List[Dict[str, Any]]: List of mock decisions
        """
        mock_decisions = [
            {
                "timestamp": datetime.now().isoformat(),
                "strategy": "rsi_strategy",
                "symbol": "AAPL",
                "action": "buy",
                "confidence": 0.75,
                "reasoning": "RSI indicates oversold conditions",
                "price": 150.25,
                "quantity": 100,
                "fallback_mode": True,
            },
            {
                "timestamp": datetime.now().isoformat(),
                "strategy": "macd_strategy",
                "symbol": "TSLA",
                "action": "sell",
                "confidence": 0.68,
                "reasoning": "MACD shows bearish crossover",
                "price": 245.50,
                "quantity": 50,
                "fallback_mode": True,
            },
            {
                "timestamp": datetime.now().isoformat(),
                "strategy": "bollinger_strategy",
                "symbol": "GOOGL",
                "action": "hold",
                "confidence": 0.82,
                "reasoning": "Price within Bollinger Bands",
                "price": 2750.00,
                "quantity": 0,
                "fallback_mode": True,
            },
        ]

        return mock_decisions

    def get_decision_by_id(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific decision by ID (fallback implementation).

        Args:
            decision_id: The decision ID to retrieve

        Returns:
            Optional[Dict[str, Any]]: The decision or None if not found
        """
        try:
            for decision in self._decisions:
                if decision.get("id") == decision_id:
                    return decision
            return None

        except Exception as e:
            logger.error(f"Error getting decision by ID {decision_id}: {e}")
            return None

    def get_decisions_by_strategy(self, strategy_name: str) -> List[Dict[str, Any]]:
        """
        Get all decisions for a specific strategy (fallback implementation).

        Args:
            strategy_name: Name of the strategy

        Returns:
            List[Dict[str, Any]]: List of decisions for the strategy
        """
        try:
            return [d for d in self._decisions if d.get("strategy") == strategy_name]
        except Exception as e:
            logger.error(f"Error getting decisions for strategy {strategy_name}: {e}")
            return []

    def get_decisions_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get all decisions for a specific symbol (fallback implementation).

        Args:
            symbol: The stock symbol

        Returns:
            List[Dict[str, Any]]: List of decisions for the symbol
        """
        try:
            return [d for d in self._decisions if d.get("symbol") == symbol]
        except Exception as e:
            logger.error(f"Error getting decisions for symbol {symbol}: {e}")
            return []

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get the health status of the fallback strategy logger.

        Returns:
            Dict[str, Any]: Health status information
        """
        try:
            return {
                "status": self._status,
                "total_decisions": len(self._decisions),
                "fallback_mode": True,
                "message": "Using fallback strategy logger",
            }
        except Exception as e:
            logger.error(f"Error getting fallback strategy logger health: {e}")
            return {"status": "error", "total_decisions": 0, "fallback_mode": True, "error": str(e)}
