"""Trade Execution Engine with Mock Simulator.

This module provides a trade execution engine with realistic simulation of
market conditions including slippage, market impact, and commission.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Types of orders."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status."""

    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Represents a trading order."""

    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    market_impact: float = 0.0

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ExecutionResult:
    """Result of order execution."""

    order: Order
    execution_price: float
    execution_quantity: float
    commission: float
    slippage: float
    market_impact: float
    total_cost: float
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None


class MarketSimulator:
    """Simulates market conditions for trade execution."""

    def __init__(
        self,
        base_slippage: float = 0.0001,
        market_impact_factor: float = 0.0001,
        commission_rate: float = 0.001,
        min_commission: float = 1.0,
        max_commission: float = 50.0,
    ):
        """Initialize market simulator.

        Args:
            base_slippage: Base slippage rate (0.01% = 0.0001)
            market_impact_factor: Market impact factor
            commission_rate: Commission rate as percentage
            min_commission: Minimum commission per trade
            max_commission: Maximum commission per trade
        """
        self.base_slippage = base_slippage
        self.market_impact_factor = market_impact_factor
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.max_commission = max_commission

        # Market volatility simulation
        self.volatility = 0.02  # 2% daily volatility
        self.spread = 0.0005  # 0.05% spread

        logger.info(
            f"Market simulator initialized with slippage={base_slippage}, "
            f"impact={market_impact_factor}, commission={commission_rate}"
        )

    def calculate_slippage(
        self, order_quantity: float, market_volume: float, price: float
    ) -> float:
        """Calculate slippage based on order size and market conditions.

        Args:
            order_quantity: Order quantity
            market_volume: Market volume
            price: Current market price

        Returns:
            Slippage amount
        """
        # Base slippage
        slippage = self.base_slippage

        # Volume-based slippage (larger orders = more slippage)
        volume_ratio = order_quantity / market_volume if market_volume > 0 else 0
        volume_slippage = min(volume_ratio * 0.001, 0.01)  # Max 1% slippage

        # Volatility-based slippage
        volatility_slippage = self.volatility * np.random.normal(0, 1) * 0.1

        # Spread component
        spread_slippage = self.spread * np.random.uniform(0.5, 1.5)

        total_slippage = (
            slippage + volume_slippage + volatility_slippage + spread_slippage
        )

        # Ensure slippage is positive
        return max(total_slippage, 0.0)

    def calculate_market_impact(
        self, order_quantity: float, market_volume: float, price: float
    ) -> float:
        """Calculate market impact of the order.

        Args:
            order_quantity: Order quantity
            market_volume: Market volume
            price: Current market price

        Returns:
            Market impact amount
        """
        # Market impact increases with order size relative to market volume
        volume_ratio = order_quantity / market_volume if market_volume > 0 else 0

        # Square root model for market impact
        impact = self.market_impact_factor * np.sqrt(volume_ratio) * price

        # Add some randomness
        impact *= np.random.uniform(0.8, 1.2)

        return max(impact, 0.0)

    def calculate_commission(self, order_value: float, order_quantity: float) -> float:
        """Calculate commission for the order.

        Args:
            order_value: Total order value
            order_quantity: Order quantity

        Returns:
            Commission amount
        """
        # Base commission
        commission = order_value * self.commission_rate

        # Apply min/max constraints
        commission = max(commission, self.min_commission)
        commission = min(commission, self.max_commission)

        # Add some variability
        commission *= np.random.uniform(0.95, 1.05)

        return commission


class TradeExecutor:
    """Main trade execution engine."""

    def __init__(
        self,
        market_simulator: Optional[MarketSimulator] = None,
        live_trading: bool = False,
    ):
        """Initialize trade executor.

        Args:
            market_simulator: Market simulator instance
            live_trading: Whether to use live trading mode
        """
        self.market_simulator = market_simulator or MarketSimulator()
        self.live_trading = live_trading
        self.orders: List[Order] = []
        self.executions: List[ExecutionResult] = []

        # Load live trading setting from environment
        if os.getenv("LIVE_TRADING", "False").lower() == "true":
            self.live_trading = True
            logger.warning("LIVE TRADING MODE ENABLED - Real money will be used!")

        logger.info(f"Trade executor initialized - Live trading: {self.live_trading}")

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Order:
        """Place a new order.

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
            order_type: Type of order
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)

        Returns:
            Order object
        """
        order_id = (
            f"order_{len(self.orders)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
        )

        self.orders.append(order)
        logger.info(f"Order placed: {order_id} - {side} {quantity} {symbol}")

        return order

    def execute_order(
        self, order: Order, market_price: float, market_volume: float = 1000000
    ) -> ExecutionResult:
        """Execute an order with market simulation.

        Args:
            order: Order to execute
            market_price: Current market price
            market_volume: Market volume for impact calculation

        Returns:
            Execution result
        """
        try:
            # Calculate execution price with slippage and market impact
            slippage = self.market_simulator.calculate_slippage(
                order.quantity, market_volume, market_price
            )
            market_impact = self.market_simulator.calculate_market_impact(
                order.quantity, market_volume, market_price
            )

            # Determine execution price based on order type
            if order.order_type == OrderType.MARKET:
                if order.side == "buy":
                    execution_price = market_price * (1 + slippage + market_impact)
                else:
                    execution_price = market_price * (1 - slippage - market_impact)
            elif order.order_type == OrderType.LIMIT:
                if order.price is None:
                    raise ValueError("Limit price required for limit orders")
                execution_price = order.price
            else:
                execution_price = market_price

            # Calculate commission
            order_value = order.quantity * execution_price
            commission = self.market_simulator.calculate_commission(
                order_value, order.quantity
            )

            # Calculate total cost
            total_cost = order_value + commission

            # Update order
            order.filled_quantity = order.quantity
            order.filled_price = execution_price
            order.commission = commission
            order.slippage = slippage
            order.market_impact = market_impact
            order.status = OrderStatus.FILLED

            # Create execution result
            result = ExecutionResult(
                order=order,
                execution_price=execution_price,
                execution_quantity=order.quantity,
                commission=commission,
                slippage=slippage,
                market_impact=market_impact,
                total_cost=total_cost,
                timestamp=datetime.now(),
                success=True,
            )

            self.executions.append(result)

            logger.info(
                f"Order executed: {order.id} - Price: {execution_price:.4f}, "
                f"Cost: {total_cost:.2f}, Slippage: {slippage:.4f}"
            )

            return result

        except Exception as e:
            logger.error(f"Order execution failed: {order.id} - {str(e)}")

            order.status = OrderStatus.REJECTED

            return ExecutionResult(
                order=order,
                execution_price=0.0,
                execution_quantity=0.0,
                commission=0.0,
                slippage=0.0,
                market_impact=0.0,
                total_cost=0.0,
                timestamp=datetime.now(),
                success=False,
                error_message=str(e),
            )

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all executions.

        Returns:
            Dictionary with execution summary
        """
        if not self.executions:
            return {
                "total_orders": 0,
                "successful_orders": 0,
                "total_volume": 0.0,
                "total_commission": 0.0,
                "total_slippage": 0.0,
                "total_market_impact": 0.0,
                "average_execution_price": 0.0,
                "success_rate": 0.0,
            }

        successful_executions = [e for e in self.executions if e.success]

        total_volume = sum(e.execution_quantity for e in successful_executions)
        total_commission = sum(e.commission for e in successful_executions)
        total_slippage = sum(
            e.slippage * e.execution_quantity * e.execution_price
            for e in successful_executions
        )
        total_market_impact = sum(
            e.market_impact * e.execution_quantity for e in successful_executions
        )

        avg_price = (
            sum(e.execution_price * e.execution_quantity for e in successful_executions)
            / total_volume
            if total_volume > 0
            else 0.0
        )

        return {
            "total_orders": len(self.orders),
            "successful_orders": len(successful_executions),
            "total_volume": total_volume,
            "total_commission": total_commission,
            "total_slippage": total_slippage,
            "total_market_impact": total_market_impact,
            "average_execution_price": avg_price,
            "success_rate": len(successful_executions) / len(self.executions),
        }

    def simulate_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        market_price: float,
        market_volume: float = 1000000,
    ) -> ExecutionResult:
        """Simulate a complete trade execution.

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Trade quantity
            market_price: Current market price
            market_volume: Market volume

        Returns:
            Execution result
        """
        # Place order
        order = self.place_order(symbol, side, quantity)

        # Execute order
        result = self.execute_order(order, market_price, market_volume)

        return result


# Global trade executor instance
trade_executor = TradeExecutor()


def get_trade_executor() -> TradeExecutor:
    """Get the global trade executor instance."""
    return trade_executor
