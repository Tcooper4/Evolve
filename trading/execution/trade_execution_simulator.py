# -*- coding: utf-8 -*-
"""
Trade Execution Simulator with realistic market impact modeling.
Enhanced with Batch 9 features: slippage model, random fill success simulation,
and real-world orderbook emulation.
"""

import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from trading.data.data_loader import DataLoader
from trading.market.market_analyzer import MarketAnalyzer


class OrderType(str, Enum):
    """Order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"
    STOP_MARKET = "stop_market"


class OrderStatus(str, Enum):
    """Order status."""

    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class FillType(str, Enum):
    """Fill types for Batch 9 enhancement."""
    IMMEDIATE = "immediate"
    PARTIAL = "partial"
    DELAYED = "delayed"
    FAILED = "failed"


@dataclass
class OrderBookLevel:
    """Order book level for Batch 9 enhancement."""
    price: float
    quantity: float
    side: str  # 'bid' or 'ask'
    timestamp: datetime


@dataclass
class OrderBook:
    """Order book snapshot for Batch 9 enhancement."""
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    spread: float
    mid_price: float


@dataclass
class Order:
    """Order information."""

    order_id: str
    symbol: str
    order_type: OrderType
    side: str  # 'buy' or 'sell'
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    market_impact: float = 0.0  # Batch 9 enhancement


@dataclass
class ExecutionResult:
    """Trade execution result."""

    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    execution_time: float
    slippage: float
    commission: float
    market_impact: float
    total_cost: float
    success: bool
    fill_type: FillType = FillType.IMMEDIATE  # Batch 9 enhancement
    failure_reason: Optional[str] = None
    orderbook_snapshot: Optional[OrderBook] = None  # Batch 9 enhancement


class TradeExecutionSimulator:
    """
    Trade Execution Simulator with:
    - Realistic slippage and spread modeling
    - Order type emulation (market, limit, stop-limit)
    - Market impact modeling
    - Commission and fee calculation
    - Execution delay simulation
    - Batch 9 enhancements: random fill success simulation, orderbook emulation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Trade Execution Simulator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.data_loader = DataLoader()
        self.market_analyzer = MarketAnalyzer()

        # Configuration
        self.base_commission = self.config.get("base_commission", 0.005)  # $5 per trade
        self.commission_rate = self.config.get(
            "commission_rate", 0.001
        )  # 0.1% of trade value
        self.min_commission = self.config.get("min_commission", 1.0)
        self.max_commission = self.config.get("max_commission", 29.95)

        # Slippage configuration
        self.base_slippage = self.config.get("base_slippage", 0.0001)  # 1 basis point
        self.volume_impact_factor = self.config.get("volume_impact_factor", 0.1)
        self.volatility_impact_factor = self.config.get("volatility_impact_factor", 0.5)
        self.slippage_model = self.config.get("slippage_model", "linear")  # Batch 9 enhancement

        # Execution delay configuration
        self.min_execution_delay = self.config.get(
            "min_execution_delay", 0.1
        )  # seconds
        self.max_execution_delay = self.config.get(
            "max_execution_delay", 2.0
        )  # seconds
        self.market_hours_only = self.config.get("market_hours_only", True)

        # Spread configuration
        self.base_spread = self.config.get("base_spread", 0.0002)  # 2 basis points
        self.volatility_spread_factor = self.config.get("volatility_spread_factor", 0.3)

        # Fill success configuration (Batch 9 enhancement)
        self.base_fill_rate = self.config.get("base_fill_rate", 0.95)
        self.volume_fill_factor = self.config.get("volume_fill_factor", 0.1)
        self.volatility_fill_factor = self.config.get("volatility_fill_factor", 0.3)
        
        # Market constraint configuration (Batch 9 enhancement)
        self.max_order_size = self.config.get("max_order_size", 1000000)  # $1M
        self.min_order_size = self.config.get("min_order_size", 100)  # $100
        self.max_daily_volume = self.config.get("max_daily_volume", 10000000)  # $10M
        
        # Orderbook configuration (Batch 9 enhancement)
        self.orderbook_levels = self.config.get("orderbook_levels", 10)
        self.spread_multiplier = self.config.get("spread_multiplier", 1.5)
        self.volume_decay_factor = self.config.get("volume_decay_factor", 0.8)

        # Storage
        self.orders: Dict[str, Order] = {}
        self.execution_history: List[ExecutionResult] = []
        self.market_data_cache: Dict[str, pd.DataFrame] = {}
        self.daily_volume: Dict[str, float] = {}  # Batch 9 enhancement
        self.volatility_cache: Dict[str, float] = {}  # Batch 9 enhancement
        self.orderbook_history: Dict[str, List[OrderBook]] = {}  # Batch 9 enhancement

    def place_order(
        self,
        symbol: str,
        order_type: OrderType,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> str:
        """
        Place a new order with enhanced validation.

        Args:
            symbol: Asset symbol
            order_type: Type of order
            side: 'buy' or 'sell'
            quantity: Order quantity
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)

        Returns:
            Order ID
        """
        try:
            # Generate order ID
            order_id = f"order_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"

            # Validate order parameters
            if not self._validate_order_parameters(
                symbol, order_type, side, quantity, price, stop_price
            ):
                raise ValueError("Invalid order parameters")

            # Check market constraints (Batch 9 enhancement)
            if not self._check_market_constraints(symbol, quantity, price):
                raise ValueError("Order violates market constraints")

            # Create order
            order = Order(
                order_id=order_id,
                symbol=symbol,
                order_type=order_type,
                side=side,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                timestamp=datetime.now(),
            )

            # Store order
            self.orders[order_id] = order

            self.logger.info(
                f"Placed {order_type.value} order: {order_id} for {quantity} {symbol} at {price}"
            )

            return order_id

        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            raise

    def execute_order(
        self, order_id: str, market_data: pd.DataFrame, orderbook_data: Optional[OrderBook] = None
    ) -> ExecutionResult:
        """
        Execute an order with realistic market simulation and Batch 9 enhancements.

        Args:
            order_id: Order ID to execute
            market_data: Current market data
            orderbook_data: Optional orderbook data for realistic fills

        Returns:
            Execution result
        """
        try:
            if order_id not in self.orders:
                raise ValueError(f"Order {order_id} not found")

            order = self.orders[order_id]

            # Check if order can be executed
            if not self._can_execute_order(order, market_data):
                return self._create_rejected_result(order, "Market conditions prevent execution")

            # Generate or use orderbook (Batch 9 enhancement)
            orderbook = orderbook_data or self._generate_orderbook(order.symbol, market_data)

            # Calculate fill success probability (Batch 9 enhancement)
            fill_probability = self._calculate_fill_probability(order, market_data, orderbook)

            # Simulate fill success (Batch 9 enhancement)
            if random.random() > fill_probability:
                return self._create_failed_result(order, "Fill failed due to market conditions")

            # Calculate execution parameters
            execution_price = self._calculate_execution_price(order, market_data, orderbook)
            slippage = self._calculate_slippage(order, market_data, orderbook)
            commission = self._calculate_commission(order, execution_price)
            market_impact = self._calculate_market_impact(order, market_data, orderbook)
            execution_delay = self._calculate_execution_delay(order)

            # Simulate execution delay (Batch 9 enhancement)
            time.sleep(execution_delay)

            # Calculate total cost
            total_cost = self._calculate_total_cost(order, execution_price, commission, slippage)

            # Update order status
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_price = execution_price
            order.commission = commission
            order.slippage = slippage
            order.market_impact = market_impact

            # Update daily volume (Batch 9 enhancement)
            self._update_daily_volume(order.symbol, order.quantity * execution_price)

            # Create execution result
            result = ExecutionResult(
                order_id=order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=execution_price,
                timestamp=datetime.now(),
                execution_time=execution_delay,
                slippage=slippage,
                commission=commission,
                market_impact=market_impact,
                total_cost=total_cost,
                success=True,
                fill_type=FillType.IMMEDIATE,
                orderbook_snapshot=orderbook
            )

            # Store execution history
            self.execution_history.append(result)

            self.logger.info(
                f"Order {order_id} executed successfully: {execution_price:.4f} "
                f"(slippage: {slippage:.4f}, commission: {commission:.2f})"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error executing order {order_id}: {str(e)}")
            return self._create_failed_result(
                self.orders.get(order_id), 
                f"Execution error: {str(e)}"
            )

    def _validate_order_parameters(
        self,
        symbol: str,
        order_type: OrderType,
        side: str,
        quantity: float,
        price: Optional[float],
        stop_price: Optional[float],
    ) -> bool:
        """Validate order parameters."""
        try:
            # Check basic parameters
            if quantity <= 0:
                return False

            if side not in ["buy", "sell"]:
                return False

            # Check order type specific requirements
            if order_type == OrderType.LIMIT and price is None:
                return False

            if order_type == OrderType.STOP_LIMIT and (
                price is None or stop_price is None
            ):
                return False

            if order_type == OrderType.STOP_MARKET and stop_price is None:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating order parameters: {str(e)}")
            return False

    def _can_execute_order(self, order: Order, market_data: pd.DataFrame) -> bool:
        """Check if order can be executed."""
        try:
            if market_data.empty or "close" not in market_data.columns:
                return False

            current_price = market_data["close"].iloc[-1]

            # Check market hours
            if self.market_hours_only:
                current_hour = order.timestamp.hour
                if current_hour < 9 or current_hour > 16:  # Simplified market hours
                    return False

            # Check order type specific conditions
            if order.order_type == OrderType.LIMIT:
                if order.side == "buy" and current_price > order.price:
                    return False
                elif order.side == "sell" and current_price < order.price:
                    return False

            elif order.order_type == OrderType.STOP_LIMIT:
                if order.side == "buy" and current_price < order.stop_price:
                    return False
                elif order.side == "sell" and current_price > order.stop_price:
                    return False

            elif order.order_type == OrderType.STOP_MARKET:
                if order.side == "buy" and current_price < order.stop_price:
                    return False
                elif order.side == "sell" and current_price > order.stop_price:
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking order execution: {str(e)}")
            return False

    def _calculate_execution_price(
        self, order: Order, market_data: pd.DataFrame, orderbook: OrderBook
    ) -> float:
        """Calculate execution price based on order type and market conditions."""
        try:
            current_price = market_data["close"].iloc[-1]
            spread = self._calculate_spread(market_data)

            if order.order_type == OrderType.MARKET:
                # Market order: execute at current price plus/minus half spread
                if order.side == "buy":
                    return current_price + (spread / 2)
                else:
                    return current_price - (spread / 2)

            elif order.order_type == OrderType.LIMIT:
                # Limit order: execute at limit price
                return order.price

            elif order.order_type == OrderType.STOP_LIMIT:
                # Stop limit: execute at limit price when stop is triggered
                return order.price

            elif order.order_type == OrderType.STOP_MARKET:
                # Stop market: execute at current price when stop is triggered
                if order.side == "buy":
                    return current_price + (spread / 2)
                else:
                    return current_price - (spread / 2)

            return current_price

        except Exception as e:
            self.logger.error(f"Error calculating execution price: {str(e)}")
            return market_data["close"].iloc[-1] if not market_data.empty else 0.0

    def _calculate_spread(self, market_data: pd.DataFrame) -> float:
        """Calculate current bid-ask spread."""
        try:
            if "high" in market_data.columns and "low" in market_data.columns:
                # Use high-low range as proxy for spread
                recent_high_low = market_data[["high", "low"]].tail(20)
                avg_range = (recent_high_low["high"] - recent_high_low["low"]).mean()
                spread = avg_range / market_data["close"].iloc[-1]
            else:
                # Use volatility-based spread
                volatility = market_data["close"].pct_change().std()
                spread = self.base_spread + (volatility * self.volatility_spread_factor)

            return max(self.base_spread, spread)

        except Exception as e:
            self.logger.error(f"Error calculating spread: {str(e)}")
            return self.base_spread

    def _calculate_slippage(self, order: Order, market_data: pd.DataFrame, orderbook: OrderBook) -> float:
        """Calculate slippage based on order size and market conditions."""
        try:
            # Base slippage
            slippage = self.base_slippage

            # Volume impact
            if "volume" in market_data.columns:
                avg_volume = market_data["volume"].tail(20).mean()
                volume_ratio = order.quantity / avg_volume
                volume_impact = volume_ratio * self.volume_impact_factor
                slippage += volume_impact

            # Volatility impact
            volatility = market_data["close"].pct_change().std()
            volatility_impact = volatility * self.volatility_impact_factor
            slippage += volatility_impact

            # Order type impact
            if order.order_type == OrderType.MARKET:
                slippage *= 1.5  # Market orders have higher slippage

            return slippage

        except Exception as e:
            self.logger.error(f"Error calculating slippage: {str(e)}")
            return self.base_slippage

    def _calculate_commission(self, order: Order, execution_price: float) -> float:
        """Calculate commission and fees."""
        try:
            trade_value = order.quantity * execution_price

            # Base commission
            commission = self.base_commission

            # Percentage commission
            commission += trade_value * self.commission_rate

            # Apply min/max limits
            commission = max(self.min_commission, min(self.max_commission, commission))

            return commission

        except Exception as e:
            self.logger.error(f"Error calculating commission: {str(e)}")
            return self.base_commission

    def _calculate_market_impact(
        self, order: Order, market_data: pd.DataFrame, orderbook: OrderBook
    ) -> float:
        """Calculate market impact of the order."""
        try:
            if "volume" in market_data.columns:
                avg_volume = market_data["volume"].tail(20).mean()
                volume_ratio = order.quantity / avg_volume

                # Market impact increases with order size relative to average volume
                impact = volume_ratio * 0.001  # 0.1% per 100% of average volume
                return impact
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"Error calculating market impact: {str(e)}")
            return 0.0

    def _calculate_execution_delay(self, order: Order) -> float:
        """Calculate execution delay."""
        try:
            # Base delay
            delay = random.uniform(self.min_execution_delay, self.max_execution_delay)

            # Order type impact
            if order.order_type == OrderType.MARKET:
                delay *= 0.5  # Market orders execute faster
            elif order.order_type == OrderType.LIMIT:
                delay *= 1.2  # Limit orders may take longer

            # Size impact
            if order.quantity > 1000:
                delay *= 1.5  # Large orders take longer

            return delay

        except Exception as e:
            self.logger.error(f"Error calculating execution delay: {str(e)}")
            return self.min_execution_delay

    def _calculate_total_cost(
        self, order: Order, execution_price: float, commission: float, slippage: float
    ) -> float:
        """Calculate total cost including slippage and commission."""
        try:
            base_cost = order.quantity * execution_price
            slippage_cost = order.quantity * execution_price * slippage
            total_cost = base_cost + slippage_cost + commission

            return total_cost

        except Exception as e:
            self.logger.error(f"Error calculating total cost: {str(e)}")
            return order.quantity * execution_price

    def _create_rejected_result(self, order: Order, reason: str) -> ExecutionResult:
        """Create rejected execution result."""
        return ExecutionResult(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=0.0,
            timestamp=order.timestamp,
            execution_time=0.0,
            slippage=0.0,
            commission=0.0,
            market_impact=0.0,
            total_cost=0.0,
            success=False,
            failure_reason=reason,
        )

    def _create_failed_result(self, order: Optional[Order], reason: str) -> ExecutionResult:
        """Create failed execution result (Batch 9 enhancement)."""
        if order is None:
            return ExecutionResult(
                order_id="unknown",
                symbol="unknown",
                side="unknown",
                quantity=0.0,
                price=0.0,
                timestamp=datetime.now(),
                execution_time=0.0,
                slippage=0.0,
                commission=0.0,
                market_impact=0.0,
                total_cost=0.0,
                success=False,
                fill_type=FillType.FAILED,
                failure_reason=reason
            )
        
        return self._create_rejected_result(order, reason)

    def _check_market_constraints(self, symbol: str, quantity: float, price: Optional[float]) -> bool:
        """Check market constraints (Batch 9 enhancement)."""
        # Check order size limits
        order_value = quantity * (price or 100)  # Use $100 as default price
        
        if order_value < self.min_order_size:
            self.logger.warning(f"Order value {order_value} below minimum {self.min_order_size}")
            return False
        
        if order_value > self.max_order_size:
            self.logger.warning(f"Order value {order_value} above maximum {self.max_order_size}")
            return False
        
        # Check daily volume limits
        daily_volume = self.daily_volume.get(symbol, 0)
        if daily_volume + order_value > self.max_daily_volume:
            self.logger.warning(f"Daily volume limit exceeded for {symbol}")
            return False
        
        return True

    def _generate_orderbook(self, symbol: str, market_data: pd.DataFrame) -> OrderBook:
        """Generate realistic orderbook using historical bid/ask spread (Batch 9 enhancement)."""
        if market_data.empty:
            # Generate default orderbook
            mid_price = 100.0
            spread = 0.01
        else:
            # Use market data to generate realistic orderbook
            mid_price = market_data.get('close', pd.Series([100.0])).iloc[-1]
            
            # Calculate historical spread
            if 'high' in market_data.columns and 'low' in market_data.columns:
                historical_spread = (market_data['high'] - market_data['low']) / mid_price
                spread = historical_spread.mean() * self.spread_multiplier
            else:
                spread = 0.002  # Default 0.2% spread
        
        # Generate bid and ask levels
        bids = []
        asks = []
        
        for i in range(self.orderbook_levels):
            # Bid levels (below mid price)
            bid_price = mid_price * (1 - spread/2 - i * spread/self.orderbook_levels)
            bid_quantity = 1000 * (self.volume_decay_factor ** i)  # Decreasing volume
            bids.append(OrderBookLevel(
                price=bid_price,
                quantity=bid_quantity,
                side='bid',
                timestamp=datetime.now()
            ))
            
            # Ask levels (above mid price)
            ask_price = mid_price * (1 + spread/2 + i * spread/self.orderbook_levels)
            ask_quantity = 1000 * (self.volume_decay_factor ** i)  # Decreasing volume
            asks.append(OrderBookLevel(
                price=ask_price,
                quantity=ask_quantity,
                side='ask',
                timestamp=datetime.now()
            ))
        
        return OrderBook(
            symbol=symbol,
            timestamp=datetime.now(),
            bids=bids,
            asks=asks,
            spread=spread,
            mid_price=mid_price
        )

    def _calculate_fill_probability(
        self, 
        order: Order, 
        market_data: pd.DataFrame,
        orderbook: OrderBook
    ) -> float:
        """Calculate probability of successful fill (Batch 9 enhancement)."""
        base_probability = self.base_fill_rate
        
        # Adjust for volume
        if order.quantity > 0:
            volume_ratio = order.quantity / sum(level.quantity for level in orderbook.bids[:3])
            volume_adjustment = max(0, 1 - volume_ratio * self.volume_fill_factor)
            base_probability *= volume_adjustment
        
        # Adjust for volatility
        volatility = self._get_volatility(order.symbol, market_data)
        volatility_adjustment = max(0, 1 - volatility * self.volatility_fill_factor)
        base_probability *= volatility_adjustment
        
        # Adjust for order type
        if order.order_type == OrderType.MARKET:
            base_probability *= 0.95  # Market orders have slightly lower fill rate
        
        return max(0.1, min(0.99, base_probability))

    def _get_volatility(self, symbol: str, market_data: pd.DataFrame) -> float:
        """Get volatility for a symbol (Batch 9 enhancement)."""
        if symbol in self.volatility_cache:
            return self.volatility_cache[symbol]
        
        if market_data.empty:
            volatility = 0.02  # Default 2% volatility
        else:
            # Calculate historical volatility
            returns = market_data.get('close', pd.Series([100.0])).pct_change().dropna()
            volatility = returns.std()
        
        self.volatility_cache[symbol] = volatility
        return volatility

    def _update_daily_volume(self, symbol: str, volume: float) -> None:
        """Update daily volume tracking (Batch 9 enhancement)."""
        today = datetime.now().date().isoformat()
        key = f"{symbol}_{today}"
        self.daily_volume[key] = self.daily_volume.get(key, 0) + volume

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        try:
            if order_id not in self.orders:
                return False

            order = self.orders[order_id]
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                self.logger.info(f"Cancelled order: {order_id}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error cancelling order: {str(e)}")
            return False

    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status."""
        try:
            return self.orders.get(order_id)
        except Exception as e:
            self.logger.error(f"Error getting order status: {str(e)}")

    def get_execution_summary(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get execution summary statistics."""
        try:
            if symbol:
                executions = [e for e in self.execution_history if e.symbol == symbol]
            else:
                executions = self.execution_history

            if not executions:
                return {}

            successful_executions = [e for e in executions if e.success]

            if not successful_executions:
                return {
                    "total_orders": len(executions),
                    "success_rate": 0.0,
                    "avg_execution_time": 0.0,
                    "avg_slippage": 0.0,
                    "avg_commission": 0.0,
                }

            return {
                "total_orders": len(executions),
                "successful_orders": len(successful_executions),
                "success_rate": len(successful_executions) / len(executions),
                "avg_execution_time": np.mean(
                    [e.execution_time for e in successful_executions]
                ),
                "avg_slippage": np.mean([e.slippage for e in successful_executions]),
                "avg_commission": np.mean(
                    [e.commission for e in successful_executions]
                ),
                "avg_market_impact": np.mean(
                    [e.market_impact for e in successful_executions]
                ),
                "total_commission": sum([e.commission for e in successful_executions]),
                "total_slippage": sum(
                    [e.slippage * e.quantity * e.price for e in successful_executions]
                ),
            }

        except Exception as e:
            self.logger.error(f"Error getting execution summary: {str(e)}")
            return {}

    def simulate_bulk_execution(
        self, orders: List[Dict[str, Any]], market_data: Dict[str, pd.DataFrame]
    ) -> List[ExecutionResult]:
        """Simulate execution of multiple orders."""
        try:
            results = []

            for order_data in orders:
                # Place order
                order_id = self.place_order(
                    symbol=order_data["symbol"],
                    order_type=OrderType(order_data["order_type"]),
                    side=order_data["side"],
                    quantity=order_data["quantity"],
                    price=order_data.get("price"),
                    stop_price=order_data.get("stop_price"),
                )

                # Execute order
                if order_data["symbol"] in market_data:
                    result = self.execute_order(
                        order_id, market_data[order_data["symbol"]]
                    )
                    results.append(result)
                else:
                    result = self._create_rejected_result(
                        self.orders[order_id], "No market data available"
                    )
                    results.append(result)

            return results

        except Exception as e:
            self.logger.error(f"Error simulating bulk execution: {str(e)}")
            return []

    def get_pending_orders(self) -> List[Order]:
        """Get all pending orders."""
        try:
            return [
                order
                for order in self.orders.values()
                if order.status == OrderStatus.PENDING
            ]
        except Exception as e:
            self.logger.error(f"Error getting pending orders: {str(e)}")
            return []

    def clear_execution_history(self):
        """Clear execution history."""
        try:
            self.execution_history.clear()
            self.logger.info("Execution history cleared")
        except Exception as e:
            self.logger.error(f"Error clearing execution history: {str(e)}")
