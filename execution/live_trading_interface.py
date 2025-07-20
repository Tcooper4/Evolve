"""Live Trading Interface for Evolve Trading Platform.

This module provides live trading capabilities with simulated execution
and optional real broker integration (Alpaca).
"""

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

# Alpaca imports
try:
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest, StopLimitOrderRequest
    from alpaca.trading.enums import OrderSide, OrderType, TimeInForce

    ALPACA_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Alpaca SDK not available: {e}")
    ALPACA_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class OrderRequest:
    """Order request structure."""

    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    order_type: str  # "market", "limit", "stop", "stop_limit"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "day"
    client_order_id: Optional[str] = None
    strategy_id: Optional[str] = None


@dataclass
class OrderStatus:
    """Order status information."""

    order_id: str
    symbol: str
    side: str
    quantity: float
    filled_quantity: float
    order_type: str
    status: str  # "pending", "filled", "cancelled", "rejected"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    filled_price: Optional[float] = None
    commission: float = 0.0
    created_at: datetime = None
    filled_at: Optional[datetime] = None
    client_order_id: Optional[str] = None
    strategy_id: Optional[str] = None


@dataclass
class Position:
    """Position information."""

    symbol: str
    quantity: float
    average_price: float
    market_value: float
    unrealized_pl: float
    realized_pl: float
    cost_basis: float
    last_updated: datetime


@dataclass
class AccountInfo:
    """Account information."""

    account_id: str
    cash: float
    buying_power: float
    equity: float
    portfolio_value: float
    day_trade_count: int
    pattern_day_trader: bool
    last_updated: datetime


class SimulatedExecutionEngine:
    """Simulated execution engine with realistic market conditions."""

    def __init__(
        self,
        initial_cash: float = 100000.0,
        commission_rate: float = 0.001,
        slippage_model: str = "proportional",
        latency_model: str = "normal",
    ):
        """Initialize simulated execution engine.

        Args:
            initial_cash: Initial cash balance
            commission_rate: Commission rate per trade
            slippage_model: Slippage model type
            latency_model: Latency model type
        """
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.slippage_model = slippage_model
        self.latency_model = latency_model

        # Account state
        self.cash = initial_cash
        self.positions = {}  # symbol -> Position
        self.orders = {}  # order_id -> OrderStatus
        self.order_counter = 0

        # Market data cache
        self.market_data = {}
        self.last_prices = {}

        # Performance tracking
        self.trade_history = []
        self.daily_pnl = []

        # Latency simulation
        self.min_latency = 0.001  # 1ms
        self.max_latency = 0.100  # 100ms

        logger.info(
            f"Initialized Simulated Execution Engine with ${initial_cash:,.2f} initial cash"
        )

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self.order_counter += 1
        return (
            f"SIM_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.order_counter:06d}"
        )

    def _simulate_latency(self) -> float:
        """Simulate execution latency."""
        if self.latency_model == "normal":
            # Normal distribution around 50ms
            latency = np.random.normal(0.050, 0.020)
        elif self.latency_model == "exponential":
            # Exponential distribution
            latency = np.random.exponential(0.050)
        else:
            # Uniform distribution
            latency = np.random.uniform(self.min_latency, self.max_latency)

        return max(self.min_latency, min(latency, self.max_latency))

    def _calculate_slippage(self, symbol: str, side: str, quantity: float) -> float:
        """Calculate slippage based on order size and market conditions."""
        if symbol not in self.last_prices:
            return 0.0

        current_price = self.last_prices[symbol]

        if self.slippage_model == "proportional":
            # Proportional to order size
            slippage_rate = 0.0001 * (quantity / 1000)  # 0.01% per 1000 shares
        elif self.slippage_model == "fixed":
            # Fixed slippage
            slippage_rate = 0.0005  # 0.05%
        else:
            # No slippage
            slippage_rate = 0.0

        # Add some randomness
        slippage_rate *= np.random.uniform(0.5, 1.5)

        return current_price * slippage_rate

    def _update_market_data(self, symbol: str, price: float, volume: float = 0):
        """Update market data for a symbol."""
        self.last_prices[symbol] = price

        if symbol not in self.market_data:
            self.market_data[symbol] = []

        self.market_data[symbol].append(
            {"timestamp": datetime.now(), "price": price, "volume": volume}
        )

        # Keep only recent data
        if len(self.market_data[symbol]) > 1000:
            self.market_data[symbol] = self.market_data[symbol][-1000:]

    def _calculate_commission(self, trade_value: float) -> float:
        """Calculate commission for a trade."""
        return trade_value * self.commission_rate

    def _update_position(self, symbol: str, quantity: float, price: float, side: str):
        """Update position after trade execution."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0.0,
                average_price=0.0,
                market_value=0.0,
                unrealized_pl=0.0,
                realized_pl=0.0,
                cost_basis=0.0,
                last_updated=datetime.now(),
            )

        position = self.positions[symbol]

        if side == "buy":
            # Buying
            if position.quantity >= 0:
                # Adding to long position
                total_cost = position.cost_basis + (quantity * price)
                position.quantity += quantity
                position.cost_basis = total_cost
                position.average_price = total_cost / position.quantity
            else:
                # Covering short position
                if abs(position.quantity) >= quantity:
                    # Partial cover
                    realized_pl = (position.average_price - price) * quantity
                    position.realized_pl += realized_pl
                    position.quantity += quantity
                    if position.quantity == 0:
                        position.average_price = 0.0
                        position.cost_basis = 0.0
                else:
                    # Full cover + new long position
                    realized_pl = (position.average_price - price) * abs(
                        position.quantity
                    )
                    position.realized_pl += realized_pl
                    remaining_quantity = quantity + position.quantity
                    position.quantity = remaining_quantity
                    position.average_price = price
                    position.cost_basis = remaining_quantity * price

        else:  # sell
            # Selling
            if position.quantity <= 0:
                # Adding to short position
                total_cost = position.cost_basis + (quantity * price)
                position.quantity -= quantity
                position.cost_basis = total_cost
                position.average_price = total_cost / abs(position.quantity)
            else:
                # Reducing long position
                if position.quantity >= quantity:
                    # Partial sale
                    realized_pl = (price - position.average_price) * quantity
                    position.realized_pl += realized_pl
                    position.quantity -= quantity
                    if position.quantity == 0:
                        position.average_price = 0.0
                        position.cost_basis = 0.0
                else:
                    # Full sale + new short position
                    realized_pl = (price - position.average_price) * position.quantity
                    position.realized_pl += realized_pl
                    remaining_quantity = quantity - position.quantity
                    position.quantity = -remaining_quantity
                    position.average_price = price
                    position.cost_basis = remaining_quantity * price

        position.last_updated = datetime.now()

    def _update_portfolio_value(self):
        """Update portfolio value based on current positions."""
        total_value = self.cash

        for symbol, position in self.positions.items():
            if symbol in self.last_prices:
                current_price = self.last_prices[symbol]
                position.market_value = position.quantity * current_price
                position.unrealized_pl = position.market_value - position.cost_basis
                total_value += position.market_value

        return total_value

    def place_order(self, order_request: OrderRequest) -> OrderStatus:
        """Place an order in simulated environment."""
        # Simulate latency
        time.sleep(self._simulate_latency())

        # Generate order ID
        order_id = self._generate_order_id()

        # Create order status
        order_status = OrderStatus(
            order_id=order_id,
            symbol=order_request.symbol,
            side=order_request.side,
            quantity=order_request.quantity,
            filled_quantity=0.0,
            order_type=order_request.order_type,
            status="pending",
            limit_price=order_request.limit_price,
            stop_price=order_request.stop_price,
            created_at=datetime.now(),
            client_order_id=order_request.client_order_id,
            strategy_id=order_request.strategy_id,
        )

        # Store order
        self.orders[order_id] = order_status

        # Process order based on type
        if order_request.order_type == "market":
            self._process_market_order(order_status)
        elif order_request.order_type == "limit":
            self._process_limit_order(order_status)
        elif order_request.order_type == "stop":
            self._process_stop_order(order_status)

        return order_status

    def _process_market_order(self, order_status: OrderStatus):
        """Process market order."""
        symbol = order_status.symbol

        if symbol not in self.last_prices:
            order_status.status = "rejected"
            return

        current_price = self.last_prices[symbol]

        # Calculate execution price with slippage
        slippage = self._calculate_slippage(
            symbol, order_status.side, order_status.quantity
        )

        if order_status.side == "buy":
            execution_price = current_price + slippage
        else:
            execution_price = current_price - slippage

        # Check if we have enough cash/position
        trade_value = order_status.quantity * execution_price
        commission = self._calculate_commission(trade_value)
        total_cost = trade_value + commission

        if order_status.side == "buy":
            if self.cash < total_cost:
                order_status.status = "rejected"
                return
            self.cash -= total_cost
        else:  # sell
            if (
                symbol not in self.positions
                or self.positions[symbol].quantity < order_status.quantity
            ):
                order_status.status = "rejected"
                return
            self.cash += trade_value - commission

        # Execute the trade
        self._update_position(
            symbol, order_status.quantity, execution_price, order_status.side
        )

        # Update order status
        order_status.status = "filled"
        order_status.filled_quantity = order_status.quantity
        order_status.filled_price = execution_price
        order_status.commission = commission
        order_status.filled_at = datetime.now()

        # Record trade
        self.trade_history.append(
            {
                "order_id": order_status.order_id,
                "symbol": symbol,
                "side": order_status.side,
                "quantity": order_status.quantity,
                "price": execution_price,
                "commission": commission,
                "timestamp": datetime.now(),
                "strategy_id": order_status.strategy_id,
            }
        )

        # Update portfolio value
        self._update_portfolio_value()

    def _process_limit_order(self, order_status: OrderStatus):
        """Process limit order."""
        # For simplicity, assume limit orders are filled immediately if price is favorable
        symbol = order_status.symbol

        if symbol not in self.last_prices:
            order_status.status = "rejected"
            return

        current_price = self.last_prices[symbol]
        limit_price = order_status.limit_price

        if order_status.side == "buy" and current_price <= limit_price:
            # Buy limit order can be filled
            self._process_market_order(order_status)
        elif order_status.side == "sell" and current_price >= limit_price:
            # Sell limit order can be filled
            self._process_market_order(order_status)
        else:
            # Order remains pending
            pass

    def _process_stop_order(self, order_status: OrderStatus):
        """Process stop order."""
        # For simplicity, assume stop orders are filled immediately if triggered
        symbol = order_status.symbol

        if symbol not in self.last_prices:
            order_status.status = "rejected"
            return

        current_price = self.last_prices[symbol]
        stop_price = order_status.stop_price

        if order_status.side == "buy" and current_price >= stop_price:
            # Buy stop order triggered
            self._process_market_order(order_status)
        elif order_status.side == "sell" and current_price <= stop_price:
            # Sell stop order triggered
            self._process_market_order(order_status)
        else:
            # Order remains pending
            pass

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status == "pending":
                order.status = "cancelled"
                return True
        return False

    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status."""
        return self.orders.get(order_id)

    def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        return self.positions.copy()

    def get_account_info(self) -> AccountInfo:
        """Get account information."""
        portfolio_value = self._update_portfolio_value()

        return AccountInfo(
            account_id="SIM_ACCOUNT",
            cash=self.cash,
            buying_power=self.cash,  # Simplified
            equity=portfolio_value,
            portfolio_value=portfolio_value,
            day_trade_count=0,  # Not tracked in simulation
            pattern_day_trader=False,
            last_updated=datetime.now(),
        )

    def update_market_data(self, market_data: Dict[str, float]):
        """Update market data for all symbols."""
        for symbol, price in market_data.items():
            self._update_market_data(symbol, price)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.trade_history:
            return {"total_trades": 0, "total_pnl": 0.0, "win_rate": 0.0}

        total_trades = len(self.trade_history)
        total_pnl = sum(trade.get("pnl", 0.0) for trade in self.trade_history)
        winning_trades = sum(
            1 for trade in self.trade_history if trade.get("pnl", 0.0) > 0
        )
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        return {
            "total_trades": total_trades,
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "initial_cash": self.initial_cash,
            "current_cash": self.cash,
            "portfolio_value": self._update_portfolio_value(),
        }


class AlpacaTradingInterface:
    """Alpaca trading interface for real order placement."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper_trading: bool = True,
    ):
        """Initialize Alpaca interface.

        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper_trading: Use paper trading account
        """
        if not ALPACA_AVAILABLE:
            raise ImportError("Alpaca SDK not available")

        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        self.paper_trading = paper_trading

        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials not provided")

        # Initialize API
        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=paper_trading
        )
        self.data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key
        )

        logger.info(
            f"Initialized Alpaca Trading Interface (Paper Trading: {paper_trading})"
        )

    def place_order(self, order_request: OrderRequest) -> OrderStatus:
        """Place order through Alpaca."""
        try:
            # Convert order side
            side = OrderSide.BUY if order_request.side == "buy" else OrderSide.SELL
            
            # Convert time in force
            time_in_force = TimeInForce.DAY if order_request.time_in_force == "day" else TimeInForce.GTC
            
            # Create appropriate order request based on order type
            if order_request.order_type == "market":
                alpaca_order_request = MarketOrderRequest(
                    symbol=order_request.symbol,
                    qty=order_request.quantity,
                    side=side,
                    time_in_force=time_in_force,
                    client_order_id=order_request.client_order_id
                )
            elif order_request.order_type == "limit":
                alpaca_order_request = LimitOrderRequest(
                    symbol=order_request.symbol,
                    qty=order_request.quantity,
                    side=side,
                    time_in_force=time_in_force,
                    limit_price=order_request.limit_price,
                    client_order_id=order_request.client_order_id
                )
            elif order_request.order_type == "stop":
                alpaca_order_request = StopOrderRequest(
                    symbol=order_request.symbol,
                    qty=order_request.quantity,
                    side=side,
                    time_in_force=time_in_force,
                    stop_price=order_request.stop_price,
                    client_order_id=order_request.client_order_id
                )
            elif order_request.order_type == "stop_limit":
                alpaca_order_request = StopLimitOrderRequest(
                    symbol=order_request.symbol,
                    qty=order_request.quantity,
                    side=side,
                    time_in_force=time_in_force,
                    limit_price=order_request.limit_price,
                    stop_price=order_request.stop_price,
                    client_order_id=order_request.client_order_id
                )
            else:
                raise ValueError(f"Unsupported order type: {order_request.order_type}")

            # Submit order
            alpaca_order = self.trading_client.submit_order(alpaca_order_request)

            # Convert to OrderStatus
            order_status = OrderStatus(
                order_id=alpaca_order.id,
                symbol=alpaca_order.symbol,
                side=alpaca_order.side.value,
                quantity=float(alpaca_order.qty),
                filled_quantity=float(alpaca_order.filled_qty),
                order_type=alpaca_order.order_type.value,
                status=alpaca_order.status.value,
                limit_price=float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
                stop_price=float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
                filled_price=float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None,
                created_at=alpaca_order.created_at,
                filled_at=alpaca_order.filled_at,
                client_order_id=alpaca_order.client_order_id,
                strategy_id=order_request.strategy_id,
            )

            return order_status

        except Exception as e:
            logger.error(f"Error placing Alpaca order: {e}")
            raise

    def cancel_order(self, order_id: str) -> bool:
        """Cancel order through Alpaca."""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            logger.error(f"Error cancelling Alpaca order: {e}")
            return False

    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status from Alpaca."""
        try:
            alpaca_order = self.trading_client.get_order_by_id(order_id)

            order_status = OrderStatus(
                order_id=alpaca_order.id,
                symbol=alpaca_order.symbol,
                side=alpaca_order.side.value,
                quantity=float(alpaca_order.qty),
                filled_quantity=float(alpaca_order.filled_qty),
                order_type=alpaca_order.order_type.value,
                status=alpaca_order.status.value,
                limit_price=float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
                stop_price=float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
                filled_price=float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None,
                created_at=alpaca_order.created_at,
                filled_at=alpaca_order.filled_at,
                client_order_id=alpaca_order.client_order_id,
            )

            return order_status

        except Exception as e:
            logger.error(f"Error getting Alpaca order status: {e}")
            return None

    def get_positions(self) -> Dict[str, Position]:
        """Get positions from Alpaca."""
        try:
            alpaca_positions = self.trading_client.get_all_positions()
            positions = {}

            for pos in alpaca_positions:
                position = Position(
                    symbol=pos.symbol,
                    quantity=float(pos.qty),
                    average_price=float(pos.avg_entry_price),
                    market_value=float(pos.market_value),
                    unrealized_pl=float(pos.unrealized_pl),
                    realized_pl=0.0,  # Not directly available from Alpaca
                    cost_basis=float(pos.qty) * float(pos.avg_entry_price),
                    last_updated=datetime.now(),
                )
                positions[pos.symbol] = position

            return positions

        except Exception as e:
            logger.error(f"Error getting Alpaca positions: {e}")
            return {}

    def get_account_info(self) -> AccountInfo:
        """Get account information from Alpaca."""
        try:
            account = self.trading_client.get_account()

            return AccountInfo(
                account_id=account.id,
                cash=float(account.cash),
                buying_power=float(account.buying_power),
                equity=float(account.equity),
                portfolio_value=float(account.portfolio_value),
                day_trade_count=int(account.daytrade_count),
                pattern_day_trader=account.pattern_day_trader,
                last_updated=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error getting Alpaca account info: {e}")
            raise


class LiveTradingInterface:
    """Main live trading interface that manages both simulated and real trading."""

    def __init__(
        self, mode: str = "simulated", alpaca_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize live trading interface.

        Args:
            mode: Trading mode ("simulated" or "live")
            alpaca_config: Alpaca configuration
        """
        self.mode = mode
        self.alpaca_config = alpaca_config or {}

        # Initialize appropriate interface
        if mode == "live" and ALPACA_AVAILABLE:
            self.trading_interface = AlpacaTradingInterface(
                api_key=self.alpaca_config.get("api_key"),
                secret_key=self.alpaca_config.get("secret_key"),
                paper_trading=self.alpaca_config.get("paper_trading", True),
            )
        else:
            self.trading_interface = SimulatedExecutionEngine(
                initial_cash=self.alpaca_config.get("initial_cash", 100000.0),
                commission_rate=self.alpaca_config.get("commission_rate", 0.001),
                slippage_model=self.alpaca_config.get("slippage_model", "proportional"),
                latency_model=self.alpaca_config.get("latency_model", "normal"),
            )

        # Order management
        self.pending_orders = {}
        self.order_history = []

        # Risk management
        self.max_position_size = self.alpaca_config.get("max_position_size", 0.1)
        self.max_daily_loss = self.alpaca_config.get("max_daily_loss", 0.05)
        self.stop_loss_pct = self.alpaca_config.get("stop_loss_pct", 0.02)

        # Performance tracking
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()

        logger.info(f"Initialized Live Trading Interface in {mode} mode")

    def place_order(self, order_request: OrderRequest) -> OrderStatus:
        """Place an order."""
        try:
            # Risk checks
            if not self._validate_order(order_request):
                raise ValueError("Order failed risk validation")

            # Place order
            order_status = self.trading_interface.place_order(order_request)

            # Track order
            if order_status.status == "pending":
                self.pending_orders[order_status.order_id] = order_status

            self.order_history.append(order_status)

            logger.info(
                f"Placed {order_status.side} order for {order_status.quantity} {order_status.symbol}"
            )

            return order_status

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise

    def _validate_order(self, order_request: OrderRequest) -> bool:
        """Validate order against risk parameters."""
        try:
            # Get account info
            account_info = self.get_account_info()

            # Check position size
            if order_request.side == "buy":
                order_value = order_request.quantity * (
                    order_request.limit_price or 100.0
                )
                if order_value > account_info.buying_power * self.max_position_size:
                    logger.warning(
                        f"Order value {order_value} exceeds max position size"
                    )
                    return False

            # Check daily loss limit
            if self.daily_pnl < -account_info.equity * self.max_daily_loss:
                logger.warning("Daily loss limit exceeded")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating order: {e}")
            return False

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            success = self.trading_interface.cancel_order(order_id)

            if success and order_id in self.pending_orders:
                del self.pending_orders[order_id]

            return success

        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False

    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status."""
        return self.trading_interface.get_order_status(order_id)

    def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        return self.trading_interface.get_positions()

    def get_account_info(self) -> AccountInfo:
        """Get account information."""
        return self.trading_interface.get_account_info()

    def update_market_data(self, market_data: Dict[str, float]):
        """Update market data."""
        if isinstance(self.trading_interface, SimulatedExecutionEngine):
            self.trading_interface.update_market_data(market_data)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if isinstance(self.trading_interface, SimulatedExecutionEngine):
            return self.trading_interface.get_performance_summary()
        else:
            # For real trading, calculate from order history
            return self._calculate_performance_from_history()

    def _calculate_performance_from_history(self) -> Dict[str, Any]:
        """Calculate performance from order history."""
        total_trades = len(self.order_history)
        filled_orders = [
            order for order in self.order_history if order.status == "filled"
        ]

        total_pnl = 0.0
        for order in filled_orders:
            if order.filled_price and order.filled_quantity:
                # Simplified PnL calculation
                if order.side == "buy":
                    total_pnl -= order.filled_price * order.filled_quantity
                else:
                    total_pnl += order.filled_price * order.filled_quantity

        return {
            "total_trades": total_trades,
            "filled_trades": len(filled_orders),
            "total_pnl": total_pnl,
            "win_rate": len([o for o in filled_orders if o.filled_price > 0])
            / len(filled_orders)
            if filled_orders
            else 0.0,
        }

    def reset_daily_tracking(self):
        """Reset daily tracking metrics."""
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = current_date


def create_live_trading_interface(
    mode: str = "simulated", config: Optional[Dict[str, Any]] = None
) -> LiveTradingInterface:
    """Create live trading interface.

    Args:
        mode: Trading mode ("simulated" or "live")
        config: Configuration dictionary

    Returns:
        Live trading interface
    """
    try:
        interface = LiveTradingInterface(mode=mode, alpaca_config=config)
        return interface
    except Exception as e:
        logger.error(f"Error creating live trading interface: {e}")
        raise
