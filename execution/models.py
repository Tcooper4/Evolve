"""
Canonical order and execution types for the Evolve execution layer.

P2 fix: Single source of truth for OrderType, OrderStatus, OrderSide, OrderRequest,
and OrderExecution. All execution and broker code should use these types.
See AUDIT_REPORT.md 2.2.
"""

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class OrderType(Enum):
    """Canonical order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    STOP_MARKET = "stop_market"
    TRAILING_STOP = "trailing_stop"
    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"


class OrderStatus(Enum):
    """Canonical order statuses."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(Enum):
    """Canonical order sides."""

    BUY = "buy"
    SELL = "sell"


@dataclass
class OrderRequest:
    """Canonical order request (use .symbol for instrument)."""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "day"
    client_order_id: Optional[str] = None
    timestamp: Optional[str] = None
    strategy_id: Optional[str] = None
    # Advanced order parameters (broker adapter)
    twap_duration_seconds: Optional[int] = None
    twap_slice_count: Optional[int] = None
    vwap_start_time: Optional[str] = None
    vwap_end_time: Optional[str] = None
    iceberg_visible_quantity: Optional[float] = None
    iceberg_reveal_quantity: Optional[float] = None

    @classmethod
    def from_legacy(
        cls,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day",
        client_order_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
    ) -> "OrderRequest":
        """Build OrderRequest from legacy (str side/order_type, limit_price) shape."""
        return cls(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=OrderSide(side) if isinstance(side, str) else side,
            order_type=OrderType(order_type) if isinstance(order_type, str) else order_type,
            quantity=quantity,
            price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
            strategy_id=strategy_id,
        )


@dataclass
class OrderExecution:
    """Canonical order execution result (use .symbol for instrument)."""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float
    executed_quantity: float
    average_price: float
    commission: float
    timestamp: str
    status: OrderStatus
    fills: List[Dict[str, Any]]
    metadata: Dict[str, Any]
