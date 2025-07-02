"""
Trade Models for Backtesting

This module contains the data structures and enums used for representing
trades and trade types in the backtesting system.
"""

from datetime import datetime
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

class TradeType(Enum):
    """Types of trades."""
    BUY = "buy"
    SELL = "sell"
    EXIT = "exit"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"

@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: datetime
    asset: str
    quantity: float
    price: float
    type: TradeType
    slippage: float
    transaction_cost: float
    spread: float
    cash_balance: float
    portfolio_value: float
    strategy: str
    position_size: float
    risk_metrics: Dict[str, float]
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    holding_period: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Post-initialization validation."""
        if self.quantity <= 0:
            raise ValueError("Trade quantity must be positive")
        if self.price <= 0:
            raise ValueError("Trade price must be positive")
        if self.slippage < 0:
            raise ValueError("Slippage cannot be negative")
        if self.transaction_cost < 0:
            raise ValueError("Transaction cost cannot be negative")
        if self.spread < 0:
            raise ValueError("Spread cannot be negative")
        if self.cash_balance < 0:
            raise ValueError("Cash balance cannot be negative")
        if self.portfolio_value < 0:
            raise ValueError("Portfolio value cannot be negative")
        if self.position_size < 0:
            raise ValueError("Position size cannot be negative")

    def calculate_total_cost(self) -> float:
        """Calculate total cost including slippage, transaction cost, and spread."""
        base_cost = self.quantity * self.price
        slippage_cost = base_cost * self.slippage
        transaction_cost = base_cost * self.transaction_cost
        spread_cost = base_cost * self.spread
        return base_cost + slippage_cost + transaction_cost + spread_cost

    def calculate_net_pnl(self) -> float:
        """Calculate net PnL including all costs."""
        if self.pnl is None:
            return 0.0
        total_cost = self.calculate_total_cost()
        return self.pnl - total_cost

    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'asset': self.asset,
            'quantity': self.quantity,
            'price': self.price,
            'type': self.type.value,
            'slippage': self.slippage,
            'transaction_cost': self.transaction_cost,
            'spread': self.spread,
            'cash_balance': self.cash_balance,
            'portfolio_value': self.portfolio_value,
            'strategy': self.strategy,
            'position_size': self.position_size,
            'risk_metrics': self.risk_metrics,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'holding_period': self.holding_period,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'metadata': self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trade':
        """Create trade from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            asset=data['asset'],
            quantity=data['quantity'],
            price=data['price'],
            type=TradeType(data['type']),
            slippage=data['slippage'],
            transaction_cost=data['transaction_cost'],
            spread=data['spread'],
            cash_balance=data['cash_balance'],
            portfolio_value=data['portfolio_value'],
            strategy=data['strategy'],
            position_size=data['position_size'],
            risk_metrics=data['risk_metrics'],
            entry_price=data.get('entry_price'),
            exit_price=data.get('exit_price'),
            holding_period=data.get('holding_period'),
            pnl=data.get('pnl'),
            pnl_pct=data.get('pnl_pct'),
            metadata=data.get('metadata')
        ) 