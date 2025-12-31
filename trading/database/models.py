"""
SQLAlchemy Database Models

Database models for the Evolve trading system, replacing JSON/Pickle persistence.
"""

import json
import logging
from datetime import datetime
from enum import Enum as PyEnum
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.types import TypeDecorator, VARCHAR

logger = logging.getLogger(__name__)

Base = declarative_base()


class JSONEncodedDict(TypeDecorator):
    """JSON-encoded dictionary type for SQLAlchemy."""
    
    impl = VARCHAR
    cache_ok = True
    
    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return None
    
    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)
        return None


class PositionStatus(PyEnum):
    """Position status enum."""
    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"


class TradeDirection(PyEnum):
    """Trade direction enum."""
    LONG = "long"
    SHORT = "short"


class PositionModel(Base):
    """SQLAlchemy model for trading positions."""
    
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolio_states.id"), nullable=False, index=True)
    
    # Position details
    symbol = Column(String(50), nullable=False, index=True)
    direction = Column(String(10), nullable=False)  # "long" or "short"
    entry_price = Column(Float, nullable=False)
    entry_time = Column(DateTime, nullable=False, index=True)
    size = Column(Float, nullable=False)
    strategy = Column(String(100), nullable=False, index=True)
    
    # Optional fields
    take_profit = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    max_holding_period_seconds = Column(Float, nullable=True)
    exit_price = Column(Float, nullable=True)
    exit_time = Column(DateTime, nullable=True, index=True)
    pnl = Column(Float, nullable=True)
    status = Column(String(20), nullable=False, default="open", index=True)
    
    # JSON fields for complex data
    rationale = Column(JSONEncodedDict, nullable=True)
    unrealized_pnl = Column(Float, nullable=True)
    risk_metrics = Column(JSONEncodedDict, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationship
    portfolio_state = relationship("PortfolioStateModel", back_populates="positions")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "size": self.size,
            "strategy": self.strategy,
            "take_profit": self.take_profit,
            "stop_loss": self.stop_loss,
            "max_holding_period": self.max_holding_period_seconds,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "pnl": self.pnl,
            "status": self.status,
            "rationale": self.rationale,
            "unrealized_pnl": self.unrealized_pnl,
            "risk_metrics": self.risk_metrics,
        }


class PortfolioStateModel(Base):
    """SQLAlchemy model for portfolio state."""
    
    __tablename__ = "portfolio_states"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_name = Column(String(100), nullable=False, index=True, default="default")
    
    # Portfolio metrics
    timestamp = Column(DateTime, nullable=False, index=True)
    cash = Column(Float, nullable=False)
    equity = Column(Float, nullable=False)
    leverage = Column(Float, nullable=False, default=1.0)
    available_capital = Column(Float, nullable=False)
    total_pnl = Column(Float, nullable=False, default=0.0)
    unrealized_pnl = Column(Float, nullable=False, default=0.0)
    
    # JSON fields for complex data
    metrics = Column(JSONEncodedDict, nullable=True)
    risk_metrics = Column(JSONEncodedDict, nullable=True)
    market_regime = Column(String(50), nullable=True)
    strategy_weights = Column(JSONEncodedDict, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    positions = relationship("PositionModel", back_populates="portfolio_state", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "cash": self.cash,
            "equity": self.equity,
            "leverage": self.leverage,
            "available_capital": self.available_capital,
            "total_pnl": self.total_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "open_positions": [p.to_dict() for p in self.positions if p.status == "open"],
            "closed_positions": [p.to_dict() for p in self.positions if p.status == "closed"],
            "metrics": self.metrics or {},
            "risk_metrics": self.risk_metrics or {},
            "market_regime": self.market_regime,
            "strategy_weights": self.strategy_weights or {},
        }


class TradingSessionModel(Base):
    """SQLAlchemy model for trading sessions/context."""
    
    __tablename__ = "trading_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    user_id = Column(String(100), nullable=True, index=True)
    
    # Session details
    status = Column(String(50), nullable=False, default="active", index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    last_activity = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    max_age_seconds = Column(Integer, nullable=True)
    
    # JSON fields
    strategies = Column(JSONEncodedDict, nullable=True)
    context_data = Column(JSONEncodedDict, nullable=True)
    metadata = Column(JSONEncodedDict, nullable=True)
    
    # Timestamps
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class StateManagerModel(Base):
    """SQLAlchemy model for state manager data."""
    
    __tablename__ = "state_manager"
    
    id = Column(Integer, primary_key=True, index=True)
    state_key = Column(String(255), unique=True, nullable=False, index=True)
    
    # State data
    state_value = Column(JSONEncodedDict, nullable=True)
    
    # Version info
    version = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False, index=True)
    
    # Metadata
    metadata = Column(JSONEncodedDict, nullable=True)


class AgentMemoryModel(Base):
    """SQLAlchemy model for agent memory."""
    
    __tablename__ = "agent_memory"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String(100), nullable=False, index=True)
    memory_key = Column(String(255), nullable=False, index=True)
    
    # Memory data
    memory_value = Column(JSONEncodedDict, nullable=True)
    memory_type = Column(String(50), nullable=True)  # "interaction", "performance", "upgrade", etc.
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Metadata
    metadata = Column(JSONEncodedDict, nullable=True)
    
    # Unique constraint
    __table_args__ = (
        {"sqlite_autoincrement": True},
    )


class TaskModel(Base):
    """SQLAlchemy model for agent tasks."""
    
    __tablename__ = "tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(100), unique=True, nullable=False, index=True)
    
    # Task details
    task_type = Column(String(50), nullable=False, index=True)
    status = Column(String(50), nullable=False, default="pending", index=True)
    priority = Column(Integer, nullable=False, default=0, index=True)
    
    # Task data
    task_data = Column(JSONEncodedDict, nullable=True)
    result = Column(JSONEncodedDict, nullable=True)
    error = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True, index=True)
    
    # Metadata
    metadata = Column(JSONEncodedDict, nullable=True)

