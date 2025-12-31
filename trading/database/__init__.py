"""
Database Backend Module

Provides SQLAlchemy-based database backend for the Evolve trading system.
Replaces JSON/Pickle file-based persistence with proper database storage.
"""

from trading.database.connection import get_db_session, init_database
from trading.database.models import (
    PortfolioStateModel,
    PositionModel,
    TradingSessionModel,
    StateManagerModel,
    AgentMemoryModel,
    TaskModel,
)

__all__ = [
    "get_db_session",
    "init_database",
    "PortfolioStateModel",
    "PositionModel",
    "TradingSessionModel",
    "StateManagerModel",
    "AgentMemoryModel",
    "TaskModel",
]

