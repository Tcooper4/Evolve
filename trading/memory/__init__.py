"""Trading Memory Module

This module provides memory and logging capabilities for the trading system.
"""

import sys
import logging

# Suppress root INFO when not in Streamlit (e.g. verify scripts, CLI)
if "streamlit" not in sys.modules:
    logging.getLogger("root").setLevel(logging.WARNING)

from .agent_logger import (
    AgentAction,
    AgentLogEntry,
    AgentLogger,
    LogLevel,
    get_agent_logger,
    log_agent_action,
)
from .memory_store import MemoryStore, MemoryType, close_memory_store, get_memory_store

__all__ = [
    "AgentLogger",
    "AgentLogEntry",
    "AgentAction",
    "LogLevel",
    "get_agent_logger",
    "log_agent_action",
    "MemoryStore",
    "MemoryType",
    "get_memory_store",
    "close_memory_store",
]
