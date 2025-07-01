"""
Core agents module.
Contains cognitive AI agents for market analysis and trading decisions.
"""

from trading.agents.base_agent_interface import BaseAgent
from trading.trading import TradingAgent

__all__ = ['BaseAgent', 'TradingAgent'] 