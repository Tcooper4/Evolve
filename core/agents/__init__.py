"""
Core agents module.
Contains cognitive AI agents for market analysis and trading decisions.
"""

# Import core agent classes
from .base_agent import BaseAgent
from .goal_planner import GoalPlannerAgent
from .router import RouterAgent
from .self_improving_agent import SelfImprovingAgent

# Import trading-specific agents
try:
    from trading.agents.base_agent_interface import BaseAgent as TradingBaseAgent
    from trading.trading import TradingAgent
    TRADING_AGENTS_AVAILABLE = True
except ImportError:
    TRADING_AGENTS_AVAILABLE = False
    TradingBaseAgent = None
    TradingAgent = None

__all__ = [
    # Core agents
    'BaseAgent',
    'GoalPlannerAgent', 
    'RouterAgent',
    'SelfImprovingAgent',
]

# Add trading agents if available
if TRADING_AGENTS_AVAILABLE:
    __all__.extend(['TradingBaseAgent', 'TradingAgent']) 