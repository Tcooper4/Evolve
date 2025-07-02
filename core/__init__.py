"""
Core module for the trading platform.
Contains fundamental AI and routing logic.
"""

# Import main components
from .agent_hub import AgentHub
from .capability_router import CapabilityRouter, get_system_health
from .sanity_checks import SanityChecker
from .session_utils import (
    initialize_session_state,
    safe_session_get,
    safe_session_set,
    update_last_updated
)

# Import agent components
from .agents.base_agent import BaseAgent
from .agents.goal_planner import GoalPlannerAgent
from .agents.router import RouterAgent
from .agents.self_improving_agent import SelfImprovingAgent

# Import model components
from .models.base_model import BaseModel

# Import utility components
from .utils.common_helpers import (
    safe_json_load,
    safe_json_save,
    validate_config,
    get_project_root
)
from .utils.technical_indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_sma,
    calculate_ema
)

__all__ = [
    # Main components
    'AgentHub',
    'CapabilityRouter',
    'get_system_health',
    'SanityChecker',
    
    # Session utilities
    'initialize_session_state',
    'safe_session_get',
    'safe_session_set',
    'update_last_updated',
    
    # Agent components
    'BaseAgent',
    'GoalPlannerAgent',
    'RouterAgent',
    'SelfImprovingAgent',
    
    # Model components
    'BaseModel',
    
    # Utility functions
    'safe_json_load',
    'safe_json_save',
    'validate_config',
    'get_project_root',
    'calculate_rsi',
    'calculate_macd',
    'calculate_bollinger_bands',
    'calculate_sma',
    'calculate_ema'
] 