"""Agents Module for Evolve Trading Platform.

This module contains various autonomous trading agents and a centralized registry.
"""

from .model_generator_agent import (
    ArxivResearchFetcher,
    AutoEvolutionaryModelGenerator,
    BenchmarkResult,
    ModelBenchmarker,
    ModelCandidate,
)
from .model_generator_agent import ModelImplementationGenerator as MIGenerator
from .model_generator_agent import (
    ResearchPaper,
    run_model_evolution,
)
from .model_innovation_agent import (
    InnovationConfig,
)
from .model_innovation_agent import ModelCandidate as InnovationModelCandidate
from .model_innovation_agent import (
    ModelEvaluation,
    ModelInnovationAgent,
    create_model_innovation_agent,
)
from .prompt_agent import PromptAgent, create_prompt_agent
try:
    from .agent_controller import AgentController, get_agent_controller
except ImportError:
    AgentController = None
    get_agent_controller = None
from .registry import (
    ALL_AGENTS,
    AgentRegistry,
    get_agent,
    get_model_builder_agent,
    get_performance_checker_agent,
    get_prompt_router_agent,
    get_registry,
    get_voice_prompt_agent,
    list_agents,
    search_agents,
)
from .strategy_research_agent import StrategyResearchAgent

# Import agent controller if available
try:
    from .agent_controller import AgentController
    _AGENT_CONTROLLER_AVAILABLE = True
except ImportError:
    _AGENT_CONTROLLER_AVAILABLE = False
    AgentController = None

# Import task router if available
try:
    from .task_router import TaskRouter
    _TASK_ROUTER_AVAILABLE = True
except ImportError:
    _TASK_ROUTER_AVAILABLE = False
    TaskRouter = None

__all__ = [
    # Legacy agents
    "AutoEvolutionaryModelGenerator",
    "ArxivResearchFetcher",
    "MIGenerator",  # Alias for ModelImplementationGenerator
    "ModelBenchmarker",
    "ResearchPaper",
    "ModelCandidate",
    "BenchmarkResult",
    "run_model_evolution",
    # Model Innovation Agent
    "ModelInnovationAgent",
    "InnovationConfig",
    "InnovationModelCandidate",
    "ModelEvaluation",
    "create_model_innovation_agent",
    # Prompt Agent
    "PromptAgent",
    "create_prompt_agent",
    # Strategy Research Agent
    "StrategyResearchAgent",
    # New registry system
    "AgentRegistry",
    "get_registry",
    "get_agent",
    "list_agents",
    "search_agents",
    "ALL_AGENTS",
    "get_prompt_router_agent",
    "get_model_builder_agent",
    "get_performance_checker_agent",
    "get_voice_prompt_agent",
]

# Add agent controller exports if available
if _AGENT_CONTROLLER_AVAILABLE:
    __all__.append("AgentController")

if _TASK_ROUTER_AVAILABLE:
    __all__.append("TaskRouter")
