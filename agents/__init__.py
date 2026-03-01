"""Agents Module for Evolve Trading Platform.

This module contains various autonomous trading agents and a centralized registry.
Legacy agents (model_generator_agent, model_innovation_agent, prompt_agent,
strategy_research_agent) may be in _dead_code; imports are optional so the
package and agents.llm load successfully without them.
"""

try:
    from .model_generator_agent import (
        ArxivResearchFetcher,
        AutoEvolutionaryModelGenerator,
        BenchmarkResult,
        ModelBenchmarker,
        ModelCandidate,
        ResearchPaper,
        run_model_evolution,
    )
    from .model_generator_agent import ModelImplementationGenerator as MIGenerator
except ImportError:
    ArxivResearchFetcher = None
    AutoEvolutionaryModelGenerator = None
    BenchmarkResult = None
    ModelBenchmarker = None
    ModelCandidate = None
    MIGenerator = None
    ResearchPaper = None
    run_model_evolution = None

try:
    from .model_innovation_agent import (
        InnovationConfig,
        ModelCandidate as InnovationModelCandidate,
        ModelEvaluation,
        ModelInnovationAgent,
        create_model_innovation_agent,
    )
except ImportError:
    InnovationConfig = None
    InnovationModelCandidate = None
    ModelEvaluation = None
    ModelInnovationAgent = None
    create_model_innovation_agent = None

try:
    from .prompt_agent import PromptAgent, create_prompt_agent
except ImportError:
    PromptAgent = None
    create_prompt_agent = None

try:
    from .agent_controller import AgentController, get_agent_controller
    _AGENT_CONTROLLER_AVAILABLE = True
except ImportError:
    AgentController = None
    get_agent_controller = None
    _AGENT_CONTROLLER_AVAILABLE = False

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

try:
    from .strategy_research_agent import StrategyResearchAgent
except ImportError:
    StrategyResearchAgent = None

# Import task router if available
try:
    from .task_router import TaskRouter
    _TASK_ROUTER_AVAILABLE = True
except ImportError:
    _TASK_ROUTER_AVAILABLE = False
    TaskRouter = None

__all__ = [
    # Legacy agents (may be None if module removed)
    "AutoEvolutionaryModelGenerator",
    "ArxivResearchFetcher",
    "MIGenerator",
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
