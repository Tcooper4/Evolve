"""Agents Module for Evolve Trading Platform.

This module contains various autonomous trading agents and a centralized registry.
Optional agents (model_generator_agent, model_innovation_agent, prompt_agent,
strategy_research_agent) are archived under _archive/; exports are None.
"""

# Archived — not on disk: agents/model_generator_agent.py
ArxivResearchFetcher = None
AutoEvolutionaryModelGenerator = None
BenchmarkResult = None
ModelBenchmarker = None
ModelCandidate = None
MIGenerator = None
ResearchPaper = None
run_model_evolution = None

# Archived — not on disk: agents/model_innovation_agent.py
InnovationConfig = None
InnovationModelCandidate = None
ModelEvaluation = None
ModelInnovationAgent = None
create_model_innovation_agent = None

# Archived — not on disk: agents/prompt_agent.py
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

# Not in agents/ (only docs/future_features/) — no import
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
