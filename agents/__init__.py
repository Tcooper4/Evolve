"""Agents Module for Evolve Trading Platform.

Optional agents (model_generator_agent, model_innovation_agent, prompt_agent,
strategy_research_agent) are archived under _archive/; exports are None.

``agent_controller`` and ``task_router`` are archived under ``_archive/agents/``.
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

# Archived under _archive/agents/
AgentController = None
get_agent_controller = None
TaskRouter = None

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

__all__ = [
    "AutoEvolutionaryModelGenerator",
    "ArxivResearchFetcher",
    "MIGenerator",
    "ModelBenchmarker",
    "ResearchPaper",
    "ModelCandidate",
    "BenchmarkResult",
    "run_model_evolution",
    "ModelInnovationAgent",
    "InnovationConfig",
    "InnovationModelCandidate",
    "ModelEvaluation",
    "create_model_innovation_agent",
    "PromptAgent",
    "create_prompt_agent",
    "StrategyResearchAgent",
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
