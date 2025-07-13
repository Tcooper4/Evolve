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
from .model_generator_agent import ResearchPaper, run_model_evolution
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
