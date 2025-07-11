"""Agents Module for Evolve Trading Platform.

This module contains various autonomous trading agents and a centralized registry.
"""

from .model_generator_agent import (
    AutoEvolutionaryModelGenerator,
    ArxivResearchFetcher,
    ModelImplementationGenerator as MIGenerator,
    ModelBenchmarker,
    ResearchPaper,
    ModelCandidate,
    BenchmarkResult,
    run_model_evolution
)

from .registry import (
    AgentRegistry,
    get_registry,
    get_agent,
    list_agents,
    search_agents,
    ALL_AGENTS,
    get_prompt_router_agent,
    get_model_builder_agent,
    get_performance_checker_agent,
    get_voice_prompt_agent
)

__all__ = [
    # Legacy agents
    'AutoEvolutionaryModelGenerator',
    'ArxivResearchFetcher',
    'MIGenerator',  # Alias for ModelImplementationGenerator
    'ModelBenchmarker',
    'ResearchPaper',
    'ModelCandidate',
    'BenchmarkResult',
    'run_model_evolution',
    
    # New registry system
    'AgentRegistry',
    'get_registry',
    'get_agent',
    'list_agents',
    'search_agents',
    'ALL_AGENTS',
    'get_prompt_router_agent',
    'get_model_builder_agent',
    'get_performance_checker_agent',
    'get_voice_prompt_agent'
] 