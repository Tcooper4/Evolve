"""Agents Module for Evolve Trading Platform.

This module contains various autonomous trading agents.
"""

from .model_generator_agent import (
    AutoEvolutionaryModelGenerator,
    ArxivResearchFetcher,
    ModelImplementationGenerator,
    ModelBenchmarker,
    ResearchPaper,
    ModelCandidate,
    BenchmarkResult,
    run_model_evolution
)

__all__ = [
    'AutoEvolutionaryModelGenerator',
    'ArxivResearchFetcher',
    'ModelImplementationGenerator', 
    'ModelBenchmarker',
    'ResearchPaper',
    'ModelCandidate',
    'BenchmarkResult',
    'run_model_evolution'
] 