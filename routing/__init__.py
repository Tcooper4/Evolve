"""
Routing Module

This module provides routing functionality for the Evolve Trading Platform.
It handles prompt routing, request classification, and result forwarding.
"""

from .prompt_router import PromptRouter, get_prompt_router, route_prompt

__all__ = ["route_prompt", "get_prompt_router", "PromptRouter"]
