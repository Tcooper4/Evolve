"""
LLM Providers Module

This module contains implementations for various LLM providers:
- Anthropic (Claude)
- OpenAI
- Local/HuggingFace models
"""

from typing import Optional

# Import providers if available
try:
    from .anthropic_provider import AnthropicProvider
    __all__ = ['AnthropicProvider']
except ImportError:
    __all__ = []

