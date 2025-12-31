"""
LLM Providers Module

This module contains implementations for various LLM providers:
- Anthropic (Claude)
- OpenAI
- Local/Ollama models
"""

from typing import Optional

# Import providers if available
__all__ = []

try:
    from .anthropic_provider import AnthropicProvider
    __all__.append('AnthropicProvider')
except ImportError:
    pass

try:
    from .local_provider import LocalLLMProvider
    __all__.append('LocalLLMProvider')
except ImportError:
    pass

