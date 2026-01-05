"""
LLM Interface Bridge Module

This module provides a bridge to the agents.llm.llm_interface module
for backward compatibility and cleaner import paths.
"""

try:
    from agents.llm.llm_interface import LLMInterface
    __all__ = ["LLMInterface"]
except ImportError as e:
    # If import fails, log the error but don't break the import
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import LLMInterface from agents.llm.llm_interface: {e}")
    
    # Create a minimal stub class to prevent import errors
    class LLMInterface:
        """Stub LLMInterface class when import fails."""
        def __init__(self, *args, **kwargs):
            raise ImportError(f"LLMInterface not available: {e}")
    
    __all__ = ["LLMInterface"]

