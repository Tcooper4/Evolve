"""
LLM Interface Bridge Module

This module provides a bridge to the agents.llm.llm_interface module
for backward compatibility and cleaner import paths.
"""

try:
    from agents.llm.llm_interface import LLMInterface
    __all__ = ["LLMInterface"]
except (ImportError, ModuleNotFoundError) as e:
    # If import fails (e.g. missing 'schedule' or other optional deps), log and degrade
    import logging
    _logger = logging.getLogger(__name__)
    _logger.warning(
        "Could not import LLMInterface from agents.llm.llm_interface: %s. Using stub.",
        e,
    )
    _llm_interface_error = e  # capture for stub

    class LLMInterface:
        """Stub when LLMInterface is unavailable (e.g. missing schedule or other deps)."""

        def __init__(self, *args, **kwargs):
            self._import_error = _llm_interface_error

        def __getattr__(self, name):
            raise ImportError(
                f"LLMInterface not available: {self._import_error}"
            ) from self._import_error

    __all__ = ["LLMInterface"]

