"""agents.llm package.

ROBUSTNESS FIX: this __init__ previously imported LLMInterface eagerly,
which pulls model_loader -> transformers (the local-model ML stack). That
meant importing ANY submodule - including active_llm_calls, which only
makes Claude/OpenAI API calls and needs none of it - failed outright when
transformers wasn't installed, and the Chat page silently lost its
tool-execution path. LLMInterface is now loaded lazily on first attribute
access (PEP 562), so API-only code paths no longer require the local-model
dependencies.
"""

__all__ = ["LLMInterface"]


def __getattr__(name):
    if name == "LLMInterface":
        from .llm_interface import LLMInterface

        return LLMInterface
    raise AttributeError(f"module 'agents.llm' has no attribute {name!r}")
