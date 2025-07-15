"""
LLM module for trading system.
"""

# Do not import LLMInterface at the top level to avoid circular import

def get_llm_interface():
    from agents.llm.llm_interface import LLMInterface
    return LLMInterface

__all__ = ["get_llm_interface"] 