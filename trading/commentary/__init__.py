"""
Commentary Module

This module provides comprehensive LLM-based commentary for trading decisions,
performance analysis, and market insights.
"""

from .commentary_engine import (
    CommentaryEngine,
    CommentaryType,
    CommentaryRequest,
    CommentaryResponse,
    create_commentary_engine
)

__all__ = [
    'CommentaryEngine',
    'CommentaryType', 
    'CommentaryRequest',
    'CommentaryResponse',
    'create_commentary_engine'
] 