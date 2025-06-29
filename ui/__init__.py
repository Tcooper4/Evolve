"""UI Module for Evolve Trading Platform.

This module contains user interface components.
"""

from .components import *
from .forecast_components import *
from .chatbox_agent import (
    ChatboxAgent,
    SpeechRecognizer,
    TextToSpeech,
    CommandParser,
    ChatMessage,
    TradingCommand,
    VoiceInput,
    create_chatbox_agent
)

__all__ = [
    'ChatboxAgent',
    'SpeechRecognizer',
    'TextToSpeech',
    'CommandParser',
    'ChatMessage',
    'TradingCommand',
    'VoiceInput',
    'create_chatbox_agent'
] 