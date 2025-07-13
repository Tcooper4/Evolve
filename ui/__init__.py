"""UI Module for Evolve Trading Platform.

This module contains user interface components.
"""

from .chatbox_agent import (
    ChatboxAgent,
    ChatMessage,
    CommandParser,
    SpeechRecognizer,
    TextToSpeech,
    TradingCommand,
    VoiceInput,
    create_chatbox_agent,
)

# Import specific components instead of wildcard imports
from .components import (
    create_benchmark_overlay,
    create_confidence_interval,
    create_export_options,
    create_forecast_display,
    create_forecast_form,
    create_model_selection,
    create_parameter_tuning,
    create_performance_metrics,
)
from .forecast_components import (
    create_forecast_chart,
    create_forecast_export,
    create_forecast_interface,
    create_forecast_metrics,
    create_model_config,
)

__all__ = [
    # Components
    "create_forecast_form",
    "create_model_selection",
    "create_parameter_tuning",
    "create_confidence_interval",
    "create_benchmark_overlay",
    "create_forecast_display",
    "create_performance_metrics",
    "create_export_options",
    # Forecast components
    "create_forecast_interface",
    "create_model_config",
    "create_forecast_chart",
    "create_forecast_metrics",
    "create_forecast_export",
    # Chatbox agent
    "ChatboxAgent",
    "SpeechRecognizer",
    "TextToSpeech",
    "CommandParser",
    "ChatMessage",
    "TradingCommand",
    "VoiceInput",
    "create_chatbox_agent",
]
