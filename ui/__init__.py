"""UI Module for Evolve Trading Platform.

This module contains user interface components.
"""

# Import from trading.ui where the actual implementations exist
try:
    from trading.ui.components import (
        create_benchmark_overlay,
        create_confidence_interval,
        create_forecast_chart,
        create_forecast_metrics,
        create_forecast_table,
        create_model_selector,
        create_parameter_inputs,
        create_performance_report,
        create_prompt_input,
        create_sidebar,
        create_strategy_chart,
        create_system_metrics_panel,
    )
    from trading.ui.forecast_components import (
        create_forecast_form,
        create_forecast_export,
        create_forecast_explanation,
    )
    from trading.ui.strategy_components import (
        create_strategy_form,
        create_performance_chart,
        create_performance_metrics,
        create_trade_list,
        create_strategy_export,
    )
    
    # Create aliases for missing functions to maintain compatibility
    create_forecast_display = create_forecast_chart
    create_model_selection = create_model_selector
    create_parameter_tuning = create_parameter_inputs
    create_performance_metrics = create_performance_report
    create_export_options = create_forecast_export
    create_forecast_interface = create_forecast_form
    create_model_config = create_model_selector
    
    # Import chatbox agent if available
    try:
        from trading.ui.chatbox_agent import (
            ChatboxAgent,
            ChatMessage,
            CommandParser,
            SpeechRecognizer,
            TextToSpeech,
            TradingCommand,
            VoiceInput,
            create_chatbox_agent,
        )
    except ImportError:
        # Use local chatbox_agent if trading.ui.chatbox_agent doesn't exist
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
            
except ImportError as e:
    # Fallback to local implementations if trading.ui is not available
    print(f"Warning: trading.ui not available: {e}")
    
    # Create placeholder functions for all required exports
    def create_benchmark_overlay(*a, **kw):
        return {"status": "placeholder", "message": "Benchmark overlay not available"}
    
    def create_confidence_interval(*a, **kw):
        return {"status": "placeholder", "message": "Confidence interval not available"}
    
    def create_forecast_chart(*a, **kw):
        return {"status": "placeholder", "message": "Forecast chart not available"}
    
    def create_forecast_metrics(*a, **kw):
        return {"status": "placeholder", "message": "Forecast metrics not available"}
    
    def create_forecast_table(*a, **kw):
        return {"status": "placeholder", "message": "Forecast table not available"}
    
    def create_model_selector(*a, **kw):
        return {"status": "placeholder", "message": "Model selector not available"}
    
    def create_parameter_inputs(*a, **kw):
        return {"status": "placeholder", "message": "Parameter inputs not available"}
    
    def create_performance_report(*a, **kw):
        return {"status": "placeholder", "message": "Performance report not available"}
    
    def create_prompt_input(*a, **kw):
        return {"status": "placeholder", "message": "Prompt input not available"}
    
    def create_sidebar(*a, **kw):
        return {"status": "placeholder", "message": "Sidebar not available"}
    
    def create_strategy_chart(*a, **kw):
        return {"status": "placeholder", "message": "Strategy chart not available"}
    
    def create_system_metrics_panel(*a, **kw):
        return {"status": "placeholder", "message": "System metrics not available"}
    
    def create_forecast_form(*a, **kw):
        return {"status": "placeholder", "message": "Forecast form not available"}
    
    def create_forecast_export(*a, **kw):
        return {"status": "placeholder", "message": "Forecast export not available"}
    
    def create_forecast_explanation(*a, **kw):
        return {"status": "placeholder", "message": "Forecast explanation not available"}
    
    def create_strategy_form(*a, **kw):
        return {"status": "placeholder", "message": "Strategy form not available"}
    
    def create_performance_chart(*a, **kw):
        return {"status": "placeholder", "message": "Performance chart not available"}
    
    def create_performance_metrics(*a, **kw):
        return {"status": "placeholder", "message": "Performance metrics not available"}
    
    def create_trade_list(*a, **kw):
        return {"status": "placeholder", "message": "Trade list not available"}
    
    def create_strategy_export(*a, **kw):
        return {"status": "placeholder", "message": "Strategy export not available"}
    
    # Create aliases for missing functions
    create_forecast_display = create_forecast_chart
    create_model_selection = create_model_selector
    create_parameter_tuning = create_parameter_inputs
    create_export_options = create_forecast_export
    create_forecast_interface = create_forecast_form
    create_model_config = create_model_selector
    
    # Import chatbox agent if available
    try:
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
    except ImportError:
        # Create placeholder classes if chatbox_agent is not available
        class ChatboxAgent:
            pass
        class ChatMessage:
            pass
        class CommandParser:
            pass
        class SpeechRecognizer:
            pass
        class TextToSpeech:
            pass
        class TradingCommand:
            pass
        class VoiceInput:
            pass
        def create_chatbox_agent():
            return None

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
    "create_forecast_table",
    "create_forecast_explanation",
    # Strategy components
    "create_strategy_form",
    "create_performance_chart",
    "create_trade_list",
    "create_strategy_export",
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
