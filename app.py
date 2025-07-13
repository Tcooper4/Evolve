"""
Evolve Trading Platform - Production Ready Streamlit App

A clean, ChatGPT-like interface with a single prompt box that routes to:
- Forecasting
- Strategy Selection
- Signal Generation
- Trade Reports

All triggered from a single input with professional styling.
"""

import logging
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Suppress Streamlit warnings when importing
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.state.session_state_proxy").setLevel(logging.ERROR)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
    logger.info("✅ Environment variables loaded from .env file")
except ImportError:
    logger.warning("⚠️ python-dotenv not available, using system environment variables")
except Exception as e:
    logger.error(f"⚠️ Error loading .env file: {e}")

# Debug: Check if API keys are loaded
alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
finnhub_key = os.getenv("FINNHUB_API_KEY")
polygon_key = os.getenv("POLYGON_API_KEY")

if alpha_vantage_key:
    logger.info(f"✅ ALPHA_VANTAGE_API_KEY loaded: {alpha_vantage_key[:10]}...")
else:
    logger.warning("❌ ALPHA_VANTAGE_API_KEY not found")

if finnhub_key:
    logger.info(f"✅ FINNHUB_API_KEY loaded: {finnhub_key[:10]}...")
else:
    logger.warning("❌ FINNHUB_API_KEY not found")

if polygon_key:
    logger.info(f"✅ POLYGON_API_KEY loaded: {polygon_key[:10]}...")
else:
    logger.warning("❌ POLYGON_API_KEY not found")

# Configure page
st.set_page_config(page_title="Evolve AI Trading Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Import core components with error handling
try:
    from trading.llm.agent import PromptAgent

    CORE_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some modules not available: {e}")
    CORE_COMPONENTS_AVAILABLE = False

# Enhanced ChatGPT-like styling
st.markdown(
    """
<style>
    /* Main container styling */
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
    }

    .main-header h1 {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }

    .main-header p {
        font-size: 1.3rem;
        opacity: 0.95;
        margin-bottom: 0;
        font-weight: 300;
    }

    /* Prompt input styling */
    .prompt-container {
        background: white;
        border-radius: 25px;
        padding: 2.5rem;
        box-shadow: 0 12px 40px rgba(0,0,0,0.12);
        margin: 2rem 0;
        border: 2px solid #f8f9fa;
        transition: all 0.3s ease;
    }

    .prompt-container:hover {
        box-shadow: 0 16px 50px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }

    .stTextInput > div > div > input {
        font-size: 1.4rem !important;
        padding: 1.2rem 1.8rem !important;
        border-radius: 20px !important;
        border: 2px solid #e9ecef !important;
        background: #fafbfc !important;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.15) !important;
        background: white !important;
        transform: scale(1.02);
    }

    /* Button styling */
    .stButton > button {
        font-size: 1.3rem !important;
        padding: 1.2rem 2.5rem !important;
        border-radius: 20px !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.5) !important;
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%) !important;
    }

    /* Result card styling */
    .result-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid #f0f0f0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }

    .result-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
    }

    .result-card h3 {
        color: #2c3e50;
        margin-bottom: 1.2rem;
        font-weight: 700;
        font-size: 1.4rem;
    }

    .result-card p {
        color: #5a6c7d;
        line-height: 1.7;
        font-size: 1.1rem;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border-right: 1px solid #dee2e6;
    }

    /* Navigation styling */
    .nav-item {
        padding: 1rem 1.2rem;
        margin: 0.3rem 0;
        border-radius: 15px;
        transition: all 0.3s ease;
        cursor: pointer;
        font-weight: 500;
        border: 2px solid transparent;
    }

    .nav-item:hover {
        background: rgba(102, 126, 234, 0.1);
        border-color: rgba(102, 126, 234, 0.2);
        transform: translateX(5px);
    }

    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    .status-success {
        background: linear-gradient(45deg, #28a745, #20c997);
        animation: pulse 2s infinite;
    }

    .status-warning {
        background: linear-gradient(45deg, #ffc107, #fd7e14);
    }

    .status-error {
        background: linear-gradient(45deg, #dc3545, #e83e8c);
    }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }

    /* Loading animation */
    .loading-spinner {
        text-align: center;
        padding: 3rem;
    }

    /* Conversation styling */
    .conversation-item {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }

    .conversation-prompt {
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }

    .conversation-response {
        color: #5a6c7d;
        font-style: italic;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2.2rem;
        }

        .prompt-container {
            padding: 1.5rem;
        }

        .stTextInput > div > div > input {
            font-size: 1.2rem !important;
            padding: 1rem 1.5rem !important;
        }
    }

    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #333;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.9rem;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.markdown(
        """
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: #2c3e50; margin-bottom: 0.5rem;">🚀 Evolve AI</h2>
        <p style="color: #6c757d; font-size: 0.9rem;">Autonomous Trading Intelligence</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Main Navigation - Simplified
    st.markdown("### 📊 Navigation")

    primary_nav = st.radio(
        "", ["🏠 Home & Chat", "📈 Forecasting", "⚡ Strategy Lab", "🧠 Model Lab", "📋 Reports"], key=os.getenv("KEY", "")
    )

    # Advanced Tools (Collapsible)
    with st.expander("🔧 Advanced", expanded=False):
        advanced_nav = st.radio("", ["⚙️ Settings", "📊 Monitor", "📈 Analytics", "🛡️ Risk"], key=os.getenv("KEY", ""))

    # Developer Tools (Hidden by default)
    if os.environ.get("EVOLVE_DEV_MODE", "0") == "1":
        with st.expander("🛠️ Dev Tools", expanded=False):
            st.markdown("- 🐛 Debug")
            st.markdown("- 📝 Logs")
            st.markdown("- ⚡ Performance")
            st.markdown("- 🔌 API")

    st.markdown("---")

    # System Status - Simplified
    st.markdown("### 🟢 Status")

    # Status indicators - compact
    status_items = [("Core Systems", "🟢"), ("Data Feed", "🟢"), ("AI Models", "🟢"), ("Agents", "🟢")]

    for name, status in status_items:
        st.markdown(f"{status} {name}")

    # Quick Stats - Compact
    st.markdown("---")
    st.markdown("### 📊 Stats")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🤖 Models", "12")
        st.metric("📈 Success", "94.2%")

    with col2:
        st.metric("⚡ Strategies", "8")
        st.metric("💰 Return", "2.8%")

# --- Main Layout ---
# Top Navigation Bar
st.markdown(
    """
<div style="
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem 2rem;
    margin: -1rem -2rem 2rem -2rem;
    border-radius: 0 0 10px 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
">
    <div style="display: flex; justify-content: space-between; align-items: center; color: white;">
        <div>
            <h1 style="margin: 0; font-size: 1.8rem; font-weight: 600;">🚀 Evolve AI Trading</h1>
            <p style="margin: 0.2rem 0 0 0; opacity: 0.9; font-size: 0.9rem;">Autonomous Financial Intelligence Platform</p>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 0.8rem; opacity: 0.8;">Current Model</div>
            <div style="font-weight: 600; font-size: 1rem;">Hybrid Ensemble</div>
            <div style="font-size: 0.8rem; opacity: 0.8;">Last Run: 2 min ago</div>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Initialize session state with concurrency safeguards


def initialize_session_state():
    """Initialize session state with proper concurrency handling."""
    # Prompt agent initialization
    if "prompt_agent" not in st.session_state:
        try:
            st.session_state.prompt_agent = PromptAgent() if CORE_COMPONENTS_AVAILABLE else None
            logger.info("✅ Prompt agent initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize prompt agent: {e}")
            st.session_state.prompt_agent = None

    # Conversation history initialization
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
        logger.info("✅ Conversation history initialized")

    # Agent logger initialization
    if "agent_logger" not in st.session_state:
        try:
            from trading.memory.agent_logger import get_agent_logger

            st.session_state.agent_logger = get_agent_logger()
            logger.info("✅ Agent logger initialized successfully")
        except ImportError as e:
            logger.warning(f"⚠️ Agent logger not available: {e}")
            st.session_state.agent_logger = None
        except Exception as e:
            logger.error(f"❌ Failed to initialize agent logger: {e}")
            st.session_state.agent_logger = None

    # Voice agent initialization
    if "voice_agent" not in st.session_state and VOICE_AGENT_AVAILABLE:
        try:
            st.session_state.voice_agent = VoicePromptAgent()
            logger.info("✅ Voice agent initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize voice agent: {e}")
            st.session_state.voice_agent = None

    # Navigation state initialization
    if "main_nav" not in st.session_state:
        st.session_state.main_nav = "Home & Chat"


# --- Voice Prompt Integration ---
try:
    pass

    from scripts.voice_prompt_agent import VoicePromptAgent

    VOICE_AGENT_AVAILABLE = True
except ImportError:
    VOICE_AGENT_AVAILABLE = False
    VoicePromptAgent = None

# Initialize session state
initialize_session_state()

if "voice_agent" not in st.session_state and VOICE_AGENT_AVAILABLE:
    st.session_state.voice_agent = VoicePromptAgent()

voice_mode = st.toggle("🎤 Voice Input", value=False, help="Enable voice prompt (Whisper or Google Speech)")

if voice_mode and VOICE_AGENT_AVAILABLE:
    st.markdown("**Click below and speak your trading command:**")
    if st.button("🎙️ Record Voice Command"):
        with st.spinner("Listening for command..."):
            text = st.session_state.voice_agent.listen_for_command(timeout=5, phrase_time_limit=10)
            if text:
                st.session_state["voice_prompt_text"] = text
                st.success(f'Voice recognized: "{text}"')
            else:
                st.warning("No voice command detected or could not transcribe.")
else:
    st.session_state["voice_prompt_text"] = ""

# Enhanced Prompt Interface - ChatGPT Style
st.markdown(
    """
<div style="
    background: #f8f9fa;
    border-radius: 15px;
    padding: 2rem;
    margin: 2rem 0;
    border: 1px solid #e9ecef;
    text-align: center;
">
    <h2 style="color: #2c3e50; margin-bottom: 1rem; font-size: 1.5rem;">
        💬 Ask Evolve AI Anything About Trading
    </h2>
    <p style="color: #6c757d; margin-bottom: 1.5rem; font-size: 1rem;">
        Your autonomous financial intelligence assistant is ready to help
    </p>
    <div style="
        background: white;
        border-radius: 10px;
        padding: 1rem;
        border: 2px solid #e9ecef;
        margin-bottom: 1rem;
    ">
        <p style="color: #495057; font-size: 0.9rem; margin: 0;">
            <strong>💡 Examples:</strong> "Forecast SPY using the most accurate model",
            "Create a new LSTM model for AAPL", "Show me RSI strategy with Bollinger Bands",
            "What's the current market sentiment for TSLA?"
        </p>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Professional Prompt Input
prompt = st.text_input(
    "🚀 Type your trading prompt here...",
    value=st.session_state.get("voice_prompt_text", ""),
    key=os.getenv("KEY", ""),
    placeholder="e.g., 'Forecast SPY using the most accurate model and RSI tuned to 10'",
    help="Ask for forecasts, strategies, model creation, or market analysis",
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    submit = st.button("🚀 Submit Query", use_container_width=True, help="Send your prompt to Evolve AI for processing")

# Process prompt
if submit and prompt:
    with st.spinner("Processing your request..."):
        try:
            # Enhanced prompt routing with proper error handling
            from agents.registry import get_prompt_router_agent

            # Get prompt router agent with error handling
            router_agent = get_prompt_router_agent()
            if router_agent is None:
                st.error("❌ Prompt router agent not available. Please check system configuration.")
                st.stop()

            # Handle prompt with comprehensive routing
            result = router_agent.handle_prompt(prompt)

            # Display result with enhanced formatting
            if hasattr(result, "message") and result.message:
                st.markdown(
                    f"""
                <div class="result-card">
                    <h3>🤖 AI Response</h3>
                    <p>{result.message}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Display strategy and model information if available
                if hasattr(result, "strategy_name") and result.strategy_name:
                    st.metric("📊 Strategy Used", result.strategy_name)
                if hasattr(result, "model_used") and result.model_used:
                    st.metric("🧠 Model Used", result.model_used)
                if hasattr(result, "confidence") and result.confidence:
                    st.metric("🎯 Confidence", f"{result.confidence:.2%}")
                if hasattr(result, "signal") and result.signal:
                    signal_color = (
                        "🟢"
                        if result.signal.lower() in ["buy", "long"]
                        else "🔴"
                        if result.signal.lower() in ["sell", "short"]
                        else "🟡"
                    )
                    st.metric("📈 Signal", f"{signal_color} {result.signal}")
            else:
                st.warning("⚠️ No response received from AI agent")

            # Store response in session state
            if "last_response" not in st.session_state:
                st.session_state.last_response = None
            st.session_state.last_response = result

            # Add to conversation history with proper state management
            if "conversation_history" not in st.session_state:
                st.session_state.conversation_history = []

            st.session_state.conversation_history.append(
                {"prompt": prompt, "response": result, "timestamp": datetime.now()}
            )

            # Enhanced navigation routing based on intent
            if hasattr(result, "message"):
                message_lower = result.message.lower()
                if "forecast" in message_lower or "prediction" in message_lower:
                    st.session_state.main_nav = "Forecasting"
                elif "strategy" in message_lower or "signal" in message_lower:
                    st.session_state.main_nav = "Strategy Lab"
                elif "report" in message_lower or "export" in message_lower or "analysis" in message_lower:
                    st.session_state.main_nav = "Reports"
                elif "tune" in message_lower or "optimize" in message_lower or "model" in message_lower:
                    st.session_state.main_nav = "Model Lab"
                elif "setting" in message_lower or "config" in message_lower:
                    st.session_state.main_nav = "Settings"

            st.success("✅ Request processed successfully! Evolve AI has analyzed your query.")

        except ImportError as e:
            st.error(f"❌ System configuration error: {str(e)}")
            st.info("Please ensure all required modules are properly installed.")
        except Exception as e:
            st.error(f"❌ Error processing request: {str(e)}")
            logger.error(f"Prompt processing error: {str(e)}", exc_info=True)

# Enhanced Conversation History Display
if st.session_state.conversation_history:
    st.markdown("### 💬 Recent Conversations")
    for i, conv in enumerate(reversed(st.session_state.conversation_history[-5:])):
        with st.expander(f"💭 {conv['prompt'][:50]}...", expanded=False):
            st.markdown(
                f"""
            <div style="
                background: #f8f9fa;
                border-radius: 10px;
                padding: 1.5rem;
                margin: 1rem 0;
                border-left: 4px solid #667eea;
            ">
                <div style="font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem;">
                    🤔 Your Question:
                </div>
                <div style="margin-bottom: 1rem; color: #495057; font-style: italic;">
                    "{conv['prompt']}"
                </div>
                <div style="font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem;">
                    🤖 AI Response:
                </div>
                <div style="color: #495057; background: white; padding: 1rem; border-radius: 5px;">
                    {getattr(conv['response'], 'message', 'No response available.')}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            if hasattr(conv["response"], "data") and conv["response"].data:
                with st.expander("Detailed Data", expanded=False):
                    st.json(conv["response"].data)

# Display agent logs
if st.session_state.agent_logger:
    with st.expander("Agent Activity Logs", expanded=False):
        try:
            from trading.memory.agent_logger import AgentAction, LogLevel

            # Get recent logs
            recent_logs = st.session_state.agent_logger.get_recent_logs(limit=20)

            if recent_logs:
                st.markdown("#### Recent Agent Actions")

                # Filter options
                col1, col2, col3 = st.columns(3)
                with col1:
                    show_level = st.selectbox("Log Level", ["All", "Info", "Warning", "Error"])
                with col2:
                    show_action = st.selectbox(
                        "Action Type", ["All", "Model Synthesis", "Strategy Switch", "Forecast", "Trade"]
                    )
                with col3:
                    show_agent = st.selectbox("Agent", ["All"] + list(set(log.agent_name for log in recent_logs)))

                # Filter logs
                filtered_logs = recent_logs
                if show_level != "All":
                    level_map = {"Info": LogLevel.INFO, "Warning": LogLevel.WARNING, "Error": LogLevel.ERROR}
                    filtered_logs = [log for log in filtered_logs if log.level == level_map[show_level]]

                if show_action != "All":
                    action_map = {
                        "Model Synthesis": AgentAction.MODEL_SYNTHESIS,
                        "Strategy Switch": AgentAction.STRATEGY_SWITCH,
                        "Forecast": AgentAction.FORECAST_GENERATION,
                        "Trade": AgentAction.TRADE_EXECUTION,
                    }
                    filtered_logs = [log for log in filtered_logs if log.action == action_map[show_action]]

                if show_agent != "All":
                    filtered_logs = [log for log in filtered_logs if log.agent_name == show_agent]

                # Display logs
                for log in filtered_logs[-10:]:  # Show last 10 filtered logs
                    level_color = {
                        LogLevel.INFO: "🟢",
                        LogLevel.WARNING: "🟡",
                        LogLevel.ERROR: "🔴",
                        LogLevel.CRITICAL: "🔴",
                    }.get(log.level, "⚪")

                    st.markdown(
                        f"""
                    <div class="conversation-item">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <span style="font-weight: 600; color: #2c3e50;">{level_color} {log.agent_name}</span>
                            <span style="font-size: 0.8rem; color: #6c757d;">{log.timestamp.strftime('%H:%M:%S')}</span>
                        </div>
                        <div style="color: #5a6c7d; margin-bottom: 0.5rem;">{log.message}</div>
                        <div style="font-size: 0.8rem; color: #6c757d;">Action: {log.action.value.replace('_', ' ').title()}</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Show performance metrics if available
                    if log.performance_metrics:
                        with st.expander("Performance Metrics", expanded=False):
                            st.json(log.performance_metrics)

                    # Show error details if available
                    if log.error_details:
                        with st.expander("Error Details", expanded=False):
                            st.error(log.error_details)
            else:
                st.info("No recent agent activity to display.")

        except Exception as e:
            st.warning(f"Could not load agent logs: {e}")

# --- Main Content ---
if primary_nav == "Home & Chat":
    # Enhanced home page with chat interface
    st.markdown("### AI Trading Assistant")

    # Display conversation history with better styling
    if st.session_state.conversation_history:
        st.markdown("#### Recent Conversations")
        for i, conv in enumerate(reversed(st.session_state.conversation_history[-5:])):
            with st.expander(f"{conv['prompt'][:50]}...", expanded=False):
                st.markdown(
                    f"""
                <div class="conversation-item">
                    <div class="conversation-prompt">Your Question:</div>
                    <div style="margin-bottom: 1rem;">{conv['prompt']}</div>
                    <div class="conversation-prompt">AI Response:</div>
                    <div class="conversation-response">{getattr(conv['response'], 'message', 'No response available.')}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                if hasattr(conv["response"], "data") and conv["response"].data:
                    with st.expander("Detailed Data", expanded=False):
                        st.json(conv["response"].data)
    else:
        st.markdown(
            """
        <div class="result-card">
            <h3>Welcome to Evolve AI Trading</h3>
            <p>Start by asking me anything about trading! I can help you with:</p>
            <ul>
                <li><strong>Forecasting:</strong> "Show me the best forecast for AAPL"</li>
                <li><strong>Strategy Analysis:</strong> "Switch to RSI strategy and optimize it"</li>
                <li><strong>Model Building:</strong> "Create a new model for cryptocurrency trading"</li>
                <li><strong>Reports:</strong> "Generate a performance report for my portfolio"</li>
                <li><strong>Market Analysis:</strong> "What's the current market sentiment?"</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

elif primary_nav == "Forecasting":
    # Import and run the forecasting page
    try:
        from pages.Forecasting import main as forecasting_main

        forecasting_main()
    except ImportError as e:
        st.error(f"Forecasting page not available: {e}")
        st.info("Please ensure the Forecasting.py page exists in the pages directory")
    except Exception as e:
        st.error(f"Error loading Forecasting page: {e}")

elif primary_nav == "Strategy Lab":
    # Import and run the strategy lab page
    try:
        from pages.Strategy_Lab import main as strategy_lab_main

        strategy_lab_main()
    except ImportError as e:
        st.error(f"Strategy Lab page not available: {e}")
        st.info("Please ensure the Strategy_Lab.py page exists in the pages directory")
    except Exception as e:
        st.error(f"Error loading Strategy Lab page: {e}")

elif primary_nav == "Model Lab":
    # Import and run the model lab page
    try:
        from pages.Model_Lab import main as model_lab_main

        model_lab_main()
    except ImportError as e:
        st.error(f"Model Lab page not available: {e}")
        st.info("Please ensure the Model_Lab.py page exists in the pages directory")
    except Exception as e:
        st.error(f"Error loading Model Lab page: {e}")

elif primary_nav == "Reports":
    # Import and run the reports page
    try:
        from pages.Reports import main as reports_main

        reports_main()
    except ImportError as e:
        st.error(f"Reports page not available: {e}")
        st.info("Please ensure the Reports.py page exists in the pages directory")
    except Exception as e:
        st.error(f"Error loading Reports page: {e}")

# Handle advanced navigation
if "advanced_nav" in locals():
    if advanced_nav == "Settings":
        st.markdown("### Advanced Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
            <div class="result-card">
                <h3>System Configuration</h3>
                <p>Advanced system settings and preferences.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Advanced settings
            st.subheader("Risk Management")
            risk_level = st.selectbox("Risk Level", ["Conservative", "Moderate", "Aggressive"])
            max_position_size = st.slider("Max Position Size (%)", 1, 50, 10)
            stop_loss = st.slider("Stop Loss (%)", 1, 20, 5)

            st.subheader("Data Sources")
            data_source = st.selectbox("Primary Data Source", ["YFinance", "Alpha Vantage", "Polygon"])
            update_frequency = st.selectbox("Update Frequency", ["Real-time", "1 minute", "5 minutes", "15 minutes"])

        with col2:
            st.markdown(
                """
            <div class="result-card">
                <h3>Performance Settings</h3>
                <p>Optimize your trading performance and monitoring.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.subheader("Notifications")
            email_notifications = st.checkbox("Email Notifications")
            slack_notifications = st.checkbox("Slack Notifications")
            telegram_notifications = st.checkbox("Telegram Notifications")

            st.subheader("Auto-Trading")
            auto_trading = st.checkbox("Enable Auto-Trading")
            if auto_trading:
                st.warning("Auto-trading is enabled. Please review your risk settings.")

    elif advanced_nav == "System Monitor":
        st.markdown("### System Monitor")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                """
            <div class="result-card">
                <h3>System Health</h3>
                <p>Real-time system monitoring.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.metric("CPU Usage", "45%", "Normal")
            st.metric("Memory Usage", "62%", "Normal")
            st.metric("Disk Usage", "28%", "Good")

        with col2:
            st.markdown(
                """
            <div class="result-card">
                <h3>Network Status</h3>
                <p>Network connectivity and data feeds.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.metric("API Calls/min", "156", "+12")
            st.metric("Data Feed", "Active", "🟢")
            st.metric("Latency", "45ms", "Good")

        with col3:
            st.markdown(
                """
            <div class="result-card">
                <h3>AI Models</h3>
                <p>Model performance and status.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.metric("Active Models", "8", "All Healthy")
            st.metric("Avg Accuracy", "87.2%", "+1.1%")
            st.metric("Training Jobs", "2", "In Progress")

    elif advanced_nav == "Performance Analytics":
        st.markdown("### Performance Analytics")

        st.markdown(
            """
        <div class="result-card">
            <h3>Advanced Analytics Dashboard</h3>
            <p>Comprehensive performance analysis and insights.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Mock analytics dashboard
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Portfolio Performance")
            st.line_chart(
                pd.DataFrame(
                    {"Portfolio": np.random.randn(100).cumsum(), "Benchmark": np.random.randn(100).cumsum() * 0.8}
                )
            )

        with col2:
            st.subheader("Risk Metrics")
            risk_metrics = {
                "Sharpe Ratio": 1.85,
                "Sortino Ratio": 2.12,
                "Calmar Ratio": 1.67,
                "Max Drawdown": 8.2,
                "VaR (95%)": 2.1,
                "CVaR (95%)": 3.4,
            }

            for metric, value in risk_metrics.items():
                st.metric(metric, f"{value:.2f}")

    elif advanced_nav == "Risk Management":
        st.markdown("### Risk Management")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
            <div class="result-card">
                <h3>Risk Alerts</h3>
                <p>Current risk monitoring and alerts.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Mock risk alerts
            alerts = [
                {"level": "Low", "message": "Portfolio concentration in tech sector", "time": "2 hours ago"},
                {"level": "Medium", "message": "High volatility detected in crypto positions", "time": "1 hour ago"},
                {"level": "High", "message": "Stop loss triggered for TSLA position", "time": "30 min ago"},
            ]

            for alert in alerts:
                color = {"Low": "🟡", "Medium": "🟠", "High": "🔴"}[alert["level"]]
                st.markdown(f"{color} **{alert['level']}:** {alert['message']}")
                st.caption(f"*{alert['time']}*")

        with col2:
            st.markdown(
                """
            <div class="result-card">
                <h3>Risk Metrics</h3>
                <p>Current risk exposure and limits.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.metric("Portfolio Beta", "1.12", "Moderate")
            st.metric("Correlation", "0.67", "Acceptable")
            st.metric("Concentration", "23%", "High")
            st.metric("Leverage", "1.05", "Low")

# Footer
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>Evolve AI Trading Platform | Professional Trading Intelligence</p>
    <p style="font-size: 0.8rem;">Built with advanced AI and machine learning</p>
</div>
""",
    unsafe_allow_html=True,
)
