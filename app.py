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
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(
    logging.ERROR
)
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
st.set_page_config(
    page_title="Evolve AI Trading Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Import core components with error handling
try:
    from agents.llm.agent import PromptAgent

    CORE_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some modules not available: {e}")
    CORE_COMPONENTS_AVAILABLE = False

# Add Task Orchestrator integration after the existing imports
try:
    from core.orchestrator.task_orchestrator import TaskOrchestrator
    from system.orchestrator_integration import get_system_integration_status
    ORCHESTRATOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Task Orchestrator not available: {e}")
    ORCHESTRATOR_AVAILABLE = False

# Add Agent Controller integration
try:
    from agents.agent_controller import get_agent_controller
    AGENT_CONTROLLER_AVAILABLE = True
    logger.info("✅ Agent Controller initialized successfully")
except ImportError as e:
    logger.warning(f"Agent Controller not available: {e}")
    AGENT_CONTROLLER_AVAILABLE = False

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

# --- Sidebar and Navigation ---
primary_nav, advanced_nav = render_sidebar()

# --- Top Navigation Bar ---
render_top_navigation()

# --- Voice Input (if enabled) ---
render_voice_input()

# --- Prompt Input and Result ---
prompt, submit = render_prompt_interface()

if submit and prompt:
    with st.spinner("Processing your request..."):
        try:
            from routing.prompt_router import route_prompt
            import asyncio

            # Run the async route_prompt function
            result = asyncio.run(route_prompt(prompt, llm_type="default"))
            render_prompt_result(result)
            # Store response in session state
            if "last_response" not in st.session_state:
                st.session_state.last_response = None
            st.session_state.last_response = result
            if "conversation_history" not in st.session_state:
                st.session_state.conversation_history = []
            st.session_state.conversation_history.append(
                {"prompt": prompt, "response": result, "timestamp": datetime.now()}
            )
            navigation_info = result.get("navigation_info", {})
            if navigation_info.get("main_nav"):
                st.session_state.main_nav = navigation_info["main_nav"]
            st.success("✅ Request processed successfully! Evolve AI has analyzed your query.")
        except ImportError as e:
            st.error(f"❌ System configuration error: {str(e)}")
            st.info("Please ensure all required modules are properly installed.")
        except Exception as e:
            st.error(f"❌ Error processing request: {str(e)}")

# --- Conversation History ---
render_conversation_history()

# --- Agent Logs ---
render_agent_logs()

# --- Main Content ---
if primary_nav == "🏠 Home & Chat":
    render_home_page()
elif primary_nav == "📈 Forecasting":
    render_forecasting_page()
elif primary_nav == "⚡ Strategy Lab":
    render_strategy_page()
elif primary_nav == "🧠 Model Lab":
    render_model_page()
elif primary_nav == "📋 Reports":
    render_reports_page()

# --- Advanced Navigation ---
if advanced_nav == "⚙️ Settings":
    render_settings_page()
elif advanced_nav == "📊 Monitor":
    render_system_monitor_page()
elif advanced_nav == "📈 Analytics":
    render_performance_analytics_page()
elif advanced_nav == "🛡️ Risk":
    render_risk_management_page()
    elif advanced_nav == "🤖 Orchestrator":
    render_orchestrator_page()

# --- Footer ---
render_footer()

