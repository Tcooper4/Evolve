"""
Evolve Trading Platform - Production Ready Streamlit App

A clean, ChatGPT-like interface with a single prompt box that routes to:
- Forecasting
- Strategy Selection  
- Signal Generation
- Trade Reports

All triggered from a single input with professional styling.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️ python-dotenv not available, using system environment variables")
except Exception as e:
    print(f"⚠️ Error loading .env file: {e}")

# Debug: Check if API keys are loaded
alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
finnhub_key = os.getenv('FINNHUB_API_KEY')
polygon_key = os.getenv('POLYGON_API_KEY')

if alpha_vantage_key:
    print(f"✅ ALPHA_VANTAGE_API_KEY loaded: {alpha_vantage_key[:10]}...")
else:
    print("❌ ALPHA_VANTAGE_API_KEY not found")

if finnhub_key:
    print(f"✅ FINNHUB_API_KEY loaded: {finnhub_key[:10]}...")
else:
    print("❌ FINNHUB_API_KEY not found")

if polygon_key:
    print(f"✅ POLYGON_API_KEY loaded: {polygon_key[:10]}...")
else:
    print("❌ POLYGON_API_KEY not found")

# Configure page
st.set_page_config(
    page_title="Evolve AI Trading Dashboard", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import core components with error handling
try:
    from trading.llm.agent import PromptAgent
    from strategies.strategy_engine import StrategyEngine
    from models.forecast_router import ForecastRouter
    from trading.report.report_export_engine import ReportExportEngine
    from trading.memory.model_monitor import ModelMonitor
    from trading.memory.strategy_logger import StrategyLogger
    from trading.agents.model_improver_agent import ModelImprovementRequest
    CORE_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some modules not available: {e}")
    CORE_COMPONENTS_AVAILABLE = False

# Enhanced ChatGPT-like styling
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 0;
    }
    
    /* Prompt input styling */
    .prompt-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 2rem 0;
        border: 2px solid #f0f0f0;
    }
    
    .stTextInput > div > div > input {
        font-size: 1.3rem !important;
        padding: 1rem 1.5rem !important;
        border-radius: 15px !important;
        border: 2px solid #e0e0e0 !important;
        background: #fafafa !important;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        background: white !important;
    }
    
    /* Button styling */
    .stButton > button {
        font-size: 1.2rem !important;
        padding: 1rem 2rem !important;
        border-radius: 15px !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Result card styling */
    .result-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid #f0f0f0;
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    .result-card h3 {
        color: #333;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .result-card p {
        color: #666;
        line-height: 1.6;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Navigation styling */
    .nav-item {
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border-radius: 10px;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .nav-item:hover {
        background: rgba(102, 126, 234, 0.1);
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-success { background: #28a745; }
    .status-warning { background: #ffc107; }
    .status-error { background: #dc3545; }
    
    /* Loading animation */
    .loading-spinner {
        text-align: center;
        padding: 2rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .prompt-container {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2>📊 Evolve AI</h2>
        <p style="color: #666; font-size: 0.9rem;">Professional Trading Suite</p>
    </div>
    """, unsafe_allow_html=True)
    
    nav = st.radio(
        "Navigation",
        [
            "📊 Forecast & Trade",
            "🧠 Strategy Builder", 
            "📈 Model Tuner",
            "📁 Reports & Exports",
            "⚙️ Settings"
        ],
        key="main_nav"
    )
    
    st.markdown("---")
    
    # System status
    st.markdown("### System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<span class="status-indicator status-success"></span>Core Systems', unsafe_allow_html=True)
    with col2:
        st.markdown('<span class="status-indicator status-success"></span>Data Feed', unsafe_allow_html=True)
    
    # Hide dev pages unless in dev mode
    if os.environ.get('EVOLVE_DEV_MODE', '0') == '1':
        st.markdown("---")
        st.markdown("### 🛠️ Developer Tools")
        st.markdown("- [Debug Console](#)")
        st.markdown("- [System Logs](#)")
        st.markdown("- [Performance Monitor](#)")

# --- Main Layout ---
st.markdown("""
<div class="main-header">
    <h1>🚀 Evolve AI Trading</h1>
    <p>Natural Language Trading Intelligence</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'prompt_agent' not in st.session_state:
    st.session_state.prompt_agent = PromptAgent() if CORE_COMPONENTS_AVAILABLE else None

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Prompt input section
st.markdown("""
<div class="prompt-container">
    <h3 style="text-align: center; margin-bottom: 1.5rem; color: #333;">
        💬 Ask me anything about trading
    </h3>
    <p style="text-align: center; color: #666; margin-bottom: 2rem;">
        Examples: "Show me the best forecast for AAPL", "Switch to RSI strategy and optimize it", 
        "Export my last report", "What's the current market sentiment?"
    </p>
</div>
""", unsafe_allow_html=True)

# Prompt input
prompt = st.text_input(
    "Type your trading prompt here...",
    key="main_prompt",
    placeholder="e.g., 'Show me the best forecast for AAPL'"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    submit = st.button("🚀 Submit", use_container_width=True)

# Process prompt
if submit and prompt:
    with st.spinner("🤖 Processing your request..."):
        try:
            if st.session_state.prompt_agent:
                response = st.session_state.prompt_agent.process_prompt(prompt)
                st.session_state.last_response = response
                
                # Add to conversation history
                st.session_state.conversation_history.append({
                    'prompt': prompt,
                    'response': response,
                    'timestamp': datetime.now()
                })
                
                # Route navigation based on intent
                if hasattr(response, 'message'):
                    message_lower = response.message.lower()
                    if 'forecast' in message_lower:
                        st.session_state.main_nav = "📊 Forecast & Trade"
                    elif 'strategy' in message_lower:
                        st.session_state.main_nav = "🧠 Strategy Builder"
                    elif 'report' in message_lower or 'export' in message_lower:
                        st.session_state.main_nav = "📁 Reports & Exports"
                    elif 'tune' in message_lower or 'optimize' in message_lower:
                        st.session_state.main_nav = "📈 Model Tuner"
                    elif 'setting' in message_lower:
                        st.session_state.main_nav = "⚙️ Settings"
                
                st.success("✅ Request processed successfully!")
            else:
                st.error("❌ Core components not available. Please check system configuration.")
        except Exception as e:
            st.error(f"❌ Error processing request: {str(e)}")

# Display conversation history
if st.session_state.conversation_history:
    st.markdown("### 📝 Recent Conversations")
    for i, conv in enumerate(reversed(st.session_state.conversation_history[-5:])):
        with st.expander(f"💬 {conv['prompt'][:50]}..."):
            st.markdown(f"**Prompt:** {conv['prompt']}")
            if hasattr(conv['response'], 'message'):
                st.markdown(f"**Response:** {conv['response'].message}")
            if hasattr(conv['response'], 'data') and conv['response'].data:
                st.json(conv['response'].data)

# --- Main Content ---
if nav == "📊 Forecast & Trade":
    st.markdown("### 📊 Forecast & Trade Results")
    if 'last_response' in st.session_state and st.session_state.last_response:
        resp = st.session_state.last_response
        with st.container():
            st.markdown(f'''
            <div class="result-card">
                <h3>🎯 Forecast Results</h3>
                <p>{getattr(resp, "message", "No result available.")}</p>
            </div>
            ''', unsafe_allow_html=True)
            if hasattr(resp, 'data') and resp.data:
                st.json(resp.data)
    else:
        st.info("💡 Try asking for a forecast like 'Show me the best forecast for AAPL'")

elif nav == "🧠 Strategy Builder":
    st.markdown("### 🧠 Strategy Builder")
    if 'last_response' in st.session_state and st.session_state.last_response:
        resp = st.session_state.last_response
        with st.container():
            st.markdown(f'''
            <div class="result-card">
                <h3>🧠 Strategy Analysis</h3>
                <p>{getattr(resp, "message", "No result available.")}</p>
            </div>
            ''', unsafe_allow_html=True)
            if hasattr(resp, 'data') and resp.data:
                st.json(resp.data)
    else:
        st.info("💡 Try asking for a strategy like 'Switch to RSI strategy and optimize it'")

elif nav == "📈 Model Tuner":
    st.markdown("### 📈 Model Tuner (Advanced)")
    st.markdown("""
    <div class="result-card">
        <h3>🔧 Model Optimization Tools</h3>
        <p>Advanced model tuning and optimization features coming soon.</p>
        <ul>
            <li>Hyperparameter optimization</li>
            <li>Model performance comparison</li>
            <li>Automated model selection</li>
            <li>Real-time model monitoring</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

elif nav == "📁 Reports & Exports":
    st.markdown("### 📁 Reports & Exports")
    st.markdown("""
    <div class="result-card">
        <h3>📊 Report Management</h3>
        <p>Comprehensive reporting and export tools coming soon.</p>
        <ul>
            <li>Performance reports</li>
            <li>Risk analysis reports</li>
            <li>Strategy backtest reports</li>
            <li>Export to PDF/Excel</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

elif nav == "⚙️ Settings":
    st.markdown("### ⚙️ Settings & Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="result-card">
            <h3>🔧 System Settings</h3>
            <p>Configure your trading environment and preferences.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Settings controls
        st.subheader("Risk Management")
        risk_level = st.selectbox("Risk Level", ["Conservative", "Moderate", "Aggressive"])
        max_position_size = st.slider("Max Position Size (%)", 1, 50, 10)
        
        st.subheader("Data Sources")
        data_source = st.selectbox("Primary Data Source", ["YFinance", "Alpha Vantage", "Polygon"])
        
    with col2:
        st.markdown("""
        <div class="result-card">
            <h3>📈 Performance Settings</h3>
            <p>Optimize your trading performance and monitoring.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Notifications")
        email_notifications = st.checkbox("Email Notifications")
        slack_notifications = st.checkbox("Slack Notifications")
        
        st.subheader("Auto-Trading")
        auto_trading = st.checkbox("Enable Auto-Trading")
        if auto_trading:
            st.warning("⚠️ Auto-trading is enabled. Please review your risk settings.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>🚀 Evolve AI Trading Platform | Professional Trading Intelligence</p>
    <p style="font-size: 0.8rem;">Built with advanced AI and machine learning</p>
</div>
""", unsafe_allow_html=True) 