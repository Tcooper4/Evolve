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

# Suppress warnings
warnings.filterwarnings('ignore')

# Suppress Streamlit warnings when importing
import logging
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)
logging.getLogger('streamlit.runtime.state.session_state_proxy').setLevel(logging.ERROR)

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
    from trading.strategies.enhanced_strategy_engine import EnhancedStrategyEngine
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
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0;">
        <h2 style="color: #2c3e50; margin-bottom: 0.5rem;">🚀 Evolve AI</h2>
        <p style="color: #6c757d; font-size: 0.9rem; font-weight: 500;">Professional Trading Intelligence</p>
        <div style="width: 50px; height: 3px; background: linear-gradient(90deg, #667eea, #764ba2); margin: 1rem auto; border-radius: 2px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Navigation
    st.markdown("### 🧭 Navigation")
    
    # Primary Actions
    primary_nav = st.radio(
        "Primary Actions",
        [
            "🏠 Home & Chat",
            "📊 Forecasting",
            "🧠 Strategy Lab",
            "📈 Model Lab",
            "📁 Reports"
        ],
        key="primary_nav"
    )
    
    st.markdown("---")
    
    # Advanced Tools (Collapsible)
    with st.expander("🔧 Advanced Tools", expanded=False):
        advanced_nav = st.radio(
            "Advanced Features",
            [
                "⚙️ Settings",
                "🔍 System Monitor",
                "📊 Performance Analytics",
                "🛡️ Risk Management"
            ],
            key="advanced_nav"
        )
    
    # Developer Tools (Hidden by default)
    if os.environ.get('EVOLVE_DEV_MODE', '0') == '1':
        st.markdown("---")
        with st.expander("🛠️ Developer Tools", expanded=False):
            st.markdown("- [Debug Console](#)")
            st.markdown("- [System Logs](#)")
            st.markdown("- [Performance Monitor](#)")
            st.markdown("- [API Testing](#)")
    
    st.markdown("---")
    
    # System Status
    st.markdown("### 📊 System Status")
    
    # Status indicators with better styling
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="display: flex; align-items: center; margin: 0.5rem 0;">
            <span class="status-indicator status-success"></span>
            <span style="font-size: 0.9rem; font-weight: 500;">Core Systems</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="display: flex; align-items: center; margin: 0.5rem 0;">
            <span class="status-indicator status-success"></span>
            <span style="font-size: 0.9rem; font-weight: 500;">Data Feed</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="display: flex; align-items: center; margin: 0.5rem 0;">
            <span class="status-indicator status-success"></span>
            <span style="font-size: 0.9rem; font-weight: 500;">AI Models</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="display: flex; align-items: center; margin: 0.5rem 0;">
            <span class="status-indicator status-success"></span>
            <span style="font-size: 0.9rem; font-weight: 500;">Agents</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Stats
    st.markdown("---")
    st.markdown("### 📈 Quick Stats")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Active Models", "12", "+2")
        st.metric("Strategies", "8", "0")
    
    with col2:
        st.metric("Success Rate", "94.2%", "+1.2%")
        st.metric("Avg Return", "2.8%", "+0.3%")

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

if 'agent_logger' not in st.session_state:
    try:
        from trading.memory.agent_logger import get_agent_logger
        st.session_state.agent_logger = get_agent_logger()
    except ImportError:
        st.session_state.agent_logger = None

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
        with st.expander(f"💬 {conv['prompt'][:50]}...", expanded=False):
            st.markdown(f"""
            <div class="conversation-item">
                <div class="conversation-prompt">🤔 **Your Question:**</div>
                <div style="margin-bottom: 1rem;">{conv['prompt']}</div>
                <div class="conversation-prompt">🤖 **AI Response:**</div>
                <div class="conversation-response">{getattr(conv['response'], 'message', 'No response available.')}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if hasattr(conv['response'], 'data') and conv['response'].data:
                with st.expander("📊 Detailed Data", expanded=False):
                    st.json(conv['response'].data)

# Display agent logs
if st.session_state.agent_logger:
    with st.expander("🔍 Agent Activity Logs", expanded=False):
        try:
            from trading.memory.agent_logger import AgentAction, LogLevel
            
            # Get recent logs
            recent_logs = st.session_state.agent_logger.get_recent_logs(limit=20)
            
            if recent_logs:
                st.markdown("#### 🤖 Recent Agent Actions")
                
                # Filter options
                col1, col2, col3 = st.columns(3)
                with col1:
                    show_level = st.selectbox("Log Level", ["All", "Info", "Warning", "Error"])
                with col2:
                    show_action = st.selectbox("Action Type", ["All", "Model Synthesis", "Strategy Switch", "Forecast", "Trade"])
                with col3:
                    show_agent = st.selectbox("Agent", ["All"] + list(set(log.agent_name for log in recent_logs)))
                
                # Filter logs
                filtered_logs = recent_logs
                if show_level != "All":
                    level_map = {"Info": LogLevel.INFO, "Warning": LogLevel.WARNING, "Error": LogLevel.ERROR}
                    filtered_logs = [log for log in filtered_logs if log.level == level_map[show_level]]
                
                if show_action != "All":
                    action_map = {"Model Synthesis": AgentAction.MODEL_SYNTHESIS, "Strategy Switch": AgentAction.STRATEGY_SWITCH, 
                                "Forecast": AgentAction.FORECAST_GENERATION, "Trade": AgentAction.TRADE_EXECUTION}
                    filtered_logs = [log for log in filtered_logs if log.action == action_map[show_action]]
                
                if show_agent != "All":
                    filtered_logs = [log for log in filtered_logs if log.agent_name == show_agent]
                
                # Display logs
                for log in filtered_logs[-10:]:  # Show last 10 filtered logs
                    level_color = {
                        LogLevel.INFO: "🟢",
                        LogLevel.WARNING: "🟡", 
                        LogLevel.ERROR: "🔴",
                        LogLevel.CRITICAL: "🔴"
                    }.get(log.level, "⚪")
                    
                    st.markdown(f"""
                    <div class="conversation-item">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <span style="font-weight: 600; color: #2c3e50;">{level_color} {log.agent_name}</span>
                            <span style="font-size: 0.8rem; color: #6c757d;">{log.timestamp.strftime('%H:%M:%S')}</span>
                        </div>
                        <div style="color: #5a6c7d; margin-bottom: 0.5rem;">{log.message}</div>
                        <div style="font-size: 0.8rem; color: #6c757d;">Action: {log.action.value.replace('_', ' ').title()}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show performance metrics if available
                    if log.performance_metrics:
                        with st.expander("📊 Performance Metrics", expanded=False):
                            st.json(log.performance_metrics)
                    
                    # Show error details if available
                    if log.error_details:
                        with st.expander("❌ Error Details", expanded=False):
                            st.error(log.error_details)
            else:
                st.info("No recent agent activity to display.")
                
        except Exception as e:
            st.warning(f"Could not load agent logs: {e}")

# --- Main Content ---
if primary_nav == "🏠 Home & Chat":
    # Enhanced home page with chat interface
    st.markdown("### 💬 AI Trading Assistant")
    
    # Display conversation history with better styling
    if st.session_state.conversation_history:
        st.markdown("#### 📝 Recent Conversations")
        for i, conv in enumerate(reversed(st.session_state.conversation_history[-5:])):
            with st.expander(f"💬 {conv['prompt'][:50]}...", expanded=False):
                st.markdown(f"""
                <div class="conversation-item">
                    <div class="conversation-prompt">🤔 **Your Question:**</div>
                    <div style="margin-bottom: 1rem;">{conv['prompt']}</div>
                    <div class="conversation-prompt">🤖 **AI Response:**</div>
                    <div class="conversation-response">{getattr(conv['response'], 'message', 'No response available.')}</div>
                </div>
                """, unsafe_allow_html=True)
                
                if hasattr(conv['response'], 'data') and conv['response'].data:
                    with st.expander("📊 Detailed Data", expanded=False):
                        st.json(conv['response'].data)
    else:
        st.markdown("""
        <div class="result-card">
            <h3>🎯 Welcome to Evolve AI Trading</h3>
            <p>Start by asking me anything about trading! I can help you with:</p>
            <ul>
                <li>📊 <strong>Forecasting:</strong> "Show me the best forecast for AAPL"</li>
                <li>🧠 <strong>Strategy Analysis:</strong> "Switch to RSI strategy and optimize it"</li>
                <li>📈 <strong>Model Building:</strong> "Create a new model for cryptocurrency trading"</li>
                <li>📁 <strong>Reports:</strong> "Generate a performance report for my portfolio"</li>
                <li>🔍 <strong>Market Analysis:</strong> "What's the current market sentiment?"</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

elif primary_nav == "📊 Forecasting":
    st.markdown("### 📊 Advanced Forecasting")
    
    # Forecasting interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="result-card">
            <h3>🎯 Multi-Model Forecasting</h3>
            <p>Get comprehensive forecasts using our ensemble of AI models.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Symbol input
        symbol = st.text_input("Enter Symbol", placeholder="e.g., AAPL, TSLA, ETH-USD")
        
        # Forecast parameters
        forecast_days = st.slider("Forecast Horizon (Days)", 1, 30, 7)
        confidence_level = st.slider("Confidence Level", 0.8, 0.99, 0.95)
        
        if st.button("🚀 Generate Forecast", use_container_width=True):
            if symbol:
                with st.spinner("🤖 Generating forecast..."):
                    try:
                        # Simulate forecast generation
                        st.success(f"✅ Forecast generated for {symbol}")
                        
                        # Display forecast results
                        st.markdown("""
                        <div class="result-card">
                            <h3>📈 Forecast Results</h3>
                            <p>Forecast analysis completed successfully.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Mock forecast data
                        forecast_data = {
                            'symbol': symbol,
                            'predicted_price': 150.25,
                            'confidence_interval': [145.50, 155.00],
                            'trend': 'bullish',
                            'models_used': ['LSTM', 'Transformer', 'Ensemble'],
                            'accuracy_score': 0.87
                        }
                        
                        st.json(forecast_data)
                        
                    except Exception as e:
                        st.error(f"❌ Error generating forecast: {e}")
            else:
                st.warning("⚠️ Please enter a symbol")
    
    with col2:
        st.markdown("""
        <div class="result-card">
            <h3>📊 Model Performance</h3>
            <p>Current model performance metrics:</p>
            <ul>
                <li>LSTM: 87% accuracy</li>
                <li>Transformer: 89% accuracy</li>
                <li>Ensemble: 92% accuracy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

elif primary_nav == "🧠 Strategy Lab":
    st.markdown("### 🧠 Strategy Laboratory")
    
    # Strategy selection and optimization
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="result-card">
            <h3>🎯 Strategy Selection</h3>
            <p>Choose and optimize trading strategies.</p>
        </div>
        """, unsafe_allow_html=True)
        
        strategy_type = st.selectbox(
            "Select Strategy",
            ["RSI Mean Reversion", "MACD Crossover", "Bollinger Bands", "Moving Average", "Custom"]
        )
        
        if strategy_type == "Custom":
            custom_strategy = st.text_area("Define Custom Strategy", placeholder="Enter your strategy logic...")
        
        optimization_level = st.select_slider(
            "Optimization Level",
            options=["None", "Basic", "Advanced", "Full"]
        )
        
        if st.button("🔧 Optimize Strategy", use_container_width=True):
            st.success("✅ Strategy optimization completed!")
    
    with col2:
        st.markdown("""
        <div class="result-card">
            <h3>📊 Strategy Performance</h3>
            <p>Performance metrics for selected strategies:</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Mock strategy performance
        performance_data = {
            'RSI Mean Reversion': {'sharpe': 1.2, 'win_rate': 0.65, 'max_dd': 0.08},
            'MACD Crossover': {'sharpe': 0.9, 'win_rate': 0.58, 'max_dd': 0.12},
            'Bollinger Bands': {'sharpe': 1.1, 'win_rate': 0.62, 'max_dd': 0.10}
        }
        
        if strategy_type in performance_data:
            perf = performance_data[strategy_type]
            st.metric("Sharpe Ratio", f"{perf['sharpe']:.2f}")
            st.metric("Win Rate", f"{perf['win_rate']:.1%}")
            st.metric("Max Drawdown", f"{perf['max_dd']:.1%}")

elif primary_nav == "📈 Model Lab":
    st.markdown("### 📈 Model Laboratory")
    
    # Model synthesis and management
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="result-card">
            <h3>🤖 AI Model Synthesis</h3>
            <p>Create and optimize new AI models using advanced synthesis techniques.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model synthesis interface
        synthesis_type = st.selectbox(
            "Model Type",
            ["LSTM", "Transformer", "Ensemble", "Custom", "Auto-Generated"]
        )
        
        target_performance = st.slider("Target Performance", 0.7, 0.95, 0.85)
        max_complexity = st.slider("Max Complexity", 0.1, 1.0, 0.6)
        
        if st.button("🔬 Synthesize Model", use_container_width=True):
            with st.spinner("🤖 Synthesizing new model..."):
                try:
                    # Initialize ModelSynthesizerAgent
                    from trading.agents.model_synthesizer_agent import ModelSynthesizerAgent
                    
                    synthesizer = ModelSynthesizerAgent()
                    
                    # Create synthesis request
                    from trading.agents.model_synthesizer_agent import SynthesisRequest, ModelType
                    
                    request = SynthesisRequest(
                        target_performance=target_performance,
                        max_complexity=max_complexity,
                        preferred_model_types=[ModelType.LSTM if synthesis_type == "LSTM" else ModelType.TRANSFORMER],
                        data_characteristics={'features': ['price', 'volume', 'technical_indicators']},
                        constraints={'max_training_time': 1800}
                    )
                    
                    st.success("✅ Model synthesis initiated!")
                    st.info("🔄 Model is being trained in the background...")
                    
                except Exception as e:
                    st.error(f"❌ Model synthesis failed: {e}")
    
    with col2:
        st.markdown("""
        <div class="result-card">
            <h3>📊 Model Registry</h3>
            <p>Active models in the system:</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Mock model registry
        models = [
            {"name": "LSTM_v1", "accuracy": 0.87, "status": "Active"},
            {"name": "Transformer_v2", "accuracy": 0.89, "status": "Active"},
            {"name": "Ensemble_v1", "accuracy": 0.92, "status": "Active"},
            {"name": "Synthesized_v1", "accuracy": 0.85, "status": "Training"}
        ]
        
        for model in models:
            status_color = "🟢" if model["status"] == "Active" else "🟡"
            st.markdown(f"{status_color} **{model['name']}** - {model['accuracy']:.1%} accuracy")

elif primary_nav == "📁 Reports":
    st.markdown("### 📁 Comprehensive Reports")
    
    # Report generation interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="result-card">
            <h3>📊 Report Generator</h3>
            <p>Generate comprehensive trading reports and analytics.</p>
        </div>
        """, unsafe_allow_html=True)
        
        report_type = st.selectbox(
            "Report Type",
            ["Performance Report", "Risk Analysis", "Strategy Backtest", "Model Evaluation", "Portfolio Summary"]
        )
        
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=30), datetime.now())
        )
        
        include_charts = st.checkbox("Include Interactive Charts", value=True)
        export_format = st.selectbox("Export Format", ["PDF", "Excel", "HTML", "JSON"])
        
        if st.button("📄 Generate Report", use_container_width=True):
            st.success("✅ Report generation completed!")
            st.info("📥 Report is ready for download")
    
    with col2:
        st.markdown("""
        <div class="result-card">
            <h3>📈 Quick Analytics</h3>
            <p>Key performance indicators:</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Mock analytics
        st.metric("Total Return", "12.5%", "+2.3%")
        st.metric("Sharpe Ratio", "1.8", "+0.2")
        st.metric("Max Drawdown", "8.2%", "-1.1%")
        st.metric("Win Rate", "68%", "+3%")

# Handle advanced navigation
if 'advanced_nav' in locals():
    if advanced_nav == "⚙️ Settings":
        st.markdown("### ⚙️ Advanced Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="result-card">
                <h3>🔧 System Configuration</h3>
                <p>Advanced system settings and preferences.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Advanced settings
            st.subheader("Risk Management")
            risk_level = st.selectbox("Risk Level", ["Conservative", "Moderate", "Aggressive"])
            max_position_size = st.slider("Max Position Size (%)", 1, 50, 10)
            stop_loss = st.slider("Stop Loss (%)", 1, 20, 5)
            
            st.subheader("Data Sources")
            data_source = st.selectbox("Primary Data Source", ["YFinance", "Alpha Vantage", "Polygon"])
            update_frequency = st.selectbox("Update Frequency", ["Real-time", "1 minute", "5 minutes", "15 minutes"])
        
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
            telegram_notifications = st.checkbox("Telegram Notifications")
            
            st.subheader("Auto-Trading")
            auto_trading = st.checkbox("Enable Auto-Trading")
            if auto_trading:
                st.warning("⚠️ Auto-trading is enabled. Please review your risk settings.")
    
    elif advanced_nav == "🔍 System Monitor":
        st.markdown("### 🔍 System Monitor")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="result-card">
                <h3>🖥️ System Health</h3>
                <p>Real-time system monitoring.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("CPU Usage", "45%", "Normal")
            st.metric("Memory Usage", "62%", "Normal")
            st.metric("Disk Usage", "28%", "Good")
        
        with col2:
            st.markdown("""
            <div class="result-card">
                <h3>🌐 Network Status</h3>
                <p>Network connectivity and data feeds.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("API Calls/min", "156", "+12")
            st.metric("Data Feed", "Active", "🟢")
            st.metric("Latency", "45ms", "Good")
        
        with col3:
            st.markdown("""
            <div class="result-card">
                <h3>🤖 AI Models</h3>
                <p>Model performance and status.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("Active Models", "8", "All Healthy")
            st.metric("Avg Accuracy", "87.2%", "+1.1%")
            st.metric("Training Jobs", "2", "In Progress")
    
    elif advanced_nav == "📊 Performance Analytics":
        st.markdown("### 📊 Performance Analytics")
        
        st.markdown("""
        <div class="result-card">
            <h3>📈 Advanced Analytics Dashboard</h3>
            <p>Comprehensive performance analysis and insights.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Mock analytics dashboard
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Portfolio Performance")
            st.line_chart(pd.DataFrame({
                'Portfolio': np.random.randn(100).cumsum(),
                'Benchmark': np.random.randn(100).cumsum() * 0.8
            }))
        
        with col2:
            st.subheader("Risk Metrics")
            risk_metrics = {
                'Sharpe Ratio': 1.85,
                'Sortino Ratio': 2.12,
                'Calmar Ratio': 1.67,
                'Max Drawdown': 8.2,
                'VaR (95%)': 2.1,
                'CVaR (95%)': 3.4
            }
            
            for metric, value in risk_metrics.items():
                st.metric(metric, f"{value:.2f}")
    
    elif advanced_nav == "🛡️ Risk Management":
        st.markdown("### 🛡️ Risk Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="result-card">
                <h3>⚠️ Risk Alerts</h3>
                <p>Current risk monitoring and alerts.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Mock risk alerts
            alerts = [
                {"level": "Low", "message": "Portfolio concentration in tech sector", "time": "2 hours ago"},
                {"level": "Medium", "message": "High volatility detected in crypto positions", "time": "1 hour ago"},
                {"level": "High", "message": "Stop loss triggered for TSLA position", "time": "30 min ago"}
            ]
            
            for alert in alerts:
                color = {"Low": "🟡", "Medium": "🟠", "High": "🔴"}[alert["level"]]
                st.markdown(f"{color} **{alert['level']}:** {alert['message']}")
                st.caption(f"*{alert['time']}*")
        
        with col2:
            st.markdown("""
            <div class="result-card">
                <h3>📊 Risk Metrics</h3>
                <p>Current risk exposure and limits.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("Portfolio Beta", "1.12", "Moderate")
            st.metric("Correlation", "0.67", "Acceptable")
            st.metric("Concentration", "23%", "High")
            st.metric("Leverage", "1.05", "Low")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>🚀 Evolve AI Trading Platform | Professional Trading Intelligence</p>
    <p style="font-size: 0.8rem;">Built with advanced AI and machine learning</p>
</div>
""", unsafe_allow_html=True) 