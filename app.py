"""
Main Streamlit Application

This is the entry point for the Agentic Forecasting System dashboard.
Provides a web interface for financial forecasting and trading strategy analysis.
"""

import streamlit as st
import importlib
from typing import Dict, Any, Optional
import logging
import pandas as pd
from datetime import datetime
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
import warnings
from pathlib import Path

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pandas_ta")

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import shared utilities
from core.session_utils import (
    initialize_session_state, 
    initialize_system_modules, 
    display_system_status,
    safe_session_get,
    safe_session_set,
    update_last_updated
)

# Import AgentHub for unified agent routing
try:
    from core.agent_hub import AgentHub
    AGENT_HUB_AVAILABLE = True
except ImportError as e:
    logger.warning(f"AgentHub import failed: {e}")
    AGENT_HUB_AVAILABLE = False

# Import trading components
from trading.memory.model_monitor import ModelMonitor
from trading.memory.strategy_logger import StrategyLogger
from trading.optimization.strategy_selection_agent import StrategySelectionAgent
from trading.portfolio.portfolio_manager import PortfolioManager
from trading.portfolio.llm_utils import LLMInterface
from trading.optimization.performance_logger import PerformanceLogger
from trading.agents.prompt_router_agent import PromptRouterAgent


def main():
    """Main application function."""
    # Page configuration
    st.set_page_config(
        page_title="Evolve Clean Trading Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    initialize_session_state()

    # Initialize AgentHub if available
    if AGENT_HUB_AVAILABLE and 'agent_hub' not in st.session_state:
        try:
            st.session_state['agent_hub'] = AgentHub()
            logger.info("AgentHub initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AgentHub: {e}")
            st.session_state["status"] = "fallback activated"

    # Initialize system modules
    module_status = initialize_system_modules()

    # Display system status
    display_system_status(module_status)

    # Main dashboard content
    st.title("🚀 Evolve Clean Trading Dashboard")
    st.markdown("Advanced AI-powered trading system with real-time market analysis and automated decision making.")

    # Update last updated timestamp
    update_last_updated()

    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        [
            "🏠 Home",
            "📈 Forecast & Trade", 
            "📊 Portfolio Dashboard",
            "⚡ Strategy History",
            "🎯 Performance Tracker",
            "🛡️ Risk Dashboard",
            "⚙️ Settings",
            "📋 System Scorecard"
        ]
    )

    # Page routing
    if page == "🏠 Home":
        show_home_page(module_status)
    elif page == "📈 Forecast & Trade":
        st.switch_page("pages/1_Forecast_Trade.py")
    elif page == "📊 Portfolio Dashboard":
        st.switch_page("pages/portfolio_dashboard.py")
    elif page == "⚡ Strategy History":
        st.switch_page("pages/6_Strategy_History.py")
    elif page == "🎯 Performance Tracker":
        st.switch_page("pages/performance_tracker.py")
    elif page == "🛡️ Risk Dashboard":
        st.switch_page("pages/risk_dashboard.py")
    elif page == "⚙️ Settings":
        st.switch_page("pages/settings.py")
    elif page == "📋 System Scorecard":
        st.switch_page("pages/5_📊_System_Scorecard.py")


def show_home_page(module_status: Dict[str, Any]):
    """Display the home page with system overview and quick actions."""
    st.header("🏠 System Overview")
    
    # System status summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="System Status",
            value="🟢 Operational" if all(status == 'SUCCESS' for status in module_status.values()) else "🟡 Degraded",
            delta="All systems online" if all(status == 'SUCCESS' for status in module_status.values()) else "Some issues detected"
        )
    
    with col2:
        # Get model trust levels
        try:
            model_monitor = ModelMonitor()
            trust_levels = model_monitor.get_model_trust_levels()
            avg_trust = sum(trust_levels.values()) / len(trust_levels) if trust_levels else 0
            st.metric(
                label="Avg Model Trust",
                value=f"{avg_trust:.1%}",
                delta=f"{len(trust_levels)} models active" if trust_levels else "No models"
            )
        except Exception as e:
            logger.warning(f"Could not get model trust levels: {e}")
            st.metric(label="Avg Model Trust", value="N/A", delta="Error")
    
    with col3:
        # Get portfolio status
        try:
            portfolio_manager = safe_session_get('portfolio_manager')
            if portfolio_manager and hasattr(portfolio_manager, 'get_position_summary'):
                summary = portfolio_manager.get_position_summary()
                total_positions = len(summary.get('positions', []))
                st.metric(
                    label="Active Positions",
                    value=total_positions,
                    delta="Portfolio active" if total_positions > 0 else "No positions"
                )
            else:
                st.metric(label="Active Positions", value="0", delta="Portfolio not initialized")
        except Exception as e:
            logger.warning(f"Could not get portfolio status: {e}")
            st.metric(label="Active Positions", value="N/A", delta="Error")
    
    with col4:
        # Get recent strategy decisions
        try:
            strategy_logger = StrategyLogger()
            recent_decisions = strategy_logger.get_recent_decisions(limit=5)
            decision_count = len(recent_decisions) if recent_decisions else 0
            st.metric(
                label="Recent Decisions",
                value=decision_count,
                delta="Strategy active" if decision_count > 0 else "No recent decisions"
            )
        except Exception as e:
            logger.warning(f"Could not get strategy decisions: {e}")
            st.metric(label="Recent Decisions", value="N/A", delta="Error")
    
    st.markdown("---")
    
    # Global AI Agent Interface
    if AGENT_HUB_AVAILABLE and 'agent_hub' in st.session_state:
        st.subheader("🤖 Global AI Agent Interface")
        st.markdown("Ask questions or request actions across the entire system:")
        
        user_prompt = st.text_area(
            "What would you like to know or do?",
            placeholder="e.g., 'Show me the best performing strategy' or 'Analyze market conditions for tech stocks'",
            height=100,
            key="global_agent_prompt"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🚀 Ask AI Agent", type="primary"):
                if user_prompt:
                    with st.spinner("Processing your request..."):
                        try:
                            agent_hub = st.session_state['agent_hub']
                            response = agent_hub.route(user_prompt)
                            
                            st.subheader("🤖 AI Response")
                            
                            if response['type'] == 'forecast':
                                st.success("📈 Forecast Response")
                            elif response['type'] == 'trading':
                                st.success("💰 Trading Response")
                            elif response['type'] == 'analysis':
                                st.success("📊 Analysis Response")
                            elif response['type'] == 'quantitative':
                                st.success("🧮 Quantitative Response")
                            elif response['type'] == 'llm':
                                st.success("💬 General Response")
                            else:
                                st.info("📋 Response")
                            
                            st.write(response['content'])
                            
                            # Show which agent was used
                            st.caption(f"Response from: {response['agent']} agent")
                            
                        except Exception as e:
                            st.error(f"Error processing request: {str(e)}")
                            logger.error(f"Global AgentHub error: {e}")
                else:
                    st.warning("Please enter a prompt to process.")
        
        with col2:
            if st.button("🔄 Reset"):
                st.session_state['global_agent_prompt'] = ""
                st.experimental_rerun()
        
        # Show recent agent interactions
        if 'agent_hub' in st.session_state:
            agent_hub = st.session_state['agent_hub']
            recent_interactions = agent_hub.get_recent_interactions(limit=5)
            if recent_interactions:
                st.subheader("🕒 Recent AI Interactions")
                for interaction in recent_interactions:
                    success_icon = "OK" if interaction['success'] else "FAIL"
                    st.write(f"{success_icon} **{interaction['agent_type']}**: {interaction['prompt'][:60]}...")
    
    st.markdown("---")
    
    # Quick actions
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📈 Generate Forecast", type="primary"):
            st.switch_page("pages/1_Forecast_Trade.py")
    
    with col2:
        if st.button("📊 View Portfolio"):
            st.switch_page("pages/portfolio_dashboard.py")
    
    with col3:
        if st.button("🎯 Track Performance"):
            st.switch_page("pages/performance_tracker.py")
    
    st.markdown("---")
    
    # Agentic prompt routing
    st.subheader("🤖 AI Assistant")
    
    # Initialize prompt router agent
    try:
        prompt_router = PromptRouterAgent()
        
        # Get user input
        user_query = st.text_input(
            "Ask me anything about trading, market analysis, or system status:",
            placeholder="e.g., 'What's the best model for AAPL?' or 'Show me recent performance'"
        )
        
        if user_query:
            with st.spinner("🤖 Processing your request..."):
                try:
                    # Route the prompt to appropriate agent
                    response = prompt_router.route_prompt(user_query)
                    
                    if response:
                        st.success("Response generated")
                        st.write(response)
                    else:
                        st.warning("No response generated - using fallback")
                        # Fallback response
                        st.info("I understand your query. Please use the navigation menu to access specific features:")
                        st.write("- Use 'Forecast & Trade' for market predictions")
                        st.write("- Use 'Portfolio Dashboard' for position management")
                        st.write("- Use 'Performance Tracker' for historical analysis")
                        
                except Exception as e:
                    logger.error(f"Prompt routing failed: {e}")
                    st.error(f"Error processing request: {str(e)}")
                    st.info("Please use the navigation menu to access specific features.")
                    
    except Exception as e:
        logger.warning(f"⚠️ Agentic routing initialization failed: {e}")
        st.info("🤖 AI Assistant temporarily unavailable. Please use the navigation menu to access features.")
    
    st.markdown("---")
    
    # Recent activity
    st.subheader("📋 Recent Activity")
    
    try:
        # Get recent strategy decisions
        strategy_logger = StrategyLogger()
        recent_decisions = strategy_logger.get_recent_decisions(limit=10)
        
        if recent_decisions:
            for decision in recent_decisions:
                timestamp = decision.get('timestamp', 'Unknown')
                strategy = decision.get('strategy', 'Unknown')
                decision_type = decision.get('decision', 'Unknown')
                confidence = decision.get('confidence', 0)
                
                st.write(f"🕒 **{timestamp}** - {strategy}: {decision_type} (Confidence: {confidence:.1%})")
        else:
            st.info("No recent activity to display.")
            
    except Exception as e:
        logger.warning(f"Could not load recent activity: {e}")
        st.info("Recent activity temporarily unavailable.")
    
    st.markdown("---")
    
    # System information
    st.subheader("ℹ️ System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Version:** 1.0.0")
        st.write("**Last Updated:**", safe_session_get('last_updated', 'Unknown'))
        st.write("**Status:**", "🟢 Operational" if all(status == 'SUCCESS' for status in module_status.values()) else "🟡 Degraded")
    
    with col2:
        st.write("**Python:**", sys.version.split()[0])
        st.write("**Streamlit:**", st.__version__)
        st.write("**Pandas:**", pd.__version__)


if __name__ == "__main__":
    main()
