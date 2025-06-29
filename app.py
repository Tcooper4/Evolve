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

# Import data feed
try:
    from data.live_feed import get_data_feed
    DATA_FEED_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Data feed import failed: {e}")
    DATA_FEED_AVAILABLE = False

# Import capability router
try:
    from core.capability_router import get_system_health as get_capability_health
    CAPABILITY_ROUTER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Capability router import failed: {e}")
    CAPABILITY_ROUTER_AVAILABLE = False


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
        result = show_home_page(module_status)
        return result
    elif page == "📈 Forecast & Trade":
        st.switch_page("pages/1_Forecast_Trade.py")
        return {'status': 'redirected', 'page': 'forecast_trade'}
    elif page == "📊 Portfolio Dashboard":
        st.switch_page("pages/portfolio_dashboard.py")
        return {'status': 'redirected', 'page': 'portfolio_dashboard'}
    elif page == "⚡ Strategy History":
        st.switch_page("pages/6_Strategy_History.py")
        return {'status': 'redirected', 'page': 'strategy_history'}
    elif page == "🎯 Performance Tracker":
        st.switch_page("pages/performance_tracker.py")
        return {'status': 'redirected', 'page': 'performance_tracker'}
    elif page == "🛡️ Risk Dashboard":
        st.switch_page("pages/risk_dashboard.py")
        return {'status': 'redirected', 'page': 'risk_dashboard'}
    elif page == "⚙️ Settings":
        st.switch_page("pages/settings.py")
        return {'status': 'redirected', 'page': 'settings'}
    elif page == "📋 System Scorecard":
        st.switch_page("pages/5_📊_System_Scorecard.py")
        return {'status': 'redirected', 'page': 'system_scorecard'}
    
    return {'status': 'unknown_page', 'page': page}


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
            decision_count = len(recent_decisions)
            st.metric(
                label="Recent Decisions",
                value=decision_count,
                delta="Strategy active" if decision_count > 0 else "No recent decisions"
            )
        except Exception as e:
            logger.warning(f"Could not get strategy decisions: {e}")
            st.metric(label="Recent Decisions", value="N/A", delta="Error")
    
    # System Health Dashboard
    st.subheader("🔧 System Health")
    
    # Get comprehensive system health
    health_data = get_comprehensive_system_health()
    
    # Display health metrics
    health_col1, health_col2, health_col3 = st.columns(3)
    
    with health_col1:
        st.metric(
            label="Data Feed Status",
            value=health_data.get('data_feed_status', 'Unknown'),
            delta=health_data.get('data_feed_providers', 0)
        )
    
    with health_col2:
        st.metric(
            label="Model Engine Status",
            value=health_data.get('model_engine_status', 'Unknown'),
            delta=health_data.get('active_models', 0)
        )
    
    with health_col3:
        st.metric(
            label="Strategy Engine Status",
            value=health_data.get('strategy_engine_status', 'Unknown'),
            delta=health_data.get('active_strategies', 0)
        )
    
    # Quick Actions
    st.subheader("⚡ Quick Actions")
    
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("📈 Run Forecast", use_container_width=True):
            st.switch_page("pages/1_Forecast_Trade.py")
    
    with action_col2:
        if st.button("📊 View Portfolio", use_container_width=True):
            st.switch_page("pages/portfolio_dashboard.py")
    
    with action_col3:
        if st.button("⚡ Strategy History", use_container_width=True):
            st.switch_page("pages/6_Strategy_History.py")
    
    # Recent Activity
    st.subheader("📋 Recent Activity")
    
    try:
        # Get recent agent interactions
        if AGENT_HUB_AVAILABLE and 'agent_hub' in st.session_state:
            agent_hub = st.session_state['agent_hub']
            recent_interactions = agent_hub.get_recent_interactions(limit=10)
            
            if recent_interactions:
                for interaction in recent_interactions:
                    with st.expander(f"{interaction['timestamp']} - {interaction['agent_type']}"):
                        st.write(f"**Prompt:** {interaction['prompt']}")
                        st.write(f"**Response:** {interaction['response'][:200]}...")
                        st.write(f"**Confidence:** {interaction.get('confidence', 'N/A')}")
            else:
                st.info("No recent agent interactions")
        else:
            st.info("Agent hub not available")
            
    except Exception as e:
        logger.warning(f"Could not get recent activity: {e}")
        st.info("Could not load recent activity")
    
    return {
        'status': 'success',
        'module_status': module_status,
        'health_data': health_data,
        'timestamp': datetime.now().isoformat()
    }


def get_comprehensive_system_health() -> Dict[str, Any]:
    """Get comprehensive system health information."""
    health_data = {
        'data_feed_status': 'Unknown',
        'data_feed_providers': 0,
        'model_engine_status': 'Unknown',
        'active_models': 0,
        'strategy_engine_status': 'Unknown',
        'active_strategies': 0,
        'overall_status': 'Unknown'
    }
    
    try:
        # Data feed health
        if DATA_FEED_AVAILABLE:
            data_feed = get_data_feed()
            feed_health = data_feed.get_system_health()
            health_data['data_feed_status'] = feed_health.get('status', 'Unknown')
            health_data['data_feed_providers'] = feed_health.get('available_providers', 0)
        
        # Model engine health
        try:
            model_monitor = ModelMonitor()
            trust_levels = model_monitor.get_model_trust_levels()
            health_data['active_models'] = len(trust_levels) if trust_levels else 0
            health_data['model_engine_status'] = 'Healthy' if health_data['active_models'] > 0 else 'Degraded'
        except Exception as e:
            logger.warning(f"Model engine health check failed: {e}")
            health_data['model_engine_status'] = 'Error'
        
        # Strategy engine health
        try:
            strategy_logger = StrategyLogger()
            recent_strategies = strategy_logger.get_recent_decisions(limit=10)
            health_data['active_strategies'] = len(recent_strategies) if recent_strategies else 0
            health_data['strategy_engine_status'] = 'Healthy' if health_data['active_strategies'] > 0 else 'Degraded'
        except Exception as e:
            logger.warning(f"Strategy engine health check failed: {e}")
            health_data['strategy_engine_status'] = 'Error'
        
        # Overall status
        healthy_components = sum([
            health_data['data_feed_status'] == 'healthy',
            health_data['model_engine_status'] == 'Healthy',
            health_data['strategy_engine_status'] == 'Healthy'
        ])
        
        if healthy_components == 3:
            health_data['overall_status'] = 'Healthy'
        elif healthy_components >= 1:
            health_data['overall_status'] = 'Degraded'
        else:
            health_data['overall_status'] = 'Critical'
            
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        health_data['overall_status'] = 'Error'
    
    return health_data


if __name__ == "__main__":
    main()
