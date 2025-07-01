"""
Main Streamlit Application

This is the entry point for the Agentic Forecasting System dashboard.
Provides a web interface for financial forecasting and trading strategy analysis.
"""

import streamlit as st
from typing import Dict, Any
import logging
from datetime import datetime
import sys
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
    update_last_updated
)

# Import AgentHub for unified agent routing
try:
    from core.agent_hub import AgentHub
    AGENT_HUB_AVAILABLE = True
except ImportError as e:
    logger.warning(f"AgentHub import failed: {e}")
    AGENT_HUB_AVAILABLE = False

# Import page modules
from pages.home import render_home_page

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
        result = render_home_page(module_status, AGENT_HUB_AVAILABLE)
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
        return {'success': True, 'result': {'status': 'redirected', 'page': 'system_scorecard'}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    return {'status': 'unknown_page', 'page': page}



if __name__ == "__main__":
    main()
