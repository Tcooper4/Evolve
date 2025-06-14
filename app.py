"""Main Streamlit application for the trading platform."""

import streamlit as st
import sys
import platform
from datetime import datetime
from pathlib import Path
import json

from trading.utils.auto_repair import auto_repair
from trading.utils.error_logger import error_logger
from trading.utils.diagnostics import diagnostics
from trading.llm.llm_interface import LLMInterface
from trading.agents.router import AgentRouter
from trading.agents.updater import ModelUpdater
from trading.memory.performance_memory import PerformanceMemory

def show_startup_banner():
    """Display system startup banner."""
    st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
        <h1>🚀 Advanced Financial Forecasting Platform</h1>
        <p style='font-size: 1.2em;'>
            Version 1.0.0 | Python {python_version} | {platform}
        </p>
        <p style='font-size: 1.1em;'>
            Last Build: {build_time} | Models: {model_count}
        </p>
    </div>
    """.format(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        build_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        model_count=len(list(Path("trading/models").glob("*.pkl")))
    ), unsafe_allow_html=True)

def initialize_system():
    """Initialize system components."""
    # Run auto-repair
    repair_results = auto_repair.run_repair()
    if repair_results['status'] != 'success':
        error_logger.log_error(
            "System repair required",
            context=repair_results
        )
    
    # Run health check
    health_status = diagnostics.run_health_check()
    if health_status['status'] != 'healthy':
        error_logger.log_error(
            "Health check failed",
            context=health_status
        )
    
    # Initialize components
    if 'llm' not in st.session_state:
        st.session_state.llm = LLMInterface()
    
    if 'router' not in st.session_state:
        st.session_state.router = AgentRouter()
    
    if 'updater' not in st.session_state:
        st.session_state.updater = ModelUpdater()
    
    if 'memory' not in st.session_state:
        st.session_state.memory = PerformanceMemory()

def main():
    """Main application entry point."""
    # Set page config
    st.set_page_config(
        page_title="Trading Platform",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Show startup banner
    show_startup_banner()
    
    # Initialize system
    initialize_system()
    
    # Main content
    st.title("Trading Dashboard")
    
    # Add your main application content here
    # ...

if __name__ == "__main__":
    main()