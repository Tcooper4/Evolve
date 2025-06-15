"""Main Streamlit application for the trading platform."""

import streamlit as st
import sys
import platform
from datetime import datetime
from pathlib import Path
import json

from trading.utils.system_startup import (
    run_system_checks,
    initialize_components,
    get_system_status,
    clear_session_state
)
from trading.utils.error_logger import error_logger
from trading.config import config

def show_startup_banner():
    """Display system startup banner with enhanced status information."""
    status = get_system_status()
    
    # Determine status badge
    if status["health_status"] == "healthy" and status["repair_status"] == "success":
        status_badge = "🟢 Healthy"
    else:
        status_badge = "🔴 Needs Attention"
    
    st.markdown(f"""
    <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
        <h1>🚀 Advanced Financial Forecasting Platform</h1>
        <p style='font-size: 1.2em;'>
            Version 1.0.0 | Python {sys.version.split()[0]} | {platform.platform()}
        </p>
        <p style='font-size: 1.1em;'>
            Last Build: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | 
            Models: {status["model_count"]} | 
            LLM: {status["llm_provider"]} | 
            Status: {status_badge}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # System logs expander
    with st.expander("System Logs"):
        if status["last_check"]:
            st.write(f"Last System Check: {status['last_check']}")
        st.write("Health Status:", status["health_status"])
        st.write("Repair Status:", status["repair_status"])

def initialize_system():
    """Initialize system components with error handling and status tracking."""
    # Run system checks
    check_results = run_system_checks()
    st.session_state["health_status"] = check_results["health"]["status"]
    st.session_state["repair_status"] = check_results["repair"]["status"]
    
    # Initialize components
    init_results = initialize_components()
    
    # Store initialization results
    st.session_state["init_errors"] = init_results["errors"]
    st.session_state["last_system_check"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Show results
    if init_results["errors"]:
        for error in init_results["errors"]:
            st.error(error)
    else:
        st.success("System initialized successfully!")

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
    
    # Add restart button
    if st.button("🔄 Restart System"):
        clear_session_state()
        st.rerun()
    
    # Initialize system if not already done
    if "health_status" not in st.session_state:
        initialize_system()
    
    # Main content
    st.title("Trading Dashboard")
    
    # Add your main application content here
    # ...

if __name__ == "__main__":
    main()