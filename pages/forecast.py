# -*- coding: utf-8 -*-
"""Forecast page for the trading dashboard."""

# Standard library imports
import sys
from datetime import datetime
from pathlib import Path

# Third-party imports
import pandas as pd
import streamlit as st

# Local imports
from trading.agents.router import AgentRouter
from trading.ui.components import (
    create_prompt_input,
    create_sidebar,
    create_forecast_chart,
    create_forecast_metrics,
    create_forecast_table
)
from memory.performance_log import log_performance
from memory.goals.status import load_goals

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def render_forecast_page():
    """Render the forecast page."""
    st.title("📈 Price Forecast")
    
    # Initialize session state variables if they don't exist
    if "signals" not in st.session_state:
        st.session_state.signals = None
    if "results" not in st.session_state:
        st.session_state.results = None
    if "forecast_results" not in st.session_state:
        st.session_state.forecast_results = None
    
    # Create sidebar
    create_sidebar()
    
    # Create prompt input
    prompt = create_prompt_input()
    
    if prompt:
        try:
            # Initialize router
            router = AgentRouter()
            
            # Get forecast
            forecast_results = router.get_forecast(prompt)
            st.session_state.forecast_results = forecast_results
            
            if forecast_results:
                # Display forecast chart
                create_forecast_chart(forecast_results)
                
                # Display metrics
                create_forecast_metrics(forecast_results)
                
                # Display forecast table
                create_forecast_table(forecast_results)
                
                # Log performance
                if "metrics" in forecast_results:
                    log_performance(
                        ticker=forecast_results.get("ticker", "unknown"),
                        model=forecast_results.get("model", "unknown"),
                        strategy=forecast_results.get("strategy", "unknown"),
                        metrics=forecast_results["metrics"]
                    )
            else:
                st.warning("No forecast results available.")
                
        except Exception as e:
            st.error(f"Error generating forecast: {str(e)}")
            st.session_state.forecast_results = None

if __name__ == "__main__":
    render_forecast_page()

