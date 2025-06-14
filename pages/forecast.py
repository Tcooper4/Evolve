"""Forecast page for the trading dashboard."""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from trading.ui.components import (
    create_prompt_input,
    create_sidebar,
    create_forecast_chart,
    create_error_block,
    create_loading_spinner
)

def render_forecast_page():
    """Render the forecast page."""
    st.title("Price Forecasting")
    
    # Create sidebar
    sidebar_config = create_sidebar()
    
    # Create prompt input
    prompt = create_prompt_input()
    
    # Process prompt if provided
    if prompt:
        try:
            with create_loading_spinner("Generating forecast..."):
                # Get forecast from router
                result = st.session_state.router.route(
                    prompt,
                    model=sidebar_config['model'],
                    strategies=sidebar_config['strategies'],
                    expert_mode=sidebar_config['expert_mode'],
                    data_source=sidebar_config['data_source'],
                    uploaded_file=sidebar_config['uploaded_file']
                )
                
                if result.status == "success":
                    # Display forecast chart
                    if result.visual:
                        st.plotly_chart(result.visual, use_container_width=True)
                    
                    # Display forecast explanation
                    if result.explanation:
                        st.info(result.explanation)
                    
                    # Display forecast metrics
                    if result.data and 'metrics' in result.data:
                        metrics = result.data['metrics']
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Predicted Price", f"${metrics.get('predicted_price', 0):.2f}")
                        with col2:
                            st.metric("Confidence", f"{metrics.get('confidence', 0):.1%}")
                        with col3:
                            st.metric("Horizon", f"{metrics.get('horizon', 'N/A')}")
                else:
                    create_error_block(result.error or "Failed to generate forecast")
                    
        except Exception as e:
            create_error_block(str(e))

if __name__ == "__main__":
    render_forecast_page()

