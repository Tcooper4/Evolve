"""Streamlit dashboard for risk visualization."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
from trading.risk.risk_metrics import (
    calculate_rolling_metrics,
    calculate_advanced_metrics,
    plot_risk_metrics,
    plot_drawdown_heatmap
)
from trading.risk.risk_analyzer import RiskAnalyzer
from trading.risk.risk_logger import RiskLogger

# Page config
st.set_page_config(
    page_title="Risk Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize components
risk_analyzer = RiskAnalyzer()
risk_logger = RiskLogger()

# Sidebar
st.sidebar.title("Risk Dashboard Settings")

# Time range selection
time_range = st.sidebar.selectbox(
    "Time Range",
    ["1D", "1W", "1M", "3M", "6M", "1Y", "All"],
    index=2
)

# Model selection
model_name = st.sidebar.selectbox(
    "Model",
    ["All Models", "Model 1", "Model 2", "Model 3"]
)

# Metric frequency
frequency = st.sidebar.selectbox(
    "Metric Frequency",
    ["Daily", "Weekly", "Monthly"],
    index=0
)

# Main content
st.title("Risk Dashboard")

# Get metrics
@st.cache_data(ttl=300)
def load_metrics():
    """Load risk metrics."""
    try:
        # Get recent metrics
        df = risk_logger.get_recent_metrics(
            model_name=None if model_name == "All Models" else model_name
        )
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by time range
        if time_range != "All":
            cutoff = {
                "1D": timedelta(days=1),
                "1W": timedelta(weeks=1),
                "1M": timedelta(days=30),
                "3M": timedelta(days=90),
                "6M": timedelta(days=180),
                "1Y": timedelta(days=365)
            }[time_range]
            df = df[df['timestamp'] > datetime.now() - cutoff]
        
        # Resample if needed
        if frequency != "Daily":
            freq_map = {
                "Weekly": "W",
                "Monthly": "M"
            }
            df = df.set_index('timestamp').resample(freq_map[frequency]).mean().reset_index()
        
        return df
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return pd.DataFrame()

# Load metrics
metrics_df = load_metrics()

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "Risk Metrics",
    "Regime Analysis",
    "Risk Summary"
])

# Risk Metrics Tab
with tab1:
    st.header("Risk Metrics")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    if not metrics_df.empty:
        latest = metrics_df.iloc[-1]
        
        with col1:
            st.metric(
                "Sharpe Ratio",
                f"{latest['metrics']['sharpe_ratio']:.2f}"
            )
        
        with col2:
            st.metric(
                "Volatility",
                f"{latest['metrics']['volatility']:.2%}"
            )
        
        with col3:
            st.metric(
                "Max Drawdown",
                f"{latest['metrics']['max_drawdown']:.2%}"
            )
        
        with col4:
            st.metric(
                "VaR (95%)",
                f"{latest['metrics']['var_95']:.2%}"
            )
        
        # Plot metrics
        st.plotly_chart(
            plot_risk_metrics(metrics_df),
            use_container_width=True
        )
        
        # Drawdown heatmap
        st.subheader("Drawdown Heatmap")
        st.plotly_chart(
            plot_drawdown_heatmap(metrics_df),
            use_container_width=True
        )

# Regime Analysis Tab
with tab2:
    st.header("Market Regime Analysis")
    
    if not metrics_df.empty:
        # Get latest assessment
        latest_assessment = risk_analyzer.get_last_assessment()
        
        if latest_assessment:
            # Display regime info
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Current Regime")
                st.write(f"**Regime:** {latest_assessment.regime}")
                st.write(f"**Risk Level:** {latest_assessment.risk_level}")
                st.write(f"**Forecast Risk Score:** {latest_assessment.forecast_risk_score:.2f}")
            
            with col2:
                st.subheader("Regime Metrics")
                for metric, value in latest_assessment.metrics.items():
                    if isinstance(value, float):
                        st.write(f"**{metric}:** {value:.2f}")
                    else:
                        st.write(f"**{metric}:** {value}")
            
            # Display recommendations
            st.subheader("Recommendations")
            for rec in latest_assessment.recommendations:
                st.write(f"- {rec}")

# Risk Summary Tab
with tab3:
    st.header("Risk Summary")
    
    if not metrics_df.empty:
        # Display risk explanation
        latest_assessment = risk_analyzer.get_last_assessment()
        
        if latest_assessment:
            st.subheader("Risk Analysis")
            st.write(latest_assessment.explanation)
            
            # Export options
            st.subheader("Export Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Export CSV"):
                    csv = metrics_df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        "risk_metrics.csv",
                        "text/csv"
                    )
            
            with col2:
                if st.button("Export JSON"):
                    json_str = metrics_df.to_json(orient="records", indent=2)
                    st.download_button(
                        "Download JSON",
                        json_str,
                        "risk_metrics.json",
                        "application/json"
                    )
            
            with col3:
                if st.button("Export Charts"):
                    # Save charts as HTML
                    fig1 = plot_risk_metrics(metrics_df)
                    fig2 = plot_drawdown_heatmap(metrics_df)
                    
                    fig1.write_html("risk_metrics.html")
                    fig2.write_html("drawdown_heatmap.html")
                    
                    st.success("Charts exported as HTML files")

# Footer
st.markdown("---")
st.markdown(
    "Risk Dashboard | Last Updated: " +
    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
) 