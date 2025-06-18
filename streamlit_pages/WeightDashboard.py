"""
Model Weight History Dashboard

This Streamlit page provides interactive visualization of model weight history,
including trends, drift detection, and performance correlation.
"""

import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import streamlit as st

def load_weight_history():
    """Load and parse weight history data."""
    if not os.path.exists("memory/weight_history.json"):
        return None
    
    with open("memory/weight_history.json", "r") as f:
        return json.load(f)

def load_audit_log():
    """Load weight audit log data."""
    if not os.path.exists("memory/weight_audit_log.json"):
        return []
    
    with open("memory/weight_audit_log.json", "r") as f:
        return json.load(f)

def main():
    st.title("📈 Model Weight History Dashboard")
    
    # Load data
    data = load_weight_history()
    if not data:
        st.warning("No weight history found.")
        return
    
    # Get unique tickers
    tickers = sorted({t for v in data.values() for t in v})
    ticker = st.selectbox("Select Ticker", tickers)
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Weight Trends", "Drift Analysis", "Audit Log"])
    
    with tab1:
        st.subheader("📊 Weight Trends Over Time")
        
        # Prepare data
        records = {ts: entry[ticker] for ts, entry in data.items() if ticker in entry}
        df = pd.DataFrame.from_dict(records, orient="index").sort_index()
        df.index = pd.to_datetime(df.index)
        
        # Create line plot
        fig = px.line(df, 
                     labels={"value": "Weight", "index": "Timestamp"},
                     title=f"{ticker} Model Weights")
        fig.update_layout(
            xaxis_title="Timestamp",
            yaxis_title="Weight",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show current weights
        st.subheader("Current Weights")
        latest_weights = df.iloc[-1].sort_values(ascending=False)
        fig2 = px.bar(x=latest_weights.index, 
                     y=latest_weights.values,
                     title="Latest Weight Distribution")
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.subheader("🔍 Weight Drift Analysis")
        
        # Calculate weight changes
        weight_changes = df.diff().abs()
        drift_threshold = st.slider("Drift Threshold", 0.01, 0.2, 0.1)
        
        # Plot drift
        fig3 = px.line(weight_changes,
                      title="Weight Changes Over Time")
        fig3.add_hline(y=drift_threshold, line_dash="dash", line_color="red",
                      annotation_text="Drift Threshold")
        st.plotly_chart(fig3, use_container_width=True)
        
        # Show models with significant drift
        significant_drift = weight_changes[weight_changes > drift_threshold].any()
        if significant_drift.any():
            st.warning("Models with significant drift detected:")
            for model in significant_drift[significant_drift].index:
                st.write(f"- {model}")
    
    with tab3:
        st.subheader("📝 Weight Update History")
        
        # Load and filter audit log
        audit_log = load_audit_log()
        ticker_logs = [log for log in audit_log if log.get("ticker") == ticker]
        
        if ticker_logs:
            # Create a DataFrame for the audit log
            audit_df = pd.DataFrame(ticker_logs)
            audit_df["time"] = pd.to_datetime(audit_df["time"])
            audit_df = audit_df.sort_values("time", ascending=False)
            
            # Display audit entries
            for _, entry in audit_df.iterrows():
                with st.expander(f"Update at {entry['time'].strftime('%Y-%m-%d %H:%M:%S')}"):
                    st.write("Weights:", entry["weights"])
                    st.write("Justification:", entry["justification"])
        else:
            st.info("No audit logs found for this ticker.")

if __name__ == "__main__":
    main() 