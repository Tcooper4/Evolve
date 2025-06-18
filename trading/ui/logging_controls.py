"""Streamlit interface for logging controls."""

import streamlit as st
from typing import Optional, Dict, Any
import json
from pathlib import Path
import pandas as pd

from ..settings import (
    set_log_level,
    set_metric_logging,
    set_metrics_path,
    get_settings,
    METRICS_PATH
)

def render_logging_controls() -> None:
    """Render logging control widgets in Streamlit sidebar."""
    st.sidebar.header("Logging Controls")
    
    # Log level control
    current_level = get_settings()["LOG_LEVEL"]
    new_level = st.sidebar.selectbox(
        "Log Level",
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        index=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].index(current_level)
    )
    
    if new_level != current_level:
        set_log_level(new_level)
        st.sidebar.success(f"Log level set to {new_level}")
    
    # Metric logging toggle
    current_metrics = get_settings()["METRIC_LOGGING_ENABLED"]
    new_metrics = st.sidebar.checkbox(
        "Enable Metric Logging",
        value=current_metrics
    )
    
    if new_metrics != current_metrics:
        set_metric_logging(new_metrics)
        st.sidebar.success(
            "Metric logging " + ("enabled" if new_metrics else "disabled")
        )
    
    # Metrics path
    current_path = get_settings()["METRICS_PATH"]
    new_path = st.sidebar.text_input(
        "Metrics File Path",
        value=current_path
    )
    
    if new_path != current_path:
        set_metrics_path(new_path)
        st.sidebar.success(f"Metrics path set to {new_path}")

def render_metrics_viewer() -> None:
    """Render metrics viewer in main Streamlit area."""
    st.header("Metrics Viewer")
    
    # Load and display metrics
    if Path(METRICS_PATH).exists():
        metrics = []
        with open(METRICS_PATH, "r") as f:
            for line in f:
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        if metrics:
            # Group metrics by type
            metric_types = {}
            for metric in metrics:
                metric_type = metric.get("type", "unknown")
                if metric_type not in metric_types:
                    metric_types[metric_type] = []
                metric_types[metric_type].append(metric)
            
            # Display metrics by type
            for metric_type, type_metrics in metric_types.items():
                st.subheader(metric_type.replace("_", " ").title())
                
                # Convert to DataFrame for better display
                df = pd.DataFrame(type_metrics)
                st.dataframe(df)
                
                # Add charts if applicable
                if metric_type == "llm_metrics":
                    st.line_chart(df.set_index("timestamp")["latency"])
                elif metric_type == "backtest_metrics":
                    st.line_chart(df.set_index("timestamp")[["sharpe_ratio", "win_rate"]])
                elif metric_type == "agent_metrics":
                    st.line_chart(df.set_index("timestamp")["confidence"])
        else:
            st.info("No metrics available")
    else:
        st.warning("Metrics file not found")

def render_log_viewer() -> None:
    """Render log viewer in main Streamlit area."""
    st.header("Log Viewer")
    
    # Load and display logs
    log_file = get_settings()["LOG_FILE"]
    if Path(log_file).exists():
        with open(log_file, "r") as f:
            logs = f.readlines()
        
        if logs:
            # Filter logs by level
            log_level = st.selectbox(
                "Filter by Level",
                ["ALL", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            )
            
            filtered_logs = logs
            if log_level != "ALL":
                filtered_logs = [
                    log for log in logs
                    if f"[{log_level}]" in log
                ]
            
            # Display logs
            st.text_area("Logs", "".join(filtered_logs), height=400)
        else:
            st.info("No logs available")
    else:
        st.warning("Log file not found")

def main() -> None:
    """Main Streamlit app."""
    st.title("Trading System Logging Controls")
    
    # Render controls in sidebar
    render_logging_controls()
    
    # Render viewers in main area
    tab1, tab2 = st.tabs(["Metrics", "Logs"])
    
    with tab1:
        render_metrics_viewer()
    
    with tab2:
        render_log_viewer()

if __name__ == "__main__":
    main() 