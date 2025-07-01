"""
Strategy History Dashboard

This page provides comprehensive analysis of strategy performance history,
including agentic vs manual decision tracking and performance comparisons.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import io
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import shared utilities
from core.session_utils import (
    initialize_session_state, 
    safe_session_get,
    safe_session_set,
    update_last_updated
)

# Import FPDF for PDF generation
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False
    st.warning("FPDF not available - PDF export will be disabled")

# Constants
LOG_PATH = "memory/strategy_override_log.json"
PERF_LOG = "memory/performance_log.json"

def load_strategy_log():
    """Load and process strategy log data."""
    if not os.path.exists(LOG_PATH):

    with open(LOG_PATH, "r") as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Process timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["agentic_label"] = df["agentic"].map({True: "Agentic", False: "Manual"})
    
    # Add derived fields if not present
    if "sector" not in df.columns:
        df["sector"] = "Unknown"
    if "asset_class" not in df.columns:
        df["asset_class"] = "Unknown"
    if "regime" not in df.columns:
        df["regime"] = "Unknown"
    
    return df

def get_top_models(df):
    """Get top performing models for each metric."""
    # Group by model and calculate mean metrics
    metrics_df = df.groupby("model").agg({
        "sharpe": "mean",
        "accuracy": "mean",
        "win_rate": "mean"
    }).round(3)
    
    # Get top model for each metric
    top_models = {}
    for metric in ["sharpe", "accuracy", "win_rate"]:
        top = metrics_df.sort_values(metric, ascending=False).iloc[0]
        top_models[metric] = f"{top.name} ({top[metric]:.3f})"
    
    return top_models

def plot_metric_timeseries(df, metric, timeframe):
    """Create interactive time series plot for a specific metric."""
    # Create figure
    fig = go.Figure()
    
    # Add traces for each model and agentic status
    for model_name in df["model"].unique():
        for agentic in [True, False]:
            mask = (df["model"] == model_name) & (df["agentic"] == agentic)
            subset = df[mask]
            
            if not subset.empty:
                fig.add_trace(go.Scatter(
                    x=subset["timestamp"],
                    y=subset[metric],
                    name=f"{model_name} ({'Agentic' if agentic else 'Manual'})",
                    mode="lines+markers",
                    hovertemplate="<b>%{x}</b><br>" +
                                f"{metric.title()}: %{{y:.2f}}<br>" +
                                "<extra></extra>"
                ))
    
    # Update layout
    fig.update_layout(
        title=f"{metric.title()} Over Time ({timeframe})",
        xaxis_title="Date",
        yaxis_title=metric.title(),
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def plot_metric_distribution(df, metric, timeframe):
    """Create distribution plot for a specific metric."""
    # Create figure
    fig = go.Figure()
    
    # Add box plots for each model
    for model_name in df["model"].unique():
        model_data = df[df["model"] == model_name]
        
        fig.add_trace(go.Box(
            y=model_data[metric],
            name=model_name,
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8,
            hovertemplate="<b>%{x}</b><br>" +
                         f"{metric.title()}: %{{y:.2f}}<br>" +
                         "<extra></extra>"
        ))
    
    # Update layout
    fig.update_layout(
        title=f"{metric.title()} Distribution by Model ({timeframe})",
        yaxis_title=metric.title(),
        showlegend=False,
        boxmode="group"
    )
    
    return fig

def generate_pdf_report(df, timeframe, selected_metric, top_models, leaderboard_df):
    """Generate a PDF report with strategy analysis."""
    if not FPDF_AVAILABLE:
        st.error("FPDF not available - cannot generate PDF report")

    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    
    # Add title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Strategy Performance Report ({timeframe})", ln=True, align="C")
    pdf.ln(10)
    
    # Add timestamp
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(10)
    
    # Add top models
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Top Performing Models", ln=True)
    pdf.set_font("Arial", "", 10)
    for metric, model in top_models.items():
        pdf.cell(0, 10, f"- {metric.title()}: {model}", ln=True)
    pdf.ln(10)
    
    # Add leaderboard table
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Performance Leaderboard", ln=True)
    
    # Table headers
    pdf.set_font("Arial", "B", 10)
    headers = leaderboard_df.columns
    col_width = pdf.w / len(headers)
    for header in headers:
        pdf.cell(col_width, 10, str(header), 1)
    pdf.ln()
    
    # Table data
    pdf.set_font("Arial", "", 10)
    for _, row in leaderboard_df.iterrows():
        for value in row:
            pdf.cell(col_width, 10, f"{value:.3f}", 1)
        pdf.ln()
    pdf.ln(10)
    
    # Add time series plot
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"{selected_metric.title()} Over Time", ln=True)
    
    # Create and save plot
    fig = plot_metric_timeseries(df, selected_metric, timeframe)
    img_bytes = fig.to_image(format="png")
    img = Image.open(io.BytesIO(img_bytes))
    img_path = "temp_plot.png"
    img.save(img_path)
    
    # Add plot to PDF
    pdf.image(img_path, x=10, y=None, w=190)
    os.remove(img_path)
    
    return pdf.output(dest="S").encode("latin1")

def plot_regime_comparison(df, regime):
    """Create comparison plots for agentic vs manual performance in a regime."""
    # Filter data for regime
    regime_data = df[df["regime"] == regime]
    
    # Calculate metrics
    metrics = ["sharpe", "accuracy", "win_rate"]
    comparison_data = []
    
    for metric in metrics:
        agentic_mean = regime_data[regime_data["agentic"]][metric].mean()
        manual_mean = regime_data[~regime_data["agentic"]][metric].mean()
        comparison_data.append({
            "metric": metric,
            "Agentic": agentic_mean,
            "Manual": manual_mean
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create bar plot
    fig = px.bar(
        comparison_df.melt(id_vars="metric", var_name="Decision Type", value_name="Value"),
        x="metric",
        y="Value",
        color="Decision Type",
        barmode="group",
        title=f"Performance Comparison in {regime} Regime"
    )
    
    return fig

def plot_sector_performance(df, group_by):
    """Create sector/asset class performance plots."""
    if group_by not in df.columns:
        st.warning(f"Column '{group_by}' not found in data")

    # Group by sector/asset class and calculate mean metrics
    sector_metrics = df.groupby(group_by).agg({
        "sharpe": "mean",
        "accuracy": "mean",
        "win_rate": "mean"
    }).round(3)
    
    # Create heatmap
    fig = px.imshow(
        sector_metrics,
        title=f"Performance Heatmap by {group_by.title()}",
        aspect="auto",
        color_continuous_scale="RdYlGn"
    )
    
    return fig

def main():
    """Main entry point for the Strategy History dashboard."""
    st.set_page_config(
        page_title="Strategy History",
        page_icon="⚡",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Update last updated timestamp
    update_last_updated()
    
    st.title("⚡ Strategy History")
    st.markdown("Comprehensive analysis of strategy performance history and decision tracking.")

    # Load data
    df = load_strategy_log()
    if df is None:
        st.error("No strategy log data found. Please run some strategies first.")
        return

    # Sidebar filters
    st.sidebar.header("🔧 Filters")
    
    # Timeframe filter
    timeframe_options = ["All Time", "Last 30 Days", "Last 90 Days", "Last 6 Months", "Last Year"]
    timeframe = st.sidebar.selectbox("📅 Timeframe", timeframe_options, index=0)
    
    # Apply timeframe filter
    if timeframe != "All Time":
        days_map = {
            "Last 30 Days": 30,
            "Last 90 Days": 90,
            "Last 6 Months": 180,
            "Last Year": 365
        }
        cutoff_date = datetime.now() - timedelta(days=days_map[timeframe])
        df = df[df["timestamp"] >= cutoff_date]

    # Model filter
    models = df["model"].unique()
    selected_models = st.sidebar.multiselect("🎯 Models", models, default=list(models))
    df = df[df["model"].isin(selected_models)]

    # Agentic filter
    agentic_filter = st.sidebar.selectbox("🤖 Decision Type", ["All", "Agentic", "Manual"])
    if agentic_filter != "All":
        agentic_bool = agentic_filter == "Agentic"
        df = df[df["agentic"] == agentic_bool]

    # Main content
    if df.empty:
        st.warning("No data available for the selected filters.")

    # Top models summary
    st.subheader("🏆 Top Performing Models")
    top_models = get_top_models(df)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Sharpe", top_models["sharpe"])
    with col2:
        st.metric("Best Accuracy", top_models["accuracy"])
    with col3:
        st.metric("Best Win Rate", top_models["win_rate"])

    # Performance leaderboard
    st.subheader("📊 Performance Leaderboard")
    leaderboard_df = df.groupby("model").agg({
        "sharpe": "mean",
        "accuracy": "mean",
        "win_rate": "mean",
        "agentic": "sum"
    }).round(3)
    leaderboard_df["total_decisions"] = df.groupby("model").size()
    leaderboard_df["agentic_ratio"] = (leaderboard_df["agentic"] / leaderboard_df["total_decisions"]).round(3)
    leaderboard_df = leaderboard_df.drop("agentic", axis=1)
    
    st.dataframe(leaderboard_df)

    # Metric selection for detailed analysis
    st.subheader("📈 Detailed Analysis")
    selected_metric = st.selectbox("Select Metric", ["sharpe", "accuracy", "win_rate"])

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Time Series", "📊 Distribution", "🔄 Regime Analysis", "📋 Export"])

    with tab1:
        st.plotly_chart(plot_metric_timeseries(df, selected_metric, timeframe), use_container_width=True)

    with tab2:
        st.plotly_chart(plot_metric_distribution(df, selected_metric, timeframe), use_container_width=True)

    with tab3:
        st.subheader("🔄 Regime Analysis")
        
        # Regime comparison
        regimes = df["regime"].unique()
        if len(regimes) > 1:
            selected_regime = st.selectbox("Select Regime", regimes)
            st.plotly_chart(plot_regime_comparison(df, selected_regime), use_container_width=True)
        else:
            st.info("Only one regime found in the data.")

        # Sector/Asset class analysis
        st.subheader("📊 Sector/Asset Class Performance")
        group_by = st.selectbox("Group by", ["sector", "asset_class"])
        sector_fig = plot_sector_performance(df, group_by)
        if sector_fig:
            st.plotly_chart(sector_fig, use_container_width=True)

    with tab4:
        st.subheader("📋 Export Options")
        
        # CSV export
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="⬇ Download CSV",
            data=csv_data,
            file_name=f"strategy_history_{timeframe.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )
        
        # PDF export
        if FPDF_AVAILABLE:
            pdf_data = generate_pdf_report(df, timeframe, selected_metric, top_models, leaderboard_df)
            if pdf_data:
                st.download_button(
                    label="⬇ Download PDF Report",
                    data=pdf_data,
                    file_name=f"strategy_report_{timeframe.lower().replace(' ', '_')}.pdf",
                    mime="application/pdf"
                )
        else:
            st.warning("PDF export not available - FPDF not installed")

    # Summary statistics
    st.subheader("📊 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Decisions", len(df))
    with col2:
        agentic_count = df["agentic"].sum()
        st.metric("Agentic Decisions", f"{agentic_count} ({agentic_count/len(df)*100:.1f}%)")
    with col3:
        avg_sharpe = df["sharpe"].mean()
        st.metric("Avg Sharpe Ratio", f"{avg_sharpe:.3f}")
    with col4:
        avg_accuracy = df["accuracy"].mean()
        st.metric("Avg Accuracy", f"{avg_accuracy:.3f}")

if __name__ == "__main__":
    main() 