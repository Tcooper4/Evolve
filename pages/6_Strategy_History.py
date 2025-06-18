"""
Strategy History Page

This page provides visualization of strategy selection history and performance metrics,
including timeframe analysis, model comparisons, and performance breakdowns.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from fpdf import FPDF
import io
from PIL import Image

# Constants
LOG_PATH = "memory/strategy_override_log.json"
PERF_LOG = "memory/performance_log.json"

def load_strategy_log():
    """Load and process strategy log data."""
    if not os.path.exists(LOG_PATH):
        return None
    
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
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create figure
    fig = go.Figure()
    
    # Add bars for agentic and manual
    fig.add_trace(go.Bar(
        name="Agentic",
        x=comparison_df["metric"],
        y=comparison_df["Agentic"],
        text=comparison_df["Agentic"].round(3),
        textposition="auto",
    ))
    
    fig.add_trace(go.Bar(
        name="Manual",
        x=comparison_df["metric"],
        y=comparison_df["Manual"],
        text=comparison_df["Manual"].round(3),
        textposition="auto",
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Performance Comparison in {regime} Regime",
        xaxis_title="Metric",
        yaxis_title="Value",
        barmode="group",
        showlegend=True
    )
    
    return fig

def plot_sector_performance(df, group_by):
    """Create performance visualization by sector or asset class."""
    # Group data
    grouped_data = df.groupby(group_by).agg({
        "sharpe": "mean",
        "accuracy": "mean",
        "win_rate": "mean"
    }).round(3)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=grouped_data.values,
        x=grouped_data.columns,
        y=grouped_data.index,
        colorscale="RdYlGn",
        text=[[f"{val:.3f}" for val in row] for row in grouped_data.values],
        texttemplate="%{text}",
        textfont={"size": 14},
        hovertemplate="<b>%{y}</b><br>" +
                     "Metric: %{x}<br>" +
                     "Value: %{z:.3f}<br>" +
                     "<extra></extra>"
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Performance by {group_by.replace('_', ' ').title()}",
        xaxis_title="Metric",
        yaxis_title=group_by.replace("_", " ").title(),
        showlegend=False
    )
    
    return fig

def main():
    st.set_page_config(
        page_title="Strategy History",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("Strategy History")
    st.markdown("View and analyze strategy selection history and performance metrics.")
    
    # Load data
    perf_df = load_strategy_log()
    
    if perf_df is None:
        st.warning("No strategy log found. Please run some strategies first.")
        return
    
    # Timeframe filter
    timeframe = st.selectbox("⏱️ Select Timeframe", ["All", "Last 7 Days", "Last 30 Days", "Last 90 Days"])
    if timeframe != "All":
        days = int(timeframe.split(" ")[1])
        cutoff = pd.Timestamp.utcnow() - timedelta(days=days)
        perf_df = perf_df[perf_df["timestamp"] >= cutoff]
    
    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        
        # Ticker selection
        tickers = sorted(perf_df["ticker"].unique())
        selected_ticker = st.selectbox("Select Ticker", ["All"] + list(tickers))
        
        # Model selection
        models = sorted(perf_df["model"].unique())
        selected_models = st.multiselect(
            "Select Models",
            options=models,
            default=models
        )
        
        # Agentic filter
        agentic_filter = st.multiselect(
            "Strategy Type",
            options=["Agentic", "Manual"],
            default=["Agentic", "Manual"]
        )
        
        # Metric selection
        metrics = ["sharpe", "accuracy", "win_rate"]
        selected_metric = st.selectbox("Select Metric", metrics)
        
        # Group by selection
        group_by = st.selectbox(
            "Group By",
            ["None", "sector", "asset_class"]
        )
        
        # Regime selection
        regimes = sorted(perf_df["regime"].unique())
        selected_regime = st.selectbox(
            "Select Regime",
            ["All"] + list(regimes)
        )
    
    # Apply filters
    filtered_df = perf_df.copy()
    if selected_ticker != "All":
        filtered_df = filtered_df[filtered_df["ticker"] == selected_ticker]
    if selected_models:
        filtered_df = filtered_df[filtered_df["model"].isin(selected_models)]
    if agentic_filter:
        filtered_df = filtered_df[filtered_df["agentic_label"].isin(agentic_filter)]
    if selected_regime != "All":
        filtered_df = filtered_df[filtered_df["regime"] == selected_regime]
    
    # Main content
    if not filtered_df.empty:
        # Summary metrics
        st.subheader("Strategy Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Decisions",
                len(filtered_df)
            )
        with col2:
            st.metric(
                "Agentic Decisions",
                len(filtered_df[filtered_df["agentic"]])
            )
        with col3:
            st.metric(
                "Manual Overrides",
                len(filtered_df[~filtered_df["agentic"]])
            )
        with col4:
            st.metric(
                "Average Confidence",
                f"{filtered_df['confidence'].mean():.2%}"
            )
        
        # Top Models Summary
        st.subheader("🥇 Top Models by Metric")
        top_models = get_top_models(filtered_df)
        st.markdown(f"""
        - **Best Sharpe**: {top_models['sharpe']}  
        - **Best Accuracy**: {top_models['accuracy']}  
        - **Best Win Rate**: {top_models['win_rate']}  
        """)
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Time Series", "Distribution", "Sector Analysis", "Regime Analysis"])
        
        with tab1:
            # Time series plot
            fig = plot_metric_timeseries(filtered_df, selected_metric, timeframe)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Distribution plot
            fig = plot_metric_distribution(filtered_df, selected_metric, timeframe)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            if group_by != "None":
                # Sector/Asset Class performance
                fig = plot_sector_performance(filtered_df, group_by)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select a grouping option in the sidebar to view sector/asset class analysis.")
        
        with tab4:
            if selected_regime != "All":
                # Regime comparison
                fig = plot_regime_comparison(filtered_df, selected_regime)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select a regime in the sidebar to view performance comparison.")
        
        # Export section
        st.subheader("📄 Export Options")
        col1, col2 = st.columns(2)
        
        with col1:
            # Export CSV
            if st.button("📤 Export Data (CSV)"):
                csv = filtered_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    csv,
                    f"strategy_history_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )
        
        with col2:
            # Export PDF
            if st.button("📑 Export Full Report (PDF)"):
                # Generate leaderboard
                leaderboard_df = filtered_df.groupby("model").agg({
                    selected_metric: ["mean", "std", "count"]
                }).round(3)
                
                # Generate PDF
                pdf_bytes = generate_pdf_report(
                    filtered_df,
                    timeframe,
                    selected_metric,
                    top_models,
                    leaderboard_df
                )
                
                # Download button
                st.download_button(
                    "Download PDF Report",
                    pdf_bytes,
                    f"strategy_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    "application/pdf"
                )
    else:
        st.warning("No data available for the selected filters.")

if __name__ == "__main__":
    main() 