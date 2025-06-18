"""
Strategy History Page

This page provides visualization of strategy selection history and performance metrics,
including timeframe analysis, model comparisons, and performance breakdowns.
It supports global state tracking for ticker/model selection and allows optional GPT-based summaries for agentic insights.
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
from llm.llm_summary import generate_strategy_commentary

# Constants
LOG_PATH = "memory/strategy_override_log.json"
PERF_LOG = "memory/performance_log.json"

def initialize_session_state():
    """Ensure global state for ticker/model selection is initialized."""
    if "selected_ticker" not in st.session_state:
        st.session_state.selected_ticker = None
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None
    if "date_range" not in st.session_state:
        st.session_state.date_range = None

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
    """Main entry point for the strategy history dashboard."""
    st.set_page_config(page_title="Strategy History", layout="wide")
    st.title("Strategy History")

    initialize_session_state()
    df = load_strategy_log()
    if df is None:
        st.warning("No strategy log data found.")
        return

    # Use global state for ticker selection
    tickers = df["ticker"].unique()
    selected_ticker = st.selectbox("📈 Select Ticker", tickers, index=tickers.tolist().index(st.session_state.selected_ticker) if st.session_state.selected_ticker in tickers else 0)
    st.session_state.selected_ticker = selected_ticker

    # Filter data for selected ticker
    df = df[df["ticker"] == selected_ticker]

    # Use global state for date range
    min_date, max_date = df["timestamp"].min(), df["timestamp"].max()
    date_range = st.date_input("📅 Date Filter", [min_date, max_date])
    st.session_state.date_range = date_range
    if len(date_range) == 2:
        df = df[(df["timestamp"] >= pd.to_datetime(date_range[0])) & 
                (df["timestamp"] <= pd.to_datetime(date_range[1]))]

    # ==== Tabs ====
    tab1, tab2, tab3, tab4 = st.tabs(["📉 Metrics", "📊 Leaderboard", "📈 Regime Analysis", "📁 Export"])

    with tab1:
        st.subheader("📉 Performance Metrics")
        selected_metric = st.selectbox("Select Metric", ["sharpe", "accuracy", "win_rate"])
        fig = plot_metric_timeseries(df, selected_metric, "All Time")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("📊 Performance Leaderboard")
        top_models = get_top_models(df)
        st.write("Top Models:", top_models)
        leaderboard_df = df.groupby("model").agg({
            "sharpe": "mean",
            "accuracy": "mean",
            "win_rate": "mean"
        }).round(3)
        st.dataframe(leaderboard_df)

    with tab3:
        st.subheader("📈 Regime Analysis")
        regime = st.selectbox("Select Regime", df["regime"].unique())
        fig = plot_regime_comparison(df, regime)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("📁 Export Report")
        if st.button("Generate PDF Report"):
            pdf_bytes = generate_pdf_report(df, "All Time", selected_metric, top_models, leaderboard_df)
            st.download_button(
                "⬇ Download PDF",
                pdf_bytes,
                file_name=f"{selected_ticker}_strategy_report.pdf",
                mime="application/pdf"
            )

    # ==== AI Commentary (GPT-based, optional) ====
    st.subheader("🧠 AI Strategy Commentary")
    show_gpt = st.checkbox("Generate GPT-based summary", value=False)
    if show_gpt:
        st.markdown("```text\n" + generate_strategy_commentary(df) + "\n```")
    else:
        st.info("Enable the checkbox to generate a GPT-based summary of strategy performance.")

if __name__ == "__main__":
    main() 