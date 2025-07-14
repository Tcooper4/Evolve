"""Streamlit dashboard for optimization visualization and control.

This page provides interactive sliders for major strategies (RSI, MACD, Bollinger, SMA)
and an optional 'Auto-Tune' mode for hyperparameter optimization.
"""

import json
import logging
import os
import sys
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add file handler for debug logs
debug_handler = logging.FileHandler("trading/optimization/logs/optimization_debug.log")
debug_handler.setLevel(logging.DEBUG)
debug_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
debug_handler.setFormatter(debug_formatter)
logger.addHandler(debug_handler)


def load_optimization_results(results_dir: str = "sandbox_results") -> pd.DataFrame:
    """Load optimization results.

    Args:
        results_dir: Directory containing results files

    Returns:
        DataFrame with optimization results
    """
    results = []

    if not os.path.exists(results_dir):
        return pd.DataFrame()

    for filename in os.listdir(results_dir):
        if filename.endswith(".json"):
            with open(os.path.join(results_dir, filename), "r") as f:
                data = json.load(f)
                results.append(
                    {
                        "timestamp": datetime.fromisoformat(data["timestamp"]),
                        "optimizer": data["optimizer"],
                        "strategy": data["strategy"],
                        "sharpe_ratio": data["metrics"]["sharpe_ratio"],
                        "win_rate": data["metrics"]["win_rate"],
                        "max_drawdown": data["metrics"]["max_drawdown"],
                        "mse": data["metrics"]["mse"],
                        "alpha": data["metrics"]["alpha"],
                    }
                )

    return pd.DataFrame(results)


def plot_optimization_metrics(df: pd.DataFrame) -> go.Figure:
    """Plot optimization metrics.

    Args:
        df: DataFrame with optimization results

    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Sharpe Ratio", "Win Rate", "Max Drawdown", "Alpha"),
    )

    # Sharpe Ratio
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["sharpe_ratio"],
            mode="lines+markers",
            name="Sharpe Ratio",
        ),
        row=1,
        col=1,
    )

    # Win Rate
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["win_rate"], mode="lines+markers", name="Win Rate"
        ),
        row=1,
        col=2,
    )

    # Max Drawdown
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["max_drawdown"],
            mode="lines+markers",
            name="Max Drawdown",
        ),
        row=2,
        col=1,
    )

    # Alpha
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["alpha"], mode="lines+markers", name="Alpha"
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=800, showlegend=True, title_text="Optimization Metrics Over Time"
    )

    return fig


def plot_strategy_comparison(df: pd.DataFrame) -> go.Figure:
    """Plot strategy comparison.

    Args:
        df: DataFrame with optimization results

    Returns:
        Plotly figure
    """
    # Group by strategy
    strategy_metrics = (
        df.groupby("strategy")
        .agg(
            {
                "sharpe_ratio": "mean",
                "win_rate": "mean",
                "max_drawdown": "mean",
                "alpha": "mean",
            }
        )
        .reset_index()
    )

    # Create radar chart
    fig = go.Figure()

    for _, row in strategy_metrics.iterrows():
        fig.add_trace(
            go.Scatterpolar(
                r=[
                    row["sharpe_ratio"],
                    row["win_rate"],
                    1 - row["max_drawdown"],
                    row["alpha"],
                ],  # Invert drawdown
                theta=["Sharpe Ratio", "Win Rate", "Drawdown", "Alpha"],
                fill="toself",
                name=row["strategy"],
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title_text="Strategy Comparison",
    )

    return fig


def plot_parameter_changes(df: pd.DataFrame) -> go.Figure:
    """Plot parameter changes over time.

    Args:
        df: DataFrame with optimization results

    Returns:
        Plotly figure
    """
    # Extract parameters
    params = []
    for filename in os.listdir("sandbox_results"):
        if filename.endswith(".json"):
            with open(os.path.join("sandbox_results", filename), "r") as f:
                data = json.load(f)
                params.append(
                    {
                        "timestamp": datetime.fromisoformat(data["timestamp"]),
                        "strategy": data["strategy"],
                        **data["params"],
                    }
                )

    params_df = pd.DataFrame(params)

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=("RSI", "MACD", "Bollinger", "SMA")
    )

    # RSI
    fig.add_trace(
        go.Scatter(
            x=params_df["timestamp"],
            y=params_df["rsi_period"],
            mode="lines+markers",
            name="RSI Period",
        ),
        row=1,
        col=1,
    )

    # MACD
    fig.add_trace(
        go.Scatter(
            x=params_df["timestamp"],
            y=params_df["macd_fast"],
            mode="lines+markers",
            name="MACD Fast",
        ),
        row=1,
        col=2,
    )

    # Bollinger
    fig.add_trace(
        go.Scatter(
            x=params_df["timestamp"],
            y=params_df["bollinger_period"],
            mode="lines+markers",
            name="Bollinger Period",
        ),
        row=2,
        col=1,
    )

    # SMA
    fig.add_trace(
        go.Scatter(
            x=params_df["timestamp"],
            y=params_df["sma_period"],
            mode="lines+markers",
            name="SMA Period",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=800, showlegend=True, title_text="Parameter Changes Over Time"
    )

    return fig


def main():
    """Main entry point for the optimization dashboard."""
    st.set_page_config(page_title="Optimization Dashboard", layout="wide")
    st.title("Optimization Dashboard")

    # Load optimization results
    df = load_optimization_results()
    if df.empty:
        st.warning("No optimization results found.")

    # ==== Sliders for Major Strategies ====
    st.subheader("Strategy Parameters")
    col1, col2 = st.columns(2)
    with col1:
        st.slider("RSI Period", 5, 30, 14)
        st.slider("MACD Fast Period", 5, 30, 12)
    with col2:
        st.slider("Bollinger Period", 5, 30, 20)
        st.slider("SMA Period", 5, 30, 20)

    # ==== Auto-Tune Mode ====
    st.subheader("Auto-Tune Mode")
    auto_tune = st.checkbox("Enable Auto-Tune", value=False)
    if auto_tune:
        st.info(
            "Auto-Tune mode is enabled. The optimizer will automatically tune hyperparameters."
        )
        # Call backend optimizer here
        # optimizer = BaseOptimizer()
        # optimizer.optimize()

    # ==== Tabs ====
    tab1, tab2, tab3 = st.tabs(
        ["📉 Metrics", "📊 Strategy Comparison", "📈 Parameter Changes"]
    )

    with tab1:
        st.subheader("📉 Optimization Metrics")
        fig = plot_optimization_metrics(df)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("📊 Strategy Comparison")
        fig = plot_strategy_comparison(df)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("📈 Parameter Changes")
        fig = plot_parameter_changes(df)
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
