"""Streamlit dashboard for optimization visualization and control."""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trading.optimization.base_optimizer import BaseOptimizer, OptimizerConfig
from trading.optimization.strategy_selection_agent import StrategySelectionAgent
from trading.optimization.performance_logger import PerformanceLogger, PerformanceMetrics

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add file handler for debug logs
debug_handler = logging.FileHandler('trading/optimization/logs/optimization_debug.log')
debug_handler.setLevel(logging.DEBUG)
debug_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
                results.append({
                    "timestamp": datetime.fromisoformat(data["timestamp"]),
                    "optimizer": data["optimizer"],
                    "strategy": data["strategy"],
                    "sharpe_ratio": data["metrics"]["sharpe_ratio"],
                    "win_rate": data["metrics"]["win_rate"],
                    "max_drawdown": data["metrics"]["max_drawdown"],
                    "mse": data["metrics"]["mse"],
                    "alpha": data["metrics"]["alpha"]
                })
    
    return pd.DataFrame(results)

def plot_optimization_metrics(df: pd.DataFrame) -> go.Figure:
    """Plot optimization metrics.
    
    Args:
        df: DataFrame with optimization results
        
    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Sharpe Ratio", "Win Rate",
                       "Max Drawdown", "Alpha")
    )
    
    # Sharpe Ratio
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["sharpe_ratio"],
            mode="lines+markers",
            name="Sharpe Ratio"
        ),
        row=1, col=1
    )
    
    # Win Rate
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["win_rate"],
            mode="lines+markers",
            name="Win Rate"
        ),
        row=1, col=2
    )
    
    # Max Drawdown
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["max_drawdown"],
            mode="lines+markers",
            name="Max Drawdown"
        ),
        row=2, col=1
    )
    
    # Alpha
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["alpha"],
            mode="lines+markers",
            name="Alpha"
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Optimization Metrics Over Time"
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
    strategy_metrics = df.groupby("strategy").agg({
        "sharpe_ratio": "mean",
        "win_rate": "mean",
        "max_drawdown": "mean",
        "alpha": "mean"
    }).reset_index()
    
    # Create radar chart
    fig = go.Figure()
    
    for _, row in strategy_metrics.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[
                row["sharpe_ratio"],
                row["win_rate"],
                1 - row["max_drawdown"],  # Invert drawdown
                row["alpha"]
            ],
            theta=["Sharpe Ratio", "Win Rate", "Drawdown", "Alpha"],
            fill="toself",
            name=row["strategy"]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title_text="Strategy Comparison"
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
                params.append({
                    "timestamp": datetime.fromisoformat(data["timestamp"]),
                    "strategy": data["strategy"],
                    **data["params"]
                })
    
    params_df = pd.DataFrame(params)
    
    # Create subplots
    n_params = len(params_df.columns) - 2  # Exclude timestamp and strategy
    fig = make_subplots(rows=n_params, cols=1)
    
    # Plot each parameter
    for i, param in enumerate(params_df.columns[2:], 1):
        fig.add_trace(
            go.Scatter(
                x=params_df["timestamp"],
                y=params_df[param],
                mode="lines+markers",
                name=param
            ),
            row=i, col=1
        )
    
    fig.update_layout(
        height=300 * n_params,
        showlegend=True,
        title_text="Parameter Changes Over Time"
    )
    
    return fig

def main():
    """Main function."""
    st.set_page_config(
        page_title="Optimization Dashboard",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("Optimization Dashboard")
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Date range
    st.sidebar.subheader("Date Range")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=start_date
    )
    end_date = st.sidebar.date_input(
        "End Date",
        value=end_date
    )
    
    # Strategy filter
    st.sidebar.subheader("Strategy Filter")
    all_strategies = ["All"] + list(load_optimization_results()["strategy"].unique())
    selected_strategy = st.sidebar.selectbox(
        "Select Strategy",
        options=all_strategies
    )
    
    # Load results
    results_df = load_optimization_results()
    
    if results_df.empty:
        st.warning("No optimization results found.")
        return
    
    # Filter results
    if selected_strategy != "All":
        results_df = results_df[results_df["strategy"] == selected_strategy]
    
    results_df = results_df[
        (results_df["timestamp"] >= pd.Timestamp(start_date)) &
        (results_df["timestamp"] <= pd.Timestamp(end_date))
    ]
    
    # Display metrics
    st.header("Optimization Metrics")
    metrics_fig = plot_optimization_metrics(results_df)
    st.plotly_chart(metrics_fig, use_container_width=True)
    
    # Strategy comparison
    st.header("Strategy Comparison")
    comparison_fig = plot_strategy_comparison(results_df)
    st.plotly_chart(comparison_fig, use_container_width=True)
    
    # Parameter changes
    st.header("Parameter Changes")
    params_fig = plot_parameter_changes(results_df)
    st.plotly_chart(params_fig, use_container_width=True)
    
    # Best configurations
    st.header("Best Configurations")
    
    # Group by strategy
    best_configs = results_df.sort_values("sharpe_ratio", ascending=False).groupby("strategy").first()
    
    for strategy, row in best_configs.iterrows():
        st.subheader(strategy)
        
        # Load configuration
        config_file = f"sandbox_results/optimization_results_{row.name.strftime('%Y%m%d_%H%M%S')}.json"
        with open(config_file, "r") as f:
            config = json.load(f)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sharpe Ratio", f"{row['sharpe_ratio']:.2f}")
        col2.metric("Win Rate", f"{row['win_rate']:.2%}")
        col3.metric("Max Drawdown", f"{row['max_drawdown']:.2%}")
        col4.metric("Alpha", f"{row['alpha']:.2f}")
        
        # Display parameters
        st.json(config["params"])

if __name__ == "__main__":
    main() 