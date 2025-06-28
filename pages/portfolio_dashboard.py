"""Portfolio dashboard for interactive visualization and management."""

import os
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import streamlit as st
from typing import Dict, List, Optional, Any

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trading.portfolio.portfolio_manager import (
    PortfolioManager, PortfolioState, Position, PositionStatus, TradeDirection
)

# Page config
st.set_page_config(
    page_title="Portfolio Dashboard",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'portfolio_manager' not in st.session_state:
    st.session_state.portfolio_manager = PortfolioManager()

# Helper functions
def load_portfolio_state(filename: str) -> None:
    """Load portfolio state from file."""
    try:
        st.session_state.portfolio_manager.load(filename)
        st.success(f"Loaded portfolio state from {filename}")
    except Exception as e:
        st.error(f"Failed to load portfolio state: {e}")

def save_portfolio_state(filename: str) -> None:
    """Save portfolio state to file."""
    try:
        st.session_state.portfolio_manager.save(filename)
        st.success(f"Saved portfolio state to {filename}")
    except Exception as e:
        st.error(f"Failed to save portfolio state: {e}")

def plot_equity_curve(positions: List[Position]) -> go.Figure:
    """Plot equity curve with trade markers.
    
    Args:
        positions: List of positions
        
    Returns:
        Plotly figure
    """
    # Create DataFrame with cumulative PnL
    df = pd.DataFrame([
        {
            'timestamp': p.entry_time,
            'pnl': 0,
            'type': 'entry',
            'symbol': p.symbol,
            'direction': p.direction.value
        } for p in positions
    ] + [
        {
            'timestamp': p.exit_time,
            'pnl': p.pnl,
            'type': 'exit',
            'symbol': p.symbol,
            'direction': p.direction.value
        } for p in positions if p.exit_time is not None
    ])
    
    if df.empty:
        return go.Figure()
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Calculate cumulative PnL
    df['cumulative_pnl'] = df['pnl'].cumsum()
    
    # Create figure
    fig = go.Figure()
    
    # Add equity curve
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['cumulative_pnl'],
        mode='lines',
        name='Equity Curve'
    ))
    
    # Add entry markers
    entries = df[df['type'] == 'entry']
    fig.add_trace(go.Scatter(
        x=entries['timestamp'],
        y=entries['cumulative_pnl'],
        mode='markers',
        marker=dict(
            symbol='triangle-up',
            size=10,
            color='green'
        ),
        name='Entry'
    ))
    
    # Add exit markers
    exits = df[df['type'] == 'exit']
    fig.add_trace(go.Scatter(
        x=exits['timestamp'],
        y=exits['cumulative_pnl'],
        mode='markers',
        marker=dict(
            symbol='triangle-down',
            size=10,
            color='red'
        ),
        name='Exit'
    ))
    
    # Update layout
    fig.update_layout(
        title='Equity Curve',
        xaxis_title='Time',
        yaxis_title='Cumulative PnL',
        showlegend=True
    )
    
    return fig

def plot_rolling_metrics(positions: List[Position], window: int = 20) -> go.Figure:
    """Plot rolling performance metrics.
    
    Args:
        positions: List of positions
        window: Rolling window size
        
    Returns:
        Plotly figure
    """
    # Create DataFrame with daily returns
    df = pd.DataFrame([
        {
            'date': p.exit_time.date(),
            'return': p.pnl / (p.entry_price * p.size)
        } for p in positions if p.exit_time is not None
    ])
    
    if df.empty:
        return go.Figure()
    
    # Calculate rolling metrics
    df['rolling_sharpe'] = df['return'].rolling(window).mean() / df['return'].rolling(window).std() * np.sqrt(252)
    df['rolling_win_rate'] = df['return'].rolling(window).apply(lambda x: (x > 0).mean())
    
    # Create figure
    fig = go.Figure()
    
    # Add rolling Sharpe
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['rolling_sharpe'],
        mode='lines',
        name='Rolling Sharpe'
    ))
    
    # Add rolling win rate
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['rolling_win_rate'],
        mode='lines',
        name='Rolling Win Rate'
    ))
    
    # Update layout
    fig.update_layout(
        title='Rolling Performance Metrics',
        xaxis_title='Time',
        yaxis_title='Value',
        showlegend=True
    )
    
    return fig

def plot_strategy_performance(positions: List[Position]) -> go.Figure:
    """Plot strategy performance comparison.
    
    Args:
        positions: List of positions
        
    Returns:
        Plotly figure
    """
    # Create DataFrame with strategy metrics
    df = pd.DataFrame([
        {
            'strategy': p.strategy,
            'pnl': p.pnl,
            'return': p.pnl / (p.entry_price * p.size) if p.pnl is not None else 0
        } for p in positions if p.exit_time is not None
    ])
    
    if df.empty:
        return go.Figure()
    
    # Calculate strategy metrics
    strategy_metrics = df.groupby('strategy').agg({
        'pnl': ['sum', 'mean', 'std'],
        'return': ['mean', 'std']
    }).reset_index()
    
    strategy_metrics.columns = ['strategy', 'total_pnl', 'mean_pnl', 'std_pnl', 'mean_return', 'std_return']
    strategy_metrics['sharpe'] = strategy_metrics['mean_return'] / strategy_metrics['std_return'] * np.sqrt(252)
    
    # Create figure
    fig = go.Figure()
    
    # Add Sharpe ratio bars
    fig.add_trace(go.Bar(
        x=strategy_metrics['strategy'],
        y=strategy_metrics['sharpe'],
        name='Sharpe Ratio'
    ))
    
    # Add total PnL bars
    fig.add_trace(go.Bar(
        x=strategy_metrics['strategy'],
        y=strategy_metrics['total_pnl'],
        name='Total PnL'
    ))
    
    # Update layout
    fig.update_layout(
        title='Strategy Performance Comparison',
        xaxis_title='Strategy',
        yaxis_title='Value',
        barmode='group',
        showlegend=True
    )
    
    return fig

# Main dashboard
st.title("Portfolio Dashboard")

# Sidebar
st.sidebar.title("Controls")

# Load/Save state
st.sidebar.subheader("Portfolio State")
if st.sidebar.button("Save Portfolio"):
    save_portfolio_state("trading/portfolio/data/portfolio_state.json")

if st.sidebar.button("Load Portfolio"):
    load_portfolio_state("trading/portfolio/data/portfolio_state.json")

# Filters
st.sidebar.subheader("Filters")
selected_strategy = st.sidebar.multiselect(
    "Strategy",
    options=sorted(set(p.strategy for p in st.session_state.portfolio_manager.state.open_positions +
                      st.session_state.portfolio_manager.state.closed_positions))
)

selected_symbol = st.sidebar.multiselect(
    "Symbol",
    options=sorted(set(p.symbol for p in st.session_state.portfolio_manager.state.open_positions +
                      st.session_state.portfolio_manager.state.closed_positions))
)

# Main content
# Portfolio Overview
st.header("Portfolio Overview")

# Get performance summary
summary = st.session_state.portfolio_manager.get_performance_summary()

# Display key metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total PnL", f"{summary.get('total_pnl', 0):.2f}")
col2.metric("Sharpe Ratio", f"{summary.get('sharpe_ratio', 0):.2f}")
col3.metric("Max Drawdown", f"{summary.get('max_drawdown', 0):.2f}")
col4.metric("Win Rate", f"{summary.get('win_rate', 0):.2%}")

# Position Summary
st.header("Position Summary")

# Get position summary
positions_df = st.session_state.portfolio_manager.get_position_summary()

# Apply filters
if selected_strategy:
    positions_df = positions_df[positions_df['strategy'].isin(selected_strategy)]
if selected_symbol:
    positions_df = positions_df[positions_df['symbol'].isin(selected_symbol)]

# Display position table
st.dataframe(positions_df)

# Performance Visualization
st.header("Performance Visualization")

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["Equity Curve", "Rolling Metrics", "Strategy Performance"])

with tab1:
    st.plotly_chart(plot_equity_curve(
        st.session_state.portfolio_manager.state.closed_positions
    ), key="equity_curve")

with tab2:
    window = st.slider("Rolling Window", 5, 100, 20)
    st.plotly_chart(plot_rolling_metrics(
        st.session_state.portfolio_manager.state.closed_positions,
        window=window
    ), key="rolling_metrics")

with tab3:
    st.plotly_chart(plot_strategy_performance(
        st.session_state.portfolio_manager.state.closed_positions
    ), key="strategy_performance")

# Export Options
st.sidebar.subheader("Export Options")
if st.sidebar.button("Export Trade Report"):
    # Create trade report
    report_df = pd.DataFrame([
        {
            'Symbol': p.symbol,
            'Strategy': p.strategy,
            'Direction': p.direction.value,
            'Entry Time': p.entry_time,
            'Entry Price': p.entry_price,
            'Size': p.size,
            'Exit Time': p.exit_time,
            'Exit Price': p.exit_price,
            'PnL': p.pnl,
            'Return': p.pnl / (p.entry_price * p.size) if p.pnl is not None else None
        } for p in st.session_state.portfolio_manager.state.closed_positions
    ])
    
    # Save to CSV
    report_df.to_csv("trading/portfolio/data/trade_report.csv", index=False)
    st.sidebar.success("Exported trade report to CSV") 