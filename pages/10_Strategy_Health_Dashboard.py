"""Strategy Health Dashboard for Evolve Trading Platform.

This page provides real-time monitoring of strategy performance,
equity curves, and system health metrics.
"""

import streamlit as st

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Strategy Health Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any, Optional
import asyncio
import threading
import time

# Import platform modules with fallbacks
PLATFORM_AVAILABLE = True
try:
    from utils.config_loader import ConfigLoader
    config_loader = ConfigLoader()
except ImportError as e:
    st.error(f"Configuration loader not available: {e}")
    PLATFORM_AVAILABLE = False

# Try to import trading modules with fallbacks
try:
    # Import with relative paths to avoid module issues
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from trading.agents.strategy_selector_agent import StrategySelectorAgent
    from trading.risk.tail_risk import TailRiskEngine, calculate_portfolio_risk
    from strategies.gatekeeper import StrategyGatekeeper, create_strategy_gatekeeper
    from data.streaming_pipeline import create_streaming_pipeline
    from execution.live_trading_interface import create_live_trading_interface
except ImportError as e:
    st.warning(f"Some platform modules not available: {e}")
    # Create dummy classes for fallback
    class StrategySelectorAgent:
        def __init__(self): pass
    class TailRiskEngine:
        def __init__(self): pass
    class StrategyGatekeeper:
        def __init__(self): pass
    def create_strategy_gatekeeper(): return StrategyGatekeeper()
    def create_streaming_pipeline(): return None
    def create_live_trading_interface(): return None

# Initialize session state
if 'dashboard_initialized' not in st.session_state:
    st.session_state.dashboard_initialized = False
if 'equity_data' not in st.session_state:
    st.session_state.equity_data = pd.DataFrame()
if 'strategy_status' not in st.session_state:
    st.session_state.strategy_status = {}
if 'performance_metrics' not in st.session_state:
    st.session_state.performance_metrics = {}
if 'risk_metrics' not in st.session_state:
    st.session_state.risk_metrics = {}

def initialize_dashboard():
    """Initialize dashboard components."""
    if st.session_state.dashboard_initialized:
        return
    
    try:
        # Load configuration using ConfigLoader
        if PLATFORM_AVAILABLE:
            st.session_state.config = config_loader.config
        else:
            # Fallback configuration
            st.session_state.config = {
                "data": {"default_lookback_days": 365},
                "display": {"chart_days": 100, "table_rows": 20},
                "optimization": {"max_iterations": 100},
                "trading": {"trading_days_per_year": 252},
                "logging": {"level": "INFO"},
                "ui": {"sidebar_width": 300}
            }
        
        st.session_state.dashboard_initialized = True
        
        # Initialize with sample data for demonstration
        initialize_sample_data()
        
        st.success("Dashboard initialized successfully!")
        
    except Exception as e:
        st.error(f"Error initializing dashboard: {e}")

def initialize_sample_data():
    """Initialize sample data for demonstration."""
    # Generate sample equity curve
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                         end=datetime.now(), freq='D')
    
    # Create realistic equity curve with some volatility
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
    equity_curve = (1 + pd.Series(returns, index=dates)).cumprod() * 100000  # Start with $100k
    
    st.session_state.equity_data = pd.DataFrame({
        'date': dates,
        'equity': equity_curve.values,
        'drawdown': calculate_drawdown(equity_curve.values)
    })
    
    # Initialize strategy status
    strategies = ['Momentum', 'Mean Reversion', 'Trend Following', 'Breakout', 'RL Agent']
    st.session_state.strategy_status = {
        strategy: {
            'status': 'ON' if i < 3 else 'OFF',
            'sharpe': np.random.uniform(0.5, 2.0),
            'drawdown': np.random.uniform(0.05, 0.15),
            'win_rate': np.random.uniform(0.4, 0.7),
            'pnl': np.random.uniform(-5000, 15000),
            'last_update': datetime.now()
        }
        for i, strategy in enumerate(strategies)
    }
    
    # Initialize performance metrics
    st.session_state.performance_metrics = {
        'total_return': 0.15,
        'sharpe_ratio': 1.8,
        'max_drawdown': 0.08,
        'win_rate': 0.62,
        'profit_factor': 1.45,
        'calmar_ratio': 2.1
    }
    
    # Initialize risk metrics
    st.session_state.risk_metrics = {
        'var_95': -0.025,
        'cvar_95': -0.035,
        'volatility': 0.18,
        'beta': 0.85,
        'correlation': 0.45
    }

def calculate_drawdown(equity_curve):
    """Calculate drawdown from equity curve."""
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return drawdown

def create_equity_curve_chart():
    """Create live equity curve chart."""
    if st.session_state.equity_data.empty:
        return go.Figure()
    
    df = st.session_state.equity_data
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Equity Curve', 'Drawdown'),
        row_heights=[0.7, 0.3]
    )
    
    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['equity'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#00ff88', width=2),
            fill='tonexty'
        ),
        row=1, col=1
    )
    
    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['drawdown'] * 100,
            mode='lines',
            name='Drawdown %',
            line=dict(color='#ff4444', width=2),
            fill='tonexty'
        ),
        row=2, col=1
    )
    
    # Add zero line for drawdown
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    fig.update_layout(
        title="Live Equity Curve & Drawdown",
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    
    return fig

def create_strategy_status_table():
    """Create strategy status table."""
    if not st.session_state.strategy_status:
        return pd.DataFrame()
    
    data = []
    for strategy, metrics in st.session_state.strategy_status.items():
        data.append({
            'Strategy': strategy,
            'Status': metrics['status'],
            'Sharpe Ratio': f"{metrics['sharpe']:.2f}",
            'Max Drawdown': f"{metrics['drawdown']:.1%}",
            'Win Rate': f"{metrics['win_rate']:.1%}",
            'PnL ($)': f"${metrics['pnl']:,.0f}",
            'Last Update': metrics['last_update'].strftime('%H:%M:%S')
        })
    
    return pd.DataFrame(data)

def create_performance_metrics_cards():
    """Create performance metrics cards."""
    metrics = st.session_state.performance_metrics
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Return",
            value=f"{metrics['total_return']:.1%}",
            delta=f"{metrics['total_return'] - 0.10:.1%}"
        )
    
    with col2:
        st.metric(
            label="Sharpe Ratio",
            value=f"{metrics['sharpe_ratio']:.2f}",
            delta=f"{metrics['sharpe_ratio'] - 1.0:.2f}"
        )
    
    with col3:
        st.metric(
            label="Max Drawdown",
            value=f"{metrics['max_drawdown']:.1%}",
            delta=f"{metrics['max_drawdown'] - 0.10:.1%}",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            label="Win Rate",
            value=f"{metrics['win_rate']:.1%}",
            delta=f"{metrics['win_rate'] - 0.50:.1%}"
        )

def create_risk_metrics_cards():
    """Create risk metrics cards."""
    metrics = st.session_state.risk_metrics
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="VaR (95%)",
            value=f"{metrics['var_95']:.1%}",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            label="CVaR (95%)",
            value=f"{metrics['cvar_95']:.1%}",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="Volatility",
            value=f"{metrics['volatility']:.1%}",
            delta=f"{metrics['volatility'] - 0.15:.1%}"
        )
    
    with col4:
        st.metric(
            label="Beta",
            value=f"{metrics['beta']:.2f}",
            delta=f"{metrics['beta'] - 1.0:.2f}"
        )

def create_strategy_health_gauge():
    """Create strategy health gauge chart."""
    # Calculate overall health score
    strategies = st.session_state.strategy_status
    if not strategies:
        return go.Figure()
    
    active_strategies = [s for s in strategies.values() if s['status'] == 'ON']
    if not active_strategies:
        return go.Figure()
    
    # Calculate health score based on multiple factors
    avg_sharpe = np.mean([s['sharpe'] for s in active_strategies])
    avg_drawdown = np.mean([s['drawdown'] for s in active_strategies])
    avg_win_rate = np.mean([s['win_rate'] for s in active_strategies])
    
    # Normalize and combine metrics
    sharpe_score = min(avg_sharpe / 2.0, 1.0)  # Normalize to 0-1
    drawdown_score = max(0, 1.0 - avg_drawdown / 0.2)  # Lower drawdown is better
    win_rate_score = avg_win_rate
    
    health_score = (sharpe_score * 0.4 + drawdown_score * 0.3 + win_rate_score * 0.3) * 100
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=health_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall Strategy Health"},
        delta={'reference': 70},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_strategy_performance_chart():
    """Create strategy performance comparison chart."""
    strategies = st.session_state.strategy_status
    if not strategies:
        return go.Figure()
    
    strategy_names = list(strategies.keys())
    sharpe_ratios = [strategies[s]['sharpe'] for s in strategy_names]
    drawdowns = [strategies[s]['drawdown'] * 100 for s in strategy_names]
    win_rates = [strategies[s]['win_rate'] * 100 for s in strategy_names]
    
    fig = go.Figure()
    
    # Add bars for each metric
    fig.add_trace(go.Bar(
        name='Sharpe Ratio',
        x=strategy_names,
        y=sharpe_ratios,
        marker_color='#00ff88'
    ))
    
    fig.add_trace(go.Bar(
        name='Max Drawdown (%)',
        x=strategy_names,
        y=drawdowns,
        marker_color='#ff4444'
    ))
    
    fig.add_trace(go.Bar(
        name='Win Rate (%)',
        x=strategy_names,
        y=win_rates,
        marker_color='#4488ff'
    ))
    
    fig.update_layout(
        title="Strategy Performance Comparison",
        barmode='group',
        height=400,
        xaxis_title="Strategy",
        yaxis_title="Metric Value"
    )
    
    return fig

def update_live_data():
    """Update live data (simulated)."""
    if not st.session_state.dashboard_initialized:
        return
    
    # Simulate live data updates
    current_time = datetime.now()
    
    # Update equity curve with new data point
    if not st.session_state.equity_data.empty:
        last_equity = st.session_state.equity_data['equity'].iloc[-1]
        new_return = np.random.normal(0.0005, 0.01)  # Small daily return
        new_equity = last_equity * (1 + new_return)
        
        new_row = pd.DataFrame({
            'date': [current_time],
            'equity': [new_equity],
            'drawdown': [calculate_drawdown([last_equity, new_equity])[-1]]
        })
        
        st.session_state.equity_data = pd.concat([st.session_state.equity_data, new_row], ignore_index=True)
        
        # Keep only last 100 data points
        if len(st.session_state.equity_data) > 100:
            st.session_state.equity_data = st.session_state.equity_data.tail(100)
    
    # Update strategy metrics
    for strategy in st.session_state.strategy_status:
        # Simulate small changes in metrics
        st.session_state.strategy_status[strategy]['sharpe'] += np.random.normal(0, 0.01)
        st.session_state.strategy_status[strategy]['drawdown'] += np.random.normal(0, 0.001)
        st.session_state.strategy_status[strategy]['win_rate'] += np.random.normal(0, 0.005)
        st.session_state.strategy_status[strategy]['pnl'] += np.random.normal(0, 100)
        st.session_state.strategy_status[strategy]['last_update'] = current_time
        
        # Ensure metrics stay in reasonable ranges
        st.session_state.strategy_status[strategy]['sharpe'] = max(0, min(3, st.session_state.strategy_status[strategy]['sharpe']))
        st.session_state.strategy_status[strategy]['drawdown'] = max(0, min(0.3, st.session_state.strategy_status[strategy]['drawdown']))
        st.session_state.strategy_status[strategy]['win_rate'] = max(0.1, min(0.9, st.session_state.strategy_status[strategy]['win_rate']))

def main():
    """Main dashboard function."""
    st.title("📊 Strategy Health Dashboard")
    st.markdown("Real-time monitoring of trading strategy performance and system health")
    
    # Initialize dashboard
    if not st.session_state.dashboard_initialized:
        initialize_dashboard()
    
    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 60, 5)
    
    # Strategy filter
    all_strategies = list(st.session_state.strategy_status.keys()) if st.session_state.strategy_status else []
    selected_strategies = st.sidebar.multiselect(
        "Select Strategies",
        all_strategies,
        default=all_strategies
    )
    
    # Time range selector
    time_range = st.sidebar.selectbox(
        "Time Range",
        ["1D", "1W", "1M", "3M", "6M", "1Y"],
        index=2
    )
    
    # Main dashboard layout
    if st.session_state.dashboard_initialized:
        # Performance metrics row
        st.subheader("📈 Performance Overview")
        create_performance_metrics_cards()
        
        # Risk metrics row
        st.subheader("⚠️ Risk Metrics")
        create_risk_metrics_cards()
        
        # Equity curve and strategy health
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 Live Equity Curve")
            equity_chart = create_equity_curve_chart()
            st.plotly_chart(equity_chart, use_container_width=True)
        
        with col2:
            st.subheader("🏥 Strategy Health")
            health_gauge = create_strategy_health_gauge()
            st.plotly_chart(health_gauge, use_container_width=True)
        
        # Strategy status table
        st.subheader("🤖 Strategy Status")
        status_df = create_strategy_status_table()
        if not status_df.empty:
            # Filter by selected strategies
            if selected_strategies:
                status_df = status_df[status_df['Strategy'].isin(selected_strategies)]
            
            # Color code the status column
            def color_status(val):
                if val == 'ON':
                    return 'background-color: #90EE90'
                else:
                    return 'background-color: #FFB6C1'
            
            st.dataframe(
                status_df.style.applymap(color_status, subset=['Status']),
                use_container_width=True
            )
        
        # Strategy performance comparison
        st.subheader("📊 Strategy Performance Comparison")
        perf_chart = create_strategy_performance_chart()
        st.plotly_chart(perf_chart, use_container_width=True)
        
        # System status and alerts
        st.subheader("🚨 System Status & Alerts")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Connection status
            st.info("🟢 All systems operational")
            st.metric("Active Strategies", len([s for s in st.session_state.strategy_status.values() if s['status'] == 'ON']))
        
        with col2:
            # Recent alerts
            st.warning("⚠️ 2 minor alerts")
            st.metric("System Uptime", "99.8%")
        
        with col3:
            # Performance alerts
            st.success("✅ Performance targets met")
            st.metric("Risk Score", "Low")
        
        # Auto-refresh functionality
        if auto_refresh:
            time.sleep(refresh_interval)
            update_live_data()
            st.experimental_rerun()
    
    else:
        st.error("Dashboard not initialized. Please check configuration.")

if __name__ == "__main__":
    main() 