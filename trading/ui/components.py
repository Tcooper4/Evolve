"""Shared UI components for the trading dashboard."""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np

def create_prompt_input() -> str:
    """Create the top-level prompt input bar.
    
    Returns:
        str: User's prompt input
    """
    return st.text_input(
        "Enter your trading request",
        placeholder="e.g., 'Forecast AAPL for the next 30 days' or 'Run RSI strategy on MSFT'",
        key="prompt_input"
    )

def create_sidebar() -> Dict[str, any]:
    """Create the sidebar with model and strategy controls.
    
    Returns:
        Dict containing selected options
    """
    with st.sidebar:
        st.title("Trading Controls")
        
        # Model Selection
        st.subheader("Model")
        model = st.selectbox(
            "Select Model",
            ["ARIMA", "LSTM", "XGBoost", "Ensemble"],
            key="model_selector"
        )
        
        # Strategy Selection
        st.subheader("Strategy")
        strategies = st.multiselect(
            "Select Strategies",
            ["RSI", "MACD", "SMA", "Bollinger", "Custom"],
            key="strategy_selector"
        )
        
        # Expert Mode
        expert_mode = st.toggle("Expert Mode", key="expert_mode")
        
        # Data Source
        st.subheader("Data Source")
        data_source = st.radio(
            "Select Data Source",
            ["Live", "Upload"],
            key="data_source"
        )
        
        if data_source == "Upload":
            uploaded_file = st.file_uploader(
                "Upload CSV file",
                type=["csv"],
                key="data_upload"
            )
        
        return {
            "model": model,
            "strategies": strategies,
            "expert_mode": expert_mode,
            "data_source": data_source,
            "uploaded_file": uploaded_file if data_source == "Upload" else None
        }

def create_forecast_chart(
    historical_data: pd.DataFrame,
    forecast_data: pd.DataFrame,
    title: str = "Price Forecast"
) -> go.Figure:
    """Create an interactive forecast chart.
    
    Args:
        historical_data: Historical price data
        forecast_data: Forecast data with confidence intervals
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    
    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=historical_data.index,
            y=historical_data['close'],
            name='Historical',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Add forecast
    fig.add_trace(
        go.Scatter(
            x=forecast_data.index,
            y=forecast_data['forecast'],
            name='Forecast',
            line=dict(color='red', dash='dash')
        ),
        row=1, col=1
    )
    
    # Add confidence intervals
    fig.add_trace(
        go.Scatter(
            x=forecast_data.index,
            y=forecast_data['upper'],
            name='Upper Bound',
            line=dict(color='gray', dash='dot'),
            showlegend=False
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_data.index,
            y=forecast_data['lower'],
            name='Lower Bound',
            line=dict(color='gray', dash='dot'),
            fill='tonexty',
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price',
        showlegend=True,
        height=600
    )
    
    return fig

def create_strategy_chart(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    title: str = "Strategy Signals"
) -> go.Figure:
    """Create an interactive strategy chart.
    
    Args:
        data: Price data
        signals: Trading signals
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    
    # Add price data
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['close'],
            name='Price',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Add signals
    fig.add_trace(
        go.Scatter(
            x=signals.index,
            y=signals['signal'],
            name='Signal',
            line=dict(color='red')
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Value',
        showlegend=True,
        height=600
    )
    
    return fig

def create_performance_report(
    results: Dict[str, any],
    title: str = "Performance Report"
) -> None:
    """Create an expandable performance report section.
    
    Args:
        results: Dictionary containing performance metrics
        title: Section title
    """
    with st.expander(title, expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Return", f"{results.get('total_return', 0):.2%}")
            st.metric("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}")
            st.metric("Max Drawdown", f"{results.get('max_drawdown', 0):.2%}")
        
        with col2:
            st.metric("Win Rate", f"{results.get('win_rate', 0):.2%}")
            st.metric("Avg Trade", f"{results.get('avg_trade', 0):.2%}")
            st.metric("Profit Factor", f"{results.get('profit_factor', 0):.2f}")
        
        with col3:
            st.metric("CAGR", f"{results.get('cagr', 0):.2%}")
            st.metric("Volatility", f"{results.get('volatility', 0):.2%}")
            st.metric("Calmar Ratio", f"{results.get('calmar_ratio', 0):.2f}")
        
        # Create equity curve
        if 'equity_curve' in results:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=results['equity_curve'].index,
                    y=results['equity_curve'].values,
                    name='Equity',
                    line=dict(color='blue')
                )
            )
            fig.update_layout(
                title="Equity Curve",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Create drawdown chart
        if 'drawdowns' in results:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=results['drawdowns'].index,
                    y=results['drawdowns'].values,
                    name='Drawdown',
                    fill='tozeroy',
                    line=dict(color='red')
                )
            )
            fig.update_layout(
                title="Drawdowns",
                xaxis_title="Date",
                yaxis_title="Drawdown",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

def create_error_block(message: str) -> None:
    """Create an error message block.
    
    Args:
        message: Error message to display
    """
    st.error(message)
    st.info("Try adjusting your parameters or selecting a different strategy.")

def create_loading_spinner(message: str = "Processing..."):
    """Create a loading spinner context manager.
    
    Args:
        message: Message to display while loading
    """
    return st.spinner(message) 