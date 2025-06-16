"""UI components for strategy comparison."""

from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_strategy_input() -> Optional[Dict[str, Any]]:
    """Create input form for strategy comparison.
    
    Returns:
        User input for strategy comparison or None if not submitted
    """
    with st.form("strategy_form"):
        # Strategy selection
        strategies = st.multiselect(
            "Select Strategies",
            ["RSI", "MACD", "Bollinger Bands", "Moving Average Crossover"],
            default=["RSI", "MACD"]
        )
        
        # Ticker input
        ticker = st.text_input("Ticker Symbol", "AAPL").upper()
        
        # Timeframe selection
        timeframe = st.selectbox(
            "Timeframe",
            ["1m", "5m", "15m", "1h", "4h", "1d"],
            index=3
        )
        
        # Date range
        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")
        
        # Submit button
        submitted = st.form_submit_button("Compare Strategies")
        
        if submitted:
            return {
                "strategies": strategies,
                "ticker": ticker,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date
            }
    return None

def create_performance_chart(performance_data: Dict[str, Any]) -> None:
    """Create performance comparison chart.
    
    Args:
        performance_data: Dictionary containing strategy performance results
    """
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add cumulative returns for each strategy
    for strategy in performance_data["strategies"]:
        fig.add_trace(
            go.Scatter(
                x=performance_data["dates"],
                y=performance_data["returns"][strategy],
                name=strategy,
                line=dict(width=2)
            ),
            secondary_y=False
        )
    
    # Add benchmark if available
    if "benchmark" in performance_data:
        fig.add_trace(
            go.Scatter(
                x=performance_data["dates"],
                y=performance_data["benchmark"],
                name="Benchmark",
                line=dict(color="gray", dash="dash")
            ),
            secondary_y=False
        )
    
    # Update layout
    fig.update_layout(
        title="Strategy Performance Comparison",
        xaxis_title="Date",
        yaxis_title="Cumulative Returns",
        showlegend=True
    )
    
    # Display chart
    st.plotly_chart(fig, use_container_width=True)

def create_metrics_table(performance_data: Dict[str, Any]) -> None:
    """Display performance metrics in a table.
    
    Args:
        performance_data: Dictionary containing strategy performance results
    """
    # Create DataFrame
    metrics = []
    for strategy in performance_data["strategies"]:
        metrics.append({
            "Strategy": strategy,
            "Total Return": f"{performance_data['metrics'][strategy]['total_return']:.2%}",
            "Sharpe Ratio": f"{performance_data['metrics'][strategy]['sharpe']:.2f}",
            "Max Drawdown": f"{performance_data['metrics'][strategy]['max_drawdown']:.2%}",
            "Win Rate": f"{performance_data['metrics'][strategy]['win_rate']:.2%}",
            "Profit Factor": f"{performance_data['metrics'][strategy]['profit_factor']:.2f}"
        })
    
    df = pd.DataFrame(metrics)
    
    # Display table
    st.dataframe(df, use_container_width=True)

def create_trade_analysis(performance_data: Dict[str, Any]) -> None:
    """Display trade analysis for each strategy.
    
    Args:
        performance_data: Dictionary containing strategy performance results
    """
    for strategy in performance_data["strategies"]:
        st.subheader(f"{strategy} Trade Analysis")
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        # Display trade metrics
        col1.metric(
            "Total Trades",
            performance_data["trades"][strategy]["total"],
            delta=performance_data["trades"][strategy]["total_change"]
        )
        
        col2.metric(
            "Average Trade",
            f"{performance_data['trades'][strategy]['avg_trade']:.2%}",
            delta=f"{performance_data['trades'][strategy]['avg_trade_change']:.2%}"
        )
        
        col3.metric(
            "Best Trade",
            f"{performance_data['trades'][strategy]['best_trade']:.2%}",
            delta=f"{performance_data['trades'][strategy]['best_trade_change']:.2%}"
        )
        
        # Display trade distribution
        st.bar_chart(performance_data["trades"][strategy]["distribution"])

def create_risk_analysis(performance_data: Dict[str, Any]) -> None:
    """Display risk analysis for each strategy.
    
    Args:
        performance_data: Dictionary containing strategy performance results
    """
    st.subheader("Risk Analysis")
    
    # Create DataFrame for risk metrics
    risk_metrics = []
    for strategy in performance_data["strategies"]:
        risk_metrics.append({
            "Strategy": strategy,
            "Volatility": f"{performance_data['risk'][strategy]['volatility']:.2%}",
            "VaR (95%)": f"{performance_data['risk'][strategy]['var_95']:.2%}",
            "CVaR (95%)": f"{performance_data['risk'][strategy]['cvar_95']:.2%}",
            "Beta": f"{performance_data['risk'][strategy]['beta']:.2f}",
            "Correlation": f"{performance_data['risk'][strategy]['correlation']:.2f}"
        })
    
    df = pd.DataFrame(risk_metrics)
    
    # Display table
    st.dataframe(df, use_container_width=True)
    
    # Display drawdown chart
    fig = go.Figure()
    
    for strategy in performance_data["strategies"]:
        fig.add_trace(
            go.Scatter(
                x=performance_data["dates"],
                y=performance_data["risk"][strategy]["drawdown"],
                name=strategy,
                fill="tozeroy"
            )
        )
    
    fig.update_layout(
        title="Strategy Drawdowns",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True) 