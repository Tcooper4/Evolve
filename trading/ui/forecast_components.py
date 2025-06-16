"""UI components for the forecast page."""

from typing import Dict, Any, Optional
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_forecast_input() -> Optional[str]:
    """Create input form for forecast parameters.
    
    Returns:
        User input for forecast or None if not submitted
    """
    with st.form("forecast_form"):
        # Ticker input
        ticker = st.text_input("Ticker Symbol", "AAPL").upper()
        
        # Timeframe selection
        timeframe = st.selectbox(
            "Forecast Timeframe",
            ["1d", "5d", "1w", "1m", "3m"],
            index=2
        )
        
        # Model selection
        model = st.selectbox(
            "Forecast Model",
            ["LSTM", "TCN", "Transformer"],
            index=0
        )
        
        # Submit button
        submitted = st.form_submit_button("Generate Forecast")
        
        if submitted:
            return {
                "ticker": ticker,
                "timeframe": timeframe,
                "model": model
            }
    return None

def create_forecast_chart(forecast_data: Dict[str, Any]) -> None:
    """Create forecast visualization.
    
    Args:
        forecast_data: Dictionary containing forecast results
    """
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add historical price
    fig.add_trace(
        go.Scatter(
            x=forecast_data["dates"],
            y=forecast_data["historical"],
            name="Historical",
            line=dict(color="blue")
        ),
        secondary_y=False
    )
    
    # Add forecast
    fig.add_trace(
        go.Scatter(
            x=forecast_data["forecast_dates"],
            y=forecast_data["forecast"],
            name="Forecast",
            line=dict(color="red", dash="dash")
        ),
        secondary_y=False
    )
    
    # Add confidence intervals
    fig.add_trace(
        go.Scatter(
            x=forecast_data["forecast_dates"],
            y=forecast_data["upper_bound"],
            fill=None,
            mode="lines",
            line=dict(color="rgba(255,0,0,0.1)"),
            name="Upper Bound"
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=forecast_data["forecast_dates"],
            y=forecast_data["lower_bound"],
            fill="tonexty",
            mode="lines",
            line=dict(color="rgba(255,0,0,0.1)"),
            name="Lower Bound"
        ),
        secondary_y=False
    )
    
    # Update layout
    fig.update_layout(
        title=f"Price Forecast for {forecast_data['ticker']}",
        xaxis_title="Date",
        yaxis_title="Price",
        showlegend=True
    )
    
    # Display chart
    st.plotly_chart(fig, use_container_width=True)

def create_forecast_metrics(forecast_data: Dict[str, Any]) -> None:
    """Display forecast performance metrics.
    
    Args:
        forecast_data: Dictionary containing forecast results
    """
    # Create metrics columns
    col1, col2, col3 = st.columns(3)
    
    # Display metrics
    col1.metric(
        "MSE",
        f"{forecast_data['metrics']['mse']:.4f}",
        delta=f"{forecast_data['metrics']['mse_change']:.4f}"
    )
    
    col2.metric(
        "Accuracy",
        f"{forecast_data['metrics']['accuracy']:.2%}",
        delta=f"{forecast_data['metrics']['accuracy_change']:.2%}"
    )
    
    col3.metric(
        "Confidence",
        f"{forecast_data['metrics']['confidence']:.2%}"
    )

def create_forecast_table(forecast_data: Dict[str, Any]) -> None:
    """Display forecast results in a table.
    
    Args:
        forecast_data: Dictionary containing forecast results
    """
    # Create DataFrame
    df = pd.DataFrame({
        "Date": forecast_data["forecast_dates"],
        "Forecast": forecast_data["forecast"],
        "Lower Bound": forecast_data["lower_bound"],
        "Upper Bound": forecast_data["upper_bound"]
    })
    
    # Format numbers
    for col in ["Forecast", "Lower Bound", "Upper Bound"]:
        df[col] = df[col].round(2)
    
    # Display table
    st.dataframe(df, use_container_width=True)

def create_forecast_explanation(forecast_data: Dict[str, Any]) -> None:
    """Display forecast explanation and insights.
    
    Args:
        forecast_data: Dictionary containing forecast results
    """
    st.subheader("Forecast Analysis")
    
    # Display explanation
    st.markdown(forecast_data["explanation"])
    
    # Display key insights
    st.subheader("Key Insights")
    for insight in forecast_data["insights"]:
        st.markdown(f"- {insight}")
    
    # Display risk factors
    st.subheader("Risk Factors")
    for risk in forecast_data["risks"]:
        st.markdown(f"- {risk}") 