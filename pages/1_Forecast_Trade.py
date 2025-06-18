"""
Forecast & Trade Page

This Streamlit page provides interactive forecasting and trading capabilities
with agentic strategy selection and manual override options.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from agents.strategy_switcher import switch_strategy_if_needed, get_best_strategy
from models.forecast_router import ForecastRouter
from memory.strategy_logger import log_strategy_decision, get_strategy_analysis
from memory.performance_logger import log_strategy_performance
from memory.model_monitor import get_model_trust_levels
from memory.performance_weights import get_latest_weights
from trading.models.base_model import ModelRegistry
from trading.data.data_loader import load_market_data
from trading.utils.visualization import plot_forecast, plot_attention_heatmap, plot_shap_values, plot_model_components, plot_model_comparison, plot_performance_over_time, plot_backtest_results
from trading.utils.metrics import calculate_metrics
import json
import os

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "selected_ticker" not in st.session_state:
        st.session_state.selected_ticker = "AAPL"
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None
    if "forecast_data" not in st.session_state:
        st.session_state.forecast_data = None
    if "start_date" not in st.session_state:
        st.session_state.start_date = datetime.now().date()
    if "end_date" not in st.session_state:
        st.session_state.end_date = datetime.now().date()

def load_model_configs():
    """Load model configurations from JSON files."""
    config_dir = 'configs/models'
    configs = {}
    if os.path.exists(config_dir):
        for file in os.listdir(config_dir):
            if file.endswith('.json'):
                with open(os.path.join(config_dir, file), 'r') as f:
                    configs[file[:-5]] = json.load(f)
    return configs

def get_model_summary(model):
    """Get formatted model summary."""
    import io
    from contextlib import redirect_stdout
    
    f = io.StringIO()
    with redirect_stdout(f):
        model.summary()
    return f.getvalue()

def main():
    st.set_page_config(
        page_title="Forecast & Trade",
        page_icon="📈",
        layout="wide"
    )

    st.title("Forecast & Trade")
    st.markdown("Generate forecasts and execute trades based on model predictions.")

    # Initialize session state
    initialize_session_state()

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        
        # Ticker selection
        ticker = st.text_input("Ticker Symbol", value=st.session_state.selected_ticker).upper()
        if ticker != st.session_state.selected_ticker:
            st.session_state.selected_ticker = ticker
            st.session_state.forecast_data = None
            st.session_state.selected_model = None

        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=st.session_state.start_date,
                max_value=datetime.now().date()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=st.session_state.end_date,
                max_value=datetime.now().date()
            )

        if start_date != st.session_state.start_date or end_date != st.session_state.end_date:
            st.session_state.start_date = start_date
            st.session_state.end_date = end_date
            st.session_state.forecast_data = None
            st.session_state.selected_model = None

        # Model selection
        st.subheader("Model Selection")
        use_agentic = st.checkbox("Use Agentic Selection", value=True)
        
        if use_agentic:
            # Get model trust levels
            trust_levels = get_model_trust_levels(ticker)
            if trust_levels:
                st.info("Model Trust Levels:")
                for model, trust in trust_levels.items():
                    st.write(f"{model}: {trust:.2%}")
            
            # Get performance weights
            weights = get_latest_weights(ticker)
            if weights:
                st.info("Model Weights:")
                for model, weight in weights.items():
                    st.write(f"{model}: {weight:.2%}")
            
            # Get best strategy
            selected_model, confidence = get_best_strategy(ticker)
            st.success(f"Selected Model: {selected_model}")
            st.write(f"Confidence: {confidence:.2%}")
            
            # Log strategy decision
            log_strategy_decision(
                ticker=ticker,
                selected_model=selected_model,
                is_agentic=True,
                confidence=confidence
            )
        else:
            # Manual model selection
            selected_model = st.selectbox(
                "Select Model",
                options=["LSTM", "Transformer", "XGBoost", "Ensemble"],
                index=0
            )
            confidence = 1.0  # Manual selection has full confidence
            
            # Log strategy decision
            log_strategy_decision(
                ticker=ticker,
                selected_model=selected_model,
                is_agentic=False,
                confidence=confidence
            )

        # Advanced options
        with st.expander("Advanced Options"):
            # Model comparison
            compare_models = st.checkbox("Compare with Other Models", value=False)
            if compare_models:
                comparison_models = st.multiselect(
                    "Select Models to Compare",
                    options=["LSTM", "Transformer", "XGBoost", "Ensemble"],
                    default=[]
                )
            
            # Visualization options
            show_confidence = st.checkbox("Show Prediction Intervals", value=True)
            show_components = st.checkbox("Show Model Components", value=False)
            show_attention = st.checkbox("Show Attention Heatmap", value=False)
            show_shap = st.checkbox("Show SHAP Values", value=False)
            
            # Backtest options
            initial_capital = st.number_input("Initial Capital", value=10000.0, step=1000.0)
            position_size = st.slider("Position Size (%)", 0, 100, 50)
            stop_loss = st.number_input("Stop Loss (%)", value=2.0, step=0.1)
            take_profit = st.number_input("Take Profit (%)", value=4.0, step=0.1)

        # Generate forecast button
        if st.button("Generate Forecast", type="primary"):
            with st.spinner("Generating forecast..."):
                # Simulate forecast generation
                forecast_data = generate_forecast(ticker, selected_model)
                st.session_state.forecast_data = forecast_data
                st.session_state.selected_model = selected_model
                
                # Calculate performance metrics
                sharpe_ratio = calculate_sharpe_ratio(forecast_data)
                accuracy_score = calculate_accuracy(forecast_data)
                win_rate = calculate_win_rate(forecast_data)
                
                # Log performance metrics
                log_strategy_performance(
                    ticker=ticker,
                    model=selected_model,
                    agentic=use_agentic,
                    metrics={
                        "sharpe": sharpe_ratio,
                        "accuracy": accuracy_score,
                        "win_rate": win_rate
                    }
                )
                
                # Show success message and refresh page
                st.success("Forecast generated and performance logged successfully!")
                st.experimental_rerun()

    # Main content area
    if st.session_state.forecast_data is not None:
        # Display forecast results
        st.subheader("Forecast Results")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "Performance", "Analysis", "Backtest"])
        
        with tab1:
            # Plot forecast
            fig = plot_forecast(
                st.session_state.forecast_data,
                show_confidence=show_confidence
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display forecast metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            with col2:
                st.metric("Accuracy", f"{accuracy_score:.2%}")
            with col3:
                st.metric("Win Rate", f"{win_rate:.2%}")
            
            # Show model components if requested
            if show_components:
                st.subheader("Model Components")
                components_fig = plot_model_components(st.session_state.forecast_data)
                st.plotly_chart(components_fig, use_container_width=True)
            
            # Show attention heatmap if requested
            if show_attention:
                st.subheader("Attention Heatmap")
                attention_fig = plot_attention_heatmap(
                    st.session_state.selected_model,
                    st.session_state.forecast_data
                )
                st.plotly_chart(attention_fig, use_container_width=True)
            
            # Show SHAP values if requested
            if show_shap:
                st.subheader("SHAP Values")
                shap_fig = plot_shap_values(
                    st.session_state.selected_model,
                    st.session_state.forecast_data
                )
                st.plotly_chart(shap_fig, use_container_width=True)
        
        with tab2:
            # Display performance metrics
            st.subheader("Performance Metrics")
            
            # Create performance metrics visualization
            metrics_fig = go.Figure()
            metrics_fig.add_trace(go.Bar(
                x=["Sharpe Ratio", "Accuracy", "Win Rate"],
                y=[sharpe_ratio, accuracy_score, win_rate],
                text=[f"{sharpe_ratio:.2f}", f"{accuracy_score:.2%}", f"{win_rate:.2%}"],
                textposition="auto",
            ))
            metrics_fig.update_layout(
                title="Performance Metrics",
                yaxis_title="Value",
                showlegend=False
            )
            st.plotly_chart(metrics_fig, use_container_width=True)
            
            # Show model comparison if requested
            if compare_models and comparison_models:
                st.subheader("Model Comparison")
                comparison_fig = plot_model_comparison(
                    st.session_state.forecast_data,
                    comparison_models
                )
                st.plotly_chart(comparison_fig, use_container_width=True)
        
        with tab3:
            # Display strategy analysis
            st.subheader("Strategy Analysis")
            
            # Get strategy analysis
            analysis = get_strategy_analysis(ticker)
            if analysis:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Decisions", analysis["total_decisions"])
                    st.metric("Agentic Decisions", analysis["agentic_decisions"])
                with col2:
                    st.metric("Manual Overrides", analysis["manual_overrides"])
                    st.metric("Average Confidence", f"{analysis['avg_confidence']:.2%}")
                
                # Display model usage
                st.subheader("Model Usage")
                model_usage = pd.DataFrame(analysis["model_usage"])
                st.bar_chart(model_usage)
                
                # Display performance over time
                st.subheader("Performance Over Time")
                performance_fig = plot_performance_over_time(analysis["performance_history"])
                st.plotly_chart(performance_fig, use_container_width=True)
            else:
                st.info("No strategy analysis available yet.")
        
        with tab4:
            # Display backtest results
            st.subheader("Backtest Results")
            
            # Run backtest
            backtest_results = run_backtest(
                st.session_state.forecast_data,
                initial_capital=initial_capital,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            # Display backtest metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", f"{backtest_results['total_return']:.2%}")
            with col2:
                st.metric("Sharpe Ratio", f"{backtest_results['sharpe_ratio']:.2f}")
            with col3:
                st.metric("Max Drawdown", f"{backtest_results['max_drawdown']:.2%}")
            with col4:
                st.metric("Win Rate", f"{backtest_results['win_rate']:.2%}")
            
            # Plot backtest results
            backtest_fig = plot_backtest_results(backtest_results)
            st.plotly_chart(backtest_fig, use_container_width=True)
            
            # Display trade history
            st.subheader("Trade History")
            trade_history = pd.DataFrame(backtest_results["trade_history"])
            st.dataframe(trade_history)

    else:
        st.info("Select a ticker and generate a forecast to see results.")

def calculate_sharpe_ratio(forecast_data):
    """Calculate Sharpe ratio from forecast data."""
    # Placeholder implementation
    return np.random.normal(1.5, 0.5)

def calculate_accuracy(forecast_data):
    """Calculate accuracy score from forecast data."""
    # Placeholder implementation
    return np.random.uniform(0.6, 0.8)

def calculate_win_rate(forecast_data):
    """Calculate win rate from forecast data."""
    # Placeholder implementation
    return np.random.uniform(0.5, 0.7)

if __name__ == "__main__":
    main() 