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
from typing import Dict, Any, List, Optional, Tuple, Union
import json
import os

# FIXME: Define logger
import logging
logger = logging.getLogger(__name__)

from trading.agents.strategy_switcher import StrategySwitcher
from models.forecast_router import ForecastRouter
from trading.memory.strategy_logger import log_strategy_decision, get_strategy_analysis
from trading.memory.performance_logger import log_strategy_performance
from trading.memory.model_monitor import get_model_trust_levels
from trading.memory.performance_weights import get_latest_weights
from trading.models.base_model import ModelRegistry
from trading.data.data_loader import load_market_data
from trading.utils.visualization import plot_forecast, plot_attention_heatmap, plot_shap_values, plot_model_components, plot_model_comparison, plot_performance_over_time, plot_backtest_results
from trading.utils.metrics import calculate_metrics
from trading.utils.system_status import get_system_status
from src.analysis.market_analysis import MarketAnalysis

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "selected_ticker" not in st.session_state:
        st.session_state.selected_ticker = "AAPL"
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None
    if "forecast_data" not in st.session_state:
        st.session_state.forecast_data = None
    if "market_analysis" not in st.session_state:
        st.session_state.market_analysis = None
    if "start_date" not in st.session_state:
        st.session_state.start_date = datetime.now().date()
    if "end_date" not in st.session_state:
        st.session_state.end_date = datetime.now().date()
    if "last_updated" not in st.session_state:
        st.session_state.last_updated = datetime.now()

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

def get_status_badge(status):
    """Get HTML badge for system status."""
    colors = {
        "operational": "green",
        "degraded": "orange",
        "down": "red"
    }
    return f'<span style="color: {colors[status]}; font-weight: bold;">●</span> {status.title()}'

def analyze_market_context(ticker: str, data: pd.DataFrame) -> Dict:
    """
    Analyze market context using the MarketAnalysis module.
    
    Args:
        ticker: Ticker symbol
        data: Market data DataFrame
        
    Returns:
        Dictionary with market analysis results
    """
    try:
        market_analyzer = MarketAnalysis()
        analysis = market_analyzer.analyze_market(data)
        
        # Log the analysis
        st.session_state.market_analysis = analysis
        
        return analysis
        
    except Exception as e:
        st.error(f"Error analyzing market context: {str(e)}")
        return {}

def display_market_analysis(analysis: Dict):
    """
    Display market analysis results in the UI.
    
    Args:
        analysis: Market analysis results
    """
    if not analysis:
        return
    
    st.subheader("📊 Market Context Analysis")
    
    # Market Regime
    if 'regime' in analysis:
        regime = analysis['regime']
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Market Regime", regime.name)
        with col2:
            st.metric("Confidence", f"{regime.confidence:.1%}")
        with col3:
            st.metric("Trend Strength", f"{regime.metrics.get('trend_strength', 0):.2f}")
        
        st.info(f"**{regime.name}**: {regime.description}")
    
    # Market Conditions
    if 'conditions' in analysis and analysis['conditions']:
        st.subheader("Market Conditions")
        
        for condition in analysis['conditions']:
            with st.expander(f"{condition.name} (Strength: {condition.strength:.1%})"):
                st.write(condition.description)
                
                # Display indicators
                if condition.indicators:
                    st.write("**Key Indicators:**")
                    for indicator, value in condition.indicators.items():
                        if isinstance(value, (int, float)):
                            st.write(f"- {indicator}: {value:.4f}")
                        else:
                            st.write(f"- {indicator}: {value}")
                
                # Display signals
                if condition.signals:
                    st.write("**Signals:**")
                    for signal, value in condition.signals.items():
                        st.write(f"- {signal}: {value}")
    
    # Trading Signals
    if 'signals' in analysis:
        st.subheader("Trading Signals")
        
        signal_categories = ['trend', 'momentum', 'volatility', 'volume', 'support_resistance']
        
        for category in signal_categories:
            if category in analysis['signals']:
                signals = analysis['signals'][category]
                if signals:
                    with st.expander(f"{category.title()} Signals"):
                        for signal_name, signal_data in signals.items():
                            if isinstance(signal_data, dict):
                                st.write(f"**{signal_name}:**")
                                for key, value in signal_data.items():
                                    st.write(f"  - {key}: {value}")
                            else:
                                st.write(f"**{signal_name}:** {signal_data}")

def generate_market_commentary(analysis: Dict, forecast_data: pd.DataFrame) -> str:
    """
    Generate market commentary based on analysis and forecast.
    
    Args:
        analysis: Market analysis results
        forecast_data: Forecast data
        
    Returns:
        Generated commentary string
    """
    commentary = []
    
    if 'regime' in analysis:
        regime = analysis['regime']
        commentary.append(f"**Market Context**: The market is currently in a {regime.name.lower()} regime with {regime.confidence:.1%} confidence.")
        
        if regime.metrics.get('trend_strength', 0) > 0.7:
            commentary.append("Strong trend conditions suggest following momentum strategies.")
        elif regime.metrics.get('trend_strength', 0) < -0.7:
            commentary.append("Strong downtrend suggests defensive positioning or short opportunities.")
        else:
            commentary.append("Mixed trend conditions suggest range-bound or mean-reversion strategies.")
    
    if 'conditions' in analysis:
        conditions = analysis['conditions']
        if conditions:
            strong_conditions = [c for c in conditions if c.strength > 0.7]
            if strong_conditions:
                commentary.append(f"**Key Conditions**: {len(strong_conditions)} strong market conditions detected.")
    
    # Add forecast-specific commentary
    if not forecast_data.empty:
        latest_forecast = forecast_data.iloc[-1] if len(forecast_data) > 0 else None
        if latest_forecast is not None:
            if 'prediction' in latest_forecast:
                pred = latest_forecast['prediction']
                if pred > 0:
                    commentary.append("**Forecast**: Model predicts upward price movement.")
                else:
                    commentary.append("**Forecast**: Model predicts downward price movement.")
    
    return " ".join(commentary) if commentary else "No market commentary available."

def main():
    st.set_page_config(
        page_title="Forecast & Trade",
        page_icon="📈",
        layout="wide"
    )

    # Get system status
    system_status = get_system_status()
    
    # Header with status and timestamp
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Forecast & Trade")
    with col2:
        st.markdown(
            f"""
            <div style="text-align: right;">
                <p>System Status: {get_status_badge(system_status['status'])}</p>
                <p>Last Updated: {st.session_state.last_updated.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("Generate forecasts and execute trades based on model predictions with market context analysis.")

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
            st.session_state.market_analysis = None

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
            st.session_state.market_analysis = None

        # Market Analysis Options
        st.subheader("Market Analysis")
        enable_market_analysis = st.checkbox("Enable Market Context Analysis", value=True)
        show_market_commentary = st.checkbox("Show Market Commentary", value=True)

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
            strategy_switcher = StrategySwitcher()
            # For now, use a simple model selection - in a real implementation, 
            # you would pass actual metrics to the strategy switcher
            selected_model = "LSTM"  # Default model
            confidence = 0.8  # Default confidence
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
                # Load market data for analysis
                try:
                    market_data = load_market_data(ticker, start_date, end_date)
                    
                    # Perform market analysis if enabled
                    if enable_market_analysis:
                        with st.spinner("Analyzing market context..."):
                            market_analysis = analyze_market_context(ticker, market_data)
                            st.session_state.market_analysis = market_analysis
                            st.success("Market analysis completed!")
                except Exception as e:
                    st.warning(f"Could not load market data for analysis: {str(e)}")
                    market_analysis = {}
                
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
        
        # Market Analysis Section
        if enable_market_analysis and st.session_state.market_analysis:
            display_market_analysis(st.session_state.market_analysis)
            
            if show_market_commentary:
                commentary = generate_market_commentary(
                    st.session_state.market_analysis, 
                    st.session_state.forecast_data
                )
                st.info(commentary)
        
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

def generate_forecast(ticker, selected_model):
    """Generate forecast for selected ticker and model."""
    try:
        # Update last updated timestamp
        st.session_state.last_updated = datetime.now()
        
        # Load market data
        data = load_market_data(ticker)
        if data is None:
            st.error(f"Failed to load data for {ticker}")
            return None
            
        # Initialize router
        router = ForecastRouter()
        
        # Get forecast
        forecast = router.get_forecast(
            data=data,
            model_type=selected_model,
            forecast_horizon=30
        )
        
        return forecast
        
    except Exception as e:
        st.error(f"Error generating forecast: {str(e)}")
        return None

if __name__ == "__main__":
    main() 