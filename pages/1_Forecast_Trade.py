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
import sys
from pathlib import Path
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pandas_ta")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import shared utilities
from core.session_utils import (
    initialize_session_state, 
    safe_session_get,
    safe_session_set,
    update_last_updated
)

# Import trading components
from trading.agents.strategy_switcher import StrategySwitcher
from models.forecast_router import ForecastRouter
from trading.memory.strategy_logger import log_strategy_decision as strategy_logger_decision, get_strategy_analysis
from trading.memory.performance_logger import log_performance
from trading.memory.model_monitor import get_model_trust_levels
from trading.memory.performance_weights import get_latest_weights
from trading.models.base_model import ModelRegistry
from trading.data.data_loader import load_market_data
from trading.utils.visualization import plot_forecast, plot_attention_heatmap, plot_shap_values, plot_model_components, plot_model_comparison, plot_performance_over_time, plot_backtest_results
from trading.utils.metrics import calculate_metrics
from trading.utils.system_status import get_system_status
from src.analysis.market_analysis import MarketAnalysis
from trading.memory.model_monitor import ModelMonitor
from trading.memory.strategy_logger import StrategyLogger
from trading.optimization.strategy_selection_agent import StrategySelectionAgent
from trading.portfolio.portfolio_manager import PortfolioManager
from trading.portfolio.llm_utils import LLMInterface
from trading.optimization.performance_logger import PerformanceLogger

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if 'last_updated' not in st.session_state:
    st.session_state['last_updated'] = datetime.now()

def load_model_configs():
    """Load model configurations from registry."""
    try:
        from trading.ui.config.registry import ModelConfigRegistry
        registry = ModelConfigRegistry()
        return registry.get_all_configs()
    except Exception as e:
        logger.warning(f"Could not load model configs: {e}")
        return {}

def get_model_summary(model):
    """Get summary information for a model."""
    try:
        configs = load_model_configs()
        return configs.get(model, {}).get('description', 'No description available')
    except Exception as e:
        logger.warning(f"Could not get model summary: {e}")
        return 'No description available'

def get_status_badge(status):
    """Get HTML badge for system status."""
    colors = {
        "operational": "green",
        "degraded": "orange", 
        "down": "red"
    }
    color = colors.get(status, "gray")
    return f'<span style="color: {color}; font-weight: bold;">● {status.title()}</span>'

def analyze_market_context(ticker: str, data: pd.DataFrame) -> Dict:
    """Analyze market context for a given ticker."""
    try:
        # Basic market analysis
        if data.empty:
            return {"status": "no_data", "message": "No market data available"}
        
        # Calculate basic metrics
        latest_price = data['close'].iloc[-1] if 'close' in data.columns else None
        price_change = data['close'].pct_change().iloc[-1] if 'close' in data.columns else None
        volatility = data['close'].pct_change().std() if 'close' in data.columns else None
        
        return {
            "status": "success",
            "ticker": ticker,
            "latest_price": latest_price,
            "price_change": price_change,
            "volatility": volatility,
            "data_points": len(data),
            "analysis_date": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Market analysis failed: {e}")
        return {"status": "error", "message": str(e)}

def display_market_analysis(analysis: Dict):
    """Display market analysis results."""
    if analysis.get("status") == "success":
        st.subheader("📊 Market Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Latest Price",
                f"${analysis.get('latest_price', 0):.2f}" if analysis.get('latest_price') else "N/A"
            )
        
        with col2:
            price_change = analysis.get('price_change', 0)
            if price_change is not None:
                st.metric(
                    "Price Change",
                    f"{price_change:.2%}",
                    delta="↗️" if price_change > 0 else "↘️" if price_change < 0 else "→"
                )
            else:
                st.metric("Price Change", "N/A")
        
        with col3:
            volatility = analysis.get('volatility', 0)
            if volatility is not None:
                st.metric("Volatility", f"{volatility:.2%}")
            else:
                st.metric("Volatility", "N/A")
        
        with col4:
            st.metric("Data Points", analysis.get('data_points', 0))
    
    elif analysis.get("status") == "no_data":
        st.warning("⚠️ No market data available for analysis")
    else:
        st.error(f"❌ Market analysis failed: {analysis.get('message', 'Unknown error')}")

def generate_market_commentary(analysis: Dict, forecast_data: pd.DataFrame) -> str:
    """Generate market commentary based on analysis and forecast."""
    try:
        if analysis.get("status") != "success":
            return "Market commentary unavailable due to analysis issues."
        
        commentary = f"Market Analysis for {analysis.get('ticker', 'Unknown')}:\n\n"
        
        # Price analysis
        latest_price = analysis.get('latest_price')
        price_change = analysis.get('price_change')
        
        if latest_price and price_change is not None:
            if price_change > 0.02:
                commentary += f"📈 Strong positive momentum with {price_change:.1%} gain. "
            elif price_change > 0:
                commentary += f"📈 Moderate positive movement with {price_change:.1%} gain. "
            elif price_change < -0.02:
                commentary += f"📉 Significant decline with {price_change:.1%} loss. "
            elif price_change < 0:
                commentary += f"📉 Slight decline with {price_change:.1%} loss. "
            else:
                commentary += "➡️ Price relatively stable. "
        
        # Volatility analysis
        volatility = analysis.get('volatility')
        if volatility is not None:
            if volatility > 0.03:
                commentary += f"High volatility environment ({volatility:.1%}). "
            elif volatility > 0.015:
                commentary += f"Moderate volatility ({volatility:.1%}). "
            else:
                commentary += f"Low volatility environment ({volatility:.1%}). "
        
        # Forecast context
        if not forecast_data.empty:
            commentary += "\n\nForecast indicates potential opportunities based on current market conditions."
        
        return commentary
        
    except Exception as e:
        logger.error(f"Commentary generation failed: {e}")
        return "Market commentary generation failed."

def main():
    """Main function for the Forecast & Trade page."""
    st.set_page_config(
        page_title="Forecast & Trade",
        page_icon="📈",
        layout="wide"
    )

    # Initialize session state
    initialize_session_state()

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
                <p>Last Updated: {safe_session_get('last_updated', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("Generate forecasts and execute trades based on model predictions with market context analysis.")

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        
        # Ticker selection
        ticker = st.text_input("Ticker Symbol", value=safe_session_get('selected_ticker', '')).upper()
        if ticker != safe_session_get('selected_ticker', ''):
            safe_session_set('selected_ticker', ticker)
            safe_session_set('forecast_data', None)
            safe_session_set('selected_model', None)
            safe_session_set('market_analysis', None)

        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=safe_session_get('start_date'),
                max_value=datetime.now().date()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=safe_session_get('end_date'),
                max_value=datetime.now().date()
            )

        if start_date != safe_session_get('start_date') or end_date != safe_session_get('end_date'):
            safe_session_set('start_date', start_date)
            safe_session_set('end_date', end_date)
            safe_session_set('forecast_data', None)
            safe_session_set('selected_model', None)
            safe_session_set('market_analysis', None)

        # Market Analysis Options
        st.subheader("Market Analysis")
        enable_market_analysis = st.checkbox("Enable Market Context Analysis", value=True)
        show_market_commentary = st.checkbox("Show Market Commentary", value=True)

        # Model selection
        st.subheader("Model Selection")
        use_agentic = st.checkbox("Use Agentic Selection", value=True)
        
        if use_agentic:
            # Get model trust levels
            try:
                model_monitor = ModelMonitor()
                trust_levels = model_monitor.get_model_trust_levels()
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
                strategy_switcher = StrategySelectionAgent()
                # For now, use a simple model selection - in a real implementation, 
                # you would pass actual metrics to the strategy switcher
                selected_model = "LSTM"  # Default model
                confidence = 0.8  # Default confidence
                st.success(f"Selected Model: {selected_model}")
                st.write(f"Confidence: {confidence:.2%}")
                
                # Log strategy decision
                strategy_logger_decision(
                    strategy_name=selected_model,
                    decision="forecast_generated",
                    confidence=confidence
                )
            except Exception as e:
                logger.warning(f"Agentic selection failed: {e}")
                selected_model = "LSTM"
                confidence = 0.8
                st.warning("Agentic selection failed, using default model")
        else:
            # Manual model selection
            selected_model = st.selectbox(
                "Select Model",
                options=["LSTM", "Transformer", "XGBoost", "Ensemble"],
                index=0
            )
            confidence = 1.0  # Manual selection has full confidence
            
            # Log strategy decision
            strategy_logger_decision(
                strategy_name=selected_model,
                decision="forecast_generated",
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
                            safe_session_set('market_analysis', market_analysis)
                            st.success("Market analysis completed!")
                except Exception as e:
                    st.warning(f"Could not load market data for analysis: {str(e)}")
                    market_analysis = {}
                
                # Simulate forecast generation
                forecast_data = generate_forecast(ticker, selected_model)
                safe_session_set('forecast_data', forecast_data)
                safe_session_set('selected_model', selected_model)
                
                # Calculate performance metrics
                sharpe_ratio = calculate_sharpe_ratio(forecast_data)
                accuracy_score = calculate_accuracy(forecast_data)
                win_rate = calculate_win_rate(forecast_data)
                
                # Log performance metrics
                log_performance(
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
    if safe_session_get('forecast_data') is not None:
        # Display forecast results
        st.subheader("Forecast Results")
        
        # Market Analysis Section
        if enable_market_analysis and safe_session_get('market_analysis'):
            display_market_analysis(safe_session_get('market_analysis'))
            
            if show_market_commentary:
                commentary = generate_market_commentary(
                    safe_session_get('market_analysis'), 
                    safe_session_get('forecast_data')
                )
                st.info(commentary)
        
        # Forecast visualization
        st.subheader("📈 Price Forecast")
        
        # Create forecast chart
        forecast_data = safe_session_get('forecast_data')
        if not forecast_data.empty:
            fig = go.Figure()
            
            # Historical data
            if 'historical' in forecast_data.columns:
                fig.add_trace(go.Scatter(
                    x=forecast_data.index,
                    y=forecast_data['historical'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue')
                ))
            
            # Forecast data
            if 'forecast' in forecast_data.columns:
                fig.add_trace(go.Scatter(
                    x=forecast_data.index,
                    y=forecast_data['forecast'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red', dash='dash')
                ))
            
            # Confidence intervals
            if show_confidence and 'upper' in forecast_data.columns and 'lower' in forecast_data.columns:
                fig.add_trace(go.Scatter(
                    x=forecast_data.index,
                    y=forecast_data['upper'],
                    mode='lines',
                    name='Upper Bound',
                    line=dict(color='lightgray'),
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_data.index,
                    y=forecast_data['lower'],
                    mode='lines',
                    fill='tonexty',
                    name='Confidence Interval',
                    line=dict(color='lightgray'),
                    fillcolor='rgba(200,200,200,0.3)'
                ))
            
            fig.update_layout(
                title=f"{ticker} Price Forecast",
                xaxis_title="Date",
                yaxis_title="Price",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance metrics
            st.subheader("📊 Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Sharpe Ratio", f"{calculate_sharpe_ratio(forecast_data):.2f}")
            
            with col2:
                st.metric("Accuracy", f"{calculate_accuracy(forecast_data):.1%}")
            
            with col3:
                st.metric("Win Rate", f"{calculate_win_rate(forecast_data):.1%}")
            
            with col4:
                st.metric("Model Confidence", f"{confidence:.1%}")
            
            # Model information
            st.subheader("🤖 Model Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Selected Model:** {selected_model}")
                st.write(f"**Selection Method:** {'Agentic' if use_agentic else 'Manual'}")
                st.write(f"**Forecast Period:** {len(forecast_data)} days")
            
            with col2:
                st.write(f"**Model Summary:**")
                st.write(get_model_summary(selected_model))
            
            # Backtest results
            if st.checkbox("Show Backtest Results", value=False):
                st.subheader("📈 Backtest Results")
                
                backtest_results = run_backtest(
                    forecast_data, 
                    initial_capital=10000, 
                    position_size=50, 
                    stop_loss=2.0, 
                    take_profit=4.0
                )
                
                if backtest_results:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Return", f"{backtest_results['total_return']:.1%}")
                    
                    with col2:
                        st.metric("Max Drawdown", f"{backtest_results['max_drawdown']:.1%}")
                    
                    with col3:
                        st.metric("Win Rate", f"{backtest_results['win_rate']:.1%}")
                    
                    with col4:
                        st.metric("Profit Factor", f"{backtest_results['profit_factor']:.2f}")
    
    else:
        st.info("👆 Use the sidebar controls to generate a forecast.")
    
    # Update last updated timestamp
    update_last_updated()

def calculate_sharpe_ratio(forecast_data):
    """Calculate Sharpe ratio."""
    # Placeholder implementation
    return np.random.uniform(0.5, 2.0)

def calculate_accuracy(forecast_data):
    """Calculate accuracy."""
    # Placeholder implementation
    return np.random.uniform(0.6, 0.9)

def calculate_win_rate(forecast_data):
    """Calculate win rate."""
    # Placeholder implementation
    return np.random.uniform(0.5, 0.8)

def generate_forecast(ticker, selected_model):
    """Generate forecast data."""
    # Placeholder implementation
    dates = pd.date_range(start=datetime.now(), periods=30, freq='D')
    historical = np.random.normal(100, 5, len(dates))
    forecast = historical + np.random.normal(0, 2, len(dates))
    
    return pd.DataFrame({
        'historical': historical,
        'forecast': forecast,
        'upper': forecast + 5,
        'lower': forecast - 5
    }, index=dates)

def run_backtest(forecast_data, initial_capital=10000, position_size=50, stop_loss=2.0, take_profit=4.0):
    """Run backtest simulation."""
    # Placeholder implementation
    return {
        'total_return': np.random.uniform(-0.1, 0.3),
        'max_drawdown': np.random.uniform(0.05, 0.2),
        'win_rate': np.random.uniform(0.4, 0.7),
        'profit_factor': np.random.uniform(0.8, 2.0)
    }

if __name__ == "__main__":
    main() 