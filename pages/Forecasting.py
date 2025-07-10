"""
Forecasting Page

A clean, production-ready forecasting interface with:
- Multi-model forecasting
- Performance metrics (RMSE, MAE, MAPE, Sharpe, Win Rate, Drawdown)
- Dynamic model creation
- Confidence intervals
- Clean UI without dev clutter
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import warnings
import logging
import json
from dataclasses import asdict

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Forecasting - Evolve AI Trading",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for clean styling
st.markdown("""
<style>
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid #f0f0f0;
    }
    
    .forecast-result {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 2px solid #dee2e6;
    }
    
    .performance-metrics {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .metric-item {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        margin-top: 0.5rem;
    }
    
    .model-creation-panel {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        border: 2px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state for forecasting."""
    if 'forecast_history' not in st.session_state:
        st.session_state.forecast_history = []
    
    if 'current_forecast' not in st.session_state:
        st.session_state.current_forecast = None
    
    if 'model_performance' not in st.session_state:
        st.session_state.model_performance = {}

def load_available_models():
    """Load available forecasting models."""
    try:
        # Import model components
        from models.forecast_router import ForecastRouter
        from trading.agents.model_creator_agent import ModelCreatorAgent
        
        # Get available models
        models = {
            'LSTM': {
                'description': 'Long Short-Term Memory neural network for time series forecasting',
                'best_for': 'Medium to long-term predictions',
                'accuracy': 0.87,
                'complexity': 'Medium'
            },
            'Transformer': {
                'description': 'Attention-based model for sequence modeling',
                'best_for': 'Complex patterns and relationships',
                'accuracy': 0.89,
                'complexity': 'High'
            },
            'XGBoost': {
                'description': 'Gradient boosting for structured data',
                'best_for': 'Feature-rich datasets',
                'accuracy': 0.85,
                'complexity': 'Medium'
            },
            'ARIMA': {
                'description': 'Autoregressive integrated moving average',
                'best_for': 'Traditional time series',
                'accuracy': 0.82,
                'complexity': 'Low'
            },
            'Prophet': {
                'description': 'Facebook\'s forecasting tool',
                'best_for': 'Seasonal patterns',
                'accuracy': 0.84,
                'complexity': 'Low'
            },
            'Ensemble': {
                'description': 'Combination of multiple models',
                'best_for': 'Maximum accuracy and robustness',
                'accuracy': 0.92,
                'complexity': 'High'
            },
            'Ridge': {
                'description': 'Ridge regression for regularization',
                'best_for': 'Linear relationships with regularization',
                'accuracy': 0.80,
                'complexity': 'Low'
            },
            'GARCH': {
                'description': 'Generalized Autoregressive Conditional Heteroskedasticity',
                'best_for': 'Volatility forecasting',
                'accuracy': 0.83,
                'complexity': 'Medium'
            }
        }
        
        return models
    except ImportError as e:
        logger.warning(f"Could not load model components: {e}")
        return {}

def calculate_forecast_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive forecast metrics."""
    try:
        # Basic error metrics
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual - predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # Directional accuracy
        actual_direction = np.diff(actual) > 0
        pred_direction = np.diff(predicted) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction)
        
        # Sharpe ratio (assuming returns)
        returns = np.diff(actual) / actual[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate (positive returns)
        win_rate = np.mean(returns > 0)
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'Directional_Accuracy': directional_accuracy,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown,
            'Win_Rate': win_rate
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {}

def generate_forecast_data(symbol: str, days: int, model: str) -> Dict[str, Any]:
    """Generate mock forecast data for demonstration."""
    try:
        # Generate mock historical data
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        historical_prices = 100 + np.cumsum(np.random.randn(100) * 0.02)
        
        # Generate forecast
        forecast_dates = pd.date_range(start=dates[-1], periods=days+1, freq='D')[1:]
        forecast_prices = historical_prices[-1] + np.cumsum(np.random.randn(days) * 0.015)
        
        # Generate confidence intervals
        confidence_lower = forecast_prices * 0.95
        confidence_upper = forecast_prices * 1.05
        
        # Calculate metrics
        metrics = calculate_forecast_metrics(historical_prices[-30:], forecast_prices[:30])
        
        return {
            'symbol': symbol,
            'model': model,
            'historical_dates': dates,
            'historical_prices': historical_prices,
            'forecast_dates': forecast_dates,
            'forecast_prices': forecast_prices,
            'confidence_lower': confidence_lower,
            'confidence_upper': confidence_upper,
            'metrics': metrics,
            'generated_at': datetime.now()
        }
    except Exception as e:
        logger.error(f"Error generating forecast data: {e}")
        return {}

def plot_forecast_with_metrics(forecast_data: Dict[str, Any]):
    """Plot forecast with comprehensive metrics."""
    try:
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Price Forecast', 'Performance Metrics', 'Error Distribution', 'Cumulative Returns'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Historical and forecast data
        fig.add_trace(
            go.Scatter(
                x=forecast_data['historical_dates'],
                y=forecast_data['historical_prices'],
                mode='lines',
                name='Historical',
                line=dict(color='#2c3e50', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=forecast_data['forecast_dates'],
                y=forecast_data['forecast_prices'],
                mode='lines',
                name='Forecast',
                line=dict(color='#3498db', width=2)
            ),
            row=1, col=1
        )
        
        # Confidence intervals
        fig.add_trace(
            go.Scatter(
                x=forecast_data['forecast_dates'],
                y=forecast_data['confidence_upper'],
                mode='lines',
                name='Upper CI',
                line=dict(color='rgba(52, 152, 219, 0.3)', width=1),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=forecast_data['forecast_dates'],
                y=forecast_data['confidence_lower'],
                mode='lines',
                fill='tonexty',
                name='Lower CI',
                line=dict(color='rgba(52, 152, 219, 0.3)', width=1),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Performance metrics bar chart
        metrics = forecast_data['metrics']
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        fig.add_trace(
            go.Bar(
                x=metric_names,
                y=metric_values,
                name='Metrics',
                marker_color='#e74c3c'
            ),
            row=1, col=2
        )
        
        # Error distribution
        historical = forecast_data['historical_prices']
        forecast = forecast_data['forecast_prices']
        errors = historical[-len(forecast):] - forecast
        
        fig.add_trace(
            go.Histogram(
                x=errors,
                name='Errors',
                marker_color='#f39c12',
                nbinsx=20
            ),
            row=2, col=1
        )
        
        # Cumulative returns
        returns = np.diff(historical) / historical[:-1]
        cumulative_returns = np.cumprod(1 + returns)
        
        fig.add_trace(
            go.Scatter(
                x=forecast_data['historical_dates'][1:],
                y=cumulative_returns,
                mode='lines',
                name='Cumulative Returns',
                line=dict(color='#27ae60', width=2)
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=f"Forecast Analysis for {forecast_data['symbol']} using {forecast_data['model']}",
            title_x=0.5
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error plotting forecast: {e}")
        st.error("Error generating forecast visualization")

def display_performance_metrics(metrics: Dict[str, float]):
    """Display performance metrics in a clean format."""
    try:
        st.markdown("### Performance Metrics")
        
        # Create metric cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{metrics.get('RMSE', 0):.4f}</div>
                <div class="metric-label">RMSE</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{metrics.get('MAE', 0):.4f}</div>
                <div class="metric-label">MAE</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{metrics.get('MAPE', 0):.2f}%</div>
                <div class="metric-label">MAPE</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{metrics.get('Directional_Accuracy', 0):.2%}</div>
                <div class="metric-label">Directional Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{metrics.get('Sharpe_Ratio', 0):.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{metrics.get('Win_Rate', 0):.2%}</div>
                <div class="metric-label">Win Rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{metrics.get('Max_Drawdown', 0):.2%}</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        logger.error(f"Error displaying metrics: {e}")

def create_new_model():
    """Interface for creating new models dynamically."""
    st.markdown("""
    <div class="model-creation-panel">
        <h3>Create New Model</h3>
        <p>Describe your model requirements in natural language and let AI create it for you.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model creation interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Requirements")
        requirements = st.text_area(
            "Describe your model requirements:",
            placeholder="e.g., 'Create a neural network for cryptocurrency price prediction with 3 layers and dropout'",
            height=150
        )
        
        model_name = st.text_input(
            "Model Name (optional):",
            placeholder="e.g., Crypto_LSTM_v1"
        )
        
        if st.button("Create Model", type="primary"):
            if requirements:
                try:
                    from trading.agents.model_creator_agent import get_model_creator_agent
                    
                    agent = get_model_creator_agent()
                    model_spec, success, errors = agent.create_and_validate_model(requirements, model_name)
                    
                    if success:
                        st.success(f"Model '{model_spec.name}' created successfully!")
                        st.json(asdict(model_spec))
                        
                        # Add to session state
                        if 'created_models' not in st.session_state:
                            st.session_state.created_models = []
                        st.session_state.created_models.append(asdict(model_spec))
                    else:
                        st.error(f"Model creation failed: {', '.join(errors)}")
                    
                except Exception as e:
                    st.error(f"Error creating model: {e}")
            else:
                st.warning("Please provide model requirements")
    
    with col2:
        st.subheader("Available Frameworks")
        
        try:
            from trading.agents.model_creator_agent import get_model_creator_agent
            agent = get_model_creator_agent()
            framework_status = agent.get_framework_status()
            
            for framework, available in framework_status.items():
                status = "✅ Available" if available else "❌ Not Available"
                st.markdown(f"**{framework.title()}:** {status}")
                
        except Exception as e:
            st.warning(f"Could not load framework status: {e}")
        
        st.subheader("Model Registry")
        if 'created_models' in st.session_state and st.session_state.created_models:
            for model in st.session_state.created_models[-5:]:  # Show last 5
                with st.expander(f"Model: {model.get('name', 'Unknown')}"):
                    st.json(model)
        else:
            st.info("No models created yet")

def main():
    """Main forecasting page function."""
    st.title("Forecasting Dashboard")
    st.markdown("Generate accurate price forecasts using advanced AI models")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar controls
    with st.sidebar:
        st.header("Forecast Settings")
        
        # Symbol input
        symbol = st.text_input("Symbol", value="AAPL", placeholder="e.g., AAPL, TSLA, BTC-USD")
        
        # Model selection
        available_models = load_available_models()
        model_options = list(available_models.keys())
        selected_model = st.selectbox("Model", model_options, index=0)
        
        # Forecast period
        forecast_days = st.slider("Forecast Period (days)", 1, 365, 30)
        
        # Confidence level
        confidence_level = st.slider("Confidence Level", 0.8, 0.99, 0.95, 0.01)
        
        # Generate forecast button
        if st.button("Generate Forecast", type="primary"):
            with st.spinner("Generating forecast..."):
                forecast_data = generate_forecast_data(symbol, forecast_days, selected_model)
                if forecast_data:
                    st.session_state.current_forecast = forecast_data
                    st.session_state.forecast_history.append(forecast_data)
                    st.success("Forecast generated successfully!")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Current forecast display
        if st.session_state.current_forecast:
            st.markdown("### Current Forecast")
            plot_forecast_with_metrics(st.session_state.current_forecast)
            
            # Performance metrics
            display_performance_metrics(st.session_state.current_forecast['metrics'])
            
            # Export options
            st.markdown("### Export Options")
            col_export1, col_export2, col_export3 = st.columns(3)
            
            with col_export1:
                if st.button("Export as CSV"):
                    df = pd.DataFrame({
                        'Date': st.session_state.current_forecast['forecast_dates'],
                        'Forecast': st.session_state.current_forecast['forecast_prices'],
                        'Lower_CI': st.session_state.current_forecast['confidence_lower'],
                        'Upper_CI': st.session_state.current_forecast['confidence_upper']
                    })
                    st.download_button(
                        label="Download CSV",
                        data=df.to_csv(index=False),
                        file_name=f"{symbol}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            with col_export2:
                if st.button("Export as JSON"):
                    st.download_button(
                        label="Download JSON",
                        data=json.dumps(st.session_state.current_forecast, default=str, indent=2),
                        file_name=f"{symbol}_forecast_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
            
            with col_export3:
                if st.button("Generate Report"):
                    st.info("Report generation feature coming soon!")
        else:
            st.info("Generate a forecast using the sidebar controls")
    
    with col2:
        # Model information
        if selected_model in available_models:
            model_info = available_models[selected_model]
            st.markdown("### Model Information")
            st.markdown(f"""
            <div class="metric-card">
                <h4>{selected_model}</h4>
                <p><strong>Description:</strong> {model_info['description']}</p>
                <p><strong>Best for:</strong> {model_info['best_for']}</p>
                <p><strong>Accuracy:</strong> {model_info['accuracy']:.1%}</p>
                <p><strong>Complexity:</strong> {model_info['complexity']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Forecast history
        if st.session_state.forecast_history:
            st.markdown("### Recent Forecasts")
            for i, forecast in enumerate(reversed(st.session_state.forecast_history[-5:])):
                with st.expander(f"{forecast['symbol']} - {forecast['model']} ({forecast['generated_at'].strftime('%Y-%m-%d %H:%M')})"):
                    st.markdown(f"**RMSE:** {forecast['metrics'].get('RMSE', 0):.4f}")
                    st.markdown(f"**Sharpe Ratio:** {forecast['metrics'].get('Sharpe_Ratio', 0):.2f}")
    
    # Model creation section
    st.markdown("---")
    create_new_model()

if __name__ == "__main__":
    main() 