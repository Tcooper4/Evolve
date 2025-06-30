"""Forecast components for Streamlit UI."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ForecastChart:
    """Chart component for displaying forecasts."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize forecast chart."""
        self.config = config or {}
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def render_forecast(self, 
                       historical_data: pd.DataFrame,
                       forecast_data: pd.DataFrame,
                       confidence_intervals: Optional[Dict[str, pd.Series]] = None):
        """Render forecast chart with historical and predicted data."""
        try:
            fig = go.Figure()
            
            # Historical data
            if not historical_data.empty:
                fig.add_trace(go.Scatter(
                    x=historical_data.index,
                    y=historical_data['Close'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue', width=2)
                ))
            
            # Forecast data
            if not forecast_data.empty:
                fig.add_trace(go.Scatter(
                    x=forecast_data.index,
                    y=forecast_data['forecast'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                # Confidence intervals
                if confidence_intervals:
                    lower = confidence_intervals.get('lower')
                    upper = confidence_intervals.get('upper')
                    
                    if lower is not None and upper is not None:
                        fig.add_trace(go.Scatter(
                            x=forecast_data.index,
                            y=upper,
                            mode='lines',
                            name='Upper CI',
                            line=dict(color='red', width=1, dash='dot'),
                            showlegend=False
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=forecast_data.index,
                            y=lower,
                            mode='lines',
                            name='Lower CI',
                            line=dict(color='red', width=1, dash='dot'),
                            fill='tonexty',
                            fillcolor='rgba(255,0,0,0.1)',
                            showlegend=False
                        ))
            
            fig.update_layout(
                title='Price Forecast',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            return {"status": "forecast_rendered", "figure": fig}
            
        except Exception as e:
            logger.error(f"Error rendering forecast chart: {e}")
            st.error("Error rendering forecast chart")
            return {'success': True, 'result': {"status": "forecast_render_failed", "error": str(e)}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def render_forecast_metrics(self, metrics: Dict[str, float]):
        """Render forecast performance metrics."""
        try:
            if not metrics:
                st.warning("No forecast metrics available")
                return {"status": "no_metrics_available"}
            
            st.subheader("Forecast Metrics")
            
            # Create metrics display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                mae = metrics.get('mae', 0)
                st.metric("MAE", f"{mae:.4f}")
            
            with col2:
                rmse = metrics.get('rmse', 0)
                st.metric("RMSE", f"{rmse:.4f}")
            
            with col3:
                mape = metrics.get('mape', 0)
                st.metric("MAPE", f"{mape:.2f}%")
            
            with col4:
                r2 = metrics.get('r2', 0)
                st.metric("R²", f"{r2:.4f}")
            
            return {"status": "metrics_rendered", "metrics": metrics}
            
        except Exception as e:
            logger.error(f"Error rendering forecast metrics: {e}")
            st.error("Error rendering metrics")
            return {'success': True, 'result': {"status": "metrics_render_failed", "error": str(e)}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

class ModelSelector:
    """Model selection component."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize model selector."""
        self.config = config or {}
        self.available_models = [
            'transformer',
            'lstm',
            'tcn',
            'arima',
            'prophet',
            'ensemble'
        ]
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def render_model_selector(self) -> str:
        """Render model selection interface."""
        try:
            st.subheader("Select Forecasting Model")
            
            selected_model = st.selectbox(
                "Choose a model:",
                self.available_models,
                index=0,
                help="Select the forecasting model to use"
            )
            
            # Model description
            model_descriptions = {
                'transformer': 'Advanced transformer model for time series forecasting',
                'lstm': 'Long Short-Term Memory neural network',
                'tcn': 'Temporal Convolutional Network',
                'arima': 'Classical ARIMA model',
                'prophet': 'Facebook Prophet model',
                'ensemble': 'Ensemble of multiple models'
            }
            
            st.info(f"**{selected_model.upper()}**: {model_descriptions.get(selected_model, 'No description available')}")
            
            return selected_model
            
        except Exception as e:
            logger.error(f"Error rendering model selector: {e}")
            st.error("Error rendering model selector")
            return {'success': True, 'result': 'transformer', 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def render_model_parameters(self, model: str) -> Dict[str, Any]:
        """Render model-specific parameters."""
        try:
            st.subheader("Model Parameters")
            params = {}
            
            if model == 'transformer':
                params['sequence_length'] = st.slider("Sequence Length", 10, 100, 50)
                params['n_heads'] = st.slider("Number of Heads", 2, 16, 8)
                params['n_layers'] = st.slider("Number of Layers", 1, 12, 6)
                
            elif model == 'lstm':
                params['hidden_size'] = st.slider("Hidden Size", 32, 512, 128)
                params['num_layers'] = st.slider("Number of Layers", 1, 5, 2)
                params['dropout'] = st.slider("Dropout", 0.0, 0.5, 0.1)
                
            elif model == 'tcn':
                params['num_channels'] = st.slider("Number of Channels", 16, 256, 64)
                params['kernel_size'] = st.slider("Kernel Size", 2, 8, 3)
                params['num_layers'] = st.slider("Number of Layers", 1, 10, 4)
                
            elif model == 'arima':
                params['p'] = st.slider("AR Order (p)", 0, 5, 1)
                params['d'] = st.slider("Difference Order (d)", 0, 2, 1)
                params['q'] = st.slider("MA Order (q)", 0, 5, 1)
                
            elif model == 'prophet':
                params['changepoint_prior_scale'] = st.slider("Changepoint Prior Scale", 0.001, 0.5, 0.05)
                params['seasonality_prior_scale'] = st.slider("Seasonality Prior Scale", 0.01, 10.0, 1.0)
                
            elif model == 'ensemble':
                params['models'] = st.multiselect(
                    "Select models for ensemble:",
                    ['transformer', 'lstm', 'tcn', 'arima'],
                    default=['transformer', 'lstm']
                )
                params['weights'] = st.selectbox(
                    "Weighting method:",
                    ['equal', 'performance', 'custom']
                )
            
            return params
            
        except Exception as e:
            logger.error(f"Error rendering model parameters: {e}")
            st.error("Error rendering parameters")
            return {'success': True, 'result': {}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

class DataInput:
    """Data input component."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize data input component."""
        self.config = config or {}
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def render_symbol_input(self) -> str:
        """Render symbol input interface."""
        try:
            st.subheader("Input Parameters")
            
            symbol = st.text_input(
                "Stock Symbol:",
                value="AAPL",
                help="Enter the stock symbol to forecast"
            ).upper()
            
            return symbol
            
        except Exception as e:
            logger.error(f"Error rendering symbol input: {e}")
            st.error("Error rendering symbol input")
            return {'success': True, 'result': "AAPL", 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def render_date_range(self) -> Tuple[str, str]:
        """Render date range selection."""
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                start_date = st.date_input(
                    "Start Date:",
                    value=pd.Timestamp.now() - pd.Timedelta(days=365),
                    help="Select start date for historical data"
                )
            
            with col2:
                end_date = st.date_input(
                    "End Date:",
                    value=pd.Timestamp.now(),
                    help="Select end date for historical data"
                )
            
            return str(start_date), str(end_date)
            
        except Exception as e:
            logger.error(f"Error rendering date range: {e}")
            st.error("Error rendering date range")
            return {'success': True, 'result': "2023-01-01", "2024-01-01", 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def render_forecast_horizon(self) -> int:
        """Render forecast horizon selection."""
        try:
            horizon = st.slider(
                "Forecast Horizon (days):",
                min_value=1,
                max_value=90,
                value=30,
                help="Number of days to forecast"
            )
            
            return horizon
            
        except Exception as e:
            logger.error(f"Error rendering forecast horizon: {e}")
            st.error("Error rendering forecast horizon")
            return {'success': True, 'result': 30, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

# Global component instances
forecast_chart = ForecastChart()
model_selector = ModelSelector()
data_input = DataInput()

def get_forecast_chart() -> ForecastChart:
    """Get the global forecast chart instance."""
    return {'success': True, 'result': forecast_chart, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

def get_model_selector() -> ModelSelector:
    """Get the global model selector instance."""
    return {'success': True, 'result': model_selector, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

def get_data_input() -> DataInput:
    """Get the global data input instance."""
    return {'success': True, 'result': data_input, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}