"""
Forecasting & Market Analysis Page

Merges functionality from:
- Forecasting.py
- Forecast_with_AI_Selection.py
- 7_Market_Analysis.py

Features:
- Quick Forecast with 4 models (LSTM, XGBoost, Prophet, ARIMA)
- Advanced Forecasting with hyperparameter tuning
- AI-powered model selection
- Multi-model comparison and ensemble
- Market analysis with technical indicators
"""

import logging
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Backend imports
from trading.data.data_loader import DataLoader, DataLoadRequest
from trading.data.providers.yfinance_provider import YFinanceProvider
from trading.models.lstm_model import LSTMForecaster
from trading.models.xgboost_model import XGBoostModel
from trading.models.prophet_model import ProphetModel
from trading.models.arima_model import ARIMAModel
from trading.data.preprocessing import FeatureEngineering, DataPreprocessor
from trading.agents.model_selector_agent import ModelSelectorAgent
from trading.market.market_analyzer import MarketAnalyzer

st.set_page_config(
    page_title="Forecasting & Market Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'selected_models' not in st.session_state:
    st.session_state.selected_models = []
if 'ai_recommendation' not in st.session_state:
    st.session_state.ai_recommendation = None
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None
if 'market_regime' not in st.session_state:
    st.session_state.market_regime = None
if 'symbol' not in st.session_state:
    st.session_state.symbol = None
if 'forecast_horizon' not in st.session_state:
    st.session_state.forecast_horizon = 7

# Main page title
st.title("📈 Forecasting & Market Analysis")
st.markdown("Advanced forecasting with AI model selection and comprehensive market analysis")

# Create tabbed interface
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🚀 Quick Forecast",
    "⚙️ Advanced Forecasting", 
    "🤖 AI Model Selection",
    "📊 Model Comparison",
    "📈 Market Analysis",
    "🔗 Multi-Asset (GNN)"  # NEW TAB
])

with tab1:
    st.header("Quick Forecast")
    st.markdown("Generate fast forecasts with pre-configured models")
    
    # Data Loading Form
    with st.form("data_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbol = st.text_input(
                "Ticker Symbol",
                value="AAPL",
                help="Enter stock ticker (e.g., AAPL, MSFT, GOOGL)"
            ).upper()
        
        with col2:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365),
                max_value=datetime.now()
            )
        
        with col3:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                max_value=datetime.now()
            )
        
        submitted = st.form_submit_button("📊 Load Data", width='stretch')
    
    if submitted:
        # Validation
        if not symbol:
            st.error("Please enter a ticker symbol")
        elif start_date >= end_date:
            st.error("Start date must be before end date")
        elif (end_date - start_date).days < 30:
            st.error("Please select at least 30 days of data")
        else:
            try:
                with st.spinner(f"Fetching data for {symbol}..."):
                    # Initialize data loader
                    loader = DataLoader()
                    
                    # Create data load request
                    request = DataLoadRequest(
                        ticker=symbol,
                        start_date=start_date.strftime("%Y-%m-%d"),
                        end_date=end_date.strftime("%Y-%m-%d"),
                        interval="1d"
                    )
                    
                    # Load data
                    response = loader.load_market_data(request)
                    
                    if not response.success:
                        st.error(f"Error loading data: {response.message}")
                    elif response.data is None or len(response.data) < 30:
                        st.error(f"Insufficient data for {symbol}. Try different dates or symbol.")
                    else:
                        # Convert to standard format (lowercase column names)
                        data = response.data.copy()
                        data.columns = [col.lower() for col in data.columns]
                        
                        # Ensure we have 'close' column
                        if 'close' not in data.columns:
                            # Try to find close price column
                            close_col = None
                            for col in ['Close', 'close', 'CLOSE', 'price', 'Price']:
                                if col in data.columns:
                                    close_col = col
                                    break
                            if close_col:
                                data['close'] = data[close_col]
                            else:
                                st.error("Could not find close price column in data")
                                data = None
                        
                        if data is not None:
                            # Store in session state
                            st.session_state.forecast_data = data
                            st.session_state.symbol = symbol
                            # forecast_horizon is already in session_state (defaults to 7 if not set)
                            # No need to reassign it here unless we want to update it
                            
                            st.success(f"✅ Loaded {len(data)} days of data for {symbol}")
                            
                            # Show data quality metrics
                            try:
                                from src.utils.data_validation import DataValidator
                                
                                validator = DataValidator()
                                quality_metrics = validator.get_quality_metrics(data)
                                
                                with st.expander("📊 Data Quality Metrics", expanded=False):
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        st.metric("Completeness", f"{quality_metrics['completeness']:.1%}")
                                    with col2:
                                        st.metric("Missing Values", quality_metrics['missing_count'])
                                    with col3:
                                        st.metric("Outliers Detected", quality_metrics['outliers'])
                                    with col4:
                                        quality_score = quality_metrics['overall_quality']
                                        st.metric("Quality Score", f"{quality_score:.0f}/100")
                                    
                                    # Show issues if any
                                    if quality_metrics['issues']:
                                        st.warning("⚠️ Data Quality Issues:")
                                        for issue in quality_metrics['issues']:
                                            st.write(f"• {issue}")
                            except Exception as e:
                                # Silently fail if validation not available
                                pass
                        
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                st.info("Please check the ticker symbol and try again.")
    
    # Display loaded data
    if st.session_state.forecast_data is not None:
        data = st.session_state.forecast_data
        
        # Show data quality metrics
        try:
            from src.utils.data_validation import DataValidator
            
            validator = DataValidator()
            quality_metrics = validator.get_quality_metrics(data)
            
            with st.expander("📊 Data Quality Metrics", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Completeness", f"{quality_metrics['completeness']:.1%}")
                with col2:
                    st.metric("Missing Values", quality_metrics['missing_count'])
                with col3:
                    st.metric("Outliers Detected", quality_metrics['outliers'])
                with col4:
                    quality_score = quality_metrics['overall_quality']
                    st.metric("Quality Score", f"{quality_score:.0f}/100")
                
                # Show issues if any
                if quality_metrics['issues']:
                    st.warning("⚠️ Data Quality Issues:")
                    for issue in quality_metrics['issues']:
                        st.write(f"• {issue}")
        except Exception as e:
            # Silently fail if validation not available
            pass
        
        st.markdown("---")
        st.subheader("📊 Data Preview")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Data Points", len(data))
        with col2:
            st.metric("Current Price", f"${data['close'].iloc[-1]:.2f}")
        with col3:
            change = ((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100
            st.metric("Period Return", f"{change:.2f}%")
        with col4:
            volatility = data['close'].pct_change().std() * np.sqrt(252) * 100
            st.metric("Annualized Volatility", f"{volatility:.2f}%")
        
        # Price chart - use advanced candlestick chart if OHLCV data available
        try:
            from utils.plotting_helper import create_candlestick_chart
            
            # Check if we have OHLCV data
            has_ohlcv = all(col in data.columns for col in ['open', 'high', 'low', 'close'])
            
            if has_ohlcv:
                fig = create_candlestick_chart(
                    data=data,
                    title=f"{st.session_state.symbol} Price History",
                    show_volume='volume' in data.columns,
                    show_ma=[20, 50, 200] if len(data) >= 200 else ([20, 50] if len(data) >= 50 else [20])
                )
            else:
                # Fallback to simple line chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='blue', width=2)
                ))
                fig.update_layout(
                    title=f"{st.session_state.symbol} Price History",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    hovermode='x unified',
                    height=400
                )
            
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            # Fallback to basic chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            ))
            fig.update_layout(
                title=f"{st.session_state.symbol} Price History",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Data table (expandable)
        with st.expander("📋 View Full Data"):
            st.dataframe(data.tail(50), width='stretch')
        
        # Model Selection & Forecasting
        st.markdown("---")
        st.subheader("🎯 Generate Forecast")
        
        # Forecast horizon slider (outside form so it updates immediately)
        forecast_horizon = st.slider(
            "Forecast Horizon (days)",
            min_value=1,
            max_value=30,
            value=st.session_state.forecast_horizon,
            key="forecast_horizon_slider",
            help="Number of days to forecast into the future"
        )
        # Update session state immediately
        st.session_state.forecast_horizon = forecast_horizon
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Select Model**")
            
            # Use UI component for model selection
            try:
                from trading.ui.forecast_components import render_model_selector
                from trading.models.model_registry import get_registry
                
                registry = get_registry()
                available_models = registry.list_models()
                
                if not available_models:
                    st.warning("⚠️ No models available. Using default models.")
                    available_models = ["LSTM", "XGBoost", "Prophet", "ARIMA"]
                
                selected_model = render_model_selector(
                    available_models=available_models,
                    key="quick_forecast_model"
                )
                
                # Show model description if available
                if selected_model:
                    try:
                        model_info = registry.get_model_info(selected_model)
                        if model_info and model_info.get('description'):
                            description = model_info['description']
                            if len(description) > 200:
                                description = description[:200] + "..."
                            st.info(f"ℹ️ {description}")
                    except Exception as e:
                        logger.warning(f"Forecasting error: {e}")
                        pass
            except ImportError:
                # Fallback to original code
                try:
                    from trading.models.model_registry import get_registry
                    registry = get_registry()
                    available_models = registry.list_models()
                    if not available_models:
                        available_models = ["LSTM", "XGBoost", "Prophet", "ARIMA"]
                    selected_model = st.radio(
                        "Choose model:",
                        available_models,
                        help="Choose a forecasting model"
                    )
                except Exception as e:
                    logger.warning(f"Forecasting error: {e}")
                    available_models = ["LSTM", "XGBoost", "Prophet", "ARIMA"]
                    selected_model = st.radio(
                        "Choose model:",
                        available_models,
                        help="Different models work better for different data patterns"
                    )
            
            forecast_button = st.button(
                "🚀 Generate Forecast",
                type="primary",
                width='stretch'
            )
        
        with col2:
            if forecast_button:
                try:
                    with st.spinner(f"Training {selected_model}..."):
                        # Get data
                        data = st.session_state.forecast_data.copy()
                        horizon = st.session_state.forecast_horizon
                        
                        # Prepare data for models (ensure proper format)
                        if not isinstance(data.index, pd.DatetimeIndex):
                            data.index = pd.to_datetime(data.index)
                        
                        # Initialize model with proper config using registry
                        model = None
                        try:
                            from trading.models.model_registry import get_registry
                            registry = get_registry()
                            ModelClass = registry.get(selected_model)
                            
                            if ModelClass is None:
                                st.error(f"Model {selected_model} not available")
                            else:
                                # Model-specific initialization
                                if selected_model == 'GARCH':
                                    st.subheader("Volatility Forecasting with GARCH")
                                    st.write("GARCH models are specialized for forecasting volatility and risk metrics.")
                                    config = {
                                        "p": 1,
                                        "q": 1,
                                        "target_column": "close"
                                    }
                                    model = ModelClass(config)
                                    
                                elif selected_model == 'Autoformer':
                                    st.subheader("Advanced Transformer Forecasting")
                                    st.write("Autoformer uses advanced attention mechanisms for complex time series patterns.")
                                    config = {
                                        "seq_len": 60,
                                        "pred_len": horizon,
                                        "d_model": 128,
                                        "target_column": "close"
                                    }
                                    model = ModelClass(config)
                                    
                                elif selected_model == 'CatBoost':
                                    st.subheader("CatBoost Gradient Boosting")
                                    st.write("Alternative to XGBoost with better categorical feature handling.")
                                    config = {
                                        "target_column": "close",
                                        "iterations": 500,
                                        "depth": 6,
                                        "learning_rate": 0.03
                                    }
                                    model = ModelClass(config)
                                    
                                elif selected_model == 'TCN':
                                    st.subheader("Temporal Convolutional Network")
                                    st.write("TCN uses dilated convolutions for efficient long-range dependencies.")
                                    config = {
                                        "target_column": "close",
                                        "num_channels": [64, 128, 256],
                                        "kernel_size": 3,
                                        "dropout": 0.2
                                    }
                                    model = ModelClass(config)
                                    
                                elif selected_model == 'Ridge':
                                    st.subheader("Ridge Regression Baseline")
                                    st.write("Simple linear baseline with L2 regularization.")
                                    config = {
                                        "target_column": "close",
                                        "alpha": 1.0
                                    }
                                    model = ModelClass(config)
                                    
                                elif "LSTM" in selected_model:
                                    # LSTMForecaster uses config dictionary
                                    config = {
                                        "target_column": "close",
                                        "sequence_length": 60,
                                        "hidden_size": 64,
                                        "num_layers": 2,
                                        "dropout": 0.2,
                                        "learning_rate": 0.001,
                                        "feature_columns": list(data.columns) if len(data.columns) > 0 else ["close"],
                                        "input_size": len(data.columns) if len(data.columns) > 0 else 1
                                    }
                                    model = ModelClass(config)
                                elif "XGBoost" in selected_model:
                                    config = {
                                        "target_column": "close",
                                        "n_estimators": 100,
                                        "max_depth": 5,
                                        "learning_rate": 0.1
                                    }
                                    model = ModelClass(config)
                                elif "Prophet" in selected_model:
                                    config = {
                                        "date_column": data.index.name if data.index.name else "ds",
                                        "target_column": "close",
                                        "prophet_params": {}
                                    }
                                    model = ModelClass(config)
                                elif "ARIMA" in selected_model:
                                    config = {
                                        "order": (5, 1, 0),
                                        "use_auto_arima": True,
                                        "target_column": "close"
                                    }
                                    model = ModelClass(config)
                                else:
                                    # Generic model initialization
                                    config = {
                                        "target_column": "close",
                                        "feature_columns": list(data.columns) if len(data.columns) > 0 else ["close"]
                                    }
                                    model = ModelClass(config)
                        except Exception as e:
                            st.warning(f"Could not use model registry: {e}. Using direct imports.")
                            # Fallback to direct imports
                            if "LSTM" in selected_model:
                                config = {
                                    "target_column": "close",
                                    "sequence_length": 60,
                                    "hidden_size": 64,
                                    "num_layers": 2,
                                    "dropout": 0.2,
                                    "learning_rate": 0.001,
                                    "feature_columns": list(data.columns) if len(data.columns) > 0 else ["close"],
                                    "input_size": len(data.columns) if len(data.columns) > 0 else 1
                                }
                                model = LSTMForecaster(config)
                            elif "XGBoost" in selected_model:
                                config = {
                                    "target_column": "close",
                                    "n_estimators": 100,
                                    "max_depth": 5,
                                    "learning_rate": 0.1
                                }
                                model = XGBoostModel(config)
                            elif "Prophet" in selected_model:
                                config = {
                                    "date_column": data.index.name if data.index.name else "ds",
                                    "target_column": "close",
                                    "prophet_params": {}
                                }
                                model = ProphetModel(config)
                            elif "ARIMA" in selected_model:
                                config = {
                                    "order": (5, 1, 0),
                                    "use_auto_arima": True
                                }
                                model = ARIMAModel(config)
                        
                        if model is None:
                            st.error("Failed to initialize model")
                        else:
                            # Train model
                            if "Prophet" in selected_model:
                                # Prophet needs special format
                                train_df = pd.DataFrame({
                                    'ds': data.index,
                                    'y': data['close']
                                })
                                fit_result = model.fit(train_df)
                            elif "ARIMA" in selected_model:
                                # ARIMA needs series
                                fit_result = model.fit(data['close'])
                            else:
                                # LSTM and XGBoost use DataFrame
                                fit_result = model.fit(data, data['close'])
                            
                            # Generate forecast - try with uncertainty if available
                            forecast_result = None
                            if hasattr(model, 'forecast_with_uncertainty'):
                                try:
                                    forecast_result = model.forecast_with_uncertainty(data, horizon=horizon, num_samples=100)
                                except Exception as e:
                                    st.warning(f"Could not generate forecast with uncertainty: {e}. Using standard forecast.")
                                    forecast_result = model.forecast(data, horizon=horizon)
                            else:
                                forecast_result = model.forecast(data, horizon=horizon)
                            
                            # Store full forecast result for confidence intervals
                            st.session_state.current_forecast_result = forecast_result
                            
                            # Postprocess forecast
                            try:
                                from trading.forecasting.forecast_postprocessor import ForecastPostprocessor
                                
                                postprocessor = ForecastPostprocessor()
                                
                                # Extract forecast values for postprocessing
                                if isinstance(forecast_result, dict):
                                    forecast_vals = forecast_result.get('forecast', [])
                                else:
                                    forecast_vals = forecast_result
                                
                                # Postprocess forecast
                                processed_forecast = postprocessor.process(
                                    forecast=forecast_vals,
                                    historical_data=data,
                                    apply_smoothing=True,
                                    remove_outliers=True,
                                    ensure_realistic_bounds=True
                                )
                                
                                # Update forecast_result with processed version
                                if isinstance(forecast_result, dict):
                                    forecast_result['forecast'] = processed_forecast['values']
                                    forecast_result['postprocessing_notes'] = processed_forecast.get('notes', [])
                                else:
                                    forecast_result = processed_forecast['values']
                                
                                # Show what was done
                                if processed_forecast.get('modifications'):
                                    with st.expander("⚙️ Forecast Postprocessing", expanded=False):
                                        st.write("**Modifications applied:**")
                                        for mod in processed_forecast['modifications']:
                                            st.write(f"• {mod}")
                                
                                # Update session state with processed forecast
                                st.session_state.current_forecast_result = forecast_result
                            except ImportError:
                                pass  # Silently fail if postprocessor not available
                            except Exception as e:
                                logger.warning(f"Forecast postprocessing failed: {e}")
                            
                            # Log model performance
                            if 'model_log' in st.session_state and 'perf_logger' in st.session_state:
                                try:
                                    import time
                                    train_time = time.time() - time.time()  # Placeholder - would need actual timing
                                    
                                    # Calculate basic metrics if possible
                                    r2_score = 0.0  # Placeholder - would need actual evaluation
                                    rmse = 0.0
                                    mae = 0.0
                                    
                                    st.session_state.model_log.log_training(
                                        model_name=selected_model,
                                        model_version='1.0',
                                        training_data_size=len(data),
                                        training_time=train_time,
                                        parameters={},
                                        metrics={
                                            'r2_score': r2_score,
                                            'rmse': rmse,
                                            'mae': mae
                                        }
                                    )
                                    
                                    st.session_state.perf_logger.log_performance(
                                        model_name=selected_model,
                                        metric_name='accuracy',
                                        metric_value=r2_score,
                                        timestamp=datetime.now()
                                    )
                                except Exception as e:
                                    pass  # Silently fail if logging not available
                            
                            # Extract forecast values
                            if isinstance(forecast_result, dict):
                                forecast_values = forecast_result.get('forecast', [])
                                forecast_dates = forecast_result.get('dates', pd.date_range(
                                    start=data.index[-1] + timedelta(days=1),
                                    periods=horizon,
                                    freq='D'
                                ))
                            else:
                                forecast_values = forecast_result
                                forecast_dates = pd.date_range(
                                    start=data.index[-1] + timedelta(days=1),
                                    periods=horizon,
                                    freq='D'
                                )
                            
                            # Debug: Check what we got
                            if forecast_values is None or len(forecast_values) == 0:
                                st.warning(f"⚠️ Forecast returned empty values. Result type: {type(forecast_result)}")
                                if isinstance(forecast_result, dict):
                                    st.write("Forecast result keys:", list(forecast_result.keys()))
                                # Use last known price as fallback
                                last_price = data['close'].iloc[-1] if 'close' in data.columns else data.iloc[-1, 0]
                                forecast_values = np.full(horizon, float(last_price))
                            
                            # Ensure forecast_values is array-like
                            if isinstance(forecast_values, (list, np.ndarray)):
                                forecast_values = np.array(forecast_values).flatten()
                                # Check for NaN/None values
                                if np.any(np.isnan(forecast_values)) or np.any(forecast_values == None):
                                    st.warning("⚠️ Forecast contains NaN/None values. Replacing with last known price.")
                                    last_price = float(data['close'].iloc[-1] if 'close' in data.columns else data.iloc[-1, 0])
                                    forecast_values = np.where(
                                        np.isnan(forecast_values) | (forecast_values == None),
                                        last_price,
                                        forecast_values
                                    )
                            else:
                                # Single value case
                                if forecast_values is None or (isinstance(forecast_values, float) and np.isnan(forecast_values)):
                                    last_price = float(data['close'].iloc[-1] if 'close' in data.columns else data.iloc[-1, 0])
                                    forecast_values = np.full(horizon, last_price)
                                else:
                                    forecast_values = np.array([float(forecast_values)] * horizon)
                            
                            # Create forecast DataFrame
                            forecast_df = pd.DataFrame({
                                'forecast': forecast_values
                            }, index=forecast_dates[:len(forecast_values)])
                            
                            # Store in session state
                            st.session_state.current_forecast = forecast_df
                            st.session_state.current_model = selected_model
                            st.session_state.current_model_instance = model  # Store model instance for explainability
                            
                            st.success(f"✅ Forecast generated using {selected_model}")
                            
                            # Natural Language Insights
                            st.markdown("---")
                            st.subheader("💬 Natural Language Insights")
                            
                            if st.button("Generate Plain English Explanation", key="generate_nlg_insights"):
                                try:
                                    from nlp.natural_language_insights import NaturalLanguageInsights
                                    
                                    nlg = NaturalLanguageInsights()
                                    
                                    # Prepare forecast result for insights
                                    forecast_for_insights = {
                                        'forecast': forecast_df['forecast'].values.tolist() if hasattr(forecast_df['forecast'].values, 'tolist') else list(forecast_df['forecast'].values),
                                        'dates': forecast_df.index.tolist() if hasattr(forecast_df.index, 'tolist') else list(forecast_df.index),
                                        'model_type': selected_model,
                                        'symbol': st.session_state.symbol
                                    }
                                    
                                    # Add confidence intervals if available
                                    if isinstance(forecast_result, dict):
                                        if 'lower_bound' in forecast_result:
                                            forecast_for_insights['lower_bound'] = forecast_result['lower_bound']
                                        if 'upper_bound' in forecast_result:
                                            forecast_for_insights['upper_bound'] = forecast_result['upper_bound']
                                    
                                    # Generate explanation
                                    insights = nlg.generate_forecast_insights(
                                        forecast=forecast_for_insights,
                                        historical_data=data,
                                        model_type=selected_model,
                                        symbol=st.session_state.symbol
                                    )
                                    
                                    # Display insights
                                    st.info(insights['summary'])
                                    
                                    with st.expander("📊 Detailed Analysis", expanded=False):
                                        st.write("**Trend Analysis:**")
                                        st.write(insights['trend_analysis'])
                                        
                                        st.write("**Key Factors:**")
                                        for factor in insights['key_factors']:
                                            st.write(f"• {factor}")
                                        
                                        st.write("**Confidence Assessment:**")
                                        st.write(insights['confidence_explanation'])
                                        
                                        st.write("**Recommendations:**")
                                        for rec in insights['recommendations']:
                                            st.write(f"• {rec}")
                                
                                except ImportError:
                                    st.error("Natural Language Insights not available")
                                except Exception as e:
                                    st.error(f"Error generating insights: {e}")
                                    import traceback
                                    st.code(traceback.format_exc())
                            
                            # AI Commentary Service
                            if 'commentary_service' in st.session_state:
                                st.markdown("---")
                                st.subheader("💬 AI Commentary")
                                
                                if st.button("Generate Commentary", key="generate_forecast_commentary"):
                                    commentary_service = st.session_state.commentary_service
                                    
                                    with st.spinner("Generating commentary..."):
                                        try:
                                            # Prepare forecast result for commentary
                                            forecast_for_commentary = {
                                                'forecast': forecast_df['forecast'].values.tolist() if hasattr(forecast_df['forecast'].values, 'tolist') else list(forecast_df['forecast'].values),
                                                'dates': forecast_df.index.tolist() if hasattr(forecast_df.index, 'tolist') else list(forecast_df.index),
                                                'model_type': selected_model,
                                                'symbol': st.session_state.symbol
                                            }
                                            
                                            # Add confidence intervals if available
                                            if isinstance(forecast_result, dict):
                                                if 'lower_bound' in forecast_result:
                                                    forecast_for_commentary['lower_bound'] = forecast_result['lower_bound']
                                                if 'upper_bound' in forecast_result:
                                                    forecast_for_commentary['upper_bound'] = forecast_result['upper_bound']
                                            
                                            commentary = commentary_service.generate_forecast_commentary(
                                                symbol=st.session_state.symbol,
                                                forecast_result=forecast_for_commentary,
                                                model_type=selected_model,
                                                historical_data=data
                                            )
                                            
                                            st.info(commentary.get('summary', 'Commentary generated'))
                                            
                                            with st.expander("📊 Detailed Analysis", expanded=False):
                                                if 'trend_analysis' in commentary:
                                                    st.write("**Trend Analysis:**")
                                                    st.write(commentary['trend_analysis'])
                                                
                                                if 'insights' in commentary and commentary['insights']:
                                                    st.write("**Key Insights:**")
                                                    for insight in commentary['insights']:
                                                        st.write(f"• {insight}")
                                                
                                                if 'warnings' in commentary and commentary['warnings']:
                                                    st.write("**Warnings:**")
                                                    for warning in commentary['warnings']:
                                                        st.warning(f"⚠️ {warning}")
                                        except Exception as e:
                                            st.error(f"Error generating commentary: {e}")
                                            import traceback
                                            st.code(traceback.format_exc())
                
                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.info("Try adjusting the date range or selecting a different model.")
            
            # Display forecast using UI components
            if st.session_state.get('current_forecast') is not None:
                st.markdown(f"**Forecast using {st.session_state.current_model}**")
                
                try:
                    from trading.ui.forecast_components import render_forecast_results, render_confidence_metrics
                    
                    hist_data = st.session_state.forecast_data
                    forecast_df = st.session_state.current_forecast
                    forecast_result = st.session_state.get('current_forecast_result', {})
                    
                    # Prepare forecast data for component
                    forecast_data = {
                        'dates': forecast_df.index.tolist() if hasattr(forecast_df.index, 'tolist') else list(forecast_df.index),
                        'forecast': forecast_df['forecast'].values.tolist() if hasattr(forecast_df['forecast'].values, 'tolist') else list(forecast_df['forecast'].values),
                        'model_name': st.session_state.current_model
                    }
                    
                    # Add confidence intervals if available
                    if isinstance(forecast_result, dict):
                        if 'lower_bound' in forecast_result:
                            forecast_data['lower_bound'] = forecast_result['lower_bound']
                        if 'upper_bound' in forecast_result:
                            forecast_data['upper_bound'] = forecast_result['upper_bound']
                        if 'confidence' in forecast_result:
                            forecast_data['confidence'] = forecast_result['confidence']
                    
                    # Render forecast results
                    render_forecast_results(
                        forecast=forecast_data,
                        historical_data=hist_data,
                        symbol=st.session_state.symbol,
                        show_chart=True,
                        show_table=True
                    )
                    
                    # Show confidence metrics
                    if isinstance(forecast_result, dict) and ('lower_bound' in forecast_result or 'confidence' in forecast_result):
                        render_confidence_metrics(forecast_data)
                    
                except ImportError:
                    # Fallback to original display code
                    from utils.plotting_helper import create_forecast_chart
                    
                    hist_data = st.session_state.forecast_data
                    forecast_df = st.session_state.current_forecast
                    forecast_result = st.session_state.get('current_forecast_result', {})
                    
                    confidence_intervals = None
                    if isinstance(forecast_result, dict) and 'lower_bound' in forecast_result and 'upper_bound' in forecast_result:
                        confidence_intervals = {
                            'lower': forecast_result['lower_bound'],
                            'upper': forecast_result['upper_bound']
                        }
                    
                    fig = create_forecast_chart(
                        historical=hist_data,
                        forecast=forecast_df['forecast'].values,
                        forecast_dates=forecast_df.index,
                        confidence_intervals=confidence_intervals,
                        title=f"{st.session_state.symbol} - Historical & Forecast"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Forecast table
                    st.markdown("**Forecast Values:**")
                    display_df = forecast_df.copy()
                    display_df['forecast'] = display_df['forecast'].apply(
                        lambda x: f"${x:.2f}" if pd.notna(x) and x is not None else "N/A"
                    )
                    st.dataframe(display_df, use_container_width=True)
                
                # Download button
                csv = forecast_df.to_csv()
                st.download_button(
                    label="📥 Download Forecast CSV",
                    data=csv,
                    file_name=f"{st.session_state.symbol}_forecast.csv",
                    mime="text/csv"
                )
                
                # Add explainability section
                st.markdown("---")
                st.subheader("🔍 Model Explainability")
                
                with st.expander("View Feature Importance & Explanations", expanded=False):
                    try:
                        from trading.models.forecast_explainability import ForecastExplainability
                        
                        if st.button("Generate Explanation", key="generate_explanation"):
                            with st.spinner("Analyzing model predictions..."):
                                try:
                                    explainer = ForecastExplainability()
                                    model = st.session_state.get('current_model_instance')
                                    forecast_result = st.session_state.get('current_forecast_result', {})
                                    
                                    # Extract forecast value
                                    if isinstance(forecast_result, dict):
                                        forecast_values = forecast_result.get('forecast', [])
                                        forecast_value = float(forecast_values[0]) if len(forecast_values) > 0 else float(data['close'].iloc[-1])
                                    else:
                                        forecast_value = float(forecast_result) if isinstance(forecast_result, (int, float)) else float(data['close'].iloc[-1])
                                    
                                    # Prepare features
                                    features = data.copy()
                                    target_history = features['close'] if 'close' in features.columns else features.iloc[:, 0]
                                    
                                    # Generate explanation
                                    explanation = explainer.explain_forecast(
                                        forecast_id=f"forecast_{st.session_state.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                        symbol=st.session_state.symbol,
                                        forecast_value=forecast_value,
                                        model=model,
                                        features=features,
                                        target_history=target_history,
                                        horizon=st.session_state.forecast_horizon
                                    )
                                    
                                    # Display feature importance
                                    if explanation.get('success') and 'explanation' in explanation:
                                        expl_obj = explanation['explanation']
                                        if hasattr(expl_obj, 'feature_importance') and expl_obj.feature_importance:
                                            st.write("**Feature Importance:**")
                                            importance_data = expl_obj.feature_importance
                                            if isinstance(importance_data, dict):
                                                importance_df = pd.DataFrame(
                                                    list(importance_data.items()),
                                                    columns=['Feature', 'Importance']
                                                ).sort_values('Importance', ascending=False)
                                                
                                                import plotly.express as px
                                                fig = px.bar(
                                                    importance_df,
                                                    x='Importance',
                                                    y='Feature',
                                                    orientation='h',
                                                    title='Feature Importance (SHAP values)'
                                                )
                                                st.plotly_chart(fig, width='stretch')
                                    
                                    # Display explanation text
                                    if explanation.get('success') and 'explanation' in explanation:
                                        expl_obj = explanation['explanation']
                                        if hasattr(expl_obj, 'explanation_text') and expl_obj.explanation_text:
                                            st.write("**Explanation:**")
                                            st.write(expl_obj.explanation_text)
                                    
                                    st.success("✅ Explanation generated")
                                except Exception as e:
                                    st.error(f"Error generating explanation: {e}")
                                    import traceback
                                    st.code(traceback.format_exc())
                    
                    except ImportError:
                        st.warning("Forecast explainability requires SHAP library. Install with: pip install shap")
                    except Exception as e:
                        st.error(f"Error initializing explainability: {e}")
                
                # AI Commentary section
                st.markdown("---")
                st.subheader("🤖 AI Market Commentary")
                
                if st.button("Generate AI Commentary", key="generate_commentary"):
                    try:
                        from agents.llm.quant_gpt_commentary_agent import QuantGPTCommentaryAgent, CommentaryRequest, CommentaryType
                        import asyncio
                        
                        agent = QuantGPTCommentaryAgent()
                        
                        if not agent.available:
                            st.error("QuantGPT Commentary Agent not available. Check initialization.")
                        else:
                            with st.spinner("AI is analyzing the forecast..."):
                                # Prepare commentary request
                                request = CommentaryRequest(
                                    request_type=CommentaryType.REGIME_ANALYSIS,
                                    symbol=st.session_state.symbol,
                                    timestamp=datetime.now(),
                                    market_data=data,
                                    model_data={
                                        'model_type': st.session_state.current_model,
                                        'forecast': forecast_result,
                                        'horizon': st.session_state.forecast_horizon
                                    }
                                )
                                
                                # Generate commentary (handle async)
                                try:
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                                    commentary = loop.run_until_complete(agent.generate_commentary(request))
                                    loop.close()
                                    
                                    st.info(commentary.summary)
                                    
                                    with st.expander("📊 Detailed Analysis"):
                                        st.write("**Technical Analysis:**")
                                        st.write(commentary.detailed_analysis)
                                        
                                        if commentary.key_insights:
                                            st.write("**Key Insights:**")
                                            for insight in commentary.key_insights:
                                                st.write(f"• {insight}")
                                        
                                        if commentary.recommendations:
                                            st.write("**Recommendations:**")
                                            for rec in commentary.recommendations:
                                                st.write(f"• {rec}")
                                        
                                        if commentary.risk_warnings:
                                            st.write("**Risk Factors:**")
                                            for risk in commentary.risk_warnings:
                                                st.warning(f"⚠️ {risk}")
                                except Exception as e:
                                    st.error(f"Error generating commentary: {e}")
                    
                    except ImportError:
                        st.error("QuantGPT Commentary Agent not available")
                    except Exception as e:
                        st.error(f"Error: {e}")

with tab2:
    st.header("Advanced Forecasting")
    st.markdown("Full model configuration with hyperparameter tuning and feature engineering")
    
    if st.session_state.forecast_data is None:
        st.warning("⚠️ Please load data first in the Quick Forecast tab")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Configuration")
            
            # Model selection using registry
            try:
                from trading.models.model_registry import get_registry
                
                registry = get_registry()
                
                # Get all single-asset models (includes advanced ones, excludes GNN)
                available_models = registry.get_advanced_models()
                
                if not available_models:
                    st.warning("⚠️ No models available. Using default models.")
                    available_models = ["LSTM", "XGBoost", "Prophet", "ARIMA"]
                
                # Create friendly names
                model_display_names = {
                    'LSTM': 'LSTM (Deep Learning)',
                    'XGBoost': 'XGBoost (Gradient Boosting)',
                    'Prophet': 'Prophet (Seasonality)',
                    'ARIMA': 'ARIMA (Statistical)',
                    'Ensemble': 'Ensemble (Multi-Model)',
                    'TCN': 'TCN (Temporal Conv)',
                    'GARCH': 'GARCH (Volatility)',
                    'Autoformer': 'Autoformer (Advanced Transformer)',
                    'CatBoost': 'CatBoost (Categorical Boosting)',
                    'Ridge': 'Ridge (Linear Baseline)'
                }
                
                display_names = [model_display_names.get(m, m) for m in available_models if m in model_display_names or m not in model_display_names]
                
                selected_display = st.selectbox(
                    "Select Model Type",
                    display_names if display_names else available_models,
                    help="Advanced forecasting with all available models"
                )
                
                # Convert back to model name
                if display_names:
                    model_type = [k for k, v in model_display_names.items() if v == selected_display]
                    model_type = model_type[0] if model_type else selected_display
                else:
                    model_type = selected_display
                
                # Show model info
                model_info = registry.get_model_info(model_type)
                if model_info:
                    st.info(f"ℹ️ {model_info.get('description', 'No description')}")
                    
                    # Show requirements
                    reqs = []
                    if model_info.get('requires_gpu'):
                        reqs.append("🖥️ GPU recommended")
                    if model_info.get('min_data_points'):
                        reqs.append(f"📊 Min {model_info['min_data_points']} data points")
                    
                    if reqs:
                        st.caption(" • ".join(reqs))
            except Exception as e:
                st.warning(f"⚠️ Could not load model registry: {e}. Using default models.")
                model_type = st.selectbox(
                    "Model Type",
                    ["LSTM", "XGBoost", "Prophet", "ARIMA"]
                )
                model_info = {}
            
            st.markdown("---")
            st.markdown(f"**{model_type} Parameters**")
            
            # Model-specific hyperparameters
            params = {}
            
            if model_type == "LSTM":
                params['hidden_dim'] = st.slider("Hidden Dimension", 16, 256, 64, 16)
                params['num_layers'] = st.slider("Number of Layers", 1, 5, 2)
                params['dropout'] = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
                params['learning_rate'] = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
                params['sequence_length'] = st.slider("Sequence Length", 30, 120, 60, 10)
            
            elif model_type == "XGBoost":
                params['n_estimators'] = st.slider("Number of Trees", 50, 500, 100, 50)
                params['max_depth'] = st.slider("Max Depth", 3, 15, 5, 1)
                params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
                params['subsample'] = st.slider("Subsample Ratio", 0.5, 1.0, 0.8, 0.1)
                params['colsample_bytree'] = st.slider("Feature Sampling", 0.5, 1.0, 0.8, 0.1)
            
            elif model_type == "Prophet":
                params['changepoint_prior_scale'] = st.slider("Changepoint Prior Scale", 0.001, 0.5, 0.05, 0.001)
                params['seasonality_prior_scale'] = st.slider("Seasonality Prior Scale", 0.01, 10.0, 10.0, 0.1)
                params['holidays_prior_scale'] = st.slider("Holidays Prior Scale", 0.01, 10.0, 10.0, 0.1)
                params['seasonality_mode'] = st.selectbox("Seasonality Mode", ['additive', 'multiplicative'])
            
            elif model_type == "ARIMA":
                p = st.slider("AR Order (p)", 0, 10, 5, 1)
                d = st.slider("Differencing (d)", 0, 2, 1, 1)
                q = st.slider("MA Order (q)", 0, 10, 0, 1)
                params['order'] = (p, d, q)
                params['use_auto_arima'] = st.checkbox("Use Auto ARIMA", value=True)
            
            elif model_type == "GARCH":
                st.markdown("**GARCH Parameters:**")
                params['p'] = st.slider("p (AR order)", 1, 5, 1, 1)
                params['q'] = st.slider("q (MA order)", 1, 5, 1, 1)
            
            elif model_type == "Autoformer":
                st.markdown("**Autoformer Parameters:**")
                params['seq_len'] = st.slider("Sequence Length", 30, 120, 60, 10)
                params['pred_len'] = st.session_state.forecast_horizon
                params['d_model'] = st.slider("Model Dimension", 64, 512, 128, 32)
            
            elif model_type == "CatBoost":
                st.markdown("**CatBoost Parameters:**")
                params['iterations'] = st.slider("Iterations", 100, 1000, 500, 50)
                params['depth'] = st.slider("Tree Depth", 3, 10, 6, 1)
                params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.1, 0.03, 0.01)
            
            elif model_type == "TCN":
                st.markdown("**TCN Parameters:**")
                params['num_channels'] = [64, 128, 256]  # Can be made configurable
                params['kernel_size'] = st.slider("Kernel Size", 2, 5, 3, 1)
                params['dropout'] = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
            
            elif model_type == "Ridge":
                st.markdown("**Ridge Parameters:**")
                params['alpha'] = st.slider("Regularization (alpha)", 0.01, 10.0, 1.0, 0.1)
            
            st.markdown("---")
            st.subheader("🔧 Feature Engineering")
            
            use_technical = st.checkbox("Add Technical Indicators", value=False)
            indicators = []
            if use_technical:
                indicators = st.multiselect(
                    "Select Indicators",
                    ["SMA", "EMA", "RSI", "MACD", "Bollinger Bands", "ATR"],
                    default=["SMA", "RSI"]
                )
            
            use_lags = st.checkbox("Add Lag Features", value=False)
            lag_periods = []
            if use_lags:
                lag_periods = st.multiselect(
                    "Lag Periods",
                    [1, 2, 3, 5, 7, 14, 21, 30],
                    default=[1, 7, 14]
                )
            
            use_macro_features = st.checkbox(
                "Include macro-economic features",
                value=False,
                help="Add features like interest rates, inflation, GDP growth"
            )
            
            macro_features_list = []
            if use_macro_features:
                macro_features_list = st.multiselect(
                    "Select Macro Features",
                    ["interest_rate", "inflation", "gdp_growth", "unemployment"],
                    default=["interest_rate", "inflation"],
                    help="Choose which macroeconomic indicators to include"
                )
            
            normalize = st.checkbox("Normalize Data", value=True)
            
            train_button = st.button("🚀 Train Model", type="primary", width='stretch')
        
        with col2:
            st.subheader("📊 Results")
            
            if train_button:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Prepare data
                    data = st.session_state.forecast_data.copy()
                    
                    # Ensure proper column names (capitalize for FeatureEngineering)
                    if 'close' in data.columns:
                        data['Close'] = data['close']
                    if 'open' in data.columns:
                        data['Open'] = data['open']
                    if 'high' in data.columns:
                        data['High'] = data['high']
                    if 'low' in data.columns:
                        data['Low'] = data['low']
                    if 'volume' in data.columns:
                        data['Volume'] = data['volume']
                    
                    # Ensure we have required columns
                    if 'Open' not in data.columns:
                        data['Open'] = data['Close']
                    if 'High' not in data.columns:
                        data['High'] = data['Close']
                    if 'Low' not in data.columns:
                        data['Low'] = data['Close']
                    if 'Volume' not in data.columns:
                        data['Volume'] = 1000000
                    
                    # Feature engineering
                    if use_technical:
                        status_text.text("Adding technical indicators...")
                        progress_bar.progress(0.2)
                        fe = FeatureEngineering()
                        
                        if "SMA" in indicators or "EMA" in indicators:
                            ma_features = fe.calculate_moving_averages(data)
                            data = pd.concat([data, ma_features], axis=1)
                        
                        if "RSI" in indicators:
                            rsi_features = fe.calculate_rsi(data)
                            data = pd.concat([data, rsi_features], axis=1)
                        
                        if "MACD" in indicators:
                            macd_features = fe.calculate_macd(data)
                            data = pd.concat([data, macd_features], axis=1)
                        
                        if "Bollinger Bands" in indicators:
                            bb_features = fe.calculate_bollinger_bands(data)
                            data = pd.concat([data, bb_features], axis=1)
                    
                    if use_lags:
                        status_text.text("Adding lag features...")
                        progress_bar.progress(0.4)
                        for lag in lag_periods:
                            data[f'lag_{lag}'] = data['Close'].shift(lag)
                    
                    # Macro feature engineering
                    if use_macro_features and macro_features_list:
                        status_text.text("Adding macro-economic features...")
                        progress_bar.progress(0.5)
                        try:
                            from trading.feature_engineering.macro_feature_engineering import MacroFeatureEngineer
                            
                            engineer = MacroFeatureEngineer()
                            
                            # Add macro features to data
                            data_with_macro = engineer.add_macro_features(
                                data=data,
                                country='US',
                                features=macro_features_list
                            )
                            
                            # Show what was added
                            new_cols = set(data_with_macro.columns) - set(data.columns)
                            if new_cols:
                                st.write("**Macro features added:**")
                                for col in new_cols:
                                    st.write(f"• {col}")
                            
                            # Use data_with_macro for training
                            data = data_with_macro
                        except ImportError:
                            st.warning("Macro Feature Engineer not available. Skipping macro features.")
                        except Exception as e:
                            st.warning(f"Could not add macro features: {e}. Continuing without them.")
                    
                    # Remove NaN
                    data = data.dropna()
                    
                    if len(data) < 30:
                        st.error("Not enough data after feature engineering. Try fewer features or more historical data.")
                        progress_bar.empty()
                        status_text.empty()
                    else:
                        # Normalize
                        if normalize:
                            status_text.text("Normalizing data...")
                            progress_bar.progress(0.6)
                            preprocessor = DataPreprocessor()
                            # Normalize only numeric columns
                            numeric_cols = data.select_dtypes(include=[np.number]).columns
                            for col in numeric_cols:
                                if col != 'Close':  # Don't normalize target yet
                                    data[col] = (data[col] - data[col].mean()) / (data[col].std() + 1e-8)
                        
                        # Train model
                        status_text.text(f"Training {model_type}...")
                        progress_bar.progress(0.8)
                        
                        # Create model using registry
                        try:
                            from trading.models.model_registry import get_registry
                            registry = get_registry()
                            ModelClass = registry.get(model_type)
                            
                            if ModelClass is None:
                                st.error(f"Model {model_type} not available")
                                model = None
                            else:
                                # Create model config
                                model_config = {
                                    "target_column": "close" if "close" in data.columns else "Close",
                                    "feature_columns": list(data.columns) if len(data.columns) > 0 else ["close"]
                                }
                                
                                # Add model-specific parameters
                                if model_type == "LSTM":
                                    model_config.update({
                                        "sequence_length": params.get('sequence_length', 60),
                                        "hidden_size": params.get('hidden_dim', 64),
                                        "num_layers": params.get('num_layers', 2),
                                        "dropout": params.get('dropout', 0.2),
                                        "learning_rate": params.get('learning_rate', 0.001),
                                        "input_size": len(data.columns) if len(data.columns) > 0 else 1
                                    })
                                elif model_type == "XGBoost":
                                    model_config.update({
                                        "n_estimators": params.get('n_estimators', 100),
                                        "max_depth": params.get('max_depth', 5),
                                        "learning_rate": params.get('learning_rate', 0.1),
                                        "subsample": params.get('subsample', 0.8),
                                        "colsample_bytree": params.get('colsample_bytree', 0.8)
                                    })
                                elif model_type == "Prophet":
                                    model_config.update({
                                        "date_column": "ds",
                                        "prophet_params": {
                                            "changepoint_prior_scale": params.get('changepoint_prior_scale', 0.05),
                                            "seasonality_prior_scale": params.get('seasonality_prior_scale', 10.0),
                                            "holidays_prior_scale": params.get('holidays_prior_scale', 10.0),
                                            "seasonality_mode": params.get('seasonality_mode', 'additive')
                                        }
                                    })
                                elif model_type == "ARIMA":
                                    model_config.update({
                                        "order": params.get('order', (5, 1, 0)),
                                        "use_auto_arima": params.get('use_auto_arima', True)
                                    })
                                elif model_type == "GARCH":
                                    model_config.update({
                                        "p": params.get('p', 1),
                                        "q": params.get('q', 1)
                                    })
                                elif model_type == "Autoformer":
                                    model_config.update({
                                        "seq_len": params.get('seq_len', 60),
                                        "pred_len": params.get('pred_len', st.session_state.forecast_horizon),
                                        "d_model": params.get('d_model', 128)
                                    })
                                elif model_type == "CatBoost":
                                    model_config.update({
                                        "iterations": params.get('iterations', 500),
                                        "depth": params.get('depth', 6),
                                        "learning_rate": params.get('learning_rate', 0.03)
                                    })
                                elif model_type == "TCN":
                                    model_config.update({
                                        "num_channels": params.get('num_channels', [64, 128, 256]),
                                        "kernel_size": params.get('kernel_size', 3),
                                        "dropout": params.get('dropout', 0.2)
                                    })
                                elif model_type == "Ridge":
                                    model_config.update({
                                        "alpha": params.get('alpha', 1.0)
                                    })
                                
                                model = ModelClass(model_config)
                        except Exception as e:
                            st.warning(f"Could not use model registry: {e}. Using direct imports.")
                            # Fallback to direct imports
                            model_config = {
                                "target_column": "close" if "close" in data.columns else "Close"
                            }
                            
                            if model_type == "LSTM":
                                model_config.update({
                                    "sequence_length": params.get('sequence_length', 60),
                                    "hidden_size": params.get('hidden_dim', 64),
                                    "num_layers": params.get('num_layers', 2),
                                    "dropout": params.get('dropout', 0.2),
                                    "learning_rate": params.get('learning_rate', 0.001),
                                    "feature_columns": list(data.columns) if len(data.columns) > 0 else ["close"],
                                    "input_size": len(data.columns) if len(data.columns) > 0 else 1
                                })
                                model = LSTMForecaster(model_config)
                            elif model_type == "XGBoost":
                                model_config.update({
                                    "n_estimators": params.get('n_estimators', 100),
                                    "max_depth": params.get('max_depth', 5),
                                    "learning_rate": params.get('learning_rate', 0.1),
                                    "subsample": params.get('subsample', 0.8),
                                    "colsample_bytree": params.get('colsample_bytree', 0.8)
                                })
                                model = XGBoostModel(model_config)
                            elif model_type == "Prophet":
                                model_config.update({
                                    "date_column": "ds",
                                    "prophet_params": {
                                        "changepoint_prior_scale": params.get('changepoint_prior_scale', 0.05),
                                        "seasonality_prior_scale": params.get('seasonality_prior_scale', 10.0),
                                        "holidays_prior_scale": params.get('holidays_prior_scale', 10.0),
                                        "seasonality_mode": params.get('seasonality_mode', 'additive')
                                    }
                                })
                                model = ProphetModel(model_config)
                            elif model_type == "ARIMA":
                                model_config.update({
                                    "order": params.get('order', (5, 1, 0)),
                                    "use_auto_arima": params.get('use_auto_arima', True)
                                })
                                model = ARIMAModel(model_config)
                            else:
                                st.error(f"Model {model_type} not available with direct imports. Please use model registry.")
                                model = None
                        
                        # Train model
                        if model_type == "Prophet":
                            train_df = pd.DataFrame({
                                'ds': data.index,
                                'y': data[model_config["target_column"]]
                            })
                            fit_result = model.fit(train_df)
                        elif model_type == "ARIMA":
                            fit_result = model.fit(data[model_config["target_column"]])
                        else:
                            fit_result = model.fit(data, data[model_config["target_column"]])
                        
                        # Generate forecast - try with uncertainty if available
                        progress_bar.progress(0.9)
                        if hasattr(model, 'forecast_with_uncertainty'):
                            try:
                                forecast_result = model.forecast_with_uncertainty(data, horizon=st.session_state.forecast_horizon, num_samples=100)
                            except Exception as e:
                                st.warning(f"Could not generate forecast with uncertainty: {e}. Using standard forecast.")
                                forecast_result = model.forecast(data, horizon=st.session_state.forecast_horizon)
                        else:
                            forecast_result = model.forecast(data, horizon=st.session_state.forecast_horizon)
                        
                        # Store full forecast result for confidence intervals
                        st.session_state.current_forecast_result = forecast_result
                        st.session_state.current_model_instance = model  # Store model instance for explainability
                        
                        # Postprocess forecast
                        try:
                            from trading.forecasting.forecast_postprocessor import ForecastPostprocessor
                            
                            postprocessor = ForecastPostprocessor()
                            
                            # Extract forecast values for postprocessing
                            if isinstance(forecast_result, dict):
                                forecast_vals = forecast_result.get('forecast', [])
                            else:
                                forecast_vals = forecast_result
                            
                            # Postprocess forecast
                            processed_forecast = postprocessor.process(
                                forecast=forecast_vals,
                                historical_data=data,
                                apply_smoothing=True,
                                remove_outliers=True,
                                ensure_realistic_bounds=True
                            )
                            
                            # Update forecast_result with processed version
                            if isinstance(forecast_result, dict):
                                forecast_result['forecast'] = processed_forecast['values']
                                forecast_result['postprocessing_notes'] = processed_forecast.get('notes', [])
                            else:
                                forecast_result = processed_forecast['values']
                            
                            # Update session state
                            st.session_state.current_forecast_result = forecast_result
                        except ImportError:
                            pass  # Silently fail if postprocessor not available
                        except Exception as e:
                            logger.warning(f"Forecast postprocessing failed: {e}")
                        
                        progress_bar.progress(1.0)
                        status_text.text("Complete!")
                        
                        st.success("✅ Model trained successfully!")
                        
                        # Extract forecast
                        if isinstance(forecast_result, dict):
                            forecast_values = forecast_result.get('forecast', [])
                            forecast_dates = forecast_result.get('dates', pd.date_range(
                                start=data.index[-1] + timedelta(days=1),
                                periods=st.session_state.forecast_horizon,
                                freq='D'
                            ))
                        else:
                            forecast_values = forecast_result
                            forecast_dates = pd.date_range(
                                start=data.index[-1] + timedelta(days=1),
                                periods=st.session_state.forecast_horizon,
                                freq='D'
                            )
                        
                        if isinstance(forecast_values, (list, np.ndarray)):
                            forecast_values = np.array(forecast_values).flatten()
                        else:
                            forecast_values = np.array([forecast_values] * st.session_state.forecast_horizon)
                        
                        # Store forecast
                        forecast_df = pd.DataFrame({
                            'forecast': forecast_values
                        }, index=forecast_dates[:len(forecast_values)])
                        
                        st.session_state.advanced_forecast = forecast_df
                        st.session_state.advanced_model = model_type
                        
                        # Display forecast chart
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data[model_config["target_column"]],
                            mode='lines',
                            name='Historical',
                            line=dict(color='blue', width=2)
                        ))
                        fig.add_trace(go.Scatter(
                            x=forecast_df.index,
                            y=forecast_df['forecast'],
                            mode='lines+markers',
                            name='Forecast',
                            line=dict(color='red', width=2, dash='dash'),
                            marker=dict(size=8)
                        ))
                        fig.update_layout(
                            title=f"{st.session_state.symbol} - Advanced Forecast ({model_type})",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            hovermode='x unified',
                            height=500
                        )
                        st.plotly_chart(fig, width='stretch')
                        
                        # Display forecast table
                        st.markdown("**Forecast Values:**")
                        display_df = forecast_df.copy()
                        display_df['forecast'] = display_df['forecast'].apply(
                            lambda x: f"${x:.2f}" if pd.notna(x) and x is not None else "N/A"
                        )
                        st.dataframe(display_df, width='stretch')
                        
                        # Add explainability section
                        st.markdown("---")
                        st.subheader("🔍 Model Explainability")
                        
                        with st.expander("📊 View Feature Importance & Explanations", expanded=False):
                            st.write("Understand what drives the model's predictions")
                            
                            if st.button("Generate Explanation", key="explain_btn_tab2"):
                                try:
                                    with st.spinner("Analyzing model predictions..."):
                                        from trading.models.forecast_explainability import ForecastExplainability
                                        
                                        # Initialize explainer
                                        explainer = ForecastExplainability()
                                        
                                        # Extract forecast value and prepare features
                                        forecast_value = forecast_result.get('forecast', [])
                                        if isinstance(forecast_value, (list, np.ndarray)) and len(forecast_value) > 0:
                                            forecast_value = float(forecast_value[0])
                                        else:
                                            forecast_value = float(forecast_value) if isinstance(forecast_value, (int, float)) else float(data[model_config["target_column"]].iloc[-1])
                                        
                                        # Prepare features DataFrame
                                        features = data.copy()
                                        if model_config["target_column"] in features.columns:
                                            target_history = features[model_config["target_column"]]
                                        else:
                                            target_history = features.iloc[:, 0]
                                        
                                        explanation = explainer.explain_forecast(
                                            forecast_id=f"forecast_{st.session_state.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                            symbol=st.session_state.symbol,
                                            forecast_value=forecast_value,
                                            model=model,
                                            features=features,
                                            target_history=target_history,
                                            horizon=st.session_state.forecast_horizon
                                        )
                                        
                                        # Display feature importance
                                        if explanation.get('success') and 'explanation' in explanation:
                                            expl_obj = explanation['explanation']
                                            if hasattr(expl_obj, 'feature_importance') and expl_obj.feature_importance:
                                                st.write("**📊 Feature Importance:**")
                                                
                                                importance_data = expl_obj.feature_importance
                                                if isinstance(importance_data, dict):
                                                    importance_df = pd.DataFrame(
                                                        importance_data.items(),
                                                        columns=['Feature', 'Importance']
                                                    ).sort_values('Importance', ascending=False)
                                                    
                                                    # Bar chart
                                                    fig_imp = px.bar(
                                                        importance_df,
                                                        x='Importance',
                                                        y='Feature',
                                                        orientation='h',
                                                        title='What Drives the Predictions?',
                                                        labels={'Importance': 'Impact on Prediction'}
                                                    )
                                                    st.plotly_chart(fig_imp, width='stretch')
                                        
                                        # Text explanation
                                        if explanation.get('success') and 'explanation' in explanation:
                                            expl_obj = explanation['explanation']
                                            if hasattr(expl_obj, 'explanation_text') and expl_obj.explanation_text:
                                                st.write("**💬 Plain English Explanation:**")
                                                st.info(expl_obj.explanation_text)
                                        
                                        st.success("✅ Explanation generated successfully!")
                                
                                except ImportError:
                                    st.warning("⚠️ Forecast explainability requires SHAP library")
                                    st.code("pip install shap")
                                except Exception as e:
                                    st.error(f"Error generating explanation: {e}")
                
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                finally:
                    progress_bar.empty()
                    status_text.empty()
            
            # Display previous forecast if exists
            if st.session_state.get('advanced_forecast') is not None:
                st.markdown("---")
                st.markdown(f"**Previous Forecast ({st.session_state.get('advanced_model', 'Unknown')})**")
                prev_forecast = st.session_state.advanced_forecast
                st.dataframe(prev_forecast.tail(10), width='stretch')

with tab3:
    st.header("AI Model Selection")
    st.markdown("""
    🤖 Let AI analyze your data and recommend the best forecasting model.
    
    The AI considers:
    - Data characteristics (trend, seasonality, volatility)
    - Historical model performance
    - Forecast horizon
    - Computational efficiency
    """)
    
    if st.session_state.forecast_data is None:
        st.warning("⚠️ Please load data first in the Quick Forecast tab")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Data Analysis")
            
            analyze_button = st.button(
                "🔍 Analyze Data & Recommend Model",
                type="primary",
                width='stretch'
            )
            
            if analyze_button:
                with st.spinner("AI analyzing data characteristics..."):
                    try:
                        # Initialize agent
                        agent = ModelSelectorAgent()
                        
                        # Detect market regime from data
                        price_data = st.session_state.forecast_data['close'].values
                        horizon = st.session_state.forecast_horizon
                        
                        # Determine horizon enum
                        from trading.agents.model_selector_agent import ForecastingHorizon, MarketRegime
                        
                        if horizon <= 7:
                            horizon_enum = ForecastingHorizon.SHORT_TERM
                        elif horizon <= 30:
                            horizon_enum = ForecastingHorizon.MEDIUM_TERM
                        else:
                            horizon_enum = ForecastingHorizon.LONG_TERM
                        
                        # Simple market regime detection
                        price_trend = (price_data[-1] - price_data[0]) / price_data[0]
                        volatility = price_data.std()
                        
                        if price_trend > 0.05:
                            regime_enum = MarketRegime.TRENDING_UP
                        elif price_trend < -0.05:
                            regime_enum = MarketRegime.TRENDING_DOWN
                        elif volatility > price_data.mean() * 0.1:
                            regime_enum = MarketRegime.VOLATILE
                        else:
                            regime_enum = MarketRegime.SIDEWAYS
                        
                        # Use agent's select_model method
                        selected_model_id, confidence = agent.select_model(
                            horizon=horizon_enum,
                            market_regime=regime_enum,
                            data_length=len(price_data),
                            required_features=[],
                            performance_weight=0.6,
                            capability_weight=0.4
                        )
                        
                        # Get recommendations
                        recommendations = agent.get_model_recommendations(
                            horizon_enum, regime_enum, top_k=3
                        )
                        
                        # Map model types to display names
                        model_name_map = {
                            'lstm': 'LSTM (Deep Learning)',
                            'xgboost': 'XGBoost (Gradient Boosting)',
                            'prophet': 'Prophet (Facebook)',
                            'arima': 'ARIMA (Statistical)',
                            'transformer': 'Transformer (Attention)',
                            'ensemble': 'Ensemble (Multiple Models)'
                        }
                        
                        # Extract model type from selected_model_id
                        model_type = selected_model_id.split('_')[0] if '_' in selected_model_id else selected_model_id
                        display_name = model_name_map.get(model_type.lower(), selected_model_id)
                        
                        # Create recommendation dict
                        recommendation = {
                            'model_name': display_name,
                            'confidence': confidence,
                            'reasoning': f"Selected based on data characteristics and forecast horizon of {horizon} days. Market regime: {regime_enum.value}.",
                            'data_characteristics': {
                                'Data Points': len(price_data),
                                'Forecast Horizon': f"{horizon} days",
                                'Volatility': f"{volatility:.2f}",
                                'Trend': f"{price_trend*100:.2f}%",
                                'Market Regime': regime_enum.value
                            },
                            'alternatives': [
                                {
                                    'model_name': model_name_map.get(rec.get('model_type', '').lower(), rec.get('model_id', '')),
                                    'confidence': rec.get('total_score', 0.5),
                                    'reason': f"Score: {rec.get('total_score', 0):.2f}"
                                }
                                for rec in recommendations[:3]
                            ] if recommendations else []
                        }
                        
                        st.session_state.ai_recommendation = recommendation
                    
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        
        with col2:
            st.subheader("💡 AI Recommendation")
            
            if st.session_state.get('ai_recommendation'):
                rec = st.session_state.ai_recommendation
                
                # Main recommendation
                st.success(f"**Recommended Model: {rec['model_name']}**")
                
                # Confidence
                confidence_pct = rec.get('confidence', 0.75) * 100
                st.metric("Confidence", f"{confidence_pct:.1f}%")
                
                # Progress bar for confidence
                st.progress(rec.get('confidence', 0.75))
                
                # Reasoning
                with st.expander("🧠 Why this model?", expanded=True):
                    st.markdown(rec.get('reasoning', 'Model selected based on data analysis'))
                    
                    if 'data_characteristics' in rec:
                        st.markdown("**Data Characteristics:**")
                        chars = rec['data_characteristics']
                        for key, value in chars.items():
                            st.text(f"• {key}: {value}")
                
                # Alternatives
                if rec.get('alternatives'):
                    with st.expander("🔄 Alternative Models"):
                        for alt in rec['alternatives']:
                            st.markdown(f"**{alt['model_name']}**")
                            st.caption(f"Confidence: {alt.get('confidence', 0)*100:.1f}%")
                            st.caption(alt.get('reason', ''))
                            st.markdown("---")
                
                # Action buttons
                col_a, col_b = st.columns(2)
                
                with col_a:
                    if st.button("✅ Use Recommended Model", width='stretch'):
                        st.session_state.selected_model = rec['model_name']
                        st.success(f"Selected: {rec['model_name']}")
                        st.info("Go to Quick Forecast tab to generate forecast")
                
                with col_b:
                    if st.button("🔄 Choose Different Model", width='stretch'):
                        st.session_state.show_override = True
                
                if st.session_state.get('show_override'):
                    st.markdown("**Override AI Selection:**")
                    override = st.selectbox(
                        "Select model:",
                        ["LSTM (Deep Learning)", "XGBoost (Gradient Boosting)", 
                         "Prophet (Facebook)", "ARIMA (Statistical)"]
                    )
                    if st.button("Confirm Override"):
                        st.session_state.selected_model = override
                        st.session_state.show_override = False
                        st.success(f"Selected: {override}")
        
        # Hybrid Model Selector Section
        st.markdown("---")
        st.subheader("🎯 Hybrid Model Selection")
        
        st.write("""
        Hybrid selection combines multiple model types and automatically 
        switches between them based on market conditions.
        """)
        
        if st.button("Use Hybrid Selector"):
            try:
                from trading.forecasting.hybrid_model_selector import HybridModelSelector
                
                data = st.session_state.forecast_data
                selector = HybridModelSelector()
                
                with st.spinner("Analyzing market conditions and selecting models..."):
                    # Detect market regime
                    regime = selector.detect_market_regime(data)
                    
                    st.info(f"📊 Detected market regime: **{regime}**")
                    
                    # Select best models for this regime
                    selected_models = selector.select_models_for_regime(regime)
                    
                    st.write("**Recommended models for current conditions:**")
                    for i, model_config in enumerate(selected_models, 1):
                        st.write(f"{i}. {model_config['name']} - {model_config['reason']}")
                    
                    # Store selected models
                    st.session_state.hybrid_selected_models = selected_models
                    st.session_state.hybrid_regime = regime
                    
                    # Auto-train hybrid ensemble
                    if st.button("Train Hybrid Ensemble"):
                        results = selector.train_hybrid_ensemble(
                            data=data,
                            models=selected_models,
                            horizon=st.session_state.forecast_horizon
                        )
                        
                        st.success("✅ Hybrid ensemble trained!")
                        
                        # Show results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Ensemble R²", f"{results.get('r2', 0):.4f}")
                        with col2:
                            st.metric("RMSE", f"{results.get('rmse', 0):.2f}")
                        with col3:
                            st.metric("Models Used", len(selected_models))
                        
                        # Store results
                        st.session_state.hybrid_ensemble_results = results
            
            except ImportError:
                st.error("Hybrid Model Selector not available")
            except Exception as e:
                st.error(f"Error using hybrid selector: {e}")
                import traceback
                st.code(traceback.format_exc())

with tab4:
    st.header("Model Comparison")
    st.markdown("Compare multiple models side-by-side and create ensemble forecasts")
    
    if st.session_state.forecast_data is None:
        st.warning("⚠️ Please load data first in the Quick Forecast tab")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📊 Select Models")
            
            models_to_compare = st.multiselect(
                "Choose models to compare:",
                ["LSTM", "XGBoost", "Prophet", "ARIMA"],
                default=["LSTM", "XGBoost"],
                help="Select 2-4 models to compare"
            )
            
            if len(models_to_compare) < 2:
                st.warning("Please select at least 2 models")
            elif len(models_to_compare) > 4:
                st.warning("Please select no more than 4 models")
            
            create_ensemble = st.checkbox("Create Ensemble Forecast", value=True)
            
            compare_button = st.button(
                "🚀 Compare Models",
                type="primary",
                width='stretch',
                disabled=len(models_to_compare) < 2
            )
        
        with col2:
            st.subheader("📈 Comparison Results")
            
            if compare_button and len(models_to_compare) >= 2:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    data = st.session_state.forecast_data.copy()
                    horizon = st.session_state.forecast_horizon
                    
                    # Ensure proper format
                    if not isinstance(data.index, pd.DatetimeIndex):
                        data.index = pd.to_datetime(data.index)
                    
                    if 'close' in data.columns:
                        data['Close'] = data['close']
                    
                    forecasts = {}
                    model_configs = {}
                    
                    # Train and forecast for each model
                    total_models = len(models_to_compare)
                    for idx, model_name in enumerate(models_to_compare):
                        status_text.text(f"Training {model_name} ({idx+1}/{total_models})...")
                        progress_bar.progress((idx) / (total_models + 1))
                        
                        try:
                            # Create model config
                            if model_name == "LSTM":
                                # LSTMForecaster uses config dictionary
                                config = {
                                    "target_column": "close" if "close" in data.columns else "Close",
                                    "sequence_length": 60,
                                    "hidden_size": 64,
                                    "num_layers": 2,
                                    "dropout": 0.2,
                                    "learning_rate": 0.001,
                                    "feature_columns": list(data.columns) if len(data.columns) > 0 else ["close"],
                                    "input_size": len(data.columns) if len(data.columns) > 0 else 1
                                }
                                model = LSTMForecaster(config)
                            elif model_name == "XGBoost":
                                config = {
                                    "target_column": "close" if "close" in data.columns else "Close",
                                    "n_estimators": 100,
                                    "max_depth": 5,
                                    "learning_rate": 0.1
                                }
                                model = XGBoostModel(config)
                            elif model_name == "Prophet":
                                config = {
                                    "date_column": "ds",
                                    "target_column": "close" if 'close' in data.columns else 'Close',
                                    "prophet_params": {}
                                }
                                model = ProphetModel(config)
                            elif model_name == "ARIMA":
                                config = {
                                    "order": (5, 1, 0),
                                    "use_auto_arima": True
                                }
                                model = ARIMAModel(config)
                            
                            model_configs[model_name] = config
                            
                            # Train model
                            if model_name == "Prophet":
                                train_df = pd.DataFrame({
                                    'ds': data.index,
                                    'y': data[config["target_column"]]
                                })
                                model.fit(train_df)
                            elif model_name == "ARIMA":
                                model.fit(data[config["target_column"]])
                            else:
                                model.fit(data, data[config["target_column"]])
                            
                            # Generate forecast
                            forecast_result = model.forecast(data, horizon=horizon)
                            
                            # Extract forecast values
                            if isinstance(forecast_result, dict):
                                forecast_values = forecast_result.get('forecast', [])
                                forecast_dates = forecast_result.get('dates', pd.date_range(
                                    start=data.index[-1] + timedelta(days=1),
                                    periods=horizon,
                                    freq='D'
                                ))
                            else:
                                forecast_values = forecast_result
                                forecast_dates = pd.date_range(
                                    start=data.index[-1] + timedelta(days=1),
                                    periods=horizon,
                                    freq='D'
                                )
                            
                            if isinstance(forecast_values, (list, np.ndarray)):
                                forecast_values = np.array(forecast_values).flatten()
                            else:
                                forecast_values = np.array([forecast_values] * horizon)
                            
                            forecasts[model_name] = {
                                'values': forecast_values,
                                'dates': forecast_dates[:len(forecast_values)]
                            }
                            
                        except Exception as e:
                            st.warning(f"Failed to train {model_name}: {str(e)}")
                            continue
                    
                    progress_bar.progress(0.9)
                    
                    # Create ensemble if requested
                    if create_ensemble and len(forecasts) >= 2:
                        status_text.text("Creating ensemble forecast...")
                        ensemble_values = []
                        for i in range(horizon):
                            values_at_step = [f['values'][i] for f in forecasts.values() if i < len(f['values'])]
                            if values_at_step:
                                ensemble_values.append(np.mean(values_at_step))
                            else:
                                ensemble_values.append(0)
                        
                        forecasts['Ensemble'] = {
                            'values': np.array(ensemble_values),
                            'dates': forecasts[list(forecasts.keys())[0]]['dates']
                        }
                    
                    progress_bar.progress(1.0)
                    status_text.text("Complete!")
                    
                    # Store results
                    st.session_state.comparison_results = forecasts
                    
                    st.success(f"✅ Compared {len(forecasts)} models successfully!")
                    
                    # Display comparison chart
                    fig = go.Figure()
                    
                    # Historical data
                    hist_data = st.session_state.forecast_data
                    fig.add_trace(go.Scatter(
                        x=hist_data.index,
                        y=hist_data['close'],
                        mode='lines',
                        name='Historical',
                        line=dict(color='black', width=2)
                    ))
                    
                    # Forecasts from each model
                    colors = ['red', 'blue', 'green', 'orange', 'purple']
                    for idx, (model_name, forecast_data) in enumerate(forecasts.items()):
                        fig.add_trace(go.Scatter(
                            x=forecast_data['dates'],
                            y=forecast_data['values'],
                            mode='lines+markers',
                            name=model_name,
                            line=dict(color=colors[idx % len(colors)], width=2, dash='dash' if model_name == 'Ensemble' else None),
                            marker=dict(size=6)
                        ))
                    
                    fig.update_layout(
                        title=f"{st.session_state.symbol} - Model Comparison",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        hovermode='x unified',
                        height=600
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    # Metrics table
                    st.markdown("---")
                    st.subheader("📊 Performance Metrics")
                    
                    metrics_data = []
                    for model_name, forecast_data in forecasts.items():
                        if model_name == 'Ensemble':
                            continue
                        
                        # Calculate simple metrics
                        forecast_vals = forecast_data['values']
                        last_price = hist_data['close'].iloc[-1]
                        
                        # Mean forecast
                        mean_forecast = np.mean(forecast_vals)
                        
                        # Forecast range
                        forecast_range = np.max(forecast_vals) - np.min(forecast_vals)
                        
                        # Trend (first vs last)
                        trend = (forecast_vals[-1] - forecast_vals[0]) / forecast_vals[0] * 100 if len(forecast_vals) > 1 else 0
                        
                        metrics_data.append({
                            'Model': model_name,
                            'Mean Forecast': f"${mean_forecast:.2f}",
                            'Range': f"${forecast_range:.2f}",
                            'Trend (%)': f"{trend:.2f}%",
                            'Std Dev': f"${np.std(forecast_vals):.2f}"
                        })
                    
                    if metrics_data:
                        metrics_df = pd.DataFrame(metrics_data)
                        st.dataframe(metrics_df, width='stretch')
                    
                    # Forecast table
                    st.markdown("---")
                    st.subheader("📋 Forecast Values")
                    
                    # Create comparison DataFrame
                    comparison_df = pd.DataFrame({
                        'Date': forecasts[list(forecasts.keys())[0]]['dates']
                    })
                    
                    for model_name, forecast_data in forecasts.items():
                        comparison_df[model_name] = forecast_data['values'][:len(comparison_df)]
                    
                    # Format for display
                    display_df = comparison_df.copy()
                    for col in display_df.columns:
                        if col != 'Date' and col in forecasts:
                            display_df[col] = display_df[col].apply(
                                lambda x: f"${x:.2f}" if pd.notna(x) and x is not None else "N/A"
                            )
                    
                    st.dataframe(display_df, width='stretch')
                    
                    # Download button
                    csv = comparison_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Comparison CSV",
                        data=csv,
                        file_name=f"{st.session_state.symbol}_model_comparison.csv",
                        mime="text/csv"
                    )
                
                except Exception as e:
                    st.error(f"Comparison failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                finally:
                    progress_bar.empty()
                    status_text.empty()
            
            # Display previous comparison if exists
            if st.session_state.get('comparison_results'):
                st.markdown("---")
                st.markdown("**Previous Comparison Results**")
                prev_results = st.session_state.comparison_results
                st.info(f"Compared {len(prev_results)} models in previous run")

with tab5:
    st.header("Market Analysis")
    st.markdown("Technical indicators, market regime detection, trend analysis, and correlation tools")
    
    if st.session_state.forecast_data is None:
        st.warning("⚠️ Please load data first in the Quick Forecast tab")
    else:
        # Analysis options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_technical = st.checkbox("Technical Indicators", value=True)
        with col2:
            show_regime = st.checkbox("Market Regime", value=True)
        with col3:
            show_trend = st.checkbox("Trend Analysis", value=True)
        
        analyze_button = st.button("🔍 Run Analysis", type="primary", width='stretch')
        
        if analyze_button:
            data = st.session_state.forecast_data.copy()
            
            # Ensure proper column names
            if 'close' in data.columns:
                data['Close'] = data['close']
            if 'open' in data.columns:
                data['Open'] = data['open']
            if 'high' in data.columns:
                data['High'] = data['high']
            if 'low' in data.columns:
                data['Low'] = data['low']
            if 'volume' in data.columns:
                data['Volume'] = data['volume']
            
            # Ensure required columns exist
            if 'Open' not in data.columns:
                data['Open'] = data['Close']
            if 'High' not in data.columns:
                data['High'] = data['Close']
            if 'Low' not in data.columns:
                data['Low'] = data['Close']
            if 'Volume' not in data.columns:
                data['Volume'] = 1000000
            
            # Technical Indicators
            if show_technical:
                st.markdown("---")
                st.subheader("📊 Technical Indicators")
                
                try:
                    fe = FeatureEngineering()
                    
                    # Calculate indicators
                    indicators_data = {}
                    
                    # Moving Averages
                    ma_data = fe.calculate_moving_averages(data)
                    indicators_data.update(ma_data.to_dict('series'))
                    
                    # RSI
                    rsi_data = fe.calculate_rsi(data)
                    indicators_data.update(rsi_data.to_dict('series'))
                    
                    # MACD
                    macd_data = fe.calculate_macd(data)
                    indicators_data.update(macd_data.to_dict('series'))
                    
                    # Bollinger Bands
                    bb_data = fe.calculate_bollinger_bands(data)
                    indicators_data.update(bb_data.to_dict('series'))
                    
                    # Create indicators DataFrame
                    indicators_df = pd.DataFrame(indicators_data, index=data.index)
                    
                    # Display key indicators
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if 'RSI' in indicators_df.columns:
                            current_rsi = indicators_df['RSI'].iloc[-1]
                            st.metric("RSI", f"{current_rsi:.2f}")
                            if current_rsi > 70:
                                st.warning("Overbought")
                            elif current_rsi < 30:
                                st.info("Oversold")
                    
                    with col2:
                        if 'SMA_20' in indicators_df.columns:
                            sma_20 = indicators_df['SMA_20'].iloc[-1]
                            current_price = data['Close'].iloc[-1]
                            st.metric("SMA(20)", f"${sma_20:.2f}")
                            if current_price > sma_20:
                                st.success("Above SMA")
                            else:
                                st.error("Below SMA")
                    
                    with col3:
                        if 'MACD' in indicators_df.columns and 'MACD_Signal' in indicators_df.columns:
                            macd_val = indicators_df['MACD'].iloc[-1]
                            signal_val = indicators_df['MACD_Signal'].iloc[-1]
                            st.metric("MACD", f"{macd_val:.2f}")
                            if macd_val > signal_val:
                                st.success("Bullish")
                            else:
                                st.error("Bearish")
                    
                    with col4:
                        if 'BB_Upper' in indicators_df.columns and 'BB_Lower' in indicators_df.columns:
                            upper = indicators_df['BB_Upper'].iloc[-1]
                            lower = indicators_df['BB_Lower'].iloc[-1]
                            current = data['Close'].iloc[-1]
                            st.metric("Bollinger Position", f"{((current - lower) / (upper - lower) * 100):.1f}%")
                    
                    # Chart with indicators
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=("Price with Moving Averages", "RSI", "MACD"),
                        vertical_spacing=0.1,
                        row_heights=[0.5, 0.25, 0.25]
                    )
                    
                    # Price and MAs
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='blue', width=2)
                    ), row=1, col=1)
                    
                    if 'SMA_20' in indicators_df.columns:
                        fig.add_trace(go.Scatter(
                            x=indicators_df.index,
                            y=indicators_df['SMA_20'],
                            mode='lines',
                            name='SMA(20)',
                            line=dict(color='orange', width=1, dash='dash')
                        ), row=1, col=1)
                    
                    if 'SMA_50' in indicators_df.columns:
                        fig.add_trace(go.Scatter(
                            x=indicators_df.index,
                            y=indicators_df['SMA_50'],
                            mode='lines',
                            name='SMA(50)',
                            line=dict(color='red', width=1, dash='dash')
                        ), row=1, col=1)
                    
                    # Bollinger Bands
                    if 'BB_Upper' in indicators_df.columns:
                        fig.add_trace(go.Scatter(
                            x=indicators_df.index,
                            y=indicators_df['BB_Upper'],
                            mode='lines',
                            name='BB Upper',
                            line=dict(color='gray', width=1, dash='dot'),
                            showlegend=False
                        ), row=1, col=1)
                        
                        fig.add_trace(go.Scatter(
                            x=indicators_df.index,
                            y=indicators_df['BB_Lower'],
                            mode='lines',
                            name='BB Lower',
                            line=dict(color='gray', width=1, dash='dot'),
                            fill='tonexty',
                            fillcolor='rgba(128,128,128,0.1)',
                            showlegend=False
                        ), row=1, col=1)
                    
                    # RSI
                    if 'RSI' in indicators_df.columns:
                        fig.add_trace(go.Scatter(
                            x=indicators_df.index,
                            y=indicators_df['RSI'],
                            mode='lines',
                            name='RSI',
                            line=dict(color='purple', width=2)
                        ), row=2, col=1)
                        
                        # Add overbought/oversold lines
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                    
                    # MACD
                    if 'MACD' in indicators_df.columns:
                        fig.add_trace(go.Scatter(
                            x=indicators_df.index,
                            y=indicators_df['MACD'],
                            mode='lines',
                            name='MACD',
                            line=dict(color='blue', width=2)
                        ), row=3, col=1)
                        
                        if 'MACD_Signal' in indicators_df.columns:
                            fig.add_trace(go.Scatter(
                                x=indicators_df.index,
                                y=indicators_df['MACD_Signal'],
                                mode='lines',
                                name='Signal',
                                line=dict(color='red', width=2)
                            ), row=3, col=1)
                    
                    fig.update_layout(height=800, showlegend=True, hovermode='x unified')
                    fig.update_xaxes(title_text="Date", row=3, col=1)
                    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                    fig.update_yaxes(title_text="RSI", row=2, col=1)
                    fig.update_yaxes(title_text="MACD", row=3, col=1)
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    # Indicators table
                    with st.expander("📋 View All Indicators"):
                        st.dataframe(indicators_df.tail(50), width='stretch')
                
                except Exception as e:
                    st.error(f"Error calculating indicators: {str(e)}")
            
            # Market Regime Detection
            if show_regime:
                st.markdown("---")
                st.subheader("🎯 Market Regime Detection")
                
                try:
                    analyzer = MarketAnalyzer()
                    
                    # Detect trend regime
                    trend_regime = analyzer.detect_market_regime(data, "trend")
                    
                    # Detect volatility regime
                    volatility_regime = analyzer.detect_market_regime(data, "volatility")
                    
                    # Display regime information
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Trend Regime**")
                        regime = trend_regime.get('regime', 'unknown')
                        strength = trend_regime.get('strength', 0)
                        
                        if regime == 'up':
                            st.success(f"📈 Uptrend (Strength: {strength:.2%})")
                        elif regime == 'down':
                            st.error(f"📉 Downtrend (Strength: {abs(strength):.2%})")
                        else:
                            st.info(f"➡️ Sideways")
                        
                        st.metric("MA Short (20)", f"${trend_regime.get('ma_short', 0):.2f}")
                        st.metric("MA Long (50)", f"${trend_regime.get('ma_long', 0):.2f}")
                    
                    with col2:
                        st.markdown("**Volatility Regime**")
                        vol_regime = volatility_regime.get('regime', 'unknown')
                        current_vol = volatility_regime.get('current_volatility', 0)
                        
                        if 'high' in vol_regime:
                            st.warning(f"⚡ High Volatility ({current_vol:.2%})")
                        elif 'low' in vol_regime:
                            st.success(f"🔇 Low Volatility ({current_vol:.2%})")
                        else:
                            st.info(f"📊 Normal Volatility ({current_vol:.2%})")
                        
                        st.metric("Volatility Rank", f"{volatility_regime.get('volatility_rank', 0):.2%}")
                        st.metric("Volatility Trend", volatility_regime.get('volatility_trend', 'unknown'))
                    
                    # Store regime in session state
                    st.session_state.market_regime = {
                        'trend': trend_regime,
                        'volatility': volatility_regime
                    }
                
                except Exception as e:
                    st.error(f"Error detecting market regime: {str(e)}")
            
            # Trend Analysis
            if show_trend:
                st.markdown("---")
                st.subheader("📈 Trend Analysis")
                
                try:
                    # Calculate trend metrics
                    returns = data['Close'].pct_change()
                    
                    # Short-term trend (5 days)
                    short_trend = (data['Close'].iloc[-1] / data['Close'].iloc[-6] - 1) * 100 if len(data) >= 6 else 0
                    
                    # Medium-term trend (20 days)
                    medium_trend = (data['Close'].iloc[-1] / data['Close'].iloc[-21] - 1) * 100 if len(data) >= 21 else 0
                    
                    # Long-term trend (full period)
                    long_trend = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
                    
                    # Volatility
                    volatility = returns.std() * np.sqrt(252) * 100
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Short-term (5d)", f"{short_trend:.2f}%", 
                                delta=f"{short_trend:.2f}%")
                    
                    with col2:
                        st.metric("Medium-term (20d)", f"{medium_trend:.2f}%",
                                delta=f"{medium_trend:.2f}%")
                    
                    with col3:
                        st.metric("Long-term (Full)", f"{long_trend:.2f}%",
                                delta=f"{long_trend:.2f}%")
                    
                    with col4:
                        st.metric("Annualized Volatility", f"{volatility:.2f}%")
                    
                    # Trend chart
                    fig = go.Figure()
                    
                    # Price
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        mode='lines',
                        name='Price',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Trend lines
                    if len(data) >= 20:
                        # 20-day moving average
                        ma20 = data['Close'].rolling(20).mean()
                        fig.add_trace(go.Scatter(
                            x=ma20.index,
                            y=ma20.values,
                            mode='lines',
                            name='MA(20)',
                            line=dict(color='orange', width=1, dash='dash')
                        ))
                    
                    if len(data) >= 50:
                        # 50-day moving average
                        ma50 = data['Close'].rolling(50).mean()
                        fig.add_trace(go.Scatter(
                            x=ma50.index,
                            y=ma50.values,
                            mode='lines',
                            name='MA(50)',
                            line=dict(color='red', width=1, dash='dash')
                        ))
                    
                    fig.update_layout(
                        title=f"{st.session_state.symbol} - Trend Analysis",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        hovermode='x unified',
                        height=400
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                
                except Exception as e:
                    st.error(f"Error analyzing trend: {str(e)}")
            
            # Correlation Analysis (if multiple symbols available)
            st.markdown("---")
            st.subheader("🔗 Correlation Analysis")
            st.info("Correlation analysis requires multiple symbols. Load additional data to compare.")
            
            # News Sentiment Analysis
            st.markdown("---")
            st.subheader("📰 News Sentiment Analysis")
            
            symbol_for_sentiment = st.text_input(
                "Symbol for sentiment", 
                value=st.session_state.get('symbol', 'AAPL'),
                key="sentiment_symbol"
            )
            
            if st.button("Analyze News Sentiment", key="analyze_sentiment"):
                try:
                    from trading.nlp.sentiment_classifier import SentimentClassifier
                    
                    classifier = SentimentClassifier()
                    
                    with st.spinner("Analyzing news sentiment..."):
                        # Get recent news (you'll need a news API)
                        # For demo, use placeholder
                        sentiment_result = classifier.analyze_symbol_sentiment(
                            symbol=symbol_for_sentiment,
                            lookback_days=7
                        )
                        
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            sentiment_score = sentiment_result['overall_sentiment']
                            if sentiment_score > 0.3:
                                st.success(f"😊 Positive: {sentiment_score:.2f}")
                            elif sentiment_score < -0.3:
                                st.error(f"😞 Negative: {sentiment_score:.2f}")
                            else:
                                st.info(f"😐 Neutral: {sentiment_score:.2f}")
                        
                        with col2:
                            st.metric("Articles Analyzed", sentiment_result['num_articles'])
                        
                        with col3:
                            st.metric("Sentiment Confidence", f"{sentiment_result['confidence']:.1%}")
                        
                        # Sentiment over time
                        if 'sentiment_history' in sentiment_result:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=sentiment_result['dates'],
                                y=sentiment_result['sentiment_history'],
                                name='Sentiment',
                                line=dict(color='purple')
                            ))
                            
                            fig.update_layout(
                                title='Sentiment Over Time',
                                xaxis_title='Date',
                                yaxis_title='Sentiment Score',
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Recent headlines
                        st.subheader("📰 Recent Headlines")
                        for article in sentiment_result.get('articles', [])[:10]:
                            sentiment_emoji = "😊" if article['sentiment'] > 0 else "😞" if article['sentiment'] < 0 else "😐"
                            st.write(f"{sentiment_emoji} **{article['title']}**")
                            st.caption(f"Sentiment: {article['sentiment']:.2f} | {article['date']}")
                
                except ImportError:
                    st.error("Sentiment classifier not available")
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
            
            # Causal Analysis
            st.markdown("---")
            st.subheader("🔬 Causal Analysis")
            
            st.write("""
            Understand what actually drives price movements using causal inference.
            Goes beyond correlation to identify true causal relationships.
            """)
            
            if st.button("Run Causal Analysis", key="run_causal_analysis"):
                if 'forecast_data' not in st.session_state:
                    st.error("Please load data first")
                else:
                    try:
                        from causal.causal_model import CausalModel
                        from causal.driver_analysis import DriverAnalysis
                        
                        data = st.session_state.forecast_data.copy()
                        symbol = st.session_state.symbol
                        
                        # Ensure proper column names
                        if 'close' in data.columns:
                            data['Close'] = data['close']
                        if 'open' in data.columns:
                            data['Open'] = data['open']
                        if 'high' in data.columns:
                            data['High'] = data['high']
                        if 'low' in data.columns:
                            data['Low'] = data['low']
                        if 'volume' in data.columns:
                            data['Volume'] = data['volume']
                        
                        # Ensure required columns exist
                        if 'Open' not in data.columns:
                            data['Open'] = data['Close']
                        if 'High' not in data.columns:
                            data['High'] = data['Close']
                        if 'Low' not in data.columns:
                            data['Low'] = data['Close']
                        if 'Volume' not in data.columns:
                            data['Volume'] = 1000000
                        
                        with st.spinner("Performing causal analysis..."):
                            # Initialize models
                            causal_model = CausalModel()
                            driver_analysis = DriverAnalysis()
                            
                            # Build causal graph
                            causal_graph = causal_model.build_causal_graph(
                                data=data,
                                target='Close',
                                features=['Volume', 'High', 'Low', 'Open']
                            )
                            
                            st.success("✅ Causal analysis complete!")
                            
                            # Display causal graph
                            st.subheader("📊 Causal Graph")
                            
                            # Visualize causal relationships
                            if 'graph_visualization' in causal_graph:
                                st.image(causal_graph['graph_visualization'])
                            else:
                                # Text-based representation
                                st.write("**Causal Relationships:**")
                                if 'relationships' in causal_graph:
                                    for relationship in causal_graph['relationships']:
                                        st.write(f"• {relationship['cause']} → {relationship['effect']} (strength: {relationship['strength']:.2f})")
                                else:
                                    st.info("Causal relationships will be displayed here")
                            
                            # Identify key drivers
                            st.subheader("🎯 Key Price Drivers")
                            
                            drivers = driver_analysis.identify_drivers(
                                data=data,
                                target='Close'
                            )
                            
                            # Display drivers ranked by importance
                            if 'drivers' in drivers and len(drivers['drivers']) > 0:
                                drivers_df = pd.DataFrame(drivers['drivers'])
                                drivers_df = drivers_df.sort_values('importance', ascending=False)
                                
                                fig = px.bar(
                                    drivers_df,
                                    x='importance',
                                    y='driver',
                                    orientation='h',
                                    title='Price Drivers (Ranked by Causal Impact)',
                                    labels={'importance': 'Causal Impact', 'driver': 'Driver'}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Interpretation
                                st.subheader("💡 Interpretation")
                                
                                top_driver = drivers_df.iloc[0]
                                st.info(f"""
                                **Primary Driver:** {top_driver['driver']}
                                
                                The analysis shows that {top_driver['driver']} has the strongest causal 
                                impact on {symbol} price movements, with an importance score of {top_driver['importance']:.2f}.
                                """)
                                
                                # Driver details
                                with st.expander("📊 Detailed Driver Analysis"):
                                    for _, driver in drivers_df.iterrows():
                                        st.write(f"**{driver['driver']}**")
                                        st.write(f"- Importance: {driver['importance']:.2f}")
                                        st.write(f"- Effect type: {driver.get('effect_type', 'Unknown')}")
                                        st.write(f"- Time lag: {driver.get('lag', 0)} periods")
                                        st.write("")
                                
                                # Counterfactual analysis
                                st.subheader("🔮 Counterfactual Analysis")
                                
                                st.write("What if key drivers changed?")
                                
                                driver_to_change = st.selectbox(
                                    "Select driver to modify",
                                    options=drivers_df['driver'].tolist(),
                                    key="counterfactual_driver"
                                )
                                
                                change_amount = st.slider(
                                    "Change amount (%)",
                                    min_value=-50,
                                    max_value=50,
                                    value=10,
                                    step=5,
                                    key="counterfactual_change"
                                )
                                
                                if st.button("Simulate Counterfactual", key="simulate_counterfactual"):
                                    try:
                                        counterfactual = causal_model.counterfactual_analysis(
                                            data=data,
                                            intervention={driver_to_change: change_amount / 100}
                                        )
                                        
                                        st.write(f"**If {driver_to_change} changes by {change_amount:+d}%:**")
                                        st.write(f"Expected price change: {counterfactual.get('expected_change', 0):.2%}")
                                        st.write(f"Confidence: {counterfactual.get('confidence', 0):.1%}")
                                    except Exception as e:
                                        st.error(f"Error in counterfactual analysis: {e}")
                            else:
                                st.warning("No drivers identified. Check data quality and feature availability.")
                    
                    except ImportError:
                        st.error("Causal analysis modules not available")
                    except Exception as e:
                        st.error(f"Error in causal analysis: {e}")
                        import traceback
                        st.code(traceback.format_exc())
            
            # Summary
            st.markdown("---")
            st.subheader("📋 Analysis Summary")
            
            summary_data = {
                'Metric': [],
                'Value': []
            }
            
            if show_technical and 'RSI' in locals() and 'indicators_df' in locals():
                summary_data['Metric'].append('Current RSI')
                summary_data['Value'].append(f"{indicators_df['RSI'].iloc[-1]:.2f}")
            
            if show_regime and 'trend_regime' in locals():
                summary_data['Metric'].append('Market Regime')
                summary_data['Value'].append(trend_regime.get('regime', 'unknown').upper())
            
            if show_trend:
                summary_data['Metric'].append('Overall Trend')
                trend_direction = "UP" if long_trend > 0 else "DOWN"
                summary_data['Value'].append(f"{trend_direction} ({long_trend:.2f}%)")
            
            if summary_data['Metric']:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, width='stretch', hide_index=True)

with tab6:
    st.header("🔗 Multi-Asset Forecasting with Graph Neural Networks")
    st.markdown("""
    GNN models relationships between correlated assets. Perfect for:
    - **Portfolio-level forecasting** - Predict entire portfolio together
    - **Sector analysis** - Model how sector stocks move together
    - **Market contagion** - Understand how shocks spread
    - **Relationship-based predictions** - Use asset correlations for better forecasts
    """)
    
    st.info("💡 GNN requires 3-20 correlated assets. It won't work with single tickers.")
    
    # Multi-ticker input
    st.subheader("📊 Step 1: Select Multiple Assets")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        tickers_input = st.text_area(
            "Enter ticker symbols (one per line)",
            value="AAPL\nMSFT\nGOOGL\nAMZN\nMETA",
            height=150,
            help="Enter 3-20 correlated stocks. Tech stocks, bank stocks, etc."
        )
        
        tickers = [t.strip().upper() for t in tickers_input.split('\n') if t.strip()]
        
        if len(tickers) < 3:
            st.error("❌ GNN requires at least 3 assets")
        elif len(tickers) > 20:
            st.warning("⚠️ Too many assets (max 20). Performance may be slow.")
        else:
            st.success(f"✅ {len(tickers)} assets selected")
    
    with col2:
        st.write("**Settings:**")
        
        correlation_threshold = st.slider(
            "Correlation Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Assets with correlation > this will be connected in graph"
        )
        
        gnn_horizon = st.slider(
            "Forecast Days",
            min_value=1,
            max_value=30,
            value=7
        )
        
        gnn_epochs = st.slider(
            "Training Epochs",
            min_value=20,
            max_value=100,
            value=50,
            help="More epochs = better accuracy but slower"
        )
    
    # Load multi-asset data
    if st.button("📥 Load Multi-Asset Data", type="primary", use_container_width=True):
        if len(tickers) < 3:
            st.error("Please select at least 3 assets")
        elif len(tickers) > 20:
            st.error("Please limit to 20 assets for performance")
        else:
            try:
                with st.spinner(f"Loading data for {len(tickers)} assets..."):
                    from trading.data.data_loader import DataLoader, DataLoadRequest
                    from datetime import datetime, timedelta
                    
                    loader = DataLoader()
                    
                    # Load data for each ticker
                    multi_asset_data = {}
                    failed_tickers = []
                    
                    progress_bar = st.progress(0)
                    
                    for i, ticker in enumerate(tickers):
                        try:
                            request = DataLoadRequest(
                                ticker=ticker,
                                start_date=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                                end_date=datetime.now().strftime("%Y-%m-%d"),
                                interval="1d"
                            )
                            response = loader.load_market_data(request)
                            
                            if response.success and response.data is not None:
                                # Get close prices
                                data = response.data
                                if 'close' in data.columns:
                                    multi_asset_data[ticker] = data['close']
                                elif 'Close' in data.columns:
                                    multi_asset_data[ticker] = data['Close']
                            else:
                                failed_tickers.append(ticker)
                        except Exception as e:
                            logger.error(f"Error loading {ticker}: {e}")
                            failed_tickers.append(ticker)
                        
                        progress_bar.progress((i + 1) / len(tickers))
                    
                    progress_bar.empty()
                    
                    if len(multi_asset_data) < 3:
                        st.error(f"Could not load enough assets. Failed: {', '.join(failed_tickers)}")
                    else:
                        # Combine into DataFrame
                        multi_df = pd.DataFrame(multi_asset_data)
                        
                        # Align dates (drop rows with any NaN)
                        multi_df = multi_df.dropna()
                        
                        if len(multi_df) < 100:
                            st.error("Insufficient overlapping data. Try different tickers or longer date range.")
                        else:
                            st.session_state.gnn_data = multi_df
                            st.session_state.gnn_tickers = list(multi_df.columns)
                            
                            st.success(f"✅ Loaded {len(multi_df)} days of data for {len(multi_df.columns)} assets")
                            
                            if failed_tickers:
                                st.warning(f"⚠️ Could not load: {', '.join(failed_tickers)}")
            
            except Exception as e:
                st.error(f"Error loading data: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Display correlation matrix and train GNN
    if 'gnn_data' in st.session_state and st.session_state.gnn_data is not None:
        st.markdown("---")
        st.subheader("📊 Step 2: View Asset Correlations")
        
        multi_df = st.session_state.gnn_data
        
        # Calculate correlation matrix
        corr_matrix = multi_df.corr()
        
        # Plot heatmap
        fig_corr = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            title=f"Asset Correlation Matrix ({len(multi_df.columns)} assets)"
        )
        
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, width='stretch')
        
        # Show graph connections
        num_connections = (corr_matrix.abs() > correlation_threshold).sum().sum() - len(multi_df.columns)
        st.info(f"🔗 Graph will have {num_connections // 2} edges (connections) based on {correlation_threshold:.0%} threshold")
        
        # Generate forecast
        st.markdown("---")
        st.subheader("🚀 Step 3: Generate GNN Forecast")
        
        target_asset = st.selectbox(
            "Select primary asset to forecast",
            options=st.session_state.gnn_tickers,
            help="GNN will forecast this asset using all connected assets"
        )
        
        if st.button("🔮 Train GNN & Generate Forecast", type="primary", use_container_width=True):
            try:
                with st.spinner(f"Training Graph Neural Network on {len(multi_df.columns)} assets..."):
                    from trading.models.advanced.gnn.gnn_model import GNNForecaster
                    
                    # Progress indicator
                    progress_text = st.empty()
                    
                    # Initialize GNN
                    progress_text.text("Initializing GNN model...")
                    gnn = GNNForecaster(
                        num_assets=len(multi_df.columns),
                        hidden_size=64,
                        num_layers=2,
                        seq_length=30,
                        correlation_threshold=correlation_threshold
                    )
                    
                    # Train
                    progress_text.text(f"Training for {gnn_epochs} epochs...")
                    gnn.fit(multi_df, epochs=gnn_epochs, batch_size=32)
                    
                    # Generate forecast
                    progress_text.text("Generating forecast...")
                    forecast_result = gnn.forecast(
                        multi_df,
                        horizon=gnn_horizon,
                        target_asset=target_asset
                    )
                    
                    progress_text.empty()
                    st.success("✅ GNN forecast generated!")
                    
                    # Display forecast chart
                    st.subheader(f"📈 {target_asset} Forecast")
                    
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=multi_df.index,
                        y=multi_df[target_asset],
                        name='Historical',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Forecast
                    fig.add_trace(go.Scatter(
                        x=forecast_result['dates'],
                        y=forecast_result['forecast'],
                        name='GNN Forecast',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    
                    fig.update_layout(
                        title=f"GNN Multi-Asset Forecast for {target_asset}",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        hovermode='x unified',
                        height=500
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Forecast Horizon", f"{gnn_horizon} days")
                    with col2:
                        st.metric("Assets Used", len(multi_df.columns))
                    with col3:
                        avg_conf = forecast_result['confidence'].mean()
                        st.metric("Avg Confidence", f"{avg_conf:.1%}")
                    with col4:
                        last_price = multi_df[target_asset].iloc[-1]
                        forecast_return = ((forecast_result['forecast'][-1] / last_price) - 1) * 100
                        st.metric("Forecast Return", f"{forecast_return:+.2f}%")
                    
                    # Show relationship matrix
                    st.markdown("---")
                    st.subheader("🔗 Learned Asset Relationships")
                    
                    relationship_matrix = gnn.get_relationship_matrix()
                    
                    fig_rel = px.imshow(
                        relationship_matrix,
                        labels=dict(color="Connection Strength"),
                        x=multi_df.columns,
                        y=multi_df.columns,
                        color_continuous_scale='Viridis',
                        title="GNN Asset Relationship Graph (1 = connected, 0 = not connected)"
                    )
                    
                    fig_rel.update_layout(height=500)
                    st.plotly_chart(fig_rel, width='stretch')
                    
                    # Forecast table
                    with st.expander("📋 View Forecast Data"):
                        forecast_df = pd.DataFrame({
                            'Date': forecast_result['dates'],
                            'Forecast': forecast_result['forecast'],
                            'Confidence': forecast_result['confidence']
                        })
                        st.dataframe(forecast_df, width='stretch')
            
            except ImportError:
                st.error("❌ GNN model not available. Make sure it has been recreated using the prompts.")
            except Exception as e:
                st.error(f"Error generating GNN forecast: {e}")
                import traceback
                st.code(traceback.format_exc())

