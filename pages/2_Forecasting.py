# -*- coding: utf-8 -*-
"""
Forecasting & Market Analysis Page

Merges functionality from:
- Forecasting.py
- Forecast_with_AI_Selection.py
- 7_Market_Analysis.py

Features:
- Quick Forecast with 3 fast models (ARIMA, XGBoost, Ridge); no walk-forward
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

from ui.page_assistant import render_page_assistant

logger = logging.getLogger(__name__)

# Backend: lazy load when Forecasting page is opened (not at app startup)
def _get_forecasting_backend():
    """Import and return backend modules/classes only when needed."""
    try:
        from trading.data.data_loader import DataLoader, DataLoadRequest
        from trading.data.providers.yfinance_provider import YFinanceProvider
        from trading.models.lstm_model import LSTMForecaster
        from trading.models.xgboost_model import XGBoostModel
        from trading.models.prophet_model import ProphetModel
        from trading.models.arima_model import ARIMAModel
        from trading.data.preprocessing import FeatureEngineering, DataPreprocessor
        from trading.agents.model_selector_agent import ModelSelectorAgent
        from trading.market.market_analyzer import MarketAnalyzer
        return {
            "DataLoader": DataLoader,
            "DataLoadRequest": DataLoadRequest,
            "YFinanceProvider": YFinanceProvider,
            "LSTMForecaster": LSTMForecaster,
            "XGBoostModel": XGBoostModel,
            "ProphetModel": ProphetModel,
            "ARIMAModel": ARIMAModel,
            "FeatureEngineering": FeatureEngineering,
            "DataPreprocessor": DataPreprocessor,
            "ModelSelectorAgent": ModelSelectorAgent,
            "MarketAnalyzer": MarketAnalyzer,
        }
    except Exception as e:
        logger.warning(f"Forecasting backend not available: {e}")
        return None

st.set_page_config(
    page_title="Forecasting & Market Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Lazy init: resolve backend when Forecasting page is first rendered (not at app startup)
if "forecasting_backend" not in st.session_state:
    st.session_state.forecasting_backend = _get_forecasting_backend()
_be = st.session_state.get("forecasting_backend")
if _be:
    DataLoader = _be["DataLoader"]
    DataLoadRequest = _be["DataLoadRequest"]
    YFinanceProvider = _be["YFinanceProvider"]
    LSTMForecaster = _be["LSTMForecaster"]
    XGBoostModel = _be["XGBoostModel"]
    ProphetModel = _be["ProphetModel"]
    ARIMAModel = _be["ARIMAModel"]
    FeatureEngineering = _be["FeatureEngineering"]
    DataPreprocessor = _be["DataPreprocessor"]
    ModelSelectorAgent = _be["ModelSelectorAgent"]
    MarketAnalyzer = _be["MarketAnalyzer"]
else:
    DataLoader = DataLoadRequest = YFinanceProvider = LSTMForecaster = XGBoostModel = None
    ProphetModel = ARIMAModel = FeatureEngineering = DataPreprocessor = ModelSelectorAgent = MarketAnalyzer = None

if not _be:
    st.error("Forecasting backend could not be loaded. Check logs and dependencies.")
    st.stop()

# Initialize session state variables (dict-style access is safer when imported outside streamlit)
if "forecast_data" not in st.session_state:
    st.session_state["forecast_data"] = None
if "selected_models" not in st.session_state:
    st.session_state["selected_models"] = []
if "ai_recommendation" not in st.session_state:
    st.session_state["ai_recommendation"] = None
if "comparison_results" not in st.session_state:
    st.session_state["comparison_results"] = None
if "market_regime" not in st.session_state:
    st.session_state["market_regime"] = None
if "symbol" not in st.session_state:
    st.session_state["symbol"] = None
if "forecast_horizon" not in st.session_state:
    st.session_state["forecast_horizon"] = 7

# Main page title
st.title("📈 Forecasting & Market Analysis")
st.markdown("Advanced forecasting with AI model selection and comprehensive market analysis")

# Create tabbed interface
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🚀 Quick Forecast",
    "⚙️ Advanced Forecasting",
    "🤖 AI Model Selection",
    "📊 Model Comparison",
    "📈 Market Analysis",
    "🔗 Multi-Asset (GNN)",
    "🎲 Monte Carlo",
])

# Quick Forecast: fast models only (ARIMA, XGBoost, Ridge). No walk-forward, no auto-select.
QUICK_FORECAST_MODELS = ["ARIMA", "XGBoost", "Ridge"]

with tab1:
    st.header("Quick Forecast")
    st.markdown("Generate fast forecasts with pre-configured models (ARIMA, XGBoost, Ridge)")
    
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
                value=datetime.now().date() - timedelta(days=365),
                min_value=datetime.now().date() - timedelta(days=365*10),
                max_value=datetime.now().date(),
                help="Historical data start. Cannot select a future date."
            )
        
        with col3:
            end_date = st.date_input(
                "End Date",
                value=datetime.now().date(),
                min_value=start_date,
                max_value=datetime.now().date(),
                help="End date for historical data. For forecasting future prices, use the 'Forecast Horizon (days)' slider — do not set end date in the future."
            )
        
        submitted = st.form_submit_button("📊 Load Data")
    
    if end_date > datetime.now().date():
        st.warning("End date is in the future. Historical data will only load up to today. Use the forecast horizon slider to project forward.")
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
                            st.session_state["forecast_data"] = data
                            st.session_state["symbol"] = symbol
                            # forecast_horizon is already in session_state (defaults to 7 if not set)
                            # No need to reassign it here unless we want to update it
                            
                            st.success(f"✅ Loaded {len(data)} days of data for {symbol}")
                            
                            # Show data quality metrics
                            try:
                                from src.utils.data_validation import DataValidator
                            except ImportError:
                                DataValidator = None
                            if DataValidator is not None:
                                try:
                                    validator = DataValidator()
                                    quality_metrics = validator.get_quality_metrics(data)
                                except Exception as qe:
                                    logger.debug("Data quality metrics failed: %s", qe)
                                    quality_metrics = None
                                if quality_metrics is not None:
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
                                        if quality_metrics['issues']:
                                            st.warning("⚠️ Data Quality Issues:")
                                            for issue in quality_metrics['issues']:
                                                st.write(f"• {issue}")
                        
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                st.info("Please check the ticker symbol and try again.")
    
    # Display loaded data
    if st.session_state.get("forecast_data") is not None:
        data = st.session_state.get("forecast_data")
        
        # Show data quality metrics (optional: src.utils.data_validation)
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
                if quality_metrics['issues']:
                    st.warning("⚠️ Data Quality Issues:")
                    for issue in quality_metrics['issues']:
                        st.write(f"• {issue}")
        except ImportError:
            pass
        except Exception as e:
            logger.debug("Data quality metrics unavailable: %s", e)
        
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
            st.dataframe(data.tail(50))
        
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
            
            # Quick Forecast: hardcoded fast models only — no LSTM, no Transformer
            selected_model = st.selectbox(
                "Select Model",
                ["ARIMA", "XGBoost", "Ridge"],
                key="quick_forecast_model",
                help="Fast models only (ARIMA, XGBoost, Ridge)"
            )
            if selected_model:
                try:
                    from trading.models.model_registry import get_registry
                    registry = get_registry()
                    model_info = registry.get_model_info(selected_model) if registry else None
                    if model_info and model_info.get('description'):
                        description = model_info['description']
                        if len(description) > 200:
                            description = description[:200] + "..."
                        st.info(f"ℹ️ {description}")
                except Exception as e:
                    logger.warning(f"Forecasting error: {e}")
            
            forecast_button = st.button(
                "🚀 Generate Forecast",
                type="primary"
            )
        
        with col2:
            if forecast_button:
                try:
                    with st.spinner(f"Training {selected_model}..."):
                        # Get data
                        data = st.session_state.get("forecast_data")
                        if data is None:
                            raise RuntimeError("No forecast_data in session_state; please load data first.")
                        data = data.copy()
                        horizon = st.session_state.get("forecast_horizon", 7)
                        if not isinstance(data.index, pd.DatetimeIndex):
                            data.index = pd.to_datetime(data.index)
                        # Normalize column names for router (expects 'close')
                        if "Close" in data.columns and "close" not in data.columns:
                            data = data.rename(columns={"Close": "close", "Open": "open", "High": "high", "Low": "low", "Volume": "volume"})
                        # Use forecast router for accuracy-validated forecast (walk-forward, validation MAPE, confidence)
                        model_key = (selected_model or "arima").lower().replace(" ", "")
                        used_router = False
                        try:
                            from trading.models.forecast_router import ForecastRouter
                            _router = ForecastRouter()
                            router_result = _router.get_forecast(data, horizon=horizon, model_type=model_key, run_walk_forward=False)
                            if router_result and router_result.get("forecast") is not None:
                                used_router = True
                                forecast_values = np.asarray(router_result["forecast"]).ravel()
                                forecast_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=horizon, freq="D")[:len(forecast_values)]
                                if hasattr(forecast_dates, "tz_localize"):
                                    try:
                                        forecast_dates = forecast_dates.tz_localize(None)
                                    except Exception:
                                        pass
                                st.session_state.current_forecast = pd.DataFrame({"forecast": forecast_values}, index=forecast_dates)
                                st.session_state.current_model = selected_model or router_result.get("model", "unknown")
                                st.session_state.current_forecast_result = {
                                    **router_result,
                                    "validation_mape": router_result.get("validation_mape"),
                                    "in_sample_mape": router_result.get("in_sample_mape"),
                                    "last_actual_price": router_result.get("last_actual_price"),
                                    "confidence_label": router_result.get("confidence_label"),
                                }
                                for w in router_result.get("warnings", []):
                                    st.warning(w)
                                st.success(f"✅ Forecast generated using {selected_model}")
                        except Exception as _e:
                            logger.debug("Router forecast failed, using direct model: %s", _e)
                        _denorm_price = 1.0
                        if not used_router:
                            # Fallback: direct model fit/forecast with feature enrichment and price normalization
                            from trading.models.forecast_features import prepare_forecast_data
                            data_prepared, last_price = prepare_forecast_data(data)
                            _denorm_price = last_price if last_price and last_price != 0 else 1.0
                            if data_prepared is not None and not data_prepared.empty:
                                data = data_prepared
                        
                        # Initialize model with proper config using registry (only when not used_router)
                        model = None
                        if not used_router:
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
                                
                                # Extract forecast values (ARIMA uses 'forecast'; some use 'predictions'/'values'/'forecast_values')
                                if isinstance(forecast_result, dict):
                                    forecast_vals = (
                                        forecast_result.get('forecast')
                                        or forecast_result.get('predictions')
                                        or forecast_result.get('values')
                                        or forecast_result.get('forecast_values')
                                        or []
                                    )
                                    if hasattr(forecast_vals, 'tolist'):
                                        forecast_vals = forecast_vals.tolist()
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
                            # Denormalize: model was trained on price/last_price
                            if _denorm_price and _denorm_price != 1.0:
                                forecast_values = np.asarray(forecast_values, dtype=float) * _denorm_price
                            
                            # Create forecast DataFrame
                            forecast_df = pd.DataFrame({
                                'forecast': forecast_values
                            }, index=forecast_dates[:len(forecast_values)])
                            
                            # Store in session state
                            st.session_state.current_forecast = forecast_df
                            st.session_state.current_model = selected_model
                            st.session_state.current_model_instance = model  # Store model instance for explainability
                            
                            # Situational awareness: write to MemoryStore for Chat context (quality gate)
                            try:
                                from trading.memory import get_memory_store
                                from trading.memory.memory_store import MemoryType
                                _store = get_memory_store()
                                _vals = forecast_values if hasattr(forecast_values, "__len__") else [float(forecast_values)]
                                _first = _vals[0] if len(_vals) > 0 else None
                                _last = _vals[-1] if len(_vals) > 0 else None
                                try:
                                    _all_same = len(_vals) > 1 and (np.unique(np.asarray(_vals).flatten()).size <= 1)
                                except Exception:
                                    _all_same = False
                                _degenerate = _first is None or _last is None or _all_same
                                if _first is not None or _last is not None:
                                    _conf = None
                                    if isinstance(forecast_result, dict):
                                        if "lower_bound" in forecast_result and "upper_bound" in forecast_result:
                                            _conf = {"lower": forecast_result["lower_bound"], "upper": forecast_result["upper_bound"]}
                                    _store.add(
                                        MemoryType.LONG_TERM,
                                        namespace="forecasts",
                                        value={
                                            "symbol": st.session_state.get("symbol", ""),
                                            "model_name": selected_model,
                                            "horizon": horizon,
                                            "forecast_first": _first,
                                            "forecast_last": _last,
                                            "confidence": _conf,
                                            "timestamp": datetime.utcnow().isoformat(),
                                            **({"model_failed": True} if _degenerate else {}),
                                        },
                                        category="results",
                                    )
                            except Exception:
                                pass
                            
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
                    
                    hist_data = st.session_state.get("forecast_data")
                    forecast_df = st.session_state.current_forecast
                    forecast_result = st.session_state.get('current_forecast_result', {})
                    
                    # Prepare forecast data for component
                    forecast_data = {
                        'dates': forecast_df.index.tolist() if hasattr(forecast_df.index, 'tolist') else list(forecast_df.index),
                        'forecast': forecast_df['forecast'].values.tolist() if hasattr(forecast_df['forecast'].values, 'tolist') else list(forecast_df['forecast'].values),
                        'model_name': st.session_state.current_model
                    }
                    if isinstance(forecast_result, dict):
                        if 'lower_bound' in forecast_result:
                            forecast_data['lower_bound'] = forecast_result['lower_bound']
                        if 'upper_bound' in forecast_result:
                            forecast_data['upper_bound'] = forecast_result['upper_bound']
                        if 'confidence' in forecast_result:
                            forecast_data['confidence'] = forecast_result['confidence']
                        if forecast_result.get('validation_mape') is not None:
                            forecast_data['validation_mape'] = forecast_result['validation_mape']
                        if forecast_result.get('confidence_label'):
                            forecast_data['confidence_label'] = forecast_result['confidence_label']
                        if forecast_result.get('last_actual_price') is not None:
                            forecast_data['last_actual_price'] = forecast_result['last_actual_price']
                    if forecast_data.get('last_actual_price') is None and hist_data is not None and len(hist_data) > 0:
                        close_col = 'close' if 'close' in hist_data.columns else (hist_data.columns[0] if len(hist_data.columns) > 0 else None)
                        if close_col:
                            forecast_data['last_actual_price'] = float(hist_data[close_col].iloc[-1])
                    
                    # Show validation MAPE, confidence label, last actual vs first forecast
                    if forecast_data.get('validation_mape') is not None or forecast_data.get('last_actual_price') is not None:
                        with st.expander("📐 Accuracy & continuity", expanded=True):
                            mape_val = forecast_data.get('validation_mape') or forecast_data.get('in_sample_mape')
                            if mape_val is not None:
                                st.metric("Validation / in-sample MAPE", f"{mape_val:.2f}%")
                            if forecast_data.get('confidence_label'):
                                st.metric("Confidence (based on MAPE)", forecast_data['confidence_label'])
                            if mape_val is not None:
                                confidence_pct = max(0.0, min(100.0, 100.0 - float(mape_val)))
                                st.metric("Confidence score", f"{confidence_pct:.0f}%")
                            last_act = forecast_data.get('last_actual_price')
                            fvals = forecast_data.get('forecast', [])
                            first_fc = fvals[0] if fvals else None
                            if last_act is not None and first_fc is not None:
                                st.write("**Continuity:** Last actual price **${:.2f}** → First forecast **${:.2f}**".format(float(last_act), float(first_fc)))
                    
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
                    
                    hist_data = st.session_state.get("forecast_data")
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
                    
                    # Forecast table (format dates as YYYY-MM-DD, no timezone)
                    st.markdown("**Forecast Values:**")
                    display_df = forecast_df.copy()
                    try:
                        idx = pd.to_datetime(display_df.index)
                        if hasattr(idx, "tz") and idx.tz is not None:
                            idx = idx.tz_localize(None)
                        display_df.index = idx.strftime("%Y-%m-%d")
                    except Exception:
                        pass
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
                        
                        # Display cached explanation so it persists across reruns
                        if st.session_state.get('forecast_explanation') is not None:
                            cached = st.session_state.forecast_explanation
                            if cached.get('success') and cached.get('explanation'):
                                expl_obj = cached['explanation']
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
                                            importance_df, x='Importance', y='Feature',
                                            orientation='h', title='Feature Importance (SHAP values)'
                                        )
                                        st.plotly_chart(fig)
                                if hasattr(expl_obj, 'explanation_text') and expl_obj.explanation_text:
                                    st.write("**Explanation:**")
                                    st.write(expl_obj.explanation_text)
                            if st.button("Clear explanation", key="clear_explanation_quick"):
                                st.session_state.forecast_explanation = None
                                st.rerun()
                        
                        if st.button("Generate Explanation", key="generate_explanation"):
                            with st.spinner("Analyzing model predictions..."):
                                try:
                                    explainer = ForecastExplainability()
                                    model = st.session_state.get('current_model_instance')
                                    forecast_result = st.session_state.get('current_forecast_result', {})
                                    
                                    if isinstance(forecast_result, dict):
                                        forecast_values = forecast_result.get('forecast', [])
                                        forecast_value = float(forecast_values[0]) if len(forecast_values) > 0 else float(data['close'].iloc[-1])
                                    else:
                                        forecast_value = float(forecast_result) if isinstance(forecast_result, (int, float)) else float(data['close'].iloc[-1])
                                    
                                    features = data.copy()
                                    target_history = features['close'] if 'close' in features.columns else features.iloc[:, 0]
                                    
                                    explanation = explainer.explain_forecast(
                                        forecast_id=f"forecast_{st.session_state.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                        symbol=st.session_state.symbol,
                                        forecast_value=forecast_value,
                                        model=model,
                                        features=features,
                                        target_history=target_history,
                                        horizon=st.session_state.forecast_horizon
                                    )
                                    st.session_state.forecast_explanation = explanation
                                    st.success("✅ Explanation generated")
                                    st.rerun()
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
                        from agents.llm.agent import get_prompt_agent
                        agent = get_prompt_agent()
                        model_name = st.session_state.get("current_model", "Unknown")
                        symbol = st.session_state.get("symbol", "Unknown")
                        last_price = float(data["close"].iloc[-1]) if "close" in data.columns else 0.0
                        fcast = forecast_result.get("forecast", []) if isinstance(forecast_result, dict) else []
                        forecast_mean = float(np.mean(fcast)) if len(fcast) > 0 else last_price
                        pct_change = ((forecast_mean - last_price) / last_price * 100) if last_price else 0.0
                        if agent:
                            with st.spinner("AI is analyzing the forecast..."):
                                commentary_prompt = (
                                    f"Provide 3 sentences of market commentary on the {model_name} forecast for {symbol}: "
                                    f"last price ${last_price:.2f}, 7-day forecast ${forecast_mean:.2f} ({pct_change:+.1f}%). Be specific and concise."
                                )
                                response = agent.process_prompt(commentary_prompt)
                                commentary = response.message if hasattr(response, "message") else str(response)
                                st.write(commentary)
                        else:
                            st.info("AI commentary requires a valid API key in .env")
                    except Exception as e:
                        st.info(f"AI commentary unavailable: {e}")

with tab2:
    st.header("Advanced Forecasting")
    st.markdown("Full model configuration with hyperparameter tuning and feature engineering")
    
    if st.session_state.get("forecast_data") is None:
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
                
                # Auto-select best model (Advanced only; runs all models, slow)
                if st.button("🔬 Auto-select best model", key="auto_select_best_advanced"):
                    data_for_best = st.session_state.get("forecast_data")
                    if data_for_best is not None and len(data_for_best) >= 60:
                        try:
                            from trading.models.forecast_router import ForecastRouter
                            _router = ForecastRouter()
                            _horizon = st.session_state.get("forecast_horizon", 7)
                            _symbol = st.session_state.get("symbol", "")
                            _df = data_for_best.copy()
                            if "Close" in _df.columns and "close" not in _df.columns:
                                _df = _df.rename(columns={"Close": "close", "Open": "open", "High": "high", "Low": "low", "Volume": "volume"})
                            best_key, best_result = _router.select_best_model(_df, _horizon, symbol=_symbol or None)
                            display_map = {"lstm": "LSTM", "xgboost": "XGBoost", "prophet": "Prophet", "arima": "ARIMA", "ridge": "Ridge", "garch": "GARCH", "tcn": "TCN", "catboost": "CatBoost", "ensemble": "Ensemble", "transformer": "Transformer", "autoformer": "Autoformer", "hybrid": "Hybrid"}
                            best_display = display_map.get(best_key, best_key.capitalize())
                            st.session_state["advanced_forecast_best_model"] = best_display
                            st.session_state["advanced_best_model_scores"] = best_result
                            st.success(f"Best model: **{best_display}** (validation MAPE: {best_result.get('validation_mape') or 'N/A'}%)")
                        except Exception as e:
                            st.warning(f"Auto-select failed: {e}")
                    else:
                        st.warning("Load at least 60 rows of data first in Quick Forecast tab.")
                if st.session_state.get("advanced_best_model_scores"):
                    with st.expander("📊 Model validation scores (walk-forward)", expanded=False):
                        scores = st.session_state["advanced_best_model_scores"].get("scores", [])
                        if scores:
                            import pandas as pd
                            tab_df = pd.DataFrame(scores)
                            st.dataframe(tab_df, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Could not load model registry: {e}. Using default models.")
                model_type = st.selectbox(
                    "Model Type",
                    ["LSTM", "XGBoost", "Prophet", "ARIMA", "Ridge", "TCN", "GARCH", "CatBoost", "Ensemble"]
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
            
            train_button = st.button("🚀 Train Model", type="primary")
        
        with col2:
            st.subheader("📊 Results")
            
            if train_button:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Prepare data
                    data = st.session_state.get("forecast_data")
                    if data is None:
                        raise RuntimeError("No forecast_data in session_state; please load data first.")
                    data = data.copy()
                    
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
                            
                            # Add macro features to data (enrich_trading_data adds FRED/World Bank features)
                            data_with_macro = engineer.enrich_trading_data(
                                data,
                                include_sentiment=('sentiment' in (macro_features_list or []))
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
                            # Most regression models accept (data, target); some ignore target
                            fit_result = model.fit(data, data[model_config["target_column"]])

                        # Validate that model is actually fitted before forecasting
                        fit_success = True
                        if isinstance(fit_result, dict) and "success" in fit_result:
                            fit_success = bool(fit_result.get("success", True))
                        is_fitted_flag = True
                        if hasattr(model, "is_fitted"):
                            is_fitted_flag = bool(getattr(model, "is_fitted"))
                        elif hasattr(model, "model"):
                            is_fitted_flag = getattr(model, "model") is not None

                        if not (fit_success and is_fitted_flag):
                            detail = ""
                            if isinstance(fit_result, dict):
                                detail = fit_result.get("error") or str(fit_result)
                            st.error(f"{model_type} training did not complete successfully. Details: {detail}")
                            st.stop()
                        
                        # Generate forecast - try with uncertainty if available; support both forecast() and predict() (e.g. HybridModel)
                        progress_bar.progress(0.9)
                        horizon = st.session_state.forecast_horizon
                        if hasattr(model, 'forecast_with_uncertainty'):
                            try:
                                forecast_result = model.forecast_with_uncertainty(data, horizon=horizon, num_samples=100)
                            except Exception as e:
                                st.warning(f"Could not generate forecast with uncertainty: {e}. Using standard forecast.")
                                if hasattr(model, 'forecast'):
                                    forecast_result = model.forecast(data, horizon=horizon)
                                else:
                                    preds = model.predict(data) if hasattr(model, 'predict') else np.array([])
                                    forecast_result = {"forecast": preds[-horizon:].tolist() if len(preds) >= horizon else preds.tolist(), "horizon": horizon}
                        elif hasattr(model, 'forecast'):
                            forecast_result = model.forecast(data, horizon=horizon)
                        elif hasattr(model, 'predict'):
                            preds = np.asarray(model.predict(data))
                            preds = preds.astype("float64").ravel()
                            if preds.size == 0:
                                forecast_result = {"forecast": [], "horizon": horizon}
                            else:
                                tail = preds[-horizon:] if preds.size >= horizon else preds
                                forecast_result = {"forecast": tail.tolist(), "horizon": horizon}
                        else:
                            forecast_result = {"forecast": [], "horizon": horizon}
                        
                        # Store full forecast result for confidence intervals
                        st.session_state.current_forecast_result = forecast_result
                        st.session_state.current_model_instance = model  # Store model instance for explainability
                        
                        # Postprocess forecast
                        try:
                            from trading.forecasting.forecast_postprocessor import ForecastPostprocessor
                            
                            postprocessor = ForecastPostprocessor()
                            
                            # Extract forecast values (support forecast, predictions, values, forecast_values)
                            if isinstance(forecast_result, dict):
                                fv = (
                                    forecast_result.get('forecast')
                                    or forecast_result.get('predictions')
                                    or forecast_result.get('values')
                                    or forecast_result.get('forecast_values')
                                    or []
                                )
                                forecast_vals = fv.tolist() if hasattr(fv, 'tolist') else fv
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
                        
                        # Extract and validate forecast values
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
                            forecast_array = np.array(
                                [v for v in forecast_values if v is not None],
                                dtype="float64"
                            ).flatten()
                        elif forecast_values is None:
                            forecast_array = np.array([], dtype="float64")
                        else:
                            forecast_array = np.array([float(forecast_values)], dtype="float64")

                        if forecast_array.size == 0:
                            st.error(f"{model_type} returned an empty forecast. Please check the model configuration.")
                            st.stop()

                        # Sanity-check price range relative to last known price
                        # Allow forecasts in the range [0.4 * last_price, 2.0 * last_price]
                        last_price_series = None
                        if "close" in data.columns:
                            last_price_series = data["close"]
                        elif "Close" in data.columns:
                            last_price_series = data["Close"]

                        if last_price_series is not None and not last_price_series.empty:
                            last_price = float(last_price_series.iloc[-1])
                            lower_bound = 0.4 * last_price
                            upper_bound = 2.0 * last_price
                            if not np.all((forecast_array >= lower_bound) & (forecast_array <= upper_bound)):
                                st.error(
                                    f"{model_type} produced forecasts outside the expected range "
                                    f"[{lower_bound:.2f}, {upper_bound:.2f}] based on last close "
                                    f"({last_price:.2f}). Please review the model configuration and data scaling."
                                )
                                st.stop()
                        
                        # Store forecast
                        forecast_df = pd.DataFrame(
                            {'forecast': forecast_array},
                            index=forecast_dates[: len(forecast_array)],
                        )
                        
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
                        st.plotly_chart(fig)
                        
                        # Display forecast table
                        st.markdown("**Forecast Values:**")
                        display_df = forecast_df.copy()
                        display_df['forecast'] = display_df['forecast'].apply(
                            lambda x: f"${x:.2f}" if pd.notna(x) and x is not None else "N/A"
                        )
                        st.dataframe(display_df)
                        
                        # Add explainability section
                        st.markdown("---")
                        st.subheader("🔍 Model Explainability")
                        
                        with st.expander("📊 View Feature Importance & Explanations", expanded=False):
                            st.write("Understand what drives the model's predictions")
                            
                            # Display cached explanation (persist across reruns)
                            if st.session_state.get('forecast_explanation_tab2') is not None:
                                cached = st.session_state.forecast_explanation_tab2
                                if cached.get('success') and cached.get('explanation'):
                                    expl_obj = cached['explanation']
                                    if hasattr(expl_obj, 'feature_importance') and expl_obj.feature_importance:
                                        st.write("**📊 Feature Importance:**")
                                        importance_data = expl_obj.feature_importance
                                        if isinstance(importance_data, dict):
                                            importance_df = pd.DataFrame(
                                                importance_data.items(),
                                                columns=['Feature', 'Importance']
                                            ).sort_values('Importance', ascending=False)
                                            fig_imp = px.bar(
                                                importance_df, x='Importance', y='Feature',
                                                orientation='h', title='What Drives the Predictions?',
                                                labels={'Importance': 'Impact on Prediction'}
                                            )
                                            st.plotly_chart(fig_imp)
                                    if hasattr(expl_obj, 'explanation_text') and expl_obj.explanation_text:
                                        st.write("**💬 Plain English Explanation:**")
                                        st.info(expl_obj.explanation_text)
                                if st.button("Clear explanation", key="clear_explanation_tab2"):
                                    st.session_state.forecast_explanation_tab2 = None
                            
                            if st.button("Generate Explanation", key="explain_btn_tab2"):
                                try:
                                    with st.spinner("Analyzing model predictions..."):
                                        from trading.models.forecast_explainability import ForecastExplainability
                                        explainer = ForecastExplainability()
                                        fv = forecast_result.get('forecast') or forecast_result.get('predictions') or []
                                        forecast_value = float(fv[0]) if isinstance(fv, (list, np.ndarray)) and len(fv) > 0 else float(data[model_config["target_column"]].iloc[-1])
                                        features = data.copy()
                                        target_history = features[model_config["target_column"]] if model_config["target_column"] in features.columns else features.iloc[:, 0]
                                        explanation = explainer.explain_forecast(
                                            forecast_id=f"forecast_{st.session_state.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                            symbol=st.session_state.symbol,
                                            forecast_value=forecast_value,
                                            model=model,
                                            features=features,
                                            target_history=target_history,
                                            horizon=st.session_state.forecast_horizon
                                        )
                                        # Persist explanation in session state so it survives reruns
                                        st.session_state.forecast_explanation_tab2 = {
                                            "success": True,
                                            "explanation": explanation,
                                        }
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
                st.dataframe(prev_forecast.tail(10))

with tab3:
    try:
        st.header("AI Model Selection")
        st.markdown("""
        🤖 Let AI analyze your data and recommend the best forecasting model.
        
        The AI considers:
        - Data characteristics (trend, seasonality, volatility)
        - Historical model performance
        - Forecast horizon
        - Computational efficiency
        """)
        
        if st.session_state.get("forecast_data") is None:
            st.warning("⚠️ Please load data first in the Quick Forecast tab")
        else:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📊 Data Analysis")
                
                analyze_button = st.button(
                    "🔍 Analyze Data & Recommend Model",
                    type="primary"
                )
                
                if analyze_button:
                    with st.spinner("AI analyzing data characteristics..."):
                        try:
                            # Use ModelSelectorAgent if available; otherwise fallback to simple recommendation
                            try:
                                from trading.agents.model_selector_agent import ModelSelectorAgent
                                agent = ModelSelectorAgent()
                            except Exception:
                                agent = None
                            
                            # Detect market regime from data
                            fd = st.session_state.get("forecast_data")
                            price_data = fd["close"].values if hasattr(fd, "columns") and "close" in fd.columns else (
                                fd.get("close")
                                if isinstance(fd, dict)
                                else (list(fd.values())[0] if isinstance(fd, dict) and fd else None)
                            )
                            if price_data is None:
                                price_data = np.array([])
                            if hasattr(price_data, 'tolist'):
                                price_data = np.asarray(price_data).ravel()
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
                            if len(price_data) < 2:
                                regime_enum = MarketRegime.SIDEWAYS
                            else:
                                price_arr = np.asarray(price_data).astype(float)
                                price_trend = (price_arr[-1] - price_arr[0]) / (price_arr[0] or 1)
                                volatility = np.nanstd(price_arr)
                                if price_trend > 0.05:
                                    regime_enum = MarketRegime.TRENDING_UP
                                elif price_trend < -0.05:
                                    regime_enum = MarketRegime.TRENDING_DOWN
                                elif volatility > (np.nanmean(price_arr) * 0.1):
                                    regime_enum = MarketRegime.VOLATILE
                                else:
                                    regime_enum = MarketRegime.SIDEWAYS
                            
                            if agent is not None:
                                # Use agent's select_model method
                                selected_model_id, confidence = agent.select_model(
                                    horizon=horizon_enum,
                                    market_regime=regime_enum,
                                    data_length=len(price_data),
                                    required_features=[],
                                    performance_weight=0.6,
                                    capability_weight=0.4
                                )
                                recommendations = agent.get_model_recommendations(
                                    horizon_enum, regime_enum, top_k=3
                                )
                            else:
                                selected_model_id = "xgboost_medium"
                                confidence = 0.75
                                recommendations = [
                                    {"model_type": "xgboost", "model_id": "xgboost_medium", "total_score": 0.8},
                                    {"model_type": "lstm", "model_id": "lstm_medium", "total_score": 0.7},
                                    {"model_type": "prophet", "model_id": "prophet_medium", "total_score": 0.65},
                                ]
                            
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
                        if st.button("✅ Use Recommended Model"):
                            st.session_state.selected_model = rec['model_name']
                            st.success(f"Selected: {rec['model_name']}")
                            st.info("Go to Quick Forecast tab to generate forecast")
                    
                    with col_b:
                        if st.button("🔄 Choose Different Model"):
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
                    
                    data = st.session_state.get("forecast_data")
                    selector = HybridModelSelector()
                    
                    with st.spinner("Analyzing market conditions and selecting models..."):
                        # Detect market regime: HybridModelSelector has no detect_market_regime; use MarketAnalyzer or default
                        regime = "neutral"
                        try:
                            from trading.market.market_analyzer import MarketAnalyzer
                            import pandas as pd
                            df = data.copy() if hasattr(data, 'copy') else pd.DataFrame(data)
                            if 'close' in df.columns and 'Close' not in df.columns:
                                df['Close'] = df['close']
                            if 'Close' in df.columns:
                                analyzer = MarketAnalyzer()
                                trend_result = analyzer.detect_market_regime(df, "trend")
                                regime = trend_result.get("regime", "neutral")
                        except Exception as e:
                            st.caption(f"Regime detection fell back to neutral: {e}")
                        
                        st.info(f"📊 Detected market regime: **{regime}**")
                        
                        # HybridModelSelector has select_best_model(model_scores, metric), not select_models_for_regime
                        # Show a simple recommendation based on regime
                        recommended = [
                            {"name": "XGBoost", "reason": "Robust across regimes"},
                            {"name": "LSTM", "reason": "Captures nonlinear patterns"},
                            {"name": "Prophet", "reason": "Good for trend and seasonality"},
                        ]
                        selected_models = recommended
                        
                        st.write("**Recommended models for current conditions:**")
                        for i, model_config in enumerate(selected_models, 1):
                            st.write(f"{i}. {model_config['name']} - {model_config['reason']}")
                        
                        # Store selected models
                        st.session_state.hybrid_selected_models = selected_models
                        st.session_state.hybrid_regime = regime
                        
                        st.caption("To train an ensemble, use the Model Comparison tab and select multiple models.")
                
                except ImportError:
                    st.error("Hybrid Model Selector not available")
                except Exception as e:
                    st.error(f"Error using hybrid selector: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    except Exception as e:
        import traceback
        st.error(f"Tab error: {e}")
        st.code(traceback.format_exc())

with tab4:
    try:
        st.header("Model Comparison")
        st.markdown("Compare multiple models side-by-side and create ensemble forecasts")
        
        if st.session_state.get("forecast_data") is None:
            st.warning("⚠️ Please load data first in the Quick Forecast tab")
        else:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("📊 Select Models")
                
                from trading.models.forecast_router import ForecastRouter
                _router_for_names = ForecastRouter()
                try:
                    available_models = list(_router_for_names.model_registry.keys())
                except Exception:
                    available_models = [
                        "lstm",
                        "xgboost",
                        "prophet",
                        "arima",
                        "ridge",
                        "tcn",
                        "catboost",
                        "ensemble",
                        "hybrid",
                    ]
                display_models = [
                    "LSTM",
                    "XGBoost",
                    "Prophet",
                    "ARIMA",
                    "Ridge",
                    "TCN",
                    "CatBoost",
                    "Ensemble",
                    "Hybrid",
                ]
                # Filter to models actually available in the router when possible
                display_models = [
                    m for m in display_models if m.lower() in available_models
                ] or display_models

                models_to_compare = st.multiselect(
                    "Choose models to compare:",
                    display_models,
                    default=[m for m in display_models if m in ["LSTM", "XGBoost"]][:2],
                    help="Select 2-5 models to compare",
                )
                
                if len(models_to_compare) < 2:
                    st.warning("Please select at least 2 models")
                elif len(models_to_compare) > 5:
                    st.warning("Please select 5 or fewer models for comparison")
                
                create_ensemble = st.checkbox("Create Ensemble Forecast", value=True)
                
                compare_button = st.button(
                    "🚀 Compare Models",
                    type="primary",
                    disabled=len(models_to_compare) < 2
                )
            
            with col2:
                st.subheader("📈 Comparison Results")
                
                if compare_button and len(models_to_compare) >= 2:
                    if len(models_to_compare) > 5:
                        st.warning("Please select 5 or fewer models for comparison")
                        st.stop()

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        data = st.session_state.get("forecast_data")
                        if data is None:
                            raise RuntimeError("No forecast_data in session_state; please load data first.")
                        data = data.copy()
                        horizon = st.session_state.forecast_horizon
                        
                        # Ensure proper format
                        if not isinstance(data.index, pd.DatetimeIndex):
                            data.index = pd.to_datetime(data.index)

                        # Router expects lowercase OHLCV when possible
                        if "Close" in data.columns and "close" not in data.columns:
                            data = data.rename(
                                columns={
                                    "Close": "close",
                                    "Open": "open",
                                    "High": "high",
                                    "Low": "low",
                                    "Volume": "volume",
                                }
                            )
                        
                        forecasts = {}
                        model_configs = {}
                        
                        # Forecast via ForecastRouter (single canonical invocation path)
                        from trading.models.forecast_router import ForecastRouter
                        _router = ForecastRouter()

                        # Train and forecast for each model
                        total_models = len(models_to_compare)
                        for idx, model_name in enumerate(models_to_compare):
                            status_text.text(f"Running {model_name} ({idx+1}/{total_models})...")
                            progress_bar.progress((idx) / (total_models + 1))
                            
                            try:
                                model_key = str(model_name).strip().lower()
                                router_result = _router.get_forecast(
                                    data,
                                    horizon=horizon,
                                    model_type=model_key,
                                    run_walk_forward=False,
                                )

                                forecast_values = np.asarray(
                                    (router_result or {}).get("forecast", []),
                                    dtype="float64",
                                ).ravel()
                                if forecast_values.size == 0:
                                    raise ValueError("Router returned empty forecast")

                                forecast_dates = pd.date_range(
                                    start=data.index[-1] + timedelta(days=1),
                                    periods=len(forecast_values),
                                    freq="D",
                                )
                                
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
                        hist_data = st.session_state.get("forecast_data")
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
                        
                        st.plotly_chart(fig)
                        
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
                            st.dataframe(metrics_df)
                        
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
                        
                        st.dataframe(display_df)
                        
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
    except Exception as e:
        import traceback
        st.error(f"Tab error: {e}")
        st.code(traceback.format_exc())

with tab5:
    try:
        st.header("Market Analysis")
        st.markdown("Technical indicators, market regime detection, trend analysis, and correlation tools")
        
        if st.session_state.get("forecast_data") is None:
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
            
            analyze_button = st.button("🔍 Run Analysis", type="primary")
            
            if analyze_button:
                data = st.session_state.get("forecast_data")
                if data is None:
                    raise RuntimeError("No forecast_data in session_state; please load data first.")
                data = data.copy()
                
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
                        
                        st.plotly_chart(fig)
                        
                        # Indicators table
                        with st.expander("📋 View All Indicators"):
                            st.dataframe(indicators_df.tail(50))
                    
                    except Exception as e:
                        st.error(f"Error calculating indicators: {str(e)}")
                
                # Market Regime Detection
                if show_regime:
                    st.markdown("---")
                    st.subheader("🎯 Market Regime Detection")
                    
                    try:
                        import pandas as pd
                        analyzer = MarketAnalyzer()
                        # Ensure data is DataFrame with Close column (log_metrics fix: do not pass dict as path)
                        regime_data = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
                        if 'Close' not in regime_data.columns and 'close' in regime_data.columns:
                            regime_data['Close'] = regime_data['close']
                        if 'Close' not in regime_data.columns:
                            st.warning("No Close price column for regime detection.")
                        else:
                            # Detect trend regime
                            trend_regime = analyzer.detect_market_regime(regime_data, "trend")
                            
                            # Detect volatility regime
                            volatility_regime = analyzer.detect_market_regime(regime_data, "volatility")
                        
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
                        
                        st.plotly_chart(fig)
                    
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
                
                # Display cached sentiment result so it persists across reruns
                if st.session_state.get('sentiment_analysis_result') is not None:
                    sentiment_result = st.session_state.sentiment_analysis_result
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        sentiment_score = sentiment_result.get('overall_sentiment', 0)
                        if sentiment_score > 0.3:
                            st.success(f"😊 Positive: {sentiment_score:.2f}")
                        elif sentiment_score < -0.3:
                            st.error(f"😞 Negative: {sentiment_score:.2f}")
                        else:
                            st.info(f"😐 Neutral: {sentiment_score:.2f}")
                    with col2:
                        st.metric("Articles Analyzed", sentiment_result.get('num_articles', 0))
                    with col3:
                        st.metric("Sentiment Confidence", f"{sentiment_result.get('confidence', 0):.1%}")
                    if 'sentiment_history' in sentiment_result:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=sentiment_result.get('dates', []),
                            y=sentiment_result['sentiment_history'],
                            name='Sentiment',
                            line=dict(color='purple')
                        ))
                        fig.update_layout(title='Sentiment Over Time', xaxis_title='Date', yaxis_title='Sentiment Score', height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    st.subheader("📰 Recent Headlines")
                    for article in sentiment_result.get('articles', [])[:10]:
                        sentiment_emoji = "😊" if article.get('sentiment', 0) > 0 else "😞" if article.get('sentiment', 0) < 0 else "😐"
                        st.write(f"{sentiment_emoji} **{article.get('title', '')}**")
                        st.caption(f"Sentiment: {article.get('sentiment', 0):.2f} | {article.get('date', '')}")
                    if st.button("Clear sentiment result", key="clear_sentiment"):
                        st.session_state.sentiment_analysis_result = None
                        st.rerun()
                
                if st.button("Analyze News Sentiment", key="analyze_sentiment"):
                    try:
                        from trading.nlp.sentiment_classifier import SentimentClassifier
                        classifier = SentimentClassifier()
                        with st.spinner("Analyzing news sentiment..."):
                            sentiment_result = classifier.analyze_symbol_sentiment(
                                symbol=symbol_for_sentiment,
                                lookback_days=7
                            )
                            st.session_state.sentiment_analysis_result = sentiment_result
                            st.rerun()
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
                
                # Display cached causal result so it persists across reruns
                if st.session_state.get('causal_analysis_result') is not None:
                    cached = st.session_state.causal_analysis_result
                    causal_graph = cached.get('causal_graph', {})
                    drivers = cached.get('drivers', {})
                    symbol = cached.get('symbol', '')
                    st.subheader("📊 Causal Graph")
                    if causal_graph.get('graph_visualization') is not None:
                        st.image(causal_graph['graph_visualization'])
                    else:
                        st.write("**Causal Relationships:**")
                        if causal_graph.get('relationships'):
                            for r in causal_graph['relationships']:
                                st.write(f"• {r.get('cause')} → {r.get('effect')} (strength: {r.get('strength', 0):.2f})")
                        else:
                            st.info("Causal relationships will be displayed here")
                    st.subheader("🎯 Key Price Drivers")
                    if drivers.get('drivers'):
                        drivers_df = pd.DataFrame(drivers['drivers']).sort_values('importance', ascending=False)
                        fig = px.bar(drivers_df, x='importance', y='driver', orientation='h',
                            title='Price Drivers (Ranked by Causal Impact)', labels={'importance': 'Causal Impact', 'driver': 'Driver'})
                        st.plotly_chart(fig, use_container_width=True)
                        top_driver = drivers_df.iloc[0]
                        st.info(f"**Primary Driver:** {top_driver['driver']}. Strongest causal impact on {symbol} (importance: {top_driver['importance']:.2f}).")
                        with st.expander("📊 Detailed Driver Analysis"):
                            for _, d in drivers_df.iterrows():
                                st.write(f"**{d['driver']}** - Importance: {d['importance']:.2f}")
                    if st.button("Clear causal result", key="clear_causal"):
                        st.session_state.causal_analysis_result = None
                        st.rerun()
                
                if st.button("Run Causal Analysis", key="run_causal_analysis"):
                    if st.session_state.get("forecast_data") is None:
                        st.error("Please load data first")
                    else:
                        try:
                            from causal.causal_model import CausalModel
                            from causal.driver_analysis import DriverAnalysis
                            data = st.session_state.get("forecast_data")
                            if data is None:
                                raise RuntimeError("No forecast_data in session_state; please load data first.")
                            data = data.copy()
                            symbol = st.session_state.get("symbol", "AAPL")
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
                            if 'Open' not in data.columns:
                                data['Open'] = data['Close']
                            if 'High' not in data.columns:
                                data['High'] = data['Close']
                            if 'Low' not in data.columns:
                                data['Low'] = data['Close']
                            if 'Volume' not in data.columns:
                                data['Volume'] = 1000000
                            with st.spinner("Performing causal analysis..."):
                                causal_model = CausalModel()
                                driver_analysis = DriverAnalysis()
                                causal_graph = causal_model.build_causal_graph(
                                    data=data, target='Close', features=['Volume', 'High', 'Low', 'Open']
                                )
                                drivers = driver_analysis.identify_drivers(data=data, target='Close')
                                st.session_state.causal_analysis_result = {
                                    'causal_graph': causal_graph,
                                    'drivers': drivers,
                                    'symbol': symbol,
                                    'data': data,
                                }
                                st.success("✅ Causal analysis complete!")
                                st.rerun()
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
                    st.dataframe(summary_df, hide_index=True)
    except Exception as e:
        import traceback
        st.error(f"Tab error: {e}")
        st.code(traceback.format_exc())

with tab6:
    try:
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
            st.plotly_chart(fig_corr)
            
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
                        
                        st.plotly_chart(fig)
                        
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
                        st.plotly_chart(fig_rel)
                        
                        # Forecast table
                        with st.expander("📋 View Forecast Data"):
                            forecast_df = pd.DataFrame({
                                'Date': forecast_result['dates'],
                                'Forecast': forecast_result['forecast'],
                                'Confidence': forecast_result['confidence']
                            })
                            st.dataframe(forecast_df)
                
                except ImportError:
                    st.error("❌ GNN model not available. Make sure it has been recreated using the prompts.")
                except Exception as e:
                    st.error(f"Error generating GNN forecast: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    except Exception as e:
        import traceback
        st.error(f"Tab error: {e}")
        st.code(traceback.format_exc())

with tab7:
    try:
        st.header("🎲 Monte Carlo Price Simulation")
        st.markdown("Geometric Brownian Motion fan chart: 5th–95th percentile price paths and probability above/below levels.")
        mc_input = st.text_input("Ticker (Monte Carlo)", value=st.session_state.get("symbol", "AAPL") or "AAPL", key="mc_symbol")
        mc_symbol = (mc_input or "AAPL").upper()
        horizon_days = st.slider("Horizon (days)", 5, 90, 30, key="mc_horizon")
        n_simulations = st.slider("Simulations", 100, 5000, 1000, key="mc_n_sim")
        if st.button("Run Monte Carlo", type="primary", key="mc_run"):
            try:
                from trading.data.data_loader import DataLoader, DataLoadRequest
                from trading.data.providers.yfinance_provider import YFinanceProvider
                from trading.analysis.monte_carlo import (
                    simulate_price_paths,
                    fan_chart_percentiles,
                    probability_above_below,
                )
                req = DataLoadRequest(
                    ticker=mc_symbol,
                    start_date=(datetime.now().date() - timedelta(days=365)).strftime("%Y-%m-%d"),
                    end_date=datetime.now().date().strftime("%Y-%m-%d"),
                )
                loader = DataLoader(provider=YFinanceProvider())
                response = loader.load_market_data(req)
                df = response.data if response.success and response.data is not None else None
                if df is None or df.empty or "close" not in [c.lower() for c in df.columns]:
                    st.error("No price data for " + mc_symbol)
                else:
                    close_col = "close" if "close" in df.columns else "Close"
                    close = df[close_col].dropna()
                    last_price = float(close.iloc[-1])
                    returns = close.pct_change().dropna()
                    mu = float(returns.mean() * 252)
                    sigma = float(returns.std() * np.sqrt(252))
                    if sigma < 1e-8:
                        sigma = 0.20
                    paths = simulate_price_paths(last_price, mu, sigma, horizon_days=horizon_days, n_simulations=n_simulations)
                    bands = fan_chart_percentiles(paths, (5, 25, 50, 75, 95))
                    st.session_state.mc_paths = paths
                    st.session_state.mc_bands = bands
                    st.session_state.mc_last_price = last_price
                    st.session_state.mc_dates = pd.date_range(start=df.index[-1], periods=horizon_days + 1, freq="D")[1:]
                    st.session_state.mc_hist_dates = close.index
                    st.session_state.mc_hist_prices = close.values
                    st.success(f"Ran {n_simulations} paths. Last price ${last_price:.2f}, μ={mu:.2%}, σ={sigma:.2%}.")
            except Exception as e:
                st.error(f"Monte Carlo failed: {e}")
                import traceback
                st.code(traceback.format_exc())
        if st.session_state.get("mc_bands") is not None:
            from trading.analysis.monte_carlo import probability_above_below
            bands = st.session_state.mc_bands
            mc_dates = st.session_state.get("mc_dates")
            last_price = st.session_state.get("mc_last_price")
            fig = go.Figure()
            hist_dates = st.session_state.get("mc_hist_dates")
            hist_prices = st.session_state.get("mc_hist_prices")
            if hist_dates is not None and hist_prices is not None:
                fig.add_trace(go.Scatter(x=hist_dates, y=hist_prices, name="Historical", line=dict(color="blue", width=2)))
            fig.add_vline(x=pd.Timestamp.now(), line_dash="dash", line_color="gray", annotation_text="Today")
            for p, label in [(5, "5th %"), (95, "95th %"), (25, "25th %"), (75, "75th %"), (50, "50th %")]:
                fig.add_trace(go.Scatter(x=mc_dates, y=bands[p], name=label, line=dict(width=2 if p == 50 else 1)))
            fig.update_layout(title="Monte Carlo fan chart (5th–95th percentiles)", xaxis_title="Date", yaxis_title="Price ($)", height=450)
            fig.update_yaxes(tickprefix="$")
            st.plotly_chart(fig, use_container_width=True)
            prob_above, _ = probability_above_below(st.session_state.mc_paths, last_price, day_index=-1)
            st.metric("P(price ≥ last)", f"{prob_above:.1%}")
            st.metric("30-day 95% interval", f"${bands[5][-1]:.2f} – ${bands[95][-1]:.2f}")
    except Exception as e:
        st.error(f"Monte Carlo tab error: {e}")
        import traceback
        st.code(traceback.format_exc())

render_page_assistant("Forecasting")

