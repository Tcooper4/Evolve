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

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Backend imports
from trading.data.data_loader import DataLoader, DataLoadRequest
from trading.data.providers.yfinance_provider import YFinanceProvider
from trading.models.lstm_model import LSTMModel
from trading.models.xgboost_model import XGBoostModel
from trading.models.prophet_model import ProphetModel
from trading.models.arima_model import ARIMAModel
from trading.data.preprocessing import FeatureEngineering, DataPreprocessor

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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚀 Quick Forecast",
    "⚙️ Advanced Forecasting", 
    "🤖 AI Model Selection",
    "📊 Model Comparison",
    "📈 Market Analysis"
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
        
        forecast_horizon = st.slider(
            "Forecast Horizon (days)",
            min_value=1,
            max_value=30,
            value=7,
            help="Number of days to forecast into the future"
        )
        
        submitted = st.form_submit_button("📊 Load Data", use_container_width=True)
    
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
                            st.session_state.forecast_horizon = forecast_horizon
                            
                            st.success(f"✅ Loaded {len(data)} days of data for {symbol}")
                        
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                st.info("Please check the ticker symbol and try again.")
    
    # Display loaded data
    if st.session_state.forecast_data is not None:
        st.markdown("---")
        st.subheader("📊 Data Preview")
        
        data = st.session_state.forecast_data
        
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
        
        # Price chart
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
            st.dataframe(data.tail(50), use_container_width=True)
        
        # Model Selection & Forecasting
        st.markdown("---")
        st.subheader("🎯 Generate Forecast")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Select Model**")
            
            model_descriptions = {
                "LSTM (Deep Learning)": "Neural network for time series. Best for complex patterns.",
                "XGBoost (Gradient Boosting)": "Tree-based model. Fast and accurate for most data.",
                "Prophet (Facebook)": "Handles seasonality well. Good for business data.",
                "ARIMA (Statistical)": "Classic statistical model. Works well for stationary data."
            }
            
            selected_model = st.radio(
                "Choose model:",
                list(model_descriptions.keys()),
                help="Different models work better for different data patterns"
            )
            
            st.info(model_descriptions[selected_model])
            
            forecast_button = st.button(
                "🚀 Generate Forecast",
                type="primary",
                use_container_width=True
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
                        
                        # Initialize model with proper config
                        model = None
                        if "LSTM" in selected_model:
                            config = {
                                "target_column": "close",
                                "sequence_length": 60,
                                "hidden_dim": 64,
                                "num_layers": 2,
                                "dropout": 0.2,
                                "learning_rate": 0.001
                            }
                            model = LSTMModel(config)
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
                            
                            # Ensure forecast_values is array-like
                            if isinstance(forecast_values, (list, np.ndarray)):
                                forecast_values = np.array(forecast_values).flatten()
                            else:
                                forecast_values = np.array([forecast_values] * horizon)
                            
                            # Create forecast DataFrame
                            forecast_df = pd.DataFrame({
                                'forecast': forecast_values
                            }, index=forecast_dates[:len(forecast_values)])
                            
                            # Store in session state
                            st.session_state.current_forecast = forecast_df
                            st.session_state.current_model = selected_model
                            
                            st.success(f"✅ Forecast generated using {selected_model}")
                
                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.info("Try adjusting the date range or selecting a different model.")
            
            # Display forecast
            if st.session_state.get('current_forecast') is not None:
                st.markdown(f"**Forecast using {st.session_state.current_model}**")
                
                # Create combined chart
                fig = go.Figure()
                
                # Historical data
                hist_data = st.session_state.forecast_data
                fig.add_trace(go.Scatter(
                    x=hist_data.index,
                    y=hist_data['close'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue', width=2)
                ))
                
                # Forecast
                forecast_df = st.session_state.current_forecast
                fig.add_trace(go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df['forecast'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='red', width=2, dash='dash'),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title=f"{st.session_state.symbol} - Historical & Forecast",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast table
                st.markdown("**Forecast Values:**")
                display_df = forecast_df.copy()
                display_df['forecast'] = display_df['forecast'].map('${:.2f}'.format)
                st.dataframe(display_df, use_container_width=True)
                
                # Download button
                csv = forecast_df.to_csv()
                st.download_button(
                    label="📥 Download Forecast CSV",
                    data=csv,
                    file_name=f"{st.session_state.symbol}_forecast.csv",
                    mime="text/csv"
                )

with tab2:
    st.header("Advanced Forecasting")
    st.markdown("Full model configuration with hyperparameter tuning and feature engineering")
    
    if st.session_state.forecast_data is None:
        st.warning("⚠️ Please load data first in the Quick Forecast tab")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Configuration")
            
            # Model selection
            model_type = st.selectbox(
                "Model Type",
                ["LSTM", "XGBoost", "Prophet", "ARIMA"]
            )
            
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
            
            normalize = st.checkbox("Normalize Data", value=True)
            
            train_button = st.button("🚀 Train Model", type="primary", use_container_width=True)
        
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
                        
                        # Create model config
                        model_config = {
                            "target_column": "close" if "close" in data.columns else "Close"
                        }
                        
                        if model_type == "LSTM":
                            model_config.update({
                                "sequence_length": params.get('sequence_length', 60),
                                "hidden_dim": params.get('hidden_dim', 64),
                                "num_layers": params.get('num_layers', 2),
                                "dropout": params.get('dropout', 0.2),
                                "learning_rate": params.get('learning_rate', 0.001)
                            })
                            model = LSTMModel(model_config)
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
                                "target_column": "close" if "close" in data.columns else "Close",
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
                        
                        # Generate forecast
                        progress_bar.progress(0.9)
                        forecast_result = model.forecast(data, horizon=st.session_state.forecast_horizon)
                        
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
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display forecast table
                        st.markdown("**Forecast Values:**")
                        display_df = forecast_df.copy()
                        display_df['forecast'] = display_df['forecast'].map('${:.2f}'.format)
                        st.dataframe(display_df, use_container_width=True)
                
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
                st.dataframe(prev_forecast.tail(10), use_container_width=True)

with tab3:
    st.header("AI Model Selection")
    st.markdown("Let AI analyze your data and recommend the best model")
    st.info("Integration pending...")

with tab4:
    st.header("Model Comparison")
    st.markdown("Compare multiple models side-by-side")
    st.info("Integration pending...")

with tab5:
    st.header("Market Analysis")
    st.markdown("Technical indicators and market regime detection")
    st.info("Integration pending...")

