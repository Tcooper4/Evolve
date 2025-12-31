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
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Backend imports
from trading.data.data_loader import DataLoader, DataLoadRequest
from trading.data.providers.yfinance_provider import YFinanceProvider
from trading.models.lstm_model import LSTMModel
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
                use_container_width=True
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
                    if st.button("✅ Use Recommended Model", use_container_width=True):
                        st.session_state.selected_model = rec['model_name']
                        st.success(f"Selected: {rec['model_name']}")
                        st.info("Go to Quick Forecast tab to generate forecast")
                
                with col_b:
                    if st.button("🔄 Choose Different Model", use_container_width=True):
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
                use_container_width=True,
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
                                config = {
                                    "target_column": "close" if "close" in data.columns else "Close",
                                    "sequence_length": 60,
                                    "hidden_dim": 64,
                                    "num_layers": 2,
                                    "dropout": 0.2,
                                    "learning_rate": 0.001
                                }
                                model = LSTMModel(config)
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
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
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
                        st.dataframe(metrics_df, use_container_width=True)
                    
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
                            display_df[col] = display_df[col].map('${:.2f}'.format)
                    
                    st.dataframe(display_df, use_container_width=True)
                    
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
        
        analyze_button = st.button("🔍 Run Analysis", type="primary", use_container_width=True)
        
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
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Indicators table
                    with st.expander("📋 View All Indicators"):
                        st.dataframe(indicators_df.tail(50), use_container_width=True)
                
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
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error analyzing trend: {str(e)}")
            
            # Correlation Analysis (if multiple symbols available)
            st.markdown("---")
            st.subheader("🔗 Correlation Analysis")
            st.info("Correlation analysis requires multiple symbols. Load additional data to compare.")
            
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
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

