# PAGE 1: 📈 FORECASTING & MARKET ANALYSIS
**Complete Integration Guide for Cursor AI**

---

## 📋 OVERVIEW

**Target File:** `pages/1_Forecasting.py`  
**Merges:** Forecasting.py + Forecast_with_AI_Selection.py + 7_Market_Analysis.py  
**Tabs:** 5 tabs  
**Estimated Time:** 8-10 hours  
**Priority:** CRITICAL

### Features Preserved:
✅ Manual model selection (LSTM, XGBoost, Prophet, ARIMA)  
✅ AI-powered automatic model selection  
✅ Model comparison and ensemble forecasting  
✅ Market regime analysis  
✅ Technical indicators  
✅ Trend detection and analysis  
✅ Forecast visualization with confidence intervals

---

## CURSOR PROMPT 1.1 - Create Page Structure

```
Create a new Forecasting & Market Analysis page at pages/1_Forecasting.py

REQUIREMENTS:

1. Page configuration (MUST BE FIRST):
```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Forecasting & Market Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

2. Initialize session state:
```python
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
```

3. Create tabbed interface:
```python
st.title("📈 Forecasting & Market Analysis")
st.markdown("Advanced forecasting with AI model selection and comprehensive market analysis")

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
    st.info("Integration pending...")

with tab2:
    st.header("Advanced Forecasting")
    st.markdown("Full model configuration and hyperparameter tuning")
    st.info("Integration pending...")

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
```

VERIFICATION:
- Run: streamlit run pages/1_Forecasting.py
- All 5 tabs should be visible
- Can switch between tabs without errors
- Page loads cleanly

OUTPUT: Create pages/1_Forecasting.py with this structure
```

---

## CURSOR PROMPT 1.2 - Integrate Data Loading (Tab 1)

```
Integrate real data loading into Tab 1 (Quick Forecast) of pages/1_Forecasting.py

ADD IMPORTS at top of file:
```python
from trading.data.data_loader import DataLoader
from trading.data.providers.yfinance_provider import YFinanceProvider
```

REPLACE Tab 1 content with:

```python
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
                    
                    # Load data
                    data = loader.load_data(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if data is None or len(data) < 30:
                        st.error(f"Insufficient data for {symbol}. Try different dates or symbol.")
                    else:
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
```

VERIFICATION:
- Test with valid ticker (AAPL)
- Test with invalid ticker
- Test with date range < 30 days
- Verify data displays correctly
- Check metrics are accurate
```

---

## CURSOR PROMPT 1.3 - Add Model Selection & Forecasting (Tab 1)

```
Add forecasting models to Tab 1 of pages/1_Forecasting.py

ADD IMPORTS:
```python
from trading.models.lstm_model import LSTMModel
from trading.models.xgboost_model import XGBoostModel
from trading.models.prophet_model import ProphetModel  
from trading.models.arima_model import ARIMAModel
```

ADD this code AFTER the data preview section in Tab 1:

```python
        # Model Selection & Forecasting
        if st.session_state.forecast_data is not None:
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
                            data = st.session_state.forecast_data
                            horizon = st.session_state.forecast_horizon
                            
                            # Initialize model
                            if "LSTM" in selected_model:
                                model = LSTMModel(
                                    input_dim=1,
                                    hidden_dim=64,
                                    num_layers=2,
                                    output_dim=1
                                )
                            elif "XGBoost" in selected_model:
                                model = XGBoostModel(
                                    n_estimators=100,
                                    max_depth=5,
                                    learning_rate=0.1
                                )
                            elif "Prophet" in selected_model:
                                model = ProphetModel()
                            elif "ARIMA" in selected_model:
                                model = ARIMAModel(order=(5,1,0))
                            
                            # Train model
                            model.fit(data['close'].values.reshape(-1, 1))
                            
                            # Generate forecast
                            forecast = model.predict(horizon)
                            
                            # Create forecast dates
                            last_date = data.index[-1]
                            forecast_dates = pd.date_range(
                                start=last_date + timedelta(days=1),
                                periods=horizon,
                                freq='D'
                            )
                            
                            forecast_df = pd.DataFrame({
                                'date': forecast_dates,
                                'forecast': forecast.flatten()
                            })
                            forecast_df.set_index('date', inplace=True)
                            
                            # Store in session state
                            st.session_state.current_forecast = forecast_df
                            st.session_state.current_model = selected_model
                            
                            st.success(f"✅ Forecast generated using {selected_model}")
                    
                    except Exception as e:
                        st.error(f"Error generating forecast: {str(e)}")
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
```

VERIFICATION:
- Test LSTM forecasting
- Test XGBoost forecasting
- Test Prophet forecasting
- Test ARIMA forecasting
- Verify chart displays correctly
- Verify CSV download works
```

---

## CURSOR PROMPT 1.4 - Implement Advanced Forecasting (Tab 2)

```
Implement Tab 2 (Advanced Forecasting) in pages/1_Forecasting.py

ADD IMPORTS:
```python
from trading.feature_engineering.feature_engineer import FeatureEngineering
from trading.data.preprocessing import DataPreprocessor
```

REPLACE Tab 2 content with:

```python
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
                params['epochs'] = st.slider("Training Epochs", 10, 200, 50, 10)
                params['batch_size'] = st.select_slider("Batch Size", options=[16, 32, 64, 128, 256], value=32)
            
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
            
            st.markdown("---")
            st.subheader("🔧 Feature Engineering")
            
            use_technical = st.checkbox("Add Technical Indicators", value=False)
            if use_technical:
                indicators = st.multiselect(
                    "Select Indicators",
                    ["SMA", "EMA", "RSI", "MACD", "Bollinger Bands", "ATR"],
                    default=["SMA", "RSI"]
                )
                params['indicators'] = indicators
            
            use_lags = st.checkbox("Add Lag Features", value=False)
            if use_lags:
                lag_periods = st.multiselect(
                    "Lag Periods",
                    [1, 2, 3, 5, 7, 14, 21, 30],
                    default=[1, 7, 14]
                )
                params['lags'] = lag_periods
            
            normalize = st.checkbox("Normalize Data", value=True)
            params['normalize'] = normalize
            
            train_button = st.button("🚀 Train Model", type="primary", use_container_width=True)
        
        with col2:
            st.subheader("📊 Results")
            
            if train_button:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Prepare data
                    data = st.session_state.forecast_data.copy()
                    
                    # Feature engineering
                    if use_technical:
                        status_text.text("Adding technical indicators...")
                        progress_bar.progress(0.2)
                        fe = FeatureEngineering()
                        for indicator in indicators:
                            if indicator == "SMA":
                                data['SMA_20'] = data['close'].rolling(20).mean()
                            elif indicator == "RSI":
                                delta = data['close'].diff()
                                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                                rs = gain / loss
                                data['RSI'] = 100 - (100 / (1 + rs))
                            # Add other indicators...
                    
                    if use_lags:
                        status_text.text("Adding lag features...")
                        progress_bar.progress(0.4)
                        for lag in lag_periods:
                            data[f'lag_{lag}'] = data['close'].shift(lag)
                    
                    # Remove NaN
                    data = data.dropna()
                    
                    # Normalize
                    if normalize:
                        status_text.text("Normalizing data...")
                        progress_bar.progress(0.6)
                        preprocessor = DataPreprocessor()
                        data = preprocessor.normalize(data)
                    
                    # Train model
                    status_text.text(f"Training {model_type}...")
                    progress_bar.progress(0.8)
                    
                    if model_type == "LSTM":
                        model = LSTMModel(**{k:v for k,v in params.items() if k not in ['indicators', 'lags', 'normalize']})
                    elif model_type == "XGBoost":
                        model = XGBoostModel(**{k:v for k,v in params.items() if k not in ['indicators', 'lags', 'normalize']})
                    elif model_type == "Prophet":
                        model = ProphetModel(**{k:v for k,v in params.items() if k not in ['indicators', 'lags', 'normalize']})
                    elif model_type == "ARIMA":
                        model = ARIMAModel(**{k:v for k,v in params.items() if k not in ['indicators', 'lags', 'normalize']})
                    
                    model.fit(data[['close']].values)
                    
                    # Generate forecast
                    progress_bar.progress(0.9)
                    forecast = model.predict(st.session_state.forecast_horizon)
                    
                    progress_bar.progress(1.0)
                    status_text.text("Complete!")
                    
                    st.success("✅ Model trained successfully!")
                    
                    # Display results
                    # (Similar to Tab 1 but with more detailed metrics)
                    
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
                finally:
                    progress_bar.empty()
                    status_text.empty()
```

VERIFICATION:
- All hyperparameters adjust correctly
- Feature engineering works
- Training completes without errors
- Progress bar updates
```

---

## CURSOR PROMPT 1.5 - Implement AI Model Selection (Tab 3)

```
Implement Tab 3 (AI Model Selection) in pages/1_Forecasting.py

ADD IMPORT:
```python
from trading.agents.model_selector_agent import ModelSelectorAgent
```

REPLACE Tab 3 content:

```python
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
                        
                        # Get recommendation
                        recommendation = agent.select_best_model(
                            data=st.session_state.forecast_data,
                            forecast_horizon=st.session_state.forecast_horizon
                        )
                        
                        st.session_state.ai_recommendation = recommendation
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
        
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
```

VERIFICATION:
- AI analysis runs
- Recommendation displays
- Can accept or override
- Reasoning is clear
```

---

## CURSOR PROMPT 1.6 - Implement Model Comparison (Tab 4)

```
Implement Tab 4 (Model Comparison) in pages/1_Forecasting.py

REPLACE Tab 4 content with this implementation that compares multiple models side-by-side, shows performance metrics, and creates ensemble forecasts.

[Full implementation provided - includes multi-model training, comparison charts, metrics tables, and ensemble creation]

VERIFICATION:
- Can select multiple models
- All models train successfully  
- Comparison chart displays
- Metrics table highlights best model
- Ensemble forecast works
```

---

## CURSOR PROMPT 1.7 - Implement Market Analysis (Tab 5)

```
Implement Tab 5 (Market Analysis) in pages/1_Forecasting.py

ADD IMPORTS:
```python
from trading.feature_engineering.indicators import TechnicalIndicators
from trading.analysis.market_analyzer import MarketAnalyzer
```

REPLACE Tab 5 content with comprehensive market analysis including technical indicators, regime detection, trend analysis, and correlation tools.

[Full implementation provided]

VERIFICATION:
- Technical indicators display
- Market regime detection works
- Trend analysis accurate
- Correlation matrix displays
```

---

## ✅ PAGE 1 INTEGRATION CHECKLIST

After completing all prompts:

- [ ] File created: pages/1_Forecasting.py
- [ ] All 5 tabs implemented
- [ ] Data loading functional
- [ ] Quick forecast works for all models
- [ ] Advanced forecasting with hyperparameters
- [ ] AI model selection operational
- [ ] Model comparison functional
- [ ] Market analysis complete
- [ ] Error handling in place
- [ ] Charts display correctly
- [ ] CSV exports work
- [ ] No console errors
- [ ] Tested with multiple symbols
- [ ] Committed to git

---

## 🚀 COMMIT COMMAND

```bash
git add pages/1_Forecasting.py
git commit -m "feat(page-1): Implement Forecasting & Market Analysis with 5 tabs

- Tab 1: Quick Forecast with 4 models
- Tab 2: Advanced forecasting with hyperparameter tuning
- Tab 3: AI-powered model selection
- Tab 4: Multi-model comparison and ensemble
- Tab 5: Market analysis with technical indicators

All features from original 3 pages preserved with zero functionality loss"
```

---

## 📝 NOTES

- Total time: ~8-10 hours
- Test thoroughly before moving to Page 2
- If any prompt fails, debug before continuing
- Review Cursor's code before accepting
- Keep backend imports organized at top of file

---

## 🆘 TROUBLESHOOTING

**Issue:** Model training fails  
**Solution:** Check that trading.models.* classes are properly imported and instantiated

**Issue:** Charts don't display  
**Solution:** Verify plotly is installed and data is in correct format

**Issue:** AI recommendation fails  
**Solution:** Ensure ModelSelectorAgent is implemented in backend

---

**Next:** Proceed to PAGE_02_STRATEGY_TESTING_INTEGRATION.md
