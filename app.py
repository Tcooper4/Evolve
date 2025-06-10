import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
import json

# Add both project root and trading directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'trading'))

# Import our custom modules
from trading.strategies import StrategyManager
from trading.backtesting import BacktestEngine
from trading.risk import RiskManager
from trading.portfolio import PortfolioManager
from trading.analysis import MarketAnalyzer
from trading.optimization import Optimizer
from trading.llm import LLMInterface
from trading.feature_engineering import FeatureEngineer
from trading.evaluation import ModelEvaluator

# Set page config
st.set_page_config(
    page_title="Evolve Clean Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables first
if 'use_api' not in st.session_state:
    st.session_state.use_api = False
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

# Initialize components
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = PortfolioManager()
if 'risk_manager' not in st.session_state:
    st.session_state.risk_manager = RiskManager(pd.Series(dtype=float))
if 'strategy_manager' not in st.session_state:
    st.session_state.strategy_manager = StrategyManager()
if 'backtest_engine' not in st.session_state:
    st.session_state.backtest_engine = BacktestEngine(pd.DataFrame())
if 'market_analyzer' not in st.session_state:
    st.session_state.market_analyzer = MarketAnalyzer()
if 'optimizer' not in st.session_state:
    st.session_state.optimizer = Optimizer(state_dim=10, action_dim=5)
if 'feature_engineer' not in st.session_state:
    st.session_state.feature_engineer = FeatureEngineer()
if 'model_evaluator' not in st.session_state:
    st.session_state.model_evaluator = ModelEvaluator()

# Initialize LLM last, after all other components
if 'llm' not in st.session_state:
    api_key = st.session_state.api_key if st.session_state.use_api else None
    st.session_state.llm = LLMInterface(api_key=api_key)

# Sidebar Navigation
st.sidebar.title("Evolve Clean")

# Add prompt input to sidebar
st.sidebar.subheader("AI Assistant")
user_prompt = st.sidebar.text_area(
    "Ask me anything about trading",
    height=150,
    help="I can help with market analysis, trading strategies, risk assessment, portfolio management, and more."
)

if user_prompt:
    # Process the prompt using LLM
    response = st.session_state.llm.process_prompt(user_prompt)
    
    # Display the response in the sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Response")
    st.sidebar.write(response.get('content', response.get('error', 'Processing...')))
    
    # Automatically configure system based on response
    if 'actions' in response:
        for action in response['actions']:
            if action['type'] == 'update_chart':
                # Update relevant charts
                if 'chart_type' in action:
                    if action['chart_type'] == 'market_analysis':
                        st.session_state.market_analyzer.update_charts(action['data'])
                    elif action['chart_type'] == 'portfolio':
                        st.session_state.portfolio.update_charts(action['data'])
                    elif action['chart_type'] == 'risk':
                        st.session_state.risk_manager.update_charts(action['data'])
            
            elif action['type'] == 'update_metrics':
                # Update relevant metrics
                if 'metrics' in action:
                    if 'portfolio' in action['metrics']:
                        st.session_state.portfolio.update_metrics(action['metrics']['portfolio'])
                    if 'risk' in action['metrics']:
                        st.session_state.risk_manager.update_metrics(action['metrics']['risk'])
                    if 'strategy' in action['metrics']:
                        st.session_state.strategy_manager.update_metrics(action['metrics']['strategy'])
            
            elif action['type'] == 'show_analysis':
                # Show detailed analysis
                if 'analysis_type' in action:
                    if action['analysis_type'] == 'market':
                        st.session_state.market_analyzer.show_analysis(action['data'])
                    elif action['analysis_type'] == 'risk':
                        st.session_state.risk_manager.show_analysis(action['data'])
                    elif action['analysis_type'] == 'portfolio':
                        st.session_state.portfolio.show_analysis(action['data'])
            
            elif action['type'] == 'configure_strategy':
                # Configure trading strategy
                if 'strategy_params' in action:
                    st.session_state.strategy_manager.configure(action['strategy_params'])
            
            elif action['type'] == 'update_portfolio':
                # Update portfolio allocation
                if 'allocation' in action:
                    st.session_state.portfolio.update_allocation(action['allocation'])
            
            elif action['type'] == 'run_backtest':
                # Run backtest with specified parameters
                if 'backtest_params' in action:
                    st.session_state.backtest_engine.run_backtest(action['backtest_params'])
            
            elif action['type'] == 'optimize_portfolio':
                # Optimize portfolio
                if 'optimization_params' in action:
                    st.session_state.optimizer.optimize(action['optimization_params'])
            
            elif action['type'] == 'update_models':
                # Update ML models
                if 'model_params' in action:
                    st.session_state.model_evaluator.update_models(action['model_params'])
            
            elif action['type'] == 'generate_features':
                # Generate new features
                if 'feature_params' in action:
                    st.session_state.feature_engineer.generate_features(action['feature_params'])

# Main navigation
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Trading", "Backtesting", "Risk Management", 
     "Portfolio", "Analysis", "Optimization", "ML Models", "Settings"]
)

# Main content
if page == "Dashboard":
    st.title("Trading Dashboard")
    
    # Portfolio Overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Value", "$1,234,567", "+2.3%")
    with col2:
        st.metric("Daily P/L", "$12,345", "-1.2%")
    with col3:
        st.metric("Open Positions", "15", "+2")
    with col4:
        st.metric("Risk Score", "0.65", "-0.05")
    
    # Performance Chart
    st.subheader("Portfolio Performance")
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    portfolio_values = pd.Series(range(len(dates))) * 1000 + 1000000
    benchmark_values = pd.Series(range(len(dates))) * 800 + 1000000
    df_perf = pd.DataFrame({
        'Date': dates,
        'Portfolio': portfolio_values,
        'Benchmark': benchmark_values
    })
    fig = px.line(df_perf, x='Date', y=['Portfolio', 'Benchmark'],
                  title='Portfolio vs Benchmark')
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent Trades
    st.subheader("Recent Trades")
    trades_data = {
        'Time': pd.date_range(start='2024-01-01', periods=5, freq='H'),
        'Symbol': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'],
        'Type': ['Buy', 'Sell', 'Buy', 'Sell', 'Buy'],
        'Quantity': [100, 50, 75, 25, 150],
        'Price': [180.5, 140.2, 380.0, 150.75, 320.25],
        'P/L': [1200, -500, 800, -300, 1500]
    }
    st.dataframe(pd.DataFrame(trades_data))

elif page == "Trading":
    st.title("Trading")
    
    # Strategy Selection
    st.subheader("Active Strategies")
    strategies = ["Momentum", "Mean Reversion", "ML-Based", "Custom"]
    selected_strategies = st.multiselect("Select Strategies", strategies)
    
    # Trading Parameters
    col1, col2 = st.columns(2)
    with col1:
        st.number_input("Position Size (%)", 1, 100, 5)
        st.number_input("Stop Loss (%)", 1, 20, 5)
    with col2:
        st.number_input("Take Profit (%)", 1, 50, 10)
        st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"])
    
    # Market Scanner
    st.subheader("Market Scanner")
    scanner_options = ["Top Gainers", "Top Losers", "Volume Leaders", "Technical Signals"]
    selected_scanner = st.selectbox("Scanner Type", scanner_options)
    
    # Display scanner results
    scanner_data = {
        'Symbol': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'],
        'Price': [180.5, 140.2, 380.0, 150.75, 320.25],
        'Change %': [2.3, -1.5, 3.2, -0.8, 1.9],
        'Volume': [1000000, 800000, 1200000, 900000, 1100000],
        'Signal': ['Buy', 'Sell', 'Buy', 'Hold', 'Buy']
    }
    st.dataframe(pd.DataFrame(scanner_data))

elif page == "Backtesting":
    st.title("Backtesting")
    
    # Backtest Configuration
    col1, col2 = st.columns(2)
    with col1:
        st.date_input("Start Date", datetime(2023, 1, 1))
        st.selectbox("Strategy", ["Momentum", "Mean Reversion", "ML-Based"])
    with col2:
        st.date_input("End Date", datetime(2024, 1, 1))
        st.multiselect("Assets", ["AAPL", "GOOGL", "MSFT", "AMZN", "META"])
    
    # Run Backtest
    if st.button("Run Backtest"):
        st.subheader("Backtest Results")
        
        # Performance Metrics
        metrics = {
            'Metric': ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate'],
            'Value': ['23.5%', '1.8', '12.3%', '65%']
        }
        st.dataframe(pd.DataFrame(metrics))
        
        # Performance Chart
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='M')
        strategy_values = pd.Series(range(len(dates))) * 100 + 10000
        benchmark_values = pd.Series(range(len(dates))) * 80 + 10000
        df_bt = pd.DataFrame({
            'Date': dates,
            'Strategy': strategy_values,
            'Benchmark': benchmark_values
        })
        fig = px.line(df_bt, x='Date', y=['Strategy', 'Benchmark'],
                      title='Backtest Performance')
        st.plotly_chart(fig, use_container_width=True)

elif page == "Risk Management":
    st.title("Risk Management")
    
    # Risk Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Portfolio VaR", "2.3%", "-0.5%")
    with col2:
        st.metric("Beta", "1.2", "+0.1")
    with col3:
        st.metric("Correlation", "0.75", "-0.05")
    
    # Risk Analysis
    st.subheader("Risk Analysis")
    risk_factors = ["Market Risk", "Liquidity Risk", "Volatility Risk", "Credit Risk"]
    risk_scores = [0.65, 0.45, 0.75, 0.35]
    
    fig = go.Figure(data=[
        go.Bar(x=risk_factors, y=risk_scores)
    ])
    fig.update_layout(title="Risk Factor Analysis")
    st.plotly_chart(fig, use_container_width=True)
    
    # Position Limits
    st.subheader("Position Limits")
    limits_data = {
        'Asset': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'],
        'Current %': [15, 12, 10, 8, 7],
        'Max %': [20, 15, 15, 10, 10],
        'Risk Score': [0.65, 0.75, 0.55, 0.85, 0.70]
    }
    st.dataframe(pd.DataFrame(limits_data))

elif page == "Portfolio":
    st.title("Portfolio Management")
    
    # Portfolio Composition
    st.subheader("Portfolio Composition")
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'Others']
    weights = [25, 20, 15, 15, 15, 10]
    
    fig = px.pie(values=weights, names=assets, title="Portfolio Allocation")
    st.plotly_chart(fig, use_container_width=True)
    
    # Holdings
    st.subheader("Current Holdings")
    holdings_data = {
        'Asset': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'],
        'Quantity': [100, 50, 75, 25, 150],
        'Avg Price': [150.5, 2800.2, 280.0, 3300.75, 180.25],
        'Current Price': [180.5, 2900.2, 300.0, 3400.75, 200.25],
        'P/L': [3000, 5000, 1500, 2500, 3000],
        'P/L %': [20, 3.5, 7.1, 3.0, 11.1]
    }
    st.dataframe(pd.DataFrame(holdings_data))

elif page == "Analysis":
    st.title("Market Analysis")
    
    # Technical Analysis
    st.subheader("Technical Analysis")
    ta_options = ["Moving Averages", "RSI", "MACD", "Bollinger Bands"]
    selected_ta = st.multiselect("Select Indicators", ta_options)
    
    # Market Sentiment
    st.subheader("Market Sentiment")
    sentiment_data = {
        'Source': ['News', 'Social Media', 'Analyst Ratings', 'Options Flow'],
        'Bullish': [65, 55, 70, 60],
        'Bearish': [35, 45, 30, 40]
    }
    fig = px.bar(pd.DataFrame(sentiment_data), x='Source', y=['Bullish', 'Bearish'],
                 title="Market Sentiment Analysis")
    st.plotly_chart(fig, use_container_width=True)
    
    # Economic Calendar
    st.subheader("Economic Calendar")
    calendar_data = {
        'Date': pd.date_range(start='2024-01-01', periods=5),
        'Event': ['Fed Rate Decision', 'GDP Release', 'CPI Data', 'Jobs Report', 'Retail Sales'],
        'Impact': ['High', 'High', 'Medium', 'High', 'Medium'],
        'Forecast': ['5.25%', '2.1%', '3.1%', '200K', '0.5%']
    }
    st.dataframe(pd.DataFrame(calendar_data))

elif page == "Optimization":
    st.title("Strategy Optimization")
    
    # Optimization Parameters
    st.subheader("Optimization Parameters")
    col1, col2 = st.columns(2)
    with col1:
        st.selectbox("Optimization Method", ["Genetic Algorithm", "Bayesian", "Grid Search"])
        st.number_input("Population Size", 10, 1000, 100)
    with col2:
        st.selectbox("Objective", ["Sharpe Ratio", "Returns", "Risk-Adjusted Returns"])
        st.number_input("Generations", 10, 100, 50)
    
    # Run Optimization
    if st.button("Run Optimization"):
        st.subheader("Optimization Results")
        
        # Parameter Space
        params_data = {
            'Parameter': ['Lookback Period', 'Entry Threshold', 'Exit Threshold', 'Position Size'],
            'Original': [20, 0.02, 0.01, 0.1],
            'Optimized': [25, 0.015, 0.008, 0.12]
        }
        st.dataframe(pd.DataFrame(params_data))
        
        # Performance Comparison
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='M')
        original_values = pd.Series(range(len(dates))) * 100 + 10000
        optimized_values = pd.Series(range(len(dates))) * 120 + 10000
        df_opt = pd.DataFrame({
            'Date': dates,
            'Original': original_values,
            'Optimized': optimized_values
        })
        fig = px.line(df_opt, x='Date', y=['Original', 'Optimized'],
                      title='Optimization Results')
        st.plotly_chart(fig, use_container_width=True)

elif page == "ML Models":
    st.title("Machine Learning Models")
    
    # Model Selection
    st.subheader("Model Management")
    model_options = ["LSTM", "Random Forest", "XGBoost", "Transformer"]
    selected_model = st.selectbox("Select Model", model_options)
    
    # Model Performance
    st.subheader("Model Performance")
    performance_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value': [0.85, 0.82, 0.88, 0.85]
    }
    st.dataframe(pd.DataFrame(performance_data))
    
    # Feature Importance
    st.subheader("Feature Importance")
    features = ['Price', 'Volume', 'RSI', 'MACD', 'Sentiment']
    importance = [0.3, 0.2, 0.15, 0.2, 0.15]
    
    fig = px.bar(x=features, y=importance, title="Feature Importance")
    st.plotly_chart(fig, use_container_width=True)
    
    # Model Training
    st.subheader("Model Training")
    if st.button("Train Model"):
        st.progress(100)
        st.success("Model training completed!")

elif page == "Settings":
    st.title("Settings")
    
    # API Configuration
    st.header("API Configuration")
    use_api = st.toggle("Enable API Features", value=st.session_state.use_api)
    if use_api:
        api_key = st.text_input("OpenAI API Key", value=st.session_state.api_key, type="password")
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
            st.session_state.llm = LLMInterface(api_key=api_key)
    st.session_state.use_api = use_api
    
    # Display current LLM status
    st.header("LLM Status")
    metrics = st.session_state.llm.get_llm_metrics()
    st.json(metrics)

    # Export/Import Configuration
    st.subheader("Export/Import Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Configuration"):
            config = {
                "use_api": st.session_state.use_api,
                "api_key": st.session_state.api_key
            }
            st.download_button(
                "Download Configuration",
                data=json.dumps(config, indent=2),
                file_name="trading_config.json",
                mime="application/json"
            )
    
    with col2:
        uploaded_file = st.file_uploader("Import Configuration", type=["json"])
        if uploaded_file is not None:
            try:
                config = json.load(uploaded_file)
                st.session_state.use_api = config.get("use_api", False)
                st.session_state.api_key = config.get("api_key", "")
                st.session_state.llm = LLMInterface(api_key=st.session_state.api_key)
                st.success("Configuration imported successfully!")
            except Exception as e:
                st.error(f"Error importing configuration: {str(e)}")

    # General Settings
    st.subheader("General Settings")
    st.selectbox("Theme", ["Light", "Dark", "System"])
    st.number_input("Refresh Rate (seconds)", 1, 60, 5)
    
    # Notification Settings
    st.subheader("Notifications")
    st.checkbox("Enable Email Notifications")
    st.checkbox("Enable SMS Notifications")
    st.checkbox("Enable Desktop Notifications")
    
    # Risk Settings
    st.subheader("Risk Parameters")
    st.number_input("Max Position Size (%)", 1, 100, 10)
    st.number_input("Max Drawdown (%)", 1, 50, 20)
    st.number_input("Daily Loss Limit (%)", 1, 20, 5) 