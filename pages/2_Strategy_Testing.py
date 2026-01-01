"""
Strategy Development & Testing Page

Merges functionality from:
- 2_Strategy_Backtest.py
- Strategy_Lab.py
- Strategy_Combo_Creator.py

Features:
- Quick backtest with pre-built strategies
- Visual strategy builder
- Code editor for advanced users
- Strategy ensemble creation
- Multi-strategy comparison
- Walk-forward and Monte Carlo analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Backend imports
from trading.data.data_loader import DataLoader, DataLoadRequest
from trading.strategies.bollinger_strategy import BollingerStrategy, BollingerConfig
from trading.strategies.macd_strategy import MACDStrategy, MACDConfig
from trading.strategies.rsi_strategy import RSIStrategy
from trading.strategies.sma_strategy import SMAStrategy, SMAConfig
from trading.strategies.custom_strategy_handler import CustomStrategyHandler
from trading.strategies.ensemble import WeightedEnsembleStrategy, EnsembleConfig
from trading.strategies.ensemble import create_ensemble_strategy
from trading.backtesting.monte_carlo import MonteCarloSimulator, MonteCarloConfig
from trading.backtesting.evaluator import ModelEvaluator

st.set_page_config(
    page_title="Strategy Development & Testing",
    page_icon="🔄",
    layout="wide"
)

# Initialize session state variables
if 'loaded_data' not in st.session_state:
    st.session_state.loaded_data = None
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'custom_strategies' not in st.session_state:
    st.session_state.custom_strategies = {}
if 'selected_strategy' not in st.session_state:
    st.session_state.selected_strategy = None
if 'strategy_combos' not in st.session_state:
    st.session_state.strategy_combos = {}

# Main page title
st.title("🔄 Strategy Development & Testing")
st.markdown("Comprehensive strategy development, testing, and optimization tools")

# Create tabbed interface
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🚀 Quick Backtest",
    "🔧 Strategy Builder",
    "💻 Advanced Editor",
    "🎯 Strategy Combos",
    "📊 Strategy Comparison",
    "🔬 Advanced Analysis"
])

# Tab 1: Quick Backtest
with tab1:
    st.header("Quick Backtest")
    st.markdown("Backtest pre-built strategies with configurable parameters")
    
    # Create strategy registry
    STRATEGY_REGISTRY = {
        "Bollinger Bands": {
            "class": BollingerStrategy,
            "config_class": BollingerConfig,
            "description": "Trades based on Bollinger Band breakouts",
            "params": {
                "window": {"type": "slider", "min": 10, "max": 50, "default": 20},
                "num_std": {"type": "slider", "min": 1.0, "max": 3.0, "default": 2.0, "step": 0.1}
            }
        },
        "MACD": {
            "class": MACDStrategy,
            "config_class": MACDConfig,
            "description": "Moving Average Convergence Divergence strategy",
            "params": {
                "fast_period": {"type": "slider", "min": 5, "max": 20, "default": 12},
                "slow_period": {"type": "slider", "min": 20, "max": 50, "default": 26},
                "signal_period": {"type": "slider", "min": 5, "max": 15, "default": 9}
            }
        },
        "RSI": {
            "class": RSIStrategy,
            "config_class": None,
            "description": "Relative Strength Index mean reversion",
            "params": {
                "rsi_period": {"type": "slider", "min": 5, "max": 30, "default": 14},
                "oversold_threshold": {"type": "slider", "min": 10, "max": 40, "default": 30},
                "overbought_threshold": {"type": "slider", "min": 60, "max": 90, "default": 70}
            }
        },
        "SMA Crossover": {
            "class": SMAStrategy,
            "config_class": SMAConfig,
            "description": "Simple Moving Average crossover strategy",
            "params": {
                "short_window": {"type": "slider", "min": 5, "max": 50, "default": 20},
                "long_window": {"type": "slider", "min": 50, "max": 200, "default": 50}
            }
        }
    }
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Data & Strategy")
        
        # Data loading
        with st.form("backtest_data_form"):
            symbol = st.text_input("Symbol", value="AAPL").upper()
            
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=365)
                )
            with col_date2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now()
                )
            
            load_data = st.form_submit_button("📊 Load Data", use_container_width=True)
        
        if load_data:
            try:
                with st.spinner(f"Loading {symbol}..."):
                    loader = DataLoader()
                    request = DataLoadRequest(
                        ticker=symbol,
                        start_date=start_date.strftime("%Y-%m-%d"),
                        end_date=end_date.strftime("%Y-%m-%d"),
                        interval="1d"
                    )
                    response = loader.load_market_data(request)
                    
                    if not response.success:
                        st.error(f"Error loading data: {response.message}")
                    elif response.data is None or len(response.data) < 30:
                        st.error(f"Insufficient data for {symbol}. Try different dates or symbol.")
                    else:
                        # Convert to standard format
                        data = response.data.copy()
                        data.columns = [col.lower() for col in data.columns]
                        
                        # Ensure we have 'close' column
                        if 'close' not in data.columns:
                            for col in ['Close', 'CLOSE', 'price', 'Price']:
                                if col in data.columns:
                                    data['close'] = data[col]
                                    break
                        
                        if 'close' in data.columns:
                            # Ensure required columns exist
                            if 'volume' not in data.columns:
                                data['volume'] = 1000000
                            if 'open' not in data.columns:
                                data['open'] = data['close']
                            if 'high' not in data.columns:
                                data['high'] = data['close']
                            if 'low' not in data.columns:
                                data['low'] = data['close']
                            
                            st.session_state.loaded_data = data
                            st.session_state.backtest_symbol = symbol
                            st.success(f"✅ Loaded {len(data)} days")
                        else:
                            st.error("Could not find close price column in data")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        
        # Strategy selection
        if st.session_state.loaded_data is not None:
            st.markdown("---")
            st.subheader("🎯 Strategy Selection")
            
            strategy_name = st.selectbox(
                "Select Strategy",
                list(STRATEGY_REGISTRY.keys())
            )
            
            strategy_info = STRATEGY_REGISTRY[strategy_name]
            st.info(strategy_info["description"])
            
            # Dynamic parameter inputs
            st.markdown("**Parameters:**")
            params = {}
            for param_name, param_config in strategy_info["params"].items():
                if param_config["type"] == "slider":
                    params[param_name] = st.slider(
                        param_name.replace("_", " ").title(),
                        min_value=param_config["min"],
                        max_value=param_config["max"],
                        value=param_config["default"],
                        step=param_config.get("step", 1)
                    )
            
            # Backtest settings
            st.markdown("---")
            st.markdown("**Backtest Settings:**")
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000,
                value=10000,
                step=1000
            )
            commission = st.number_input(
                "Commission (%)",
                min_value=0.0,
                value=0.1,
                step=0.01
            )
            
            run_backtest = st.button(
                "🚀 Run Backtest",
                type="primary",
                use_container_width=True
            )
    
    with col2:
        st.subheader("📈 Results")
        
        if run_backtest and st.session_state.loaded_data is not None:
            try:
                with st.spinner("Running backtest..."):
                    # Get data
                    data = st.session_state.loaded_data.copy()
                    
                    # Initialize strategy
                    strategy_class = strategy_info["class"]
                    config_class = strategy_info.get("config_class")
                    
                    if config_class:
                        # Use config class
                        config = config_class(**params)
                        strategy = strategy_class(config)
                    else:
                        # Direct initialization (for RSI)
                        strategy = strategy_class(**params)
                    
                    # Generate signals
                    signals_df = strategy.generate_signals(data)
                    
                    # Calculate returns
                    if 'close' not in data.columns:
                        st.error("Data missing 'close' column")
                    else:
                        data['returns'] = data['close'].pct_change()
                        
                        # Get signal column (could be 'signal' or different)
                        signal_col = 'signal' if 'signal' in signals_df.columns else signals_df.columns[0]
                        
                        # Calculate strategy returns (shift by 1 to avoid lookahead)
                        data['strategy_returns'] = signals_df[signal_col].shift(1) * data['returns']
                        data['cumulative_returns'] = (1 + data['returns']).cumprod()
                        data['strategy_cumulative_returns'] = (1 + data['strategy_returns']).cumprod()
                        
                        # Calculate equity curve
                        initial_value = initial_capital
                        equity_curve = initial_value * (1 + data['strategy_returns']).cumprod()
                        equity_curve = equity_curve.fillna(initial_value)
                        
                        # Calculate metrics
                        total_return = (equity_curve.iloc[-1] / initial_value - 1) * 100
                        
                        # Sharpe ratio
                        returns_series = data['strategy_returns'].dropna()
                        if len(returns_series) > 0 and returns_series.std() > 0:
                            sharpe = (returns_series.mean() / returns_series.std()) * np.sqrt(252)
                        else:
                            sharpe = 0.0
                        
                        # Max drawdown
                        cumulative = (1 + data['strategy_returns']).cumprod()
                        running_max = cumulative.cummax()
                        drawdown = (cumulative / running_max - 1) * 100
                        max_dd = drawdown.min()
                        
                        # Win rate
                        trades = returns_series[returns_series != 0]
                        if len(trades) > 0:
                            win_rate = (trades > 0).sum() / len(trades) * 100
                        else:
                            win_rate = 0.0
                        
                        # Store results
                        results = {
                            'total_return': total_return / 100,
                            'sharpe_ratio': sharpe,
                            'max_drawdown': max_dd / 100,
                            'win_rate': win_rate / 100,
                            'equity_curve': pd.DataFrame({
                                'equity': equity_curve.values
                            }, index=data.index),
                            'trades': signals_df[signals_df[signal_col] != 0].to_dict('records') if signal_col in signals_df.columns else []
                        }
                        
                        st.session_state.backtest_results = results
                        st.session_state.backtest_strategy = strategy_name
                
                st.success("✅ Backtest complete!")
                
            except Exception as e:
                st.error(f"Backtest failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        
        # Display results
        if st.session_state.get('backtest_results'):
            results = st.session_state.backtest_results
            
            # Metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            with col_m1:
                total_return = results.get('total_return', 0) * 100
                st.metric("Total Return", f"{total_return:.2f}%")
            
            with col_m2:
                sharpe = results.get('sharpe_ratio', 0)
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            
            with col_m3:
                max_dd = results.get('max_drawdown', 0) * 100
                st.metric("Max Drawdown", f"{max_dd:.2f}%")
            
            with col_m4:
                win_rate = results.get('win_rate', 0) * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")
            
            # Equity curve
            if 'equity_curve' in results and not results['equity_curve'].empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=results['equity_curve'].index,
                    y=results['equity_curve']['equity'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='green', width=2)
                ))
                
                fig.update_layout(
                    title="Equity Curve",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Trade list
            if 'trades' in results and len(results['trades']) > 0:
                st.markdown("**Trade History:**")
                trades_df = pd.DataFrame(results['trades'])
                st.dataframe(trades_df, use_container_width=True)
                
                # Download trades
                csv = trades_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Trades",
                    data=csv,
                    file_name=f"{st.session_state.get('backtest_symbol', 'symbol')}_{st.session_state.get('backtest_strategy', 'strategy')}_trades.csv",
                    mime="text/csv"
                )

with tab2:
    st.header("Strategy Builder")
    st.markdown("Create custom trading strategies without coding using our visual builder")
    
    # Initialize custom strategy handler
    if 'strategy_handler' not in st.session_state:
        st.session_state.strategy_handler = CustomStrategyHandler()
    
    handler = st.session_state.strategy_handler
    
    # Two column layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🔧 Strategy Configuration")
        
        # Strategy metadata
        with st.expander("📝 Strategy Details", expanded=True):
            strategy_name = st.text_input(
                "Strategy Name",
                placeholder="My Custom Strategy",
                help="Give your strategy a unique name"
            )
            
            strategy_description = st.text_area(
                "Description",
                placeholder="Describe what your strategy does...",
                height=100
            )
        
        # Entry Conditions
        with st.expander("📈 Entry Conditions", expanded=True):
            st.markdown("**When should we ENTER a trade?**")
            
            entry_logic = st.radio(
                "Entry Logic",
                ["All conditions must be met (AND)", "Any condition can trigger (OR)"],
                key="entry_logic"
            )
            
            num_entry_conditions = st.number_input(
                "Number of entry conditions",
                min_value=1,
                max_value=5,
                value=1,
                key="num_entry"
            )
            
            entry_conditions = []
            for i in range(int(num_entry_conditions)):
                st.markdown(f"**Condition {i+1}:**")
                
                col_ind, col_op, col_val = st.columns([2, 1, 1])
                
                with col_ind:
                    indicator = st.selectbox(
                        "Indicator",
                        ["Price", "SMA", "EMA", "RSI", "MACD", "Bollinger Bands", "Volume"],
                        key=f"entry_ind_{i}"
                    )
                
                with col_op:
                    operator = st.selectbox(
                        "Operator",
                        [">", "<", ">=", "<=", "crosses above", "crosses below"],
                        key=f"entry_op_{i}"
                    )
                
                with col_val:
                    if indicator in ["SMA", "EMA"]:
                        value = st.selectbox(
                            "Value",
                            ["Price", "20", "50", "200"],
                            key=f"entry_val_{i}"
                        )
                    elif indicator == "RSI":
                        value = st.number_input(
                            "Value",
                            min_value=0,
                            max_value=100,
                            value=30,
                            key=f"entry_val_{i}"
                        )
                    elif indicator == "Price":
                        value = st.text_input(
                            "Value",
                            value="SMA(20)",
                            key=f"entry_val_{i}"
                        )
                    else:
                        value = st.text_input(
                            "Value",
                            value="0",
                            key=f"entry_val_{i}"
                        )
                
                entry_conditions.append({
                    "indicator": indicator,
                    "operator": operator,
                    "value": value
                })
                
                st.markdown("---")
        
        # Exit Conditions
        with st.expander("📉 Exit Conditions", expanded=True):
            st.markdown("**When should we EXIT a trade?**")
            
            exit_logic = st.radio(
                "Exit Logic",
                ["All conditions must be met (AND)", "Any condition can trigger (OR)"],
                key="exit_logic"
            )
            
            num_exit_conditions = st.number_input(
                "Number of exit conditions",
                min_value=1,
                max_value=5,
                value=1,
                key="num_exit"
            )
            
            exit_conditions = []
            for i in range(int(num_exit_conditions)):
                st.markdown(f"**Condition {i+1}:**")
                
                col_ind, col_op, col_val = st.columns([2, 1, 1])
                
                with col_ind:
                    indicator = st.selectbox(
                        "Indicator",
                        ["Price", "SMA", "EMA", "RSI", "MACD", "Profit %", "Loss %", "Time"],
                        key=f"exit_ind_{i}"
                    )
                
                with col_op:
                    if indicator in ["Profit %", "Loss %"]:
                        operator = st.selectbox(
                            "Operator",
                            [">=", "<="],
                            key=f"exit_op_{i}"
                        )
                    else:
                        operator = st.selectbox(
                            "Operator",
                            [">", "<", ">=", "<=", "crosses above", "crosses below"],
                            key=f"exit_op_{i}"
                        )
                
                with col_val:
                    if indicator == "Profit %":
                        value = st.number_input(
                            "Target %",
                            min_value=0.0,
                            value=5.0,
                            step=0.5,
                            key=f"exit_val_{i}"
                        )
                    elif indicator == "Loss %":
                        value = st.number_input(
                            "Stop Loss %",
                            min_value=0.0,
                            value=2.0,
                            step=0.5,
                            key=f"exit_val_{i}"
                        )
                    elif indicator == "Time":
                        value = st.number_input(
                            "Days",
                            min_value=1,
                            value=5,
                            key=f"exit_val_{i}"
                        )
                    else:
                        value = st.text_input(
                            "Value",
                            value="0",
                            key=f"exit_val_{i}"
                        )
                
                exit_conditions.append({
                    "indicator": indicator,
                    "operator": operator,
                    "value": value
                })
                
                st.markdown("---")
        
        # Position Sizing
        with st.expander("💰 Position Sizing", expanded=False):
            sizing_method = st.selectbox(
                "Position Sizing Method",
                [
                    "Fixed Dollar Amount",
                    "Percentage of Portfolio",
                    "Volatility-Based (ATR)",
                    "Kelly Criterion"
                ]
            )
            
            if sizing_method == "Fixed Dollar Amount":
                position_size = st.number_input(
                    "Position Size ($)",
                    min_value=100,
                    value=1000,
                    step=100
                )
            elif sizing_method == "Percentage of Portfolio":
                position_size = st.slider(
                    "Position Size (%)",
                    min_value=1,
                    max_value=100,
                    value=10
                )
            elif sizing_method == "Volatility-Based (ATR)":
                atr_multiplier = st.slider(
                    "ATR Multiplier",
                    min_value=0.5,
                    max_value=3.0,
                    value=1.5,
                    step=0.1
                )
                position_size = atr_multiplier
            else:  # Kelly Criterion
                kelly_fraction = st.slider(
                    "Kelly Fraction",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.25,
                    step=0.05
                )
                position_size = kelly_fraction
        
        # Risk Management
        with st.expander("⚠️ Risk Management", expanded=False):
            use_stop_loss = st.checkbox("Use Stop Loss", value=True)
            stop_loss_pct = 2.0
            if use_stop_loss:
                stop_loss_pct = st.number_input(
                    "Stop Loss %",
                    min_value=0.1,
                    max_value=50.0,
                    value=2.0,
                    step=0.1
                )
            
            use_take_profit = st.checkbox("Use Take Profit", value=True)
            take_profit_pct = 5.0
            if use_take_profit:
                take_profit_pct = st.number_input(
                    "Take Profit %",
                    min_value=0.1,
                    max_value=100.0,
                    value=5.0,
                    step=0.5
                )
            
            max_positions = st.number_input(
                "Max Concurrent Positions",
                min_value=1,
                max_value=20,
                value=5
            )
            
            max_daily_loss = st.number_input(
                "Max Daily Loss ($)",
                min_value=100,
                value=1000,
                step=100
            )
    
    with col2:
        st.subheader("📊 Strategy Preview")
        
        # Generate strategy code preview
        if strategy_name:
            st.markdown("**Generated Strategy Logic:**")
            
            # Build strategy description
            preview_text = f"### {strategy_name}\n\n"
            if strategy_description:
                preview_text += f"{strategy_description}\n\n"
            
            # Entry rules
            preview_text += "**Entry Rules:**\n"
            logic_word = "AND" if "All" in entry_logic else "OR"
            for i, cond in enumerate(entry_conditions):
                connector = f" {logic_word} " if i > 0 else ""
                preview_text += f"{connector}{cond['indicator']} {cond['operator']} {cond['value']}\n"
            
            # Exit rules
            preview_text += "\n**Exit Rules:**\n"
            logic_word = "AND" if "All" in exit_logic else "OR"
            for i, cond in enumerate(exit_conditions):
                connector = f" {logic_word} " if i > 0 else ""
                preview_text += f"{connector}{cond['indicator']} {cond['operator']} {cond['value']}\n"
            
            # Position sizing
            preview_text += f"\n**Position Sizing:** {sizing_method}\n"
            
            # Risk management
            preview_text += "\n**Risk Management:**\n"
            if use_stop_loss:
                preview_text += f"- Stop Loss: {stop_loss_pct}%\n"
            if use_take_profit:
                preview_text += f"- Take Profit: {take_profit_pct}%\n"
            preview_text += f"- Max Positions: {max_positions}\n"
            preview_text += f"- Max Daily Loss: ${max_daily_loss}\n"
            
            st.markdown(preview_text)
            
            # Validation
            st.markdown("---")
            st.markdown("**Strategy Validation:**")
            
            validation_issues = []
            
            if not strategy_name.strip():
                validation_issues.append("❌ Strategy name is required")
            
            if len(entry_conditions) == 0:
                validation_issues.append("❌ At least one entry condition is required")
            
            if len(exit_conditions) == 0:
                validation_issues.append("❌ At least one exit condition is required")
            
            if validation_issues:
                for issue in validation_issues:
                    st.error(issue)
            else:
                st.success("✅ Strategy is valid and ready to save!")
                
                # Save strategy button
                col_save, col_test = st.columns(2)
                
                with col_save:
                    if st.button("💾 Save Strategy", type="primary", use_container_width=True):
                        try:
                            # Build strategy configuration
                            strategy_config = {
                                "name": strategy_name,
                                "description": strategy_description,
                                "entry_conditions": entry_conditions,
                                "entry_logic": "AND" if "All" in entry_logic else "OR",
                                "exit_conditions": exit_conditions,
                                "exit_logic": "AND" if "All" in exit_logic else "OR",
                                "position_sizing": {
                                    "method": sizing_method,
                                    "size": position_size
                                },
                                "risk_management": {
                                    "stop_loss": stop_loss_pct if use_stop_loss else None,
                                    "take_profit": take_profit_pct if use_take_profit else None,
                                    "max_positions": max_positions,
                                    "max_daily_loss": max_daily_loss
                                },
                                "created_at": datetime.now().isoformat(),
                                "updated_at": datetime.now().isoformat(),
                                "is_active": True
                            }
                            
                            # Generate Python code for the strategy
                            strategy_code = generate_strategy_code(strategy_config)
                            
                            # Save using handler
                            result = handler.create_strategy_from_code(
                                name=strategy_name,
                                code=strategy_code,
                                parameters=strategy_config
                            )
                            
                            if result.get("success"):
                                st.success(f"✅ {result.get('message')}")
                                
                                # Store in session state
                                if 'custom_strategies' not in st.session_state:
                                    st.session_state.custom_strategies = {}
                                st.session_state.custom_strategies[strategy_name] = strategy_config
                                
                                # Reload strategies
                                handler.load_strategies()
                                st.rerun()
                            else:
                                st.error(f"❌ {result.get('error')}")
                        
                        except Exception as e:
                            st.error(f"Error saving strategy: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                
                with col_test:
                    if st.button("🧪 Test Strategy", use_container_width=True):
                        st.info("Testing functionality will be available after saving the strategy. You can test it in Tab 1 (Quick Backtest).")
        
        # Load existing strategies
        st.markdown("---")
        st.markdown("**Saved Custom Strategies:**")
        
        if handler.strategies:
            saved_strategies = list(handler.strategies.keys())
            
            selected_saved = st.selectbox(
                "Load existing strategy",
                [""] + saved_strategies,
                key="load_saved_strategy"
            )
            
            if selected_saved:
                col_load, col_del = st.columns(2)
                
                with col_load:
                    if st.button("📂 Load Strategy", use_container_width=True):
                        st.info(f"Loading {selected_saved}...")
                        # In a real implementation, you'd populate the form fields
                        # with the loaded strategy's parameters
                
                with col_del:
                    if st.button("🗑️ Delete Strategy", use_container_width=True):
                        try:
                            result = handler.delete_strategy(selected_saved)
                            if result.get("success"):
                                st.success(f"✅ {result.get('message')}")
                                st.rerun()
                            else:
                                st.error(f"❌ {result.get('error')}")
                        except Exception as e:
                            st.error(f"Error deleting strategy: {str(e)}")
        else:
            st.info("No saved strategies yet. Create your first one above!")

with tab3:
    st.header("Advanced Editor")
    st.markdown("Write custom strategy Python code with full control and testing capabilities")
    
    # Initialize handler if needed
    if 'strategy_handler' not in st.session_state:
        st.session_state.strategy_handler = CustomStrategyHandler()
    
    handler = st.session_state.strategy_handler
    
    # Strategy templates
    STRATEGY_TEMPLATES = {
        "Empty Template": '''"""
Custom Strategy Template

Write your strategy description here.
"""

import pandas as pd
import numpy as np
from typing import Optional

def generate_signals(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Generate trading signals.
    
    Args:
        data: DataFrame with OHLCV data (columns: open, high, low, close, volume)
        **kwargs: Additional parameters
    
    Returns:
        DataFrame with 'signal' column:
        - 1 for buy signals
        - -1 for sell signals
        - 0 for no signal
    """
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0
    
    # Your strategy logic here
    # Example: Buy when price crosses above SMA(20)
    # sma_20 = data['close'].rolling(20).mean()
    # signals.loc[data['close'] > sma_20, 'signal'] = 1
    
    return signals
''',
        "RSI Mean Reversion": '''"""
RSI Mean Reversion Strategy

Buys when RSI is oversold (< 30) and sells when overbought (> 70).
"""

import pandas as pd
import numpy as np

def generate_signals(data: pd.DataFrame, rsi_period: int = 14, oversold: float = 30, overbought: float = 70, **kwargs) -> pd.DataFrame:
    """Generate RSI-based signals."""
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0
    
    # Calculate RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Generate signals
    signals.loc[rsi < oversold, 'signal'] = 1  # Buy when oversold
    signals.loc[rsi > overbought, 'signal'] = -1  # Sell when overbought
    
    return signals
''',
        "Moving Average Crossover": '''"""
Moving Average Crossover Strategy

Buys when short MA crosses above long MA, sells when it crosses below.
"""

import pandas as pd
import numpy as np

def generate_signals(data: pd.DataFrame, short_period: int = 20, long_period: int = 50, **kwargs) -> pd.DataFrame:
    """Generate MA crossover signals."""
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0
    
    # Calculate moving averages
    sma_short = data['close'].rolling(short_period).mean()
    sma_long = data['close'].rolling(long_period).mean()
    
    # Generate signals on crossover
    signals.loc[(sma_short > sma_long) & (sma_short.shift(1) <= sma_long.shift(1)), 'signal'] = 1  # Golden cross
    signals.loc[(sma_short < sma_long) & (sma_short.shift(1) >= sma_long.shift(1)), 'signal'] = -1  # Death cross
    
    return signals
''',
        "Bollinger Bands": '''"""
Bollinger Bands Strategy

Buys when price touches lower band, sells when it touches upper band.
"""

import pandas as pd
import numpy as np

def generate_signals(data: pd.DataFrame, period: int = 20, num_std: float = 2.0, **kwargs) -> pd.DataFrame:
    """Generate Bollinger Bands signals."""
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0
    
    # Calculate Bollinger Bands
    sma = data['close'].rolling(period).mean()
    std = data['close'].rolling(period).std()
    upper_band = sma + (num_std * std)
    lower_band = sma - (num_std * std)
    
    # Generate signals
    signals.loc[data['close'] <= lower_band, 'signal'] = 1  # Buy at lower band
    signals.loc[data['close'] >= upper_band, 'signal'] = -1  # Sell at upper band
    
    return signals
''',
        "MACD Strategy": '''"""
MACD Strategy

Uses MACD line and signal line crossovers to generate buy/sell signals.
"""

import pandas as pd
import numpy as np

def generate_signals(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, **kwargs) -> pd.DataFrame:
    """Generate MACD signals."""
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0
    
    # Calculate MACD
    ema_fast = data['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    # Generate signals on crossover
    signals.loc[(macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1)), 'signal'] = 1
    signals.loc[(macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1)), 'signal'] = -1
    
    return signals
'''
    }
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Code Editor")
        
        # Template selection
        selected_template = st.selectbox(
            "Load Template",
            ["None"] + list(STRATEGY_TEMPLATES.keys()),
            help="Select a template to start with or choose None to write from scratch"
        )
        
        if selected_template != "None":
            if st.button("📋 Load Template", use_container_width=True):
                st.session_state.editor_code = STRATEGY_TEMPLATES[selected_template]
                st.rerun()
        
        # Strategy name
        editor_strategy_name = st.text_input(
            "Strategy Name",
            placeholder="MyCustomStrategy",
            key="editor_strategy_name",
            help="Name for your custom strategy"
        )
        
        # Code editor
        default_code = st.session_state.get('editor_code', STRATEGY_TEMPLATES["Empty Template"])
        
        strategy_code = st.text_area(
            "Strategy Code",
            value=default_code,
            height=500,
            key="editor_code_area",
            help="Write your strategy code here. Must include a generate_signals(data, **kwargs) function."
        )
        
        # Editor actions
        col_save, col_validate, col_test = st.columns(3)
        
        with col_save:
            save_code = st.button("💾 Save Strategy", type="primary", use_container_width=True)
        
        with col_validate:
            validate_code = st.button("✓ Validate", use_container_width=True)
        
        with col_test:
            test_code = st.button("🧪 Test", use_container_width=True)
    
    with col2:
        st.subheader("🔍 Validation & Testing")
        
        # Validation results
        if validate_code or save_code:
            validation_result = validate_strategy_code(strategy_code, editor_strategy_name)
            
            if validation_result["valid"]:
                st.success("✅ Code is valid!")
                if validation_result.get("warnings"):
                    for warning in validation_result["warnings"]:
                        st.warning(f"⚠️ {warning}")
            else:
                st.error("❌ Code validation failed:")
                st.code(validation_result["error"], language="python")
        
        # Save strategy
        if save_code:
            if not editor_strategy_name.strip():
                st.error("Please enter a strategy name")
            elif not strategy_code.strip():
                st.error("Please enter strategy code")
            else:
                try:
                    result = handler.create_strategy_from_code(
                        name=editor_strategy_name,
                        code=strategy_code,
                        parameters={}
                    )
                    
                    if result.get("success"):
                        st.success(f"✅ {result.get('message')}")
                        handler.load_strategies()
                    else:
                        st.error(f"❌ {result.get('error')}")
                except Exception as e:
                    st.error(f"Error saving strategy: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Test strategy
        if test_code:
            if st.session_state.loaded_data is None:
                st.warning("⚠️ Please load data first in Tab 1 (Quick Backtest)")
            else:
                try:
                    with st.spinner("Testing strategy..."):
                        # Create a temporary module to test the code
                        import importlib.util
                        import sys
                        
                        # Validate code first
                        validation_result = validate_strategy_code(strategy_code, "test")
                        if not validation_result["valid"]:
                            st.error("Code validation failed. Please fix errors before testing.")
                            st.code(validation_result["error"], language="python")
                        else:
                            # Execute code in a safe namespace
                            test_namespace = {
                                'pd': pd,
                                'np': np,
                                'DataFrame': pd.DataFrame,
                                'Series': pd.Series
                            }
                            
                            exec(strategy_code, test_namespace)
                            
                            # Test generate_signals function
                            if 'generate_signals' in test_namespace:
                                test_data = st.session_state.loaded_data.copy()
                                test_data.columns = [col.lower() for col in test_data.columns]
                                
                                signals = test_namespace['generate_signals'](test_data)
                                
                                st.success("✅ Strategy executed successfully!")
                                
                                # Display results
                                st.markdown("**Test Results:**")
                                
                                col_r1, col_r2, col_r3 = st.columns(3)
                                with col_r1:
                                    buy_signals = (signals['signal'] == 1).sum()
                                    st.metric("Buy Signals", buy_signals)
                                with col_r2:
                                    sell_signals = (signals['signal'] == -1).sum()
                                    st.metric("Sell Signals", sell_signals)
                                with col_r3:
                                    total_signals = (signals['signal'] != 0).sum()
                                    st.metric("Total Signals", total_signals)
                                
                                # Show signal chart
                                if not signals.empty and 'signal' in signals.columns:
                                    fig = go.Figure()
                                    
                                    # Price
                                    fig.add_trace(go.Scatter(
                                        x=test_data.index,
                                        y=test_data['close'],
                                        mode='lines',
                                        name='Price',
                                        line=dict(color='blue', width=2)
                                    ))
                                    
                                    # Buy signals
                                    buy_points = signals[signals['signal'] == 1]
                                    if not buy_points.empty:
                                        fig.add_trace(go.Scatter(
                                            x=buy_points.index,
                                            y=test_data.loc[buy_points.index, 'close'],
                                            mode='markers',
                                            name='Buy',
                                            marker=dict(color='green', size=10, symbol='triangle-up')
                                        ))
                                    
                                    # Sell signals
                                    sell_points = signals[signals['signal'] == -1]
                                    if not sell_points.empty:
                                        fig.add_trace(go.Scatter(
                                            x=sell_points.index,
                                            y=test_data.loc[sell_points.index, 'close'],
                                            mode='markers',
                                            name='Sell',
                                            marker=dict(color='red', size=10, symbol='triangle-down')
                                        ))
                                    
                                    fig.update_layout(
                                        title="Strategy Signals Test",
                                        xaxis_title="Date",
                                        yaxis_title="Price",
                                        height=400
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Show signals table
                                with st.expander("📋 View Signals"):
                                    st.dataframe(signals[signals['signal'] != 0], use_container_width=True)
                            else:
                                st.error("Code must define a generate_signals(data, **kwargs) function")
                
                except Exception as e:
                    st.error(f"Error testing strategy: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Export/Import
        st.markdown("---")
        st.subheader("📤 Export / Import")
        
        col_exp, col_imp = st.columns(2)
        
        with col_exp:
            if st.button("📥 Export Code", use_container_width=True):
                if strategy_code:
                    st.download_button(
                        label="Download Strategy Code",
                        data=strategy_code,
                        file_name=f"{editor_strategy_name or 'strategy'}.py",
                        mime="text/x-python"
                    )
        
        with col_imp:
            uploaded_file = st.file_uploader(
                "Import Code",
                type=['py', 'txt'],
                help="Upload a Python file with your strategy code"
            )
            
            if uploaded_file is not None:
                try:
                    imported_code = uploaded_file.read().decode('utf-8')
                    st.session_state.editor_code = imported_code
                    st.success("✅ Code imported successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error importing code: {str(e)}")
        
        # Load saved strategies
        st.markdown("---")
        st.subheader("📂 Load Saved Strategy")
        
        if handler.strategies:
            saved_strategies = list(handler.strategies.keys())
            selected_strategy = st.selectbox(
                "Select strategy to load",
                [""] + saved_strategies,
                key="load_strategy_editor"
            )
            
            if selected_strategy and st.button("📂 Load", use_container_width=True):
                strategy = handler.strategies[selected_strategy]
                st.session_state.editor_code = strategy.code
                st.session_state.editor_strategy_name = strategy.name
                st.success(f"✅ Loaded {selected_strategy}")
                st.rerun()
        else:
            st.info("No saved strategies available")

with tab4:
    st.header("Strategy Combos")
    st.markdown("Combine multiple strategies into powerful ensemble strategies")
    
    if st.session_state.loaded_data is None:
        st.warning("⚠️ Please load data first in Tab 1 (Quick Backtest)")
    else:
        # Available strategies (pre-built + custom)
        available_strategies = {
            "Bollinger Bands": {"type": "prebuilt", "class": BollingerStrategy, "config": BollingerConfig},
            "MACD": {"type": "prebuilt", "class": MACDStrategy, "config": MACDConfig},
            "RSI": {"type": "prebuilt", "class": RSIStrategy, "config": None},
            "SMA Crossover": {"type": "prebuilt", "class": SMAStrategy, "config": SMAConfig}
        }
        
        # Add custom strategies
        if 'strategy_handler' in st.session_state:
            handler = st.session_state.strategy_handler
            handler.load_strategies()
            for name, strategy in handler.strategies.items():
                available_strategies[name] = {"type": "custom", "strategy": strategy}
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🔧 Ensemble Configuration")
            
            # Strategy selection
            selected_strategies = st.multiselect(
                "Select Strategies to Combine",
                list(available_strategies.keys()),
                default=list(available_strategies.keys())[:2] if len(available_strategies) >= 2 else [],
                help="Select 2-5 strategies to combine"
            )
            
            if len(selected_strategies) < 2:
                st.warning("Please select at least 2 strategies")
            elif len(selected_strategies) > 5:
                st.error("Maximum 5 strategies allowed")
            else:
                # Ensemble method
                ensemble_method = st.selectbox(
                    "Combination Method",
                    ["weighted_average", "voting"],
                    help="How to combine signals from multiple strategies"
                )
                
                st.markdown("**Strategy Weights:**")
                st.caption("Weights will be normalized to sum to 1.0")
                
                strategy_weights = {}
                total_weight = 0.0
                
                for strategy_name in selected_strategies:
                    weight = st.slider(
                        f"{strategy_name} Weight",
                        min_value=0.0,
                        max_value=1.0,
                        value=1.0 / len(selected_strategies),
                        step=0.05,
                        key=f"weight_{strategy_name}"
                    )
                    strategy_weights[strategy_name] = weight
                    total_weight += weight
                
                # Display normalized weights
                if total_weight > 0:
                    st.markdown("**Normalized Weights:**")
                    normalized_weights = {k: v / total_weight for k, v in strategy_weights.items()}
                    for name, weight in normalized_weights.items():
                        st.progress(weight, text=f"{name}: {weight:.1%}")
                
                # Advanced settings
                with st.expander("⚙️ Advanced Settings", expanded=False):
                    confidence_threshold = st.slider(
                        "Confidence Threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.6,
                        step=0.05,
                        help="Minimum confidence to generate a signal"
                    )
                    
                    if ensemble_method == "voting":
                        consensus_threshold = st.slider(
                            "Consensus Threshold",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.5,
                            step=0.05,
                            help="Minimum agreement for consensus signals"
                        )
                    else:
                        consensus_threshold = 0.5
                
                # Backtest settings
                st.markdown("---")
                st.markdown("**Backtest Settings:**")
                initial_capital = st.number_input(
                    "Initial Capital ($)",
                    min_value=1000,
                    value=10000,
                    step=1000,
                    key="combo_capital"
                )
                
                # Run ensemble backtest
                run_ensemble = st.button(
                    "🚀 Run Ensemble Backtest",
                    type="primary",
                    use_container_width=True
                )
        
        with col2:
            st.subheader("📊 Ensemble Results")
            
            if run_ensemble and len(selected_strategies) >= 2:
                try:
                    with st.spinner("Running ensemble backtest..."):
                        data = st.session_state.loaded_data.copy()
                        data.columns = [col.lower() for col in data.columns]
                        
                        # Ensure required columns
                        if 'volume' not in data.columns:
                            data['volume'] = 1000000
                        if 'open' not in data.columns:
                            data['open'] = data['close']
                        if 'high' not in data.columns:
                            data['high'] = data['close']
                        if 'low' not in data.columns:
                            data['low'] = data['close']
                        
                        # Generate signals for each strategy
                        strategy_signals = {}
                        strategy_results = {}
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for idx, strategy_name in enumerate(selected_strategies):
                            status_text.text(f"Generating signals for {strategy_name}...")
                            progress_bar.progress((idx) / (len(selected_strategies) + 1))
                            
                            try:
                                strategy_info = available_strategies[strategy_name]
                                
                                if strategy_info["type"] == "prebuilt":
                                    # Initialize pre-built strategy
                                    strategy_class = strategy_info["class"]
                                    config_class = strategy_info.get("config")
                                    
                                    if config_class:
                                        config = config_class()
                                        strategy = strategy_class(config)
                                    else:
                                        # RSI doesn't use config class
                                        strategy = strategy_class()
                                    
                                    # Generate signals
                                    signals_df = strategy.generate_signals(data)
                                    
                                    # Extract signal column
                                    if 'signal' in signals_df.columns:
                                        signal_col = signals_df[['signal']].copy()
                                    else:
                                        # Try to find signal column
                                        signal_col = pd.DataFrame({'signal': signals_df.iloc[:, 0]}, index=signals_df.index)
                                    
                                    strategy_signals[strategy_name] = signal_col
                                    strategy_results[strategy_name] = {
                                        "signals": signal_col,
                                        "success": True
                                    }
                                
                                elif strategy_info["type"] == "custom":
                                    # Execute custom strategy
                                    custom_strategy = strategy_info["strategy"]
                                    result = handler.execute_strategy(custom_strategy.name, data)
                                    
                                    if result.get("success"):
                                        signals = result.get("result", {}).get("signals")
                                        if isinstance(signals, pd.DataFrame):
                                            if 'signal' in signals.columns:
                                                strategy_signals[strategy_name] = signals[['signal']]
                                            else:
                                                strategy_signals[strategy_name] = pd.DataFrame({'signal': signals.iloc[:, 0]}, index=signals.index)
                                        else:
                                            st.warning(f"Unexpected signal format for {strategy_name}")
                                    
                                    strategy_results[strategy_name] = result
                            
                            except Exception as e:
                                st.warning(f"Failed to generate signals for {strategy_name}: {str(e)}")
                                continue
                        
                        # Create ensemble
                        if len(strategy_signals) >= 2:
                            status_text.text("Combining signals into ensemble...")
                            progress_bar.progress(0.9)
                            
                            # Normalize weights
                            total_weight = sum(strategy_weights.values())
                            if total_weight > 0:
                                normalized_weights = {k: v / total_weight for k, v in strategy_weights.items() if k in strategy_signals}
                            else:
                                # Equal weights
                                normalized_weights = {k: 1.0 / len(strategy_signals) for k in strategy_signals.keys()}
                            
                            # Create ensemble config
                            ensemble_config = EnsembleConfig(
                                strategy_weights=normalized_weights,
                                combination_method=ensemble_method,
                                confidence_threshold=confidence_threshold,
                                consensus_threshold=consensus_threshold if ensemble_method == "voting" else 0.5
                            )
                            
                            # Create and run ensemble
                            ensemble = WeightedEnsembleStrategy(ensemble_config)
                            combined_signals = ensemble.combine_signals(strategy_signals)
                            
                            progress_bar.progress(1.0)
                            status_text.text("Complete!")
                            
                            st.success(f"✅ Ensemble created with {len(strategy_signals)} strategies!")
                            
                            # Calculate ensemble returns
                            data['returns'] = data['close'].pct_change()
                            data['ensemble_returns'] = combined_signals['signal'].shift(1) * data['returns']
                            data['ensemble_cumulative'] = (1 + data['ensemble_returns']).cumprod()
                            
                            # Calculate metrics
                            equity_curve = initial_capital * data['ensemble_cumulative'].fillna(1.0)
                            total_return = (equity_curve.iloc[-1] / initial_capital - 1) * 100
                            
                            returns_series = data['ensemble_returns'].dropna()
                            if len(returns_series) > 0 and returns_series.std() > 0:
                                sharpe = (returns_series.mean() / returns_series.std()) * np.sqrt(252)
                            else:
                                sharpe = 0.0
                            
                            cumulative = data['ensemble_cumulative']
                            running_max = cumulative.cummax()
                            drawdown = (cumulative / running_max - 1) * 100
                            max_dd = drawdown.min()
                            
                            trades = returns_series[returns_series != 0]
                            if len(trades) > 0:
                                win_rate = (trades > 0).sum() / len(trades) * 100
                            else:
                                win_rate = 0.0
                            
                            # Display metrics
                            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                            
                            with col_m1:
                                st.metric("Total Return", f"{total_return:.2f}%")
                            
                            with col_m2:
                                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                            
                            with col_m3:
                                st.metric("Max Drawdown", f"{max_dd:.2f}%")
                            
                            with col_m4:
                                st.metric("Win Rate", f"{win_rate:.1f}%")
                            
                            # Comparison chart
                            st.markdown("---")
                            st.subheader("📈 Ensemble vs Individual Strategies")
                            
                            fig = go.Figure()
                            
                            # Ensemble equity curve
                            fig.add_trace(go.Scatter(
                                x=equity_curve.index,
                                y=equity_curve.values,
                                mode='lines',
                                name='Ensemble',
                                line=dict(color='purple', width=3)
                            ))
                            
                            # Individual strategy equity curves
                            colors = ['blue', 'green', 'orange', 'red', 'brown']
                            for idx, strategy_name in enumerate(strategy_signals.keys()):
                                if strategy_name in strategy_results and strategy_results[strategy_name].get("success"):
                                    signals = strategy_signals[strategy_name]
                                    strategy_returns = signals['signal'].shift(1) * data['returns']
                                    strategy_cumulative = (1 + strategy_returns).cumprod()
                                    strategy_equity = initial_capital * strategy_cumulative.fillna(1.0)
                                    
                                    fig.add_trace(go.Scatter(
                                        x=strategy_equity.index,
                                        y=strategy_equity.values,
                                        mode='lines',
                                        name=strategy_name,
                                        line=dict(color=colors[idx % len(colors)], width=2, dash='dash')
                                    ))
                            
                            fig.update_layout(
                                title="Equity Curve Comparison",
                                xaxis_title="Date",
                                yaxis_title="Portfolio Value ($)",
                                height=500,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Strategy performance comparison table
                            st.markdown("---")
                            st.subheader("📊 Strategy Performance Comparison")
                            
                            comparison_data = []
                            
                            # Add ensemble
                            comparison_data.append({
                                "Strategy": "Ensemble",
                                "Return (%)": f"{total_return:.2f}",
                                "Sharpe": f"{sharpe:.2f}",
                                "Max DD (%)": f"{max_dd:.2f}",
                                "Win Rate (%)": f"{win_rate:.1f}",
                                "Weight": "N/A"
                            })
                            
                            # Add individual strategies
                            for strategy_name in strategy_signals.keys():
                                signals = strategy_signals[strategy_name]
                                strategy_returns = signals['signal'].shift(1) * data['returns']
                                strategy_cumulative = (1 + strategy_returns).cumprod()
                                strategy_equity = initial_capital * strategy_cumulative.fillna(1.0)
                                
                                strat_return = (strategy_equity.iloc[-1] / initial_capital - 1) * 100
                                
                                strat_returns = strategy_returns.dropna()
                                if len(strat_returns) > 0 and strat_returns.std() > 0:
                                    strat_sharpe = (strat_returns.mean() / strat_returns.std()) * np.sqrt(252)
                                else:
                                    strat_sharpe = 0.0
                                
                                strat_dd = ((strategy_cumulative / strategy_cumulative.cummax()) - 1).min() * 100
                                
                                strat_trades = strat_returns[strat_returns != 0]
                                if len(strat_trades) > 0:
                                    strat_win_rate = (strat_trades > 0).sum() / len(strat_trades) * 100
                                else:
                                    strat_win_rate = 0.0
                                
                                weight = normalized_weights.get(strategy_name, 0.0) * 100
                                
                                comparison_data.append({
                                    "Strategy": strategy_name,
                                    "Return (%)": f"{strat_return:.2f}",
                                    "Sharpe": f"{strat_sharpe:.2f}",
                                    "Max DD (%)": f"{strat_dd:.2f}",
                                    "Win Rate (%)": f"{strat_win_rate:.1f}",
                                    "Weight (%)": f"{weight:.1f}"
                                })
                            
                            comparison_df = pd.DataFrame(comparison_data)
                            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                            
                            # Store results
                            st.session_state.ensemble_results = {
                                "ensemble_signals": combined_signals,
                                "strategy_signals": strategy_signals,
                                "metrics": {
                                    "total_return": total_return,
                                    "sharpe": sharpe,
                                    "max_drawdown": max_dd,
                                    "win_rate": win_rate
                                },
                                "equity_curve": equity_curve
                            }
                            
                            progress_bar.empty()
                            status_text.empty()
                        
                        else:
                            st.error("Need at least 2 successful strategies to create ensemble")
                
                except Exception as e:
                    st.error(f"Ensemble backtest failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
            
            # Display previous results if available
            if st.session_state.get('ensemble_results'):
                st.markdown("---")
                st.markdown("**Previous Ensemble Results Available**")
                prev_results = st.session_state.ensemble_results
                st.info(f"Ensemble with {len(prev_results.get('strategy_signals', {}))} strategies")

with tab5:
    st.header("Strategy Comparison")
    st.markdown("Compare multiple strategies side-by-side with comprehensive metrics and statistical analysis")
    
    if st.session_state.loaded_data is None:
        st.warning("⚠️ Please load data first in Tab 1 (Quick Backtest)")
    else:
        # Available strategies
        available_strategies = {
            "Bollinger Bands": {"type": "prebuilt", "class": BollingerStrategy, "config": BollingerConfig},
            "MACD": {"type": "prebuilt", "class": MACDStrategy, "config": MACDConfig},
            "RSI": {"type": "prebuilt", "class": RSIStrategy, "config": None},
            "SMA Crossover": {"type": "prebuilt", "class": SMAStrategy, "config": SMAConfig}
        }
        
        # Add custom strategies
        if 'strategy_handler' in st.session_state:
            handler = st.session_state.strategy_handler
            handler.load_strategies()
            for name, strategy in handler.strategies.items():
                available_strategies[name] = {"type": "custom", "strategy": strategy}
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📊 Strategy Selection")
            
            # Multi-select strategies
            strategies_to_compare = st.multiselect(
                "Select Strategies to Compare",
                list(available_strategies.keys()),
                default=list(available_strategies.keys())[:3] if len(available_strategies) >= 3 else list(available_strategies.keys()),
                help="Select 2 or more strategies to compare"
            )
            
            if len(strategies_to_compare) < 2:
                st.warning("Please select at least 2 strategies")
            else:
                # Backtest settings
                st.markdown("---")
                st.markdown("**Backtest Settings:**")
                initial_capital = st.number_input(
                    "Initial Capital ($)",
                    min_value=1000,
                    value=10000,
                    step=1000,
                    key="compare_capital"
                )
                
                commission = st.number_input(
                    "Commission (%)",
                    min_value=0.0,
                    value=0.1,
                    step=0.01,
                    key="compare_commission"
                )
                
                # Run comparison
                run_comparison = st.button(
                    "🚀 Run Comparison",
                    type="primary",
                    use_container_width=True
                )
        
        with col2:
            st.subheader("📈 Comparison Results")
            
            if run_comparison and len(strategies_to_compare) >= 2:
                try:
                    with st.spinner("Running parallel backtests..."):
                        data = st.session_state.loaded_data.copy()
                        data.columns = [col.lower() for col in data.columns]
                        
                        # Ensure required columns
                        if 'volume' not in data.columns:
                            data['volume'] = 1000000
                        if 'open' not in data.columns:
                            data['open'] = data['close']
                        if 'high' not in data.columns:
                            data['high'] = data['close']
                        if 'low' not in data.columns:
                            data['low'] = data['close']
                        
                        # Run backtests for all strategies
                        comparison_results = {}
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for idx, strategy_name in enumerate(strategies_to_compare):
                            status_text.text(f"Backtesting {strategy_name}... ({idx+1}/{len(strategies_to_compare)})")
                            progress_bar.progress(idx / len(strategies_to_compare))
                            
                            try:
                                strategy_info = available_strategies[strategy_name]
                                
                                # Initialize strategy
                                if strategy_info["type"] == "prebuilt":
                                    strategy_class = strategy_info["class"]
                                    config_class = strategy_info.get("config")
                                    
                                    if config_class:
                                        config = config_class()
                                        strategy = strategy_class(config)
                                    else:
                                        strategy = strategy_class()
                                    
                                    # Generate signals
                                    signals_df = strategy.generate_signals(data)
                                    
                                    # Extract signal column
                                    if 'signal' in signals_df.columns:
                                        signal_col = signals_df['signal']
                                    else:
                                        signal_col = signals_df.iloc[:, 0]
                                
                                elif strategy_info["type"] == "custom":
                                    custom_strategy = strategy_info["strategy"]
                                    result = handler.execute_strategy(custom_strategy.name, data)
                                    
                                    if result.get("success"):
                                        signals = result.get("result", {}).get("signals")
                                        if isinstance(signals, pd.DataFrame):
                                            signal_col = signals['signal'] if 'signal' in signals.columns else signals.iloc[:, 0]
                                        else:
                                            signal_col = pd.Series(0, index=data.index)
                                    else:
                                        raise Exception(f"Strategy execution failed: {result.get('error')}")
                                
                                # Calculate returns
                                data['returns'] = data['close'].pct_change()
                                strategy_returns = signal_col.shift(1) * data['returns']
                                strategy_cumulative = (1 + strategy_returns).cumprod()
                                equity_curve = initial_capital * strategy_cumulative.fillna(1.0)
                                
                                # Calculate metrics
                                total_return = (equity_curve.iloc[-1] / initial_capital - 1) * 100
                                
                                returns_series = strategy_returns.dropna()
                                if len(returns_series) > 0 and returns_series.std() > 0:
                                    sharpe = (returns_series.mean() / returns_series.std()) * np.sqrt(252)
                                else:
                                    sharpe = 0.0
                                
                                cumulative = strategy_cumulative
                                running_max = cumulative.cummax()
                                drawdown = (cumulative / running_max - 1) * 100
                                max_dd = drawdown.min()
                                
                                trades = returns_series[returns_series != 0]
                                if len(trades) > 0:
                                    win_rate = (trades > 0).sum() / len(trades) * 100
                                    avg_win = trades[trades > 0].mean() * 100 if (trades > 0).sum() > 0 else 0
                                    avg_loss = trades[trades < 0].mean() * 100 if (trades < 0).sum() > 0 else 0
                                else:
                                    win_rate = 0.0
                                    avg_win = 0.0
                                    avg_loss = 0.0
                                
                                # Total trades
                                total_trades = len(trades)
                                
                                # Profit factor
                                if abs(avg_loss) > 0:
                                    profit_factor = (avg_win * (trades > 0).sum()) / (abs(avg_loss) * (trades < 0).sum()) if (trades < 0).sum() > 0 else float('inf')
                                else:
                                    profit_factor = float('inf') if (trades > 0).sum() > 0 else 0
                                
                                comparison_results[strategy_name] = {
                                    "total_return": total_return,
                                    "sharpe": sharpe,
                                    "max_drawdown": max_dd,
                                    "win_rate": win_rate,
                                    "total_trades": total_trades,
                                    "profit_factor": profit_factor,
                                    "equity_curve": equity_curve,
                                    "returns": strategy_returns,
                                    "signals": signal_col
                                }
                            
                            except Exception as e:
                                st.warning(f"Failed to backtest {strategy_name}: {str(e)}")
                                continue
                        
                        progress_bar.progress(1.0)
                        status_text.text("Complete!")
                        
                        if len(comparison_results) >= 2:
                            st.success(f"✅ Compared {len(comparison_results)} strategies successfully!")
                            
                            # Find best performer
                            best_return = max(comparison_results.items(), key=lambda x: x[1]["total_return"])
                            best_sharpe = max(comparison_results.items(), key=lambda x: x[1]["sharpe"])
                            best_dd = min(comparison_results.items(), key=lambda x: x[1]["max_drawdown"])
                            
                            # Display best performers
                            st.markdown("**🏆 Best Performers:**")
                            col_b1, col_b2, col_b3 = st.columns(3)
                            with col_b1:
                                st.metric("Best Return", best_return[0], f"{best_return[1]['total_return']:.2f}%")
                            with col_b2:
                                st.metric("Best Sharpe", best_sharpe[0], f"{best_sharpe[1]['sharpe']:.2f}")
                            with col_b3:
                                st.metric("Best Drawdown", best_dd[0], f"{best_dd[1]['max_drawdown']:.2f}%")
                            
                            # Comparison table
                            st.markdown("---")
                            st.subheader("📊 Performance Metrics Table")
                            
                            comparison_data = []
                            for name, results in comparison_results.items():
                                is_best_return = name == best_return[0]
                                is_best_sharpe = name == best_sharpe[0]
                                is_best_dd = name == best_dd[0]
                                
                                comparison_data.append({
                                    "Strategy": name,
                                    "Total Return (%)": f"{results['total_return']:.2f}",
                                    "Sharpe Ratio": f"{results['sharpe']:.2f}",
                                    "Max Drawdown (%)": f"{results['max_drawdown']:.2f}",
                                    "Win Rate (%)": f"{results['win_rate']:.1f}",
                                    "Total Trades": results['total_trades'],
                                    "Profit Factor": f"{results['profit_factor']:.2f}" if results['profit_factor'] != float('inf') else "∞"
                                })
                            
                            comparison_df = pd.DataFrame(comparison_data)
                            
                            # Display with highlighting
                            st.dataframe(
                                comparison_df,
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            # Equity curve comparison chart
                            st.markdown("---")
                            st.subheader("📈 Equity Curve Comparison")
                            
                            fig = go.Figure()
                            
                            colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink']
                            for idx, (name, results) in enumerate(comparison_results.items()):
                                is_best = name == best_return[0]
                                line_width = 3 if is_best else 2
                                line_dash = None if is_best else 'dash'
                                
                                fig.add_trace(go.Scatter(
                                    x=results['equity_curve'].index,
                                    y=results['equity_curve'].values,
                                    mode='lines',
                                    name=name + (" ⭐" if is_best else ""),
                                    line=dict(
                                        color=colors[idx % len(colors)],
                                        width=line_width,
                                        dash=line_dash
                                    )
                                ))
                            
                            fig.update_layout(
                                title="Equity Curve Comparison",
                                xaxis_title="Date",
                                yaxis_title="Portfolio Value ($)",
                                height=500,
                                hovermode='x unified',
                                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Drawdown comparison
                            st.markdown("---")
                            st.subheader("📉 Drawdown Comparison")
                            
                            fig_dd = go.Figure()
                            
                            for idx, (name, results) in enumerate(comparison_results.items()):
                                equity = results['equity_curve']
                                cumulative = equity / initial_capital
                                running_max = cumulative.cummax()
                                drawdown = (cumulative / running_max - 1) * 100
                                
                                fig_dd.add_trace(go.Scatter(
                                    x=drawdown.index,
                                    y=drawdown.values,
                                    mode='lines',
                                    name=name,
                                    line=dict(color=colors[idx % len(colors)], width=2),
                                    fill='tozeroy'
                                ))
                            
                            fig_dd.update_layout(
                                title="Drawdown Comparison",
                                xaxis_title="Date",
                                yaxis_title="Drawdown (%)",
                                height=400,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig_dd, use_container_width=True)
                            
                            # Returns distribution
                            st.markdown("---")
                            st.subheader("📊 Returns Distribution")
                            
                            fig_ret = go.Figure()
                            
                            for idx, (name, results) in enumerate(comparison_results.items()):
                                returns = results['returns'].dropna() * 100
                                fig_ret.add_trace(go.Histogram(
                                    x=returns,
                                    name=name,
                                    opacity=0.6,
                                    nbinsx=50
                                ))
                            
                            fig_ret.update_layout(
                                title="Returns Distribution",
                                xaxis_title="Daily Return (%)",
                                yaxis_title="Frequency",
                                height=400,
                                barmode='overlay'
                            )
                            
                            st.plotly_chart(fig_ret, use_container_width=True)
                            
                            # Statistical summary
                            st.markdown("---")
                            st.subheader("📈 Statistical Summary")
                            
                            summary_data = []
                            for name, results in comparison_results.items():
                                returns = results['returns'].dropna()
                                summary_data.append({
                                    "Strategy": name,
                                    "Mean Return (%)": f"{returns.mean() * 100:.4f}",
                                    "Std Dev (%)": f"{returns.std() * 100:.4f}",
                                    "Skewness": f"{returns.skew():.2f}",
                                    "Kurtosis": f"{returns.kurtosis():.2f}",
                                    "Min Return (%)": f"{returns.min() * 100:.2f}",
                                    "Max Return (%)": f"{returns.max() * 100:.2f}"
                                })
                            
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, use_container_width=True, hide_index=True)
                            
                            # Store results
                            st.session_state.comparison_results = comparison_results
                            
                        else:
                            st.error("Need at least 2 successful backtests to compare")
                
                except Exception as e:
                    st.error(f"Comparison failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
            
            # Display previous comparison if available
            if st.session_state.get('comparison_results'):
                st.markdown("---")
                st.markdown("**Previous Comparison Results Available**")
                prev_results = st.session_state.comparison_results
                st.info(f"Compared {len(prev_results)} strategies in previous run")

with tab6:
    st.header("Advanced Analysis")
    st.markdown("Advanced testing methodologies: Walk-forward analysis, Monte Carlo simulation, sensitivity analysis, and optimization")
    
    if st.session_state.loaded_data is None:
        st.warning("⚠️ Please load data first in Tab 1 (Quick Backtest)")
    else:
        # Analysis type selection
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Walk-Forward Analysis", "Monte Carlo Simulation", "Sensitivity Analysis", "Parameter Optimization"],
            help="Choose the type of advanced analysis to perform"
        )
        
        if analysis_type == "Walk-Forward Analysis":
            st.subheader("🔄 Walk-Forward Analysis")
            st.markdown("Test strategy stability across multiple time windows")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**Configuration:**")
                window_size = st.number_input(
                    "Training Window Size (days)",
                    min_value=30,
                    max_value=1000,
                    value=252,
                    step=21,
                    help="Size of training window"
                )
                
                test_size = st.number_input(
                    "Test Window Size (days)",
                    min_value=10,
                    max_value=200,
                    value=63,
                    step=10,
                    help="Size of test window"
                )
                
                step_size = st.number_input(
                    "Step Size (days)",
                    min_value=5,
                    max_value=100,
                    value=21,
                    step=5,
                    help="How much to move the window forward"
                )
                
                initial_capital = st.number_input(
                    "Initial Capital ($)",
                    min_value=1000,
                    value=10000,
                    step=1000,
                    key="wf_capital"
                )
            
            with col2:
                # Strategy selection for walk-forward
                st.markdown("**Strategy Selection:**")
                strategy_for_wf = st.selectbox(
                    "Select Strategy",
                    ["Bollinger Bands", "MACD", "RSI", "SMA Crossover"],
                    key="wf_strategy"
                )
                
                run_walk_forward = st.button(
                    "🚀 Run Walk-Forward Analysis",
                    type="primary",
                    use_container_width=True
                )
            
            if run_walk_forward:
                try:
                    with st.spinner("Running walk-forward analysis..."):
                        data = st.session_state.loaded_data.copy()
                        data.columns = [col.lower() for col in data.columns]
                        
                        # Ensure required columns
                        if 'volume' not in data.columns:
                            data['volume'] = 1000000
                        if 'open' not in data.columns:
                            data['open'] = data['close']
                        if 'high' not in data.columns:
                            data['high'] = data['close']
                        if 'low' not in data.columns:
                            data['low'] = data['close']
                        
                        # Initialize strategy
                        if strategy_for_wf == "Bollinger Bands":
                            strategy = BollingerStrategy(BollingerConfig())
                        elif strategy_for_wf == "MACD":
                            strategy = MACDStrategy(MACDConfig())
                        elif strategy_for_wf == "RSI":
                            strategy = RSIStrategy()
                        else:  # SMA
                            strategy = SMAStrategy(SMAConfig())
                        
                        # Run walk-forward analysis
                        evaluator = ModelEvaluator(data, initial_cash=initial_capital)
                        
                        # Simple walk-forward implementation
                        total_days = len(data)
                        num_windows = (total_days - window_size - test_size) // step_size + 1
                        
                        window_results = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i in range(num_windows):
                            status_text.text(f"Processing window {i+1}/{num_windows}...")
                            progress_bar.progress(i / num_windows)
                            
                            train_start = i * step_size
                            train_end = train_start + window_size
                            test_start = train_end
                            test_end = min(test_start + test_size, total_days)
                            
                            if test_end <= total_days:
                                train_data = data.iloc[train_start:train_end]
                                test_data = data.iloc[test_start:test_end]
                                
                                # Generate signals on test data
                                signals = strategy.generate_signals(test_data)
                                signal_col = signals['signal'] if 'signal' in signals.columns else signals.iloc[:, 0]
                                
                                # Calculate returns
                                test_data = test_data.copy()
                                test_data['returns'] = test_data['close'].pct_change()
                                strategy_returns = signal_col.shift(1) * test_data['returns']
                                
                                # Calculate metrics
                                total_return = (1 + strategy_returns).prod() - 1
                                sharpe = (strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)) if strategy_returns.std() > 0 else 0
                                
                                cumulative = (1 + strategy_returns).cumprod()
                                max_dd = ((cumulative / cumulative.cummax()) - 1).min()
                                
                                window_results.append({
                                    "window": i + 1,
                                    "train_start": train_data.index[0],
                                    "test_start": test_data.index[0],
                                    "test_end": test_data.index[-1],
                                    "return": total_return * 100,
                                    "sharpe": sharpe,
                                    "max_drawdown": max_dd * 100
                                })
                        
                        progress_bar.progress(1.0)
                        status_text.text("Complete!")
                        
                        if window_results:
                            results_df = pd.DataFrame(window_results)
                            
                            st.success(f"✅ Walk-forward analysis complete! Processed {len(window_results)} windows")
                            
                            # Summary metrics
                            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                            with col_s1:
                                st.metric("Mean Return", f"{results_df['return'].mean():.2f}%")
                            with col_s2:
                                st.metric("Mean Sharpe", f"{results_df['sharpe'].mean():.2f}")
                            with col_s3:
                                st.metric("Mean Max DD", f"{results_df['max_drawdown'].mean():.2f}%")
                            with col_s4:
                                st.metric("Std Dev Return", f"{results_df['return'].std():.2f}%")
                            
                            # Visualization
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=results_df['window'],
                                y=results_df['return'],
                                mode='lines+markers',
                                name='Return (%)',
                                line=dict(color='blue', width=2)
                            ))
                            fig.update_layout(
                                title="Walk-Forward Returns by Window",
                                xaxis_title="Window Number",
                                yaxis_title="Return (%)",
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Results table
                            st.dataframe(results_df, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Walk-forward analysis failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        elif analysis_type == "Monte Carlo Simulation":
            st.subheader("🎲 Monte Carlo Simulation")
            st.markdown("Simulate thousands of possible portfolio scenarios")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**Configuration:**")
                n_simulations = st.number_input(
                    "Number of Simulations",
                    min_value=100,
                    max_value=10000,
                    value=1000,
                    step=100,
                    help="Number of simulation paths to generate"
                )
                
                initial_capital = st.number_input(
                    "Initial Capital ($)",
                    min_value=1000,
                    value=10000,
                    step=1000,
                    key="mc_capital"
                )
                
                bootstrap_method = st.selectbox(
                    "Bootstrap Method",
                    ["historical", "block", "parametric"],
                    help="Method for resampling returns"
                )
            
            with col2:
                st.markdown("**Strategy Selection:**")
                strategy_for_mc = st.selectbox(
                    "Select Strategy",
                    ["Bollinger Bands", "MACD", "RSI", "SMA Crossover"],
                    key="mc_strategy"
                )
                
                run_monte_carlo = st.button(
                    "🚀 Run Monte Carlo Simulation",
                    type="primary",
                    use_container_width=True
                )
            
            if run_monte_carlo:
                try:
                    with st.spinner(f"Running {n_simulations} Monte Carlo simulations..."):
                        data = st.session_state.loaded_data.copy()
                        data.columns = [col.lower() for col in data.columns]
                        
                        # Initialize strategy
                        if strategy_for_mc == "Bollinger Bands":
                            strategy = BollingerStrategy(BollingerConfig())
                        elif strategy_for_mc == "MACD":
                            strategy = MACDStrategy(MACDConfig())
                        elif strategy_for_mc == "RSI":
                            strategy = RSIStrategy()
                        else:
                            strategy = SMAStrategy(SMAConfig())
                        
                        # Generate signals and returns
                        signals = strategy.generate_signals(data)
                        signal_col = signals['signal'] if 'signal' in signals.columns else signals.iloc[:, 0]
                        
                        data['returns'] = data['close'].pct_change()
                        strategy_returns = signal_col.shift(1) * data['returns']
                        strategy_returns = strategy_returns.dropna()
                        
                        # Run Monte Carlo simulation
                        config = MonteCarloConfig(
                            n_simulations=n_simulations,
                            bootstrap_method=bootstrap_method,
                            initial_capital=initial_capital
                        )
                        
                        simulator = MonteCarloSimulator(config)
                        simulated_paths = simulator.simulate_portfolio_paths(
                            strategy_returns,
                            initial_capital,
                            n_simulations
                        )
                        
                        # Calculate percentiles
                        simulator.calculate_percentiles()
                        
                        st.success("✅ Monte Carlo simulation complete!")
                        
                        # Summary statistics
                        final_values = simulated_paths.iloc[:, -1]
                        
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        with col_m1:
                            st.metric("Mean Final Value", f"${final_values.mean():,.2f}")
                        with col_m2:
                            st.metric("Median Final Value", f"${final_values.median():,.2f}")
                        with col_m3:
                            st.metric("5th Percentile", f"${final_values.quantile(0.05):,.2f}")
                        with col_m4:
                            st.metric("95th Percentile", f"${final_values.quantile(0.95):,.2f}")
                        
                        # Visualization
                        fig = go.Figure()
                        
                        # Sample paths
                        sample_paths = simulated_paths.sample(min(50, n_simulations), axis=1)
                        for col in sample_paths.columns:
                            fig.add_trace(go.Scatter(
                                x=simulated_paths.index,
                                y=sample_paths[col],
                                mode='lines',
                                name=f"Path {col}",
                                line=dict(width=1, color='lightblue'),
                                showlegend=False
                            ))
                        
                        # Percentiles
                        if simulator.percentiles is not None:
                            for percentile, values in simulator.percentiles.items():
                                fig.add_trace(go.Scatter(
                                    x=simulated_paths.index,
                                    y=values,
                                    mode='lines',
                                    name=f"{percentile*100:.0f}th Percentile",
                                    line=dict(width=2, dash='dash')
                                ))
                        
                        fig.update_layout(
                            title="Monte Carlo Simulation Paths",
                            xaxis_title="Period",
                            yaxis_title="Portfolio Value ($)",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Monte Carlo simulation failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        elif analysis_type == "Sensitivity Analysis":
            st.subheader("🔬 Sensitivity Analysis")
            st.markdown("Test how strategy performance changes with different parameters")
            
            st.info("Sensitivity analysis allows you to test parameter variations. Select a strategy and adjust parameters to see how performance changes.")
            
            strategy_for_sens = st.selectbox(
                "Select Strategy",
                ["Bollinger Bands", "MACD", "RSI", "SMA Crossover"],
                key="sens_strategy"
            )
            
            if strategy_for_sens == "Bollinger Bands":
                param_name = "Window"
                param_values = st.slider(
                    "Window Values",
                    min_value=10,
                    max_value=50,
                    value=(15, 25),
                    step=1,
                    help="Range of window values to test"
                )
            elif strategy_for_sens == "MACD":
                param_name = "Fast Period"
                param_values = st.slider(
                    "Fast Period Values",
                    min_value=5,
                    max_value=20,
                    value=(10, 15),
                    step=1
                )
            elif strategy_for_sens == "RSI":
                param_name = "RSI Period"
                param_values = st.slider(
                    "RSI Period Values",
                    min_value=5,
                    max_value=30,
                    value=(10, 20),
                    step=1
                )
            else:  # SMA
                param_name = "Short Window"
                param_values = st.slider(
                    "Short Window Values",
                    min_value=5,
                    max_value=50,
                    value=(15, 25),
                    step=1
                )
            
            run_sensitivity = st.button(
                "🚀 Run Sensitivity Analysis",
                type="primary",
                use_container_width=True
            )
            
            if run_sensitivity:
                st.info("Sensitivity analysis implementation in progress. This will test multiple parameter combinations and display results.")
        
        else:  # Parameter Optimization
            st.subheader("⚙️ Parameter Optimization")
            st.markdown("Find optimal strategy parameters using optimization algorithms")
            
            st.info("Parameter optimization helps find the best parameter values for your strategy. This feature will be enhanced with genetic algorithms and Bayesian optimization.")
            
            strategy_for_opt = st.selectbox(
                "Select Strategy",
                ["Bollinger Bands", "MACD", "RSI", "SMA Crossover"],
                key="opt_strategy"
            )
            
            optimization_method = st.selectbox(
                "Optimization Method",
                ["Grid Search", "Random Search", "Genetic Algorithm", "Bayesian Optimization"],
                help="Method to use for parameter optimization"
            )
            
            objective = st.selectbox(
                "Optimization Objective",
                ["Maximize Sharpe Ratio", "Maximize Return", "Minimize Drawdown", "Maximize Win Rate"],
                help="What metric to optimize for"
            )
            
            run_optimization = st.button(
                "🚀 Run Optimization",
                type="primary",
                use_container_width=True
            )
            
            if run_optimization:
                st.info("Parameter optimization implementation in progress. This will search for optimal parameters and display results.")


# Helper functions for Strategy Builder
def generate_strategy_code(config: dict) -> str:
    """Generate Python code for a custom strategy based on configuration."""
    
    strategy_class_name = config['name'].replace(' ', '').replace('-', '').replace('_', '')
    
    code = f'''
"""
{config.get('name', 'Custom Strategy')}
{config.get('description', '')}

Auto-generated strategy code
"""

import pandas as pd
import numpy as np
from typing import Optional

class {strategy_class_name}Strategy:
    def __init__(self):
        self.name = "{config['name']}"
        self.config = {config}
        self.position_size = {config['position_sizing']['size']}
        self.max_positions = {config['risk_management']['max_positions']}
        self.stop_loss_pct = {config['risk_management'].get('stop_loss') if config['risk_management'].get('stop_loss') else 'None'}
        self.take_profit_pct = {config['risk_management'].get('take_profit') if config['risk_management'].get('take_profit') else 'None'}
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on strategy rules."""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Ensure data has required columns (case-insensitive)
        data_lower = data.copy()
        data_lower.columns = data_lower.columns.str.lower()
        
        # Calculate indicators if needed
        if 'close' not in data_lower.columns:
            raise ValueError("Data must contain 'close' column")
        
        # Entry conditions
        entry_conditions = []
        {generate_entry_conditions_code(config)}
        
        # Exit conditions  
        exit_conditions = []
        {generate_exit_conditions_code(config)}
        
        # Apply entry logic ({config['entry_logic']})
        if len(entry_conditions) > 0:
            if "{config['entry_logic']}" == "AND":
                entry_signal = pd.concat(entry_conditions, axis=1).all(axis=1)
            else:
                entry_signal = pd.concat(entry_conditions, axis=1).any(axis=1)
            
            signals.loc[entry_signal, 'signal'] = 1
        
        # Apply exit logic ({config['exit_logic']})
        if len(exit_conditions) > 0:
            if "{config['exit_logic']}" == "AND":
                exit_signal = pd.concat(exit_conditions, axis=1).all(axis=1)
            else:
                exit_signal = pd.concat(exit_conditions, axis=1).any(axis=1)
            
            signals.loc[exit_signal, 'signal'] = -1
        
        return signals
'''
    
    return code


def generate_entry_conditions_code(config: dict) -> str:
    """Generate code for entry conditions."""
    code_lines = []
    for i, cond in enumerate(config.get('entry_conditions', [])):
        indicator = cond['indicator']
        operator = cond['operator']
        value = cond['value']
        
        # Generate pandas condition based on indicator type
        if indicator == "RSI":
            code_lines.append(f"# Calculate RSI")
            code_lines.append(f"rsi_period = 14")
            code_lines.append(f"delta = data_lower['close'].diff()")
            code_lines.append(f"gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()")
            code_lines.append(f"loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()")
            code_lines.append(f"rs = gain / loss")
            code_lines.append(f"rsi = 100 - (100 / (1 + rs))")
            code_lines.append(f"entry_conditions.append((rsi {operator} {value}).to_frame('cond_{i}'))")
        elif indicator == "SMA":
            code_lines.append(f"# Calculate SMA")
            period = value if isinstance(value, (int, float)) else 20
            code_lines.append(f"sma = data_lower['close'].rolling({period}).mean()")
            code_lines.append(f"entry_conditions.append((data_lower['close'] {operator} sma).to_frame('cond_{i}'))")
        elif indicator == "Price":
            code_lines.append(f"# Price condition")
            code_lines.append(f"entry_conditions.append((data_lower['close'] {operator} {value}).to_frame('cond_{i}'))")
        else:
            # Generic condition placeholder
            code_lines.append(f"# {indicator} {operator} {value}")
            code_lines.append(f"entry_conditions.append((data_lower['close'] > 0).to_frame('cond_{i}'))  # Placeholder")
    
    if not code_lines:
        code_lines.append("# No entry conditions defined")
        code_lines.append("pass")
    
    return "\n        ".join(code_lines)


def generate_exit_conditions_code(config: dict) -> str:
    """Generate code for exit conditions."""
    code_lines = []
    for i, cond in enumerate(config.get('exit_conditions', [])):
        indicator = cond['indicator']
        operator = cond['operator']
        value = cond['value']
        
        if indicator == "Profit %":
            code_lines.append(f"# Profit target exit")
            code_lines.append(f"# This would be calculated during backtesting based on entry price")
            code_lines.append(f"exit_conditions.append((data_lower['close'] > 0).to_frame('cond_{i}'))  # Placeholder")
        elif indicator == "Loss %":
            code_lines.append(f"# Stop loss exit")
            code_lines.append(f"# This would be calculated during backtesting based on entry price")
            code_lines.append(f"exit_conditions.append((data_lower['close'] > 0).to_frame('cond_{i}'))  # Placeholder")
        elif indicator == "RSI":
            code_lines.append(f"# RSI exit condition")
            code_lines.append(f"rsi_period = 14")
            code_lines.append(f"delta = data_lower['close'].diff()")
            code_lines.append(f"gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()")
            code_lines.append(f"loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()")
            code_lines.append(f"rs = gain / loss")
            code_lines.append(f"rsi = 100 - (100 / (1 + rs))")
            code_lines.append(f"exit_conditions.append((rsi {operator} {value}).to_frame('cond_{i}'))")
        else:
            code_lines.append(f"# {indicator} {operator} {value}")
            code_lines.append(f"exit_conditions.append((data_lower['close'] > 0).to_frame('cond_{i}'))  # Placeholder")
    
    if not code_lines:
        code_lines.append("# No exit conditions defined")
        code_lines.append("pass")
    
    return "\n        ".join(code_lines)


def validate_strategy_code(code: str, strategy_name: str = "strategy") -> dict:
    """Validate strategy code for syntax and required functions.
    
    Args:
        code: Python code string to validate
        strategy_name: Name of the strategy (for error messages)
    
    Returns:
        Dictionary with validation results:
        - valid: bool - Whether code is valid
        - error: str - Error message if invalid
        - warnings: list - List of warnings
    """
    warnings_list = []
    
    try:
        # Try to compile the code
        compile(code, f"<{strategy_name}>", "exec")
        
        # Check for required function
        if "generate_signals" not in code:
            return {
                "valid": False,
                "error": "Code must contain a 'generate_signals' function",
                "warnings": []
            }
        
        # Check for required imports
        if "import pandas" not in code and "import pd" not in code:
            warnings_list.append("Consider importing pandas for DataFrame operations")
        
        # Check for function signature
        if "def generate_signals" not in code:
            return {
                "valid": False,
                "error": "generate_signals function definition not found",
                "warnings": warnings_list
            }
        
        # Check if function accepts data parameter
        if "generate_signals(data" not in code and "generate_signals( data" not in code:
            warnings_list.append("generate_signals should accept 'data' as first parameter")
        
        return {
            "valid": True,
            "error": None,
            "warnings": warnings_list
        }
    
    except SyntaxError as e:
        return {
            "valid": False,
            "error": f"Syntax error: {str(e)}",
            "warnings": []
        }
    except Exception as e:
        return {
            "valid": False,
            "error": f"Validation error: {str(e)}",
            "warnings": []
        }

