import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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
        
        load_data = st.form_submit_button("📊 Load Data")
    
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
    
    # Strategy selection using UI component
    run_backtest = False  # Initialize before use
    if st.session_state.loaded_data is not None:
        st.markdown("---")
        st.subheader("🎯 Strategy Selection")
        
        try:
            from trading.ui.strategy_components import render_strategy_selector
            
            strategy_name = render_strategy_selector(key="quick_backtest_strategy")
            
            if strategy_name:
                strategy_info = STRATEGY_REGISTRY.get(strategy_name, {})
                if strategy_info:
                    st.info(strategy_info.get("description", "No description available"))
                else:
                    # Fallback if strategy not in registry
                    st.warning("Selected strategy not found in registry; using default configuration.")
                    strategy_info = STRATEGY_REGISTRY[list(STRATEGY_REGISTRY.keys())[0]]
                    strategy_name = list(STRATEGY_REGISTRY.keys())[0]
                    st.info(strategy_info.get("description", "No description available"))
            else:
                # Fallback if component returns None
                strategy_name = list(STRATEGY_REGISTRY.keys())[0]
                strategy_info = STRATEGY_REGISTRY[strategy_name]
                st.info(strategy_info.get("description", "No description available"))
        except ImportError:
            # Fallback to original code when shared strategy selector is unavailable
            strategy_name = st.selectbox(
                "Select Strategy",
                list(STRATEGY_REGISTRY.keys())
            )
            strategy_info = STRATEGY_REGISTRY[strategy_name]
            st.info(strategy_info.get("description", "No description available"))
        
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
        
        # Sentiment-Based Strategy
        st.markdown("---")
        st.subheader("📰 Sentiment-Based Strategy")
        
        use_sentiment = st.checkbox("Include sentiment signals", key="use_sentiment_signals")
        
        sentiment_threshold = 0.3
        sentiment_signals_data = None
        
        if use_sentiment:
            try:
                from trading.signals.sentiment_signals import SentimentSignals
                
                sentiment_threshold = st.slider(
                    "Sentiment Threshold",
                    min_value=0.1,
                    max_value=0.9,
                    value=0.3,
                    step=0.1,
                    help="Minimum sentiment score to trigger a signal"
                )
                
                if st.button("Generate Sentiment Signals", key="generate_sentiment_signals"):
                    with st.spinner("Generating sentiment signals..."):
                        sentiment_signals = SentimentSignals()
                        
                        # Get symbol from session state or use default
                        symbol = st.session_state.get('backtest_symbol', 'AAPL')
                        data = st.session_state.loaded_data.copy()
                        
                        # Generate signals (SentimentSignals uses generate_sentiment_signals(ticker) or generate_signals if present)
                        method = next((m for m in ["generate_signals", "generate_sentiment_signals", "get_signals", "compute_signals"] if hasattr(sentiment_signals, m)), None)
                        if method:
                            fn = getattr(sentiment_signals, method)
                            if method == "generate_sentiment_signals":
                                raw = fn(symbol, include_reddit=True, include_news=True)
                                signals = {"buy_count": 1 if raw.get("signals", {}).get("buy_signal") else 0, "sell_count": 1 if raw.get("signals", {}).get("sell_signal") else 0, "signal_history": [{"date": raw.get("timestamp", ""), "signal": "buy" if raw.get("signals", {}).get("buy_signal") else "sell" if raw.get("signals", {}).get("sell_signal") else "hold", "confidence": raw.get("confidence", 0)}], **raw}
                            else:
                                signals = fn(symbol=symbol, price_data=data, sentiment_threshold=sentiment_threshold) if method == "generate_signals" else fn(symbol)
                        else:
                            signals = None
                        
                        sentiment_signals_data = signals
                        
                        if signals is None:
                            st.warning("Sentiment signals not available")
                        else:
                            st.success("✅ Sentiment signals generated!")
                        
                        # Display signal summary
                        col_sig1, col_sig2 = st.columns(2)
                        
                        with col_sig1:
                            st.metric("Buy Signals", signals.get('buy_count', 0) if signals else 0)
                        with col_sig2:
                            st.metric("Sell Signals", signals.get('sell_count', 0) if signals else 0)
                        
                        # Show recent signals
                        if signals and 'signal_history' in signals and len(signals.get('signal_history', [])) > 0:
                            st.markdown("**Recent Signals:**")
                            recent_signals = signals['signal_history'][-10:]
                            signals_df = pd.DataFrame(recent_signals)
                            
                            # Format the dataframe for display
                            if not signals_df.empty:
                                # Ensure date column is formatted
                                if 'date' in signals_df.columns:
                                    signals_df['date'] = pd.to_datetime(signals_df['date']).dt.strftime('%Y-%m-%d')
                                elif 'timestamp' in signals_df.columns:
                                    signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp']).dt.strftime('%Y-%m-%d')
                                
                                st.dataframe(signals_df, width='stretch')
                        
                        # Store in session state for use in backtest
                        if signals is not None:
                            st.session_state.sentiment_signals = signals
            
            except ImportError:
                st.warning("⚠️ Sentiment signals not available. Install required dependencies.")
            except Exception as e:
                st.error(f"Error generating sentiment signals: {e}")
                import traceback
                st.code(traceback.format_exc())
        
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
            type="primary"
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
                    
                    # Store results in a standard schema
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

                    # Normalize trade records for downstream pages
                    results['trades'] = _normalize_trades(results.get('trades', []))

                    # Normalize keys for downstream pages (Performance, Reports, Portfolio)
                    try:
                        from trading.backtesting.backtester import Backtester

                        results = Backtester.normalize_results(results)
                    except Exception:
                        # If normalization fails, continue with raw results
                        pass

                    # AGENT_MEMORY_LAYER: Persist backtest outcome (long-term) — quality gate: only when metrics present
                    try:
                        from trading.memory.memory_store import MemoryType
                        _sr, _tr = results.get("sharpe_ratio"), results.get("total_return")
                        if _sr is not None and _tr is not None:
                            _get_memory_store().add(
                                MemoryType.LONG_TERM,
                                namespace="StreamlitStrategyTesting",
                                category="backtest",
                                key=f"{st.session_state.get('backtest_symbol', 'AAPL')}:{strategy_name}:{datetime.utcnow().isoformat()}",
                                value={
                                    "symbol": st.session_state.get('backtest_symbol', 'AAPL'),
                                    "strategy": strategy_name,
                                    "metrics": {
                                        "total_return": _tr,
                                        "sharpe_ratio": _sr,
                                        "max_drawdown": results.get("max_drawdown"),
                                        "win_rate": results.get("win_rate"),
                                        "total_trades": len(results.get("trades", [])),
                                    },
                                },
                                metadata={"source": "pages/3_Strategy_Testing.py quick backtest"},
                            )
                    except Exception:
                        pass
                    # Situational awareness: also write to backtests/results for Chat context (quality gate)
                    try:
                        from trading.memory.memory_store import MemoryType
                        _store = _get_memory_store()
                        _sr = results.get("sharpe_ratio")
                        _tr = results.get("total_return")
                        if _sr is not None and _tr is not None:
                            _start = data.index[0]
                            _end = data.index[-1]
                            _start_str = str(_start.date()) if hasattr(_start, 'date') else str(_start)
                            _end_str = str(_end.date()) if hasattr(_end, 'date') else str(_end)
                            _anomalous = (
                                (_sr is not None and _sr < -2) or
                                (_tr is not None and _tr < -0.8)
                            )
                            _store.add(
                                MemoryType.LONG_TERM,
                                namespace="backtests",
                                value={
                                    "strategy_name": strategy_name,
                                    "symbol": st.session_state.get("backtest_symbol", "AAPL"),
                                    "total_return": _tr,
                                    "sharpe_ratio": _sr,
                                    "max_drawdown": results.get("max_drawdown"),
                                    "win_rate": results.get("win_rate"),
                                    "start_date": _start_str,
                                    "end_date": _end_str,
                                    "timestamp": datetime.utcnow().isoformat(),
                                    **({"anomalous": True} if _anomalous else {}),
                                },
                                category="results",
                            )
                    except Exception:
                        pass
                    
                    st.session_state.backtest_results = results
                    st.session_state.backtest_strategy = strategy_name
            
            st.success("✅ Backtest complete!")
            
        except Exception as e:
            st.error(f"Backtest failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    # Display results using UI components
    if st.session_state.get('backtest_results'):
        results = st.session_state.backtest_results
        
        try:
            from trading.ui.strategy_components import render_backtest_results, render_strategy_metrics, render_trade_list
            
            # Prepare backtest results for component
            backtest_data = {
                'equity_curve': results.get('equity_curve', pd.DataFrame()),
                'returns': results.get('returns', None),
                'drawdown': results.get('drawdown', None),
                'strategy_name': st.session_state.get('backtest_strategy', 'Strategy'),
                'dates': results['equity_curve'].index if 'equity_curve' in results and not results['equity_curve'].empty else None
            }
            
            # Render backtest results (chart)
            render_backtest_results(backtest_data)
            
            # Prepare strategy stats
            strategy_stats = {
                'total_return': results.get('total_return', 0),
                'sharpe_ratio': results.get('sharpe_ratio', 0),
                'max_drawdown': results.get('max_drawdown', 0),
                'win_rate': results.get('win_rate', 0),
                'total_trades': len(results.get('trades', [])),
                'volatility': results.get('volatility', 0)
            }
            
            # Render strategy metrics
            render_strategy_metrics(strategy_stats)
            
            # Render trade list
            if 'trades' in results and len(results['trades']) > 0:
                trades_df = pd.DataFrame(results['trades'])
                render_trade_list(trades_df)
                
                # Download button
                csv = trades_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Trades",
                    data=csv,
                    file_name=f"{st.session_state.get('backtest_symbol', 'symbol')}_{st.session_state.get('backtest_strategy', 'strategy')}_trades.csv",
                    mime="text/csv"
                )
                
        except ImportError:
            # Fallback to original display code
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
                st.plotly_chart(fig, width='stretch')
            
            # Trade list
            if 'trades' in results and len(results['trades']) > 0:
                st.markdown("**Trade History:**")
                trades_df = pd.DataFrame(results['trades'])
                st.dataframe(trades_df, width='stretch')
                
                csv = trades_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Trades",
                    data=csv,
                    file_name=f"{st.session_state.get('backtest_symbol', 'symbol')}_{st.session_state.get('backtest_strategy', 'strategy')}_trades.csv",
                    mime="text/csv"
                )
        
        # Walk-Forward Analysis
        st.markdown("---")
        st.subheader("🚶 Walk-Forward Analysis")
        
        st.write("""
        Walk-forward analysis tests strategy robustness by repeatedly retraining 
        on expanding windows and testing on out-of-sample data.
        """)
        
        enable_walk_forward = st.checkbox("Enable walk-forward validation", key="enable_walk_forward")
        
        if enable_walk_forward:
            if st.session_state.loaded_data is None:
                st.warning("⚠️ Please load data first to run walk-forward analysis")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    train_window = st.slider(
                        "Training window (days)",
                        min_value=30,
                        max_value=365,
                        value=180,
                        key="wf_train_window"
                    )
                    
                    test_window = st.slider(
                        "Test window (days)",
                        min_value=7,
                        max_value=90,
                        value=30,
                        key="wf_test_window"
                    )
                
                with col2:
                    step_size = st.slider(
                        "Step size (days)",
                        min_value=7,
                        max_value=90,
                        value=30,
                        help="How far to move window each iteration",
                        key="wf_step_size"
                    )
                    
                    num_iterations = st.number_input(
                        "Number of iterations",
                        min_value=3,
                        max_value=20,
                        value=6,
                        key="wf_num_iterations"
                    )
                
                if st.button("Run Walk-Forward Analysis", key="run_walk_forward", type="primary"):
                    try:
                        validator = WalkForwardValidator()
                        
                        # Get strategy and data
                        data = st.session_state.loaded_data.copy()
                        
                        # Get selected strategy
                        if 'selected_strategy' in st.session_state and st.session_state.selected_strategy:
                            selected_strategy_name = st.session_state.selected_strategy
                        elif 'backtest_strategy' in st.session_state:
                            selected_strategy_name = st.session_state.backtest_strategy
                        else:
                            selected_strategy_name = strategy_name if 'strategy_name' in locals() else None
                        
                        if selected_strategy_name is None:
                            st.error("Please select a strategy first")
                        else:
                            try:
                                strategy_params = st.session_state.get(f'{selected_strategy_name}_params', {})
                                selected_strategy, err = _resolve_walk_forward_strategy(selected_strategy_name, strategy_params)
                                if err:
                                    st.error(err)
                                    selected_strategy = None
                            except Exception as e:
                                st.error(f"Error initializing strategy: {e}")
                                selected_strategy = None
                            
                            if selected_strategy:
                                with st.spinner("Running walk-forward validation..."):
                                    # Progress bar
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()
                                    
                                    # Progress callback
                                    def progress_callback(iteration, total):
                                        progress_bar.progress(iteration / total)
                                        status_text.text(f"Iteration {iteration}/{total}")
                                    
                                    try:
                                        results = validator.walk_forward_test(
                                            strategy=selected_strategy,
                                            data=data,
                                            train_window=train_window,
                                            test_window=test_window,
                                            step_size=step_size,
                                            num_iterations=num_iterations,
                                            progress_callback=progress_callback
                                        )
                                        
                                        progress_bar.empty()
                                        status_text.empty()
                                        
                                        st.success("✅ Walk-forward analysis complete!")
                                        
                                        # Display results
                                        st.subheader("📊 Walk-Forward Results")
                                        
                                        # Summary metrics
                                        col1, col2, col3, col4 = st.columns(4)
                                        
                                        with col1:
                                            avg_return = results.get('avg_return', 0)
                                            st.metric("Avg Return", f"{avg_return:.2%}")
                                        
                                        with col2:
                                            consistency = results.get('consistency_score', 0)
                                            st.metric("Consistency", f"{consistency:.1%}")
                                        
                                        with col3:
                                            win_rate = results.get('win_rate', 0)
                                            st.metric("Win Rate", f"{win_rate:.1%}")
                                        
                                        with col4:
                                            num_iter = results.get('num_iterations') or results.get('num_windows') or len(results.get('returns', []))
                                            st.metric("Iterations", int(num_iter) if num_iter is not None else 0)
                                        
                                        # Iteration results
                                        if 'returns' in results and results.get('returns'):
                                            iterations_df = pd.DataFrame(
                                                {
                                                    "iteration": list(range(1, len(results['returns']) + 1)),
                                                    "return": results['returns'],
                                                }
                                            )
                                            
                                            # Chart of returns by iteration
                                            fig = go.Figure()
                                            
                                            fig.add_trace(go.Bar(
                                                x=iterations_df['iteration'] if 'iteration' in iterations_df.columns else range(len(iterations_df)),
                                                y=iterations_df['return'] if 'return' in iterations_df.columns else iterations_df.iloc[:, 0],
                                                name='Return',
                                                marker_color=['green' if r > 0 else 'red' for r in (iterations_df['return'] if 'return' in iterations_df.columns else iterations_df.iloc[:, 0])]
                                            ))
                                            
                                            fig.update_layout(
                                                title='Returns by Walk-Forward Iteration',
                                                xaxis_title='Iteration',
                                                yaxis_title='Return',
                                                yaxis_tickformat='.2%',
                                                height=400
                                            )
                                            
                                            st.plotly_chart(fig, width='stretch')
                                            
                                            # Detailed results table
                                            with st.expander("📋 Detailed Results", expanded=False):
                                                st.dataframe(iterations_df, width='stretch')
                                        
                                        # Interpretation
                                        st.subheader("💡 Interpretation")
                                        
                                        consistency = results.get('consistency_score', 0)
                                        avg_return = results.get('avg_return', 0)
                                        
                                        if consistency > 0.7:
                                            st.success(f"✅ Strategy shows good consistency ({consistency:.1%})")
                                        elif consistency > 0.5:
                                            st.warning(f"⚠️ Strategy shows moderate consistency ({consistency:.1%})")
                                        else:
                                            st.error(f"❌ Strategy shows poor consistency ({consistency:.1%})")
                                        
                                        if avg_return > 0:
                                            st.info(f"Average return across all iterations: {avg_return:.2%}")
                                        else:
                                            st.warning(f"Negative average return: {avg_return:.2%}")
                                    
                                    except Exception as e:
                                        progress_bar.empty()
                                        status_text.empty()
                                        st.error(f"Error in walk-forward test: {e}")
                                        import traceback
                                        st.code(traceback.format_exc())
                            else:
                                st.error("Could not initialize strategy for walk-forward analysis")
                    
                    except ImportError:
                        st.error("Walk-forward validator not available. Make sure trading.validation.walk_forward_utils is available.")
                    except Exception as e:
                        st.error(f"Error: {e}")
                        import traceback
                        st.code(traceback.format_exc())

