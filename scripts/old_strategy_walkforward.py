import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os, sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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
                # Strategy selection for walk-forward (use full registry for consistency)
                st.markdown("**Strategy Selection:**")
                strategy_for_wf = st.selectbox(
                    "Select Strategy",
                    list(STRATEGY_REGISTRY.keys()),
                    key="wf_strategy"
                )

                run_walk_forward = st.button(
                    "🚀 Run Walk-Forward Analysis",
                    type="primary"
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
                        evaluator = BacktestEvaluator(data, initial_cash=initial_capital)

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

                            # AGENT_MEMORY_LAYER: Persist walk-forward analysis outcome (long-term)
                            try:
                                from trading.memory.memory_store import MemoryType

                                _get_memory_store().add(
                                    MemoryType.LONG_TERM,
                                    namespace="StreamlitStrategyTesting",
                                    category="walk_forward_analysis",
                                    key=f"{strategy_for_wf}:{datetime.utcnow().isoformat()}",
                                    value={
                                        "strategy": strategy_for_wf,
                                        "window_size": window_size,
                                        "step_size": step_size,
                                        "test_size": test_size,
                                        "windows": len(window_results),
                                        "summary": {
                                            "return_mean": float(results_df["return"].mean()) if "return" in results_df else None,
                                            "sharpe_mean": float(results_df["sharpe"].mean()) if "sharpe" in results_df else None,
                                            "max_drawdown_min": float(results_df["max_drawdown"].min()) if "max_drawdown" in results_df else None,
                                        },
                                    },
                                    metadata={"source": "pages/3_Strategy_Testing.py walk-forward"},
                                )
                            except Exception:
                                pass

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
                            st.plotly_chart(fig)

                            # Results table
                            st.dataframe(results_df)

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
                    list(STRATEGY_REGISTRY.keys()),
                    key="mc_strategy"
                )

                run_monte_carlo = st.button(
                    "🚀 Run Monte Carlo Simulation",
                    type="primary"
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
                            for percentile_key, values in simulator.percentiles.items():
                                # Handle percentile key format (e.g., "P5", "P50", "P95", "Mean")
                                if percentile_key == "Mean":
                                    label = "Mean"
                                elif percentile_key.startswith("P"):
                                    # Extract number from "P5" -> 5, "P50" -> 50, etc.
                                    try:
                                        percentile_num = int(percentile_key[1:])
                                        label = f"{percentile_num}th Percentile"
                                    except ValueError:
                                        label = f"{percentile_key} Percentile"
                                else:
                                    # Try to convert to float if it's a numeric string
                                    try:
                                        percentile_num = float(percentile_key) * 100
                                        label = f"{percentile_num:.0f}th Percentile"
                                    except (ValueError, TypeError):
                                        label = f"{percentile_key} Percentile"

                                fig.add_trace(go.Scatter(
                                    x=simulated_paths.index,
                                    y=values,
                                    mode='lines',
                                    name=label,
                                    line=dict(width=2, dash='dash')
                                ))

                        fig.update_layout(
                            title="Monte Carlo Simulation Paths",
                            xaxis_title="Period",
                            yaxis_title="Portfolio Value ($)",
                            height=500
                        )
                        st.plotly_chart(fig)

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
                list(STRATEGY_REGISTRY.keys()),
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
                type="primary"
            )

            if run_sensitivity:
                st.info("Sensitivity analysis implementation in progress. This will test multiple parameter combinations and display results.")

        else:  # Parameter Optimization
            st.subheader("⚙️ Parameter Optimization")
            st.markdown("Find optimal strategy parameters using optimization algorithms")

            st.info("Parameter optimization helps find the best parameter values for your strategy. This feature will be enhanced with genetic algorithms and Bayesian optimization.")

            strategy_for_opt = st.selectbox(
                "Select Strategy",
                list(STRATEGY_REGISTRY.keys()),
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
                type="primary"
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
        elif indicator == "MACD":
            code_lines.append(f"# Calculate MACD")
            code_lines.append(f"ema_12 = data_lower['close'].ewm(span=12).mean()")
            code_lines.append(f"ema_26 = data_lower['close'].ewm(span=26).mean()")
            code_lines.append(f"macd = ema_12 - ema_26")
            code_lines.append(f"signal = macd.ewm(span=9).mean()")
            if operator == ">":
                code_lines.append(f"entry_conditions.append((macd > signal).to_frame('macd_cond_{i}'))")
            elif operator == "<":
                code_lines.append(f"entry_conditions.append((macd < signal).to_frame('macd_cond_{i}'))")
            else:
                code_lines.append(f"entry_conditions.append((macd {operator} signal).to_frame('macd_cond_{i}'))")
        elif indicator == "EMA":
            period = value if isinstance(value, (int, float)) else 20
            code_lines.append(f"# Calculate EMA")
            code_lines.append(f"ema_{int(period)} = data_lower['close'].ewm(span={int(period)}).mean()")
            code_lines.append(f"entry_conditions.append((data_lower['close'] {operator} ema_{int(period)}).to_frame('ema_cond_{i}'))")
        elif indicator == "Bollinger Bands":
            period = 20
            std_dev = 2
            code_lines.append(f"# Calculate Bollinger Bands")
            code_lines.append(f"bb_period = {period}")
            code_lines.append(f"bb_std = {std_dev}")
            code_lines.append(f"sma_bb = data_lower['close'].rolling(bb_period).mean()")
            code_lines.append(f"std_bb = data_lower['close'].rolling(bb_period).std()")
            code_lines.append(f"bb_upper = sma_bb + (std_bb * bb_std)")
            code_lines.append(f"bb_lower = sma_bb - (std_bb * bb_std)")
            if operator == ">":
                code_lines.append(f"entry_conditions.append((data_lower['close'] > bb_upper).to_frame('bb_cond_{i}'))")
            elif operator == "<":
                code_lines.append(f"entry_conditions.append((data_lower['close'] < bb_lower).to_frame('bb_cond_{i}'))")
            else:
                code_lines.append(f"entry_conditions.append((data_lower['close'] {operator} {value}).to_frame('bb_cond_{i}'))")
        else:
            # Generic condition - try to use the indicator name as a column
            code_lines.append(f"# {indicator} {operator} {value}")
            if indicator.lower() in ['close', 'open', 'high', 'low', 'volume']:
                code_lines.append(f"entry_conditions.append((data_lower['{indicator.lower()}'] {operator} {value}).to_frame('cond_{i}'))")
            else:
                st.warning(f"⚠️ Strategy code generation: Unknown indicator '{indicator}'. Using default condition.")
                code_lines.append(f"# Note: Indicator '{indicator}' not recognized - using placeholder")
                code_lines.append(f"entry_conditions.append((data_lower['close'] > 0).to_frame('cond_{i}'))  # Placeholder - implement {indicator} logic")

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
            code_lines.append(f"# Note: Profit % exit requires entry price tracking during backtesting")
            code_lines.append(f"# This condition should be evaluated in the backtester with entry_price context")
            code_lines.append(f"profit_target = {value} / 100.0")
            code_lines.append(f"# exit_conditions.append((current_price >= entry_price * (1 + profit_target)).to_frame('profit_cond_{i}'))")
            code_lines.append(f"# Note: Implement profit target logic in backtester with entry_price tracking")
        elif indicator == "Loss %":
            code_lines.append(f"# Stop loss exit")
            code_lines.append(f"# Note: Loss % exit requires entry price tracking during backtesting")
            code_lines.append(f"# This condition should be evaluated in the backtester with entry_price context")
            code_lines.append(f"stop_loss = {value} / 100.0")
            code_lines.append(f"# exit_conditions.append((current_price <= entry_price * (1 - stop_loss)).to_frame('loss_cond_{i}'))")
            code_lines.append(f"# Note: Implement stop loss logic in backtester with entry_price tracking")
        elif indicator == "RSI":
            code_lines.append(f"# RSI exit condition")
            code_lines.append(f"rsi_period = 14")
            code_lines.append(f"delta = data_lower['close'].diff()")
            code_lines.append(f"gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()")
            code_lines.append(f"loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()")
            code_lines.append(f"rs = gain / loss")
            code_lines.append(f"rsi = 100 - (100 / (1 + rs))")
            code_lines.append(f"exit_conditions.append((rsi {operator} {value}).to_frame('cond_{i}'))")
        elif indicator == "MACD":
            code_lines.append(f"# MACD exit condition")
            code_lines.append(f"ema_12 = data_lower['close'].ewm(span=12).mean()")
            code_lines.append(f"ema_26 = data_lower['close'].ewm(span=26).mean()")
            code_lines.append(f"macd = ema_12 - ema_26")
            code_lines.append(f"signal = macd.ewm(span=9).mean()")
            if operator == ">":
                code_lines.append(f"exit_conditions.append((macd < signal).to_frame('macd_exit_cond_{i}'))")
            elif operator == "<":
                code_lines.append(f"exit_conditions.append((macd > signal).to_frame('macd_exit_cond_{i}'))")
            else:
                code_lines.append(f"exit_conditions.append((macd {operator} signal).to_frame('macd_exit_cond_{i}'))")
        elif indicator == "SMA":
            period = value if isinstance(value, (int, float)) else 20
            code_lines.append(f"# SMA exit condition")
            code_lines.append(f"sma_exit = data_lower['close'].rolling({int(period)}).mean()")
            code_lines.append(f"exit_conditions.append((data_lower['close'] {operator} sma_exit).to_frame('sma_exit_cond_{i}'))")
        else:
            code_lines.append(f"# {indicator} {operator} {value}")
            if indicator.lower() in ['close', 'open', 'high', 'low', 'volume']:
                code_lines.append(f"exit_conditions.append((data_lower['{indicator.lower()}'] {operator} {value}).to_frame('exit_cond_{i}'))")
            else:
                st.warning(f"⚠️ Strategy code generation: Unknown exit indicator '{indicator}'. Using default condition.")
                code_lines.append(f"# Note: Exit indicator '{indicator}' not recognized - using placeholder")
                code_lines.append(f"exit_conditions.append((data_lower['close'] > 0).to_frame('exit_cond_{i}'))  # Placeholder - implement {indicator} logic")

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

# TAB 7: AI Strategy Research
