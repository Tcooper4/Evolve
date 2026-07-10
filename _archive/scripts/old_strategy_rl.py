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

    st.write("Train an AI agent to learn optimal trading strategies")

    # RL configuration
    col1, col2 = st.columns(2)

    with col1:
        episodes = st.slider("Training Episodes", 100, 5000, 1000)
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.001, 0.01, 0.1],
            value=0.001
        )

    with col2:
        reward_type = st.selectbox(
            "Reward Function",
            ["Sharpe Ratio", "Total Return", "Risk-Adjusted Return"]
        )

        gamma = st.slider("Discount Factor (γ)", 0.9, 0.99, 0.95, 0.01)

    if st.button("🚀 Train RL Agent", type="primary"):
        if 'forecast_data' not in st.session_state and 'loaded_data' not in st.session_state:
            st.error("Please load data first")
        else:
            try:
                from rl.rl_trader import RLTrader

                # Use forecast_data if available, otherwise use loaded_data
                if 'forecast_data' in st.session_state:
                    data = st.session_state.forecast_data
                else:
                    data = st.session_state.loaded_data

                if data is None:
                    st.error("No data available. Please load data first.")
                else:
                    with st.spinner("Training RL agent..."):
                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # Initialize RL agent
                        agent = RLTrader(
                            learning_rate=learning_rate,
                            gamma=gamma,
                            reward_function=reward_type.lower().replace(' ', '_')
                        )

                        # Training loop
                        rewards_history = []

                        for episode in range(episodes):
                            # Train one episode
                            reward = agent.train_episode(data)
                            rewards_history.append(reward)

                            # Update progress
                            if episode % 10 == 0:
                                progress_bar.progress((episode + 1) / episodes)
                                status_text.text(f"Episode {episode+1}/{episodes} - Reward: {reward:.2f}")

                        progress_bar.empty()
                        status_text.empty()

                        st.success("✅ RL agent training complete!")

                        # Show results
                        st.subheader("📊 Training Results")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Final Reward", f"{rewards_history[-1]:.2f}")
                        with col2:
                            avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
                            st.metric("Avg Reward (last 100)", f"{avg_reward:.2f}")
                        with col3:
                            if len(rewards_history) > 0 and rewards_history[0] != 0:
                                improvement = ((rewards_history[-1] / rewards_history[0]) - 1) * 100
                                st.metric("Improvement", f"{improvement:+.1f}%")
                            else:
                                st.metric("Improvement", "N/A")

                        # Reward curve
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=rewards_history,
                            name='Episode Reward',
                            line=dict(color='blue', width=1)
                        ))

                        # Add moving average
                        window = 50
                        if len(rewards_history) >= window:
                            moving_avg = pd.Series(rewards_history).rolling(window).mean()
                            fig.add_trace(go.Scatter(
                                y=moving_avg,
                                name=f'MA({window})',
                                line=dict(color='red', width=2)
                            ))

                        fig.update_layout(
                            title='Training Progress',
                            xaxis_title='Episode',
                            yaxis_title='Reward'
                        )

                        st.plotly_chart(fig, width='stretch')

                        # Store agent in session state
                        st.session_state.rl_agent = agent
                        st.session_state.rl_rewards_history = rewards_history

                        # Test the trained agent
                        st.subheader("🧪 Test Trained Agent")

                        if st.button("Run Backtest with RL Agent"):
                            try:
                                test_results = agent.backtest(data)

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Return", f"{test_results.get('total_return', 0):.2%}")
                                with col2:
                                    st.metric("Sharpe Ratio", f"{test_results.get('sharpe_ratio', 0):.2f}")
                                with col3:
                                    st.metric("Max Drawdown", f"{test_results.get('max_drawdown', 0):.2%}")

                                # Equity curve
                                if 'dates' in test_results and 'equity_curve' in test_results:
                                    fig_equity = go.Figure()
                                    fig_equity.add_trace(go.Scatter(
                                        x=test_results['dates'],
                                        y=test_results['equity_curve'],
                                        name='RL Agent',
                                        line=dict(color='green')
                                    ))

                                    fig_equity.update_layout(
                                        title='RL Agent Performance',
                                        xaxis_title='Date',
                                        yaxis_title='Portfolio Value'
                                    )

                                    st.plotly_chart(fig_equity, width='stretch')

                                st.session_state.rl_backtest_results = test_results
                            except Exception as e:
                                st.error(f"Error running backtest: {e}")
                                import traceback
                                st.code(traceback.format_exc())

                        # Save model
                        if st.button("💾 Save RL Agent"):
                            try:
                                agent.save_model('rl_agent.pkl')
                                st.success("Agent saved!")
                            except Exception as e:
                                st.error(f"Error saving agent: {e}")

            except ImportError:
                st.error("RL Trader not available. Please ensure the rl module is properly installed.")
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())

    # Display saved agent info if available
    if 'rl_agent' in st.session_state:
        st.markdown("---")
        st.subheader("💾 Saved RL Agent")
        st.info("RL Agent is loaded and ready to use. You can run backtests or continue training.")

        if 'rl_backtest_results' in st.session_state:
            st.markdown("**Last Backtest Results:**")
            results = st.session_state.rl_backtest_results
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", f"{results.get('total_return', 0):.2%}")
            with col2:
                st.metric("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}")
            with col3:
                st.metric("Max Drawdown", f"{results.get('max_drawdown', 0):.2%}")
            with col4:
                st.metric("Win Rate", f"{results.get('win_rate', 0):.2%}")

render_page_assistant("Strategy Testing")

