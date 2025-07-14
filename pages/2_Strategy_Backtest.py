# -*- coding: utf-8 -*-
"""Strategy Backtest Page for Evolve Trading Platform."""

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# Page config
st.set_page_config(page_title="Strategy Backtest", page_icon="📈", layout="wide")


def plot_strategy_result_plotly(strategy_result: dict):
    """Plot strategy results using Plotly for consistency across app."""
    try:
        if not strategy_result or "data" not in strategy_result:
            st.warning("No strategy data available for plotting")
            return

        data = strategy_result["data"]

        # Create Plotly figure
        fig = go.Figure()

        # Add price data
        if "close" in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["close"],
                    mode="lines",
                    name="Price",
                    line=dict(color="blue", width=2),
                )
            )

        # Add strategy signals
        if "signal" in data.columns:
            buy_signals = data[data["signal"] == 1]
            sell_signals = data[data["signal"] == -1]

            if not buy_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals["close"],
                        mode="markers",
                        name="Buy Signal",
                        marker=dict(color="green", size=10, symbol="triangle-up"),
                    )
                )

            if not sell_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals["close"],
                        mode="markers",
                        name="Sell Signal",
                        marker=dict(color="red", size=10, symbol="triangle-down"),
                    )
                )

        # Add cumulative returns if available
        if "strategy_cumulative_returns" in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["strategy_cumulative_returns"],
                    mode="lines",
                    name="Strategy Returns",
                    line=dict(color="orange", width=2),
                    yaxis="y2",
                )
            )

        # Update layout
        fig.update_layout(
            title="Strategy Backtest Results",
            xaxis_title="Date",
            yaxis_title="Price",
            yaxis2=dict(title="Cumulative Returns", overlaying="y", side="right"),
            height=600,
            showlegend=True,
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error plotting strategy results: {e}")


def run_backtest_strategy(
    strategy_name: str,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    params: dict,
) -> dict:
    """Run backtest for a given strategy."""
    try:
        # Validate parameters
        for param, value in params.items():
            if isinstance(value, (int, float)) and value <= 0:
                return {
                    "success": False,
                    "error": f"Parameter {param} must be > 0, got {value}",
                }

        # Import strategy modules
        if strategy_name == "Bollinger Bands":
            from trading.strategies.bollinger_strategy import (
                BollingerConfig,
                BollingerStrategy,
            )

            config = BollingerConfig(**params)
            strategy = BollingerStrategy(config)
        elif strategy_name == "Moving Average Crossover":
            from trading.strategies.sma_strategy import SMAConfig, SMAStrategy

            config = SMAConfig(**params)
            strategy = SMAStrategy(config)
        elif strategy_name == "RSI Mean Reversion":
            from trading.strategies.rsi_strategy import RSIConfig, RSIStrategy

            config = RSIConfig(**params)
            strategy = RSIStrategy(config)
        elif strategy_name == "MACD Momentum":
            from trading.strategies.macd_strategy import MACDConfig, MACDStrategy

            config = MACDConfig(**params)
            strategy = MACDStrategy(config)
        else:
            return {
                "success": False,
                "error": f"Strategy {strategy_name} not implemented",
            }

        # Load data (placeholder - implement with real data provider)
        # For now, generate sample data
        dates = pd.date_range(start_date, end_date, freq="D")
        sample_data = pd.DataFrame(
            {
                "close": 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
                "volume": np.random.randint(1000, 10000, len(dates)),
                "open": 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
                "high": 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) + 2,
                "low": 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) - 2,
            },
            index=dates,
        )

        # Generate signals
        signals = strategy.generate_signals(sample_data)

        # Calculate returns
        sample_data["returns"] = sample_data["close"].pct_change()
        sample_data["strategy_returns"] = (
            signals["signal"].shift(1) * sample_data["returns"]
        )
        sample_data["cumulative_returns"] = (1 + sample_data["returns"]).cumprod()
        sample_data["strategy_cumulative_returns"] = (
            1 + sample_data["strategy_returns"]
        ).cumprod()

        # Calculate metrics
        total_return = sample_data["strategy_cumulative_returns"].iloc[-1] - 1
        sharpe_ratio = (
            sample_data["strategy_returns"].mean()
            / sample_data["strategy_returns"].std()
            * np.sqrt(252)
        )
        max_drawdown = (
            sample_data["strategy_cumulative_returns"]
            / sample_data["strategy_cumulative_returns"].cummax()
            - 1
        ).min()

        return {
            "success": True,
            "data": sample_data,
            "metrics": {
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": len(sample_data[sample_data["strategy_returns"] > 0])
                / len(sample_data[sample_data["strategy_returns"] != 0]),
            },
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    st.title("📈 Strategy Backtest")
    st.markdown("Backtest your trading strategies with historical data")

    # Sidebar for backtest configuration
    with st.sidebar:
        st.header("Backtest Configuration")

        # Strategy selection with multiselect for strategy combos
        strategy_list = [
            "Bollinger Bands",
            "Moving Average Crossover",
            "RSI Mean Reversion",
            "MACD Momentum",
        ]
        selected_strategies = st.multiselect(
            "Select Strategies",
            strategy_list,
            default=[strategy_list[0]] if strategy_list else [],
        )

        # Use first selected strategy for single strategy backtest
        strategy = selected_strategies[0] if selected_strategies else None

        # Symbol input
        symbol = st.text_input("Symbol", value="AAPL").upper()

        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date", value=datetime.now() - timedelta(days=365)
            )
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())

        # Strategy parameters
        st.subheader("Strategy Parameters")

        params = {}
        if strategy == "Bollinger Bands":
            params["window"] = st.slider("Period", 10, 50, 20)
            params["num_std"] = st.slider("Standard Deviation", 1.0, 3.0, 2.0)
        elif strategy == "Moving Average Crossover":
            params["short_period"] = st.slider("Fast Period", 5, 20, 10)
            params["long_period"] = st.slider("Slow Period", 20, 50, 30)
        elif strategy == "RSI Mean Reversion":
            params["period"] = st.slider("RSI Period", 10, 30, 14)
            params["oversold"] = st.slider("Oversold", 20, 40, 30)
            params["overbought"] = st.slider("Overbought", 60, 80, 70)
        elif strategy == "MACD Momentum":
            params["fast_period"] = st.slider("Fast Period", 8, 15, 12)
            params["slow_period"] = st.slider("Slow Period", 20, 30, 26)
            params["signal_period"] = st.slider("Signal Period", 5, 15, 9)

        # Run backtest button
        run_backtest = st.button("🚀 Run Backtest", type="primary")

    # Main content
    if run_backtest:
        with st.spinner("Running backtest..."):
            result = run_backtest_strategy(
                strategy, symbol, start_date, end_date, params
            )

            if result["success"]:
                st.success("✅ Backtest completed successfully!")

                # Display metrics
                metrics = result["metrics"]
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Return", f"{metrics['total_return']:.2%}")
                with col2:
                    st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                with col3:
                    st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
                with col4:
                    st.metric("Win Rate", f"{metrics['win_rate']:.2%}")

                # Plot results using Plotly
                plot_strategy_result_plotly(result)

            else:
                st.error(f"❌ Backtest failed: {result['error']}")
    else:
        st.info("Configure your backtest parameters and click 'Run Backtest' to start.")

        # Show example results
        st.subheader("📈 Example Backtest Results")
        st.info(
            "Real backtest results will appear here after running a backtest with actual market data."
        )


if __name__ == "__main__":
    main()
