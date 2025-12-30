"""
Strategy Combo Creator

This page allows users to create and test combinations of multiple trading strategies.
It integrates with the existing strategy pipeline and provides a user-friendly interface
for creating, testing, and optimizing strategy combinations.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure logging
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(page_title="Strategy Combo Creator", page_icon="ðŸ”—", layout="wide")


def main():
    """Main function for the Strategy Combo Creator page."""
    st.title("ðŸ”— Strategy Combo Creator")
    st.markdown("Create and test combinations of multiple trading strategies")

    # Sidebar configuration
    with st.sidebar:
        st.header("âš™ï¸ Configuration")

        # Strategy selection
        st.subheader("ðŸ“ˆ Strategy Selection")
        available_strategies = get_available_strategies()

        selected_strategies = st.multiselect(
            "Select strategies to combine:",
            options=available_strategies,
            default=(
                available_strategies[:2]
                if len(available_strategies) >= 2
                else available_strategies
            ),
        )

        # Combination mode
        st.subheader("ðŸ”„ Combination Mode")
        combine_modes = get_combine_modes()
        selected_mode = st.selectbox(
            "Select combination mode:", options=combine_modes, index=0
        )

        # Weights configuration
        st.subheader("âš–ï¸ Strategy Weights")
        weights = {}
        if selected_strategies:
            st.write("Configure weights for each strategy:")
            for strategy in selected_strategies:
                weight = st.slider(
                    f"Weight for {strategy}:",
                    min_value=0.1,
                    max_value=3.0,
                    value=1.0,
                    step=0.1,
                )
                weights[strategy] = weight

        # Advanced settings
        st.subheader("ðŸ”§ Advanced Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold:", min_value=0.1, max_value=1.0, value=0.6, step=0.1
        )

        min_agreement = st.slider(
            "Minimum Agreement:", min_value=0.1, max_value=1.0, value=0.5, step=0.1
        )

        smoothing_window = st.slider(
            "Smoothing Window:", min_value=1, max_value=20, value=5, step=1
        )

        enable_validation = st.checkbox("Enable Signal Validation", value=True)

    # Main content
    if not selected_strategies:
        st.warning("Please select at least one strategy to continue.")
        return

    # Data loading
    st.header("ðŸ“Š Data & Analysis")

    # Load sample data or allow user upload
    data_source = st.radio("Data Source:", ["Sample Data", "Upload Data", "Live Data"])

    if data_source == "Sample Data":
        data = load_sample_data()
        st.success("Loaded sample market data")
    elif data_source == "Upload Data":
        data = load_uploaded_data()
        if data is None:
            st.error("Please upload valid market data")
            return
    else:  # Live Data
        data = load_live_data()
        if data is None:
            st.error("Could not load live data")
            return

    # Display data preview
    with st.expander("ðŸ“‹ Data Preview"):
        st.dataframe(data.head())
        st.write(f"Data shape: {data.shape}")

    # Create strategy pipeline
    try:
        pipeline = create_strategy_pipeline(
            selected_strategies,
            selected_mode,
            weights,
            confidence_threshold,
            min_agreement,
            smoothing_window,
            enable_validation,
        )

        # Generate combined signals
        combined_signal, metadata = pipeline.generate_combined_signals(
            data, selected_strategies
        )

        # Display results
        display_results(data, combined_signal, metadata, pipeline)

        # Performance analysis
        display_performance_analysis(data, combined_signal, metadata)

        # Strategy comparison
        display_strategy_comparison(data, selected_strategies, pipeline)

        # Save combo
        display_save_combo(pipeline, selected_strategies, selected_mode, weights)

    except Exception as e:
        st.error(f"Error creating strategy pipeline: {e}")
        logger.error(f"Strategy pipeline error: {e}")


def get_available_strategies() -> List[str]:
    """Get list of available strategies."""
    try:
        from strategies.strategy_pipeline import get_strategy_names

        return get_strategy_names()
    except ImportError:
        return ["RSI", "MACD", "Bollinger", "SMA"]


def get_combine_modes() -> List[str]:
    """Get list of available combination modes."""
    try:
        from strategies.strategy_pipeline import get_combine_modes

        return get_combine_modes()
    except ImportError:
        return ["intersection", "union", "weighted", "voting", "confidence"]


def load_sample_data() -> pd.DataFrame:
    """Load sample market data."""
    # Generate sample data
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    np.random.seed(42)

    # Generate realistic price data
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))

    # Add some trend and volatility
    trend = np.linspace(0, 0.1, len(dates))
    prices = prices * (1 + trend)

    # Add volume
    volume = np.random.lognormal(10, 0.5, len(dates))

    data = pd.DataFrame(
        {
            "date": dates,
            "open": prices * (1 + np.random.normal(0, 0.005, len(dates))),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            "close": prices,
            "volume": volume,
        }
    )

    data.set_index("date", inplace=True)
    return data


def load_uploaded_data() -> Optional[pd.DataFrame]:
    """Load data uploaded by user."""
    uploaded_file = st.file_uploader(
        "Upload market data (CSV)", type=["csv"], help="Upload CSV file with OHLCV data"
    )

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)

            # Try to set date index
            date_columns = ["date", "Date", "time", "Time", "datetime", "Datetime"]
            for col in date_columns:
                if col in data.columns:
                    data[col] = pd.to_datetime(data[col])
                    data.set_index(col, inplace=True)
                    break

            # Ensure required columns exist
            required_columns = ["close", "Close"]
            if not any(col in data.columns for col in required_columns):
                st.error("Data must contain 'close' or 'Close' column")
                return None

            # Standardize column names
            column_mapping = {
                "Close": "close",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Volume": "volume",
            }
            data.rename(columns=column_mapping, inplace=True)

            return data

        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None

    return None


def load_live_data() -> Optional[pd.DataFrame]:
    """Load live market data."""
    st.info("Live data loading not implemented yet. Using sample data.")
    return load_sample_data()


def create_strategy_pipeline(
    strategies: List[str],
    mode: str,
    weights: Dict[str, float],
    confidence_threshold: float,
    min_agreement: float,
    smoothing_window: int,
    enable_validation: bool,
):
    """Create a strategy pipeline with the given configuration."""
    try:
        from strategies.strategy_pipeline import (
            CombinationConfig,
            StrategyConfig,
            StrategyPipeline,
        )

        # Create strategy configurations
        strategy_configs = []
        for strategy in strategies:
            config = StrategyConfig(
                name=strategy,
                weight=weights.get(strategy, 1.0),
                confidence_threshold=confidence_threshold,
            )
            strategy_configs.append(config)

        # Create combination configuration
        combination_config = CombinationConfig(
            mode=mode,
            min_agreement=min_agreement,
            confidence_threshold=confidence_threshold,
            smoothing_window=smoothing_window if smoothing_window > 1 else None,
            enable_validation=enable_validation,
        )

        # Create pipeline
        pipeline = StrategyPipeline(strategy_configs, combination_config)

        return pipeline

    except ImportError as e:
        st.error(f"Could not import strategy pipeline: {e}")
        raise


def display_results(
    data: pd.DataFrame,
    combined_signal: pd.Series,
    metadata: Dict[str, Any],
    pipeline: Any,
):
    """Display the results of strategy combination."""
    st.header("ðŸ“ˆ Combined Strategy Results")

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(
        ["ðŸ“Š Signal Chart", "ðŸ“‹ Signal Summary", "ðŸ” Metadata"]
    )

    with tab1:
        # Plot combined signal with price
        fig = go.Figure()

        # Add price data
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["close"],
                mode="lines",
                name="Price",
                line=dict(color="blue", width=1),
            )
        )

        # Add buy signals
        buy_signals = combined_signal[combined_signal == 1]
        if len(buy_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=data.loc[buy_signals.index, "close"],
                    mode="markers",
                    name="Buy Signal",
                    marker=dict(color="green", size=8, symbol="triangle-up"),
                )
            )

        # Add sell signals
        sell_signals = combined_signal[combined_signal == -1]
        if len(sell_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=data.loc[sell_signals.index, "close"],
                    mode="markers",
                    name="Sell Signal",
                    marker=dict(color="red", size=8, symbol="triangle-down"),
                )
            )

        fig.update_layout(
            title="Combined Strategy Signals",
            xaxis_title="Date",
            yaxis_title="Price",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Signal summary
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Signals", len(combined_signal[combined_signal != 0]))

        with col2:
            st.metric("Buy Signals", len(combined_signal[combined_signal == 1]))

        with col3:
            st.metric("Sell Signals", len(combined_signal[combined_signal == -1]))

        with col4:
            st.metric("Hold Periods", len(combined_signal[combined_signal == 0]))

        # Signal distribution
        signal_counts = combined_signal.value_counts()
        fig = px.pie(
            values=signal_counts.values,
            names=["Hold", "Buy", "Sell"],
            title="Signal Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Display metadata
        st.json(metadata)


def display_performance_analysis(
    data: pd.DataFrame, combined_signal: pd.Series, metadata: Dict[str, Any]
):
    """Display performance analysis of the combined strategy."""
    st.header("ðŸ“Š Performance Analysis")

    # Calculate basic performance metrics
    try:
        # Calculate returns
        price_returns = data["close"].pct_change()

        # Calculate strategy returns
        strategy_returns = combined_signal.shift(1) * price_returns

        # Calculate cumulative returns
        cumulative_returns = (1 + strategy_returns).cumprod()
        price_cumulative = (1 + price_returns).cumprod()

        # Performance metrics
        total_return = cumulative_returns.iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(data)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Return", f"{total_return:.2%}")

        with col2:
            st.metric("Annualized Return", f"{annualized_return:.2%}")

        with col3:
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

        with col4:
            st.metric("Max Drawdown", f"{max_drawdown:.2%}")

        # Performance chart
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=cumulative_returns,
                mode="lines",
                name="Strategy Returns",
                line=dict(color="green", width=2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=price_cumulative,
                mode="lines",
                name="Buy & Hold",
                line=dict(color="blue", width=1),
            )
        )

        fig.update_layout(
            title="Strategy Performance vs Buy & Hold",
            xaxis_title="Date",
            yaxis_title="Cumulative Returns",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error calculating performance metrics: {e}")


def display_strategy_comparison(
    data: pd.DataFrame, selected_strategies: List[str], pipeline: Any
):
    """Display comparison of individual strategies."""
    st.header("ðŸ” Strategy Comparison")

    try:
        # Generate individual strategy signals
        individual_signals = {}

        for strategy in selected_strategies:
            if strategy in pipeline.strategy_functions:
                function = pipeline.strategy_functions[strategy]
                signal = function(data)
                individual_signals[strategy] = signal

        if not individual_signals:
            st.warning("No individual strategy signals available")
            return

        # Create comparison chart
        fig = go.Figure()

        for strategy, signal in individual_signals.items():
            # Calculate strategy returns
            price_returns = data["close"].pct_change()
            strategy_returns = signal.shift(1) * price_returns
            cumulative_returns = (1 + strategy_returns).cumprod()

            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=cumulative_returns,
                    mode="lines",
                    name=strategy,
                    line=dict(width=1),
                )
            )

        fig.update_layout(
            title="Individual Strategy Performance Comparison",
            xaxis_title="Date",
            yaxis_title="Cumulative Returns",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Strategy agreement analysis
        if len(individual_signals) > 1:
            st.subheader("ðŸ¤ Strategy Agreement Analysis")

            # Calculate agreement matrix
            agreement_data = []
            strategies = list(individual_signals.keys())

            for i, strategy1 in enumerate(strategies):
                for j, strategy2 in enumerate(strategies[i + 1 :], i + 1):
                    signal1 = individual_signals[strategy1]
                    signal2 = individual_signals[strategy2]
                    agreement = (signal1 == signal2).mean()
                    agreement_data.append([strategy1, strategy2, agreement])

            if agreement_data:
                agreement_df = pd.DataFrame(
                    agreement_data, columns=["Strategy 1", "Strategy 2", "Agreement"]
                )
                st.dataframe(agreement_df)

    except Exception as e:
        st.error(f"Error in strategy comparison: {e}")


def display_save_combo(
    pipeline: Any,
    selected_strategies: List[str],
    selected_mode: str,
    weights: Dict[str, float],
):
    """Display options to save the strategy combination."""
    st.header("ðŸ’¾ Save Strategy Combo")

    # Combo name
    combo_name = st.text_input(
        "Strategy Combo Name:",
        value=f"Combo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Enter a name for this strategy combination",
    )

    # Combo description
    combo_description = st.text_area(
        "Description:",
        value=f"Combination of {', '.join(selected_strategies)} using {selected_mode} mode",
        help="Describe this strategy combination",
    )

    # Save options
    col1, col2 = st.columns(2)

    with col1:
        if st.button("ðŸ’¾ Save Combo"):
            try:
                # Save combo configuration
                combo_config = {
                    "name": combo_name,
                    "description": combo_description,
                    "strategies": selected_strategies,
                    "mode": selected_mode,
                    "weights": weights,
                    "created_at": datetime.now().isoformat(),
                    "pipeline_config": {
                        "confidence_threshold": pipeline.combination_config.confidence_threshold,
                        "min_agreement": pipeline.combination_config.min_agreement,
                        "smoothing_window": pipeline.combination_config.smoothing_window,
                        "enable_validation": pipeline.combination_config.enable_validation,
                    },
                }

                # Save to file (you can implement your own saving logic)
                import json
                import os

                # Create combos directory if it doesn't exist
                os.makedirs("data/strategy_combos", exist_ok=True)

                # Save combo
                filename = f"data/strategy_combos/{combo_name}.json"
                with open(filename, "w") as f:
                    json.dump(combo_config, f, indent=2)

                st.success(f"Strategy combo saved as {filename}")

            except Exception as e:
                st.error(f"Error saving combo: {e}")

    with col2:
        if st.button("ðŸ“Š Export Results"):
            try:
                # Export results to CSV
                results_data = {
                    "timestamp": datetime.now().isoformat(),
                    "combo_name": combo_name,
                    "strategies": selected_strategies,
                    "mode": selected_mode,
                    "weights": weights,
                }

                results_df = pd.DataFrame([results_data])
                csv = results_df.to_csv(index=False)

                st.download_button(
                    label="ðŸ“¥ Download Results CSV",
                    data=csv,
                    file_name=f"{combo_name}_results.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"Error exporting results: {e}")


if __name__ == "__main__":
    main()
