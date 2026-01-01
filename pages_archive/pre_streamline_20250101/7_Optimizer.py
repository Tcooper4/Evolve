"""
Optimizer Dashboard.

This module provides a Streamlit dashboard for the optimization framework,
allowing users to:
1. Select optimization methods (Grid, Bayesian, Genetic)
2. Configure optimization parameters
3. Visualize optimization results
4. Compare different optimization runs
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# Configure logging
logger = logging.getLogger(__name__)

# Import configuration system
try:
    from utils.config_loader import config

    CONFIG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Config loader import failed: {e}")
    CONFIG_AVAILABLE = False

# Import from consolidated trading.optimization module
try:
    from trading.optimization import OptimizationVisualizer, StrategyOptimizer

    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Optimization module import failed: {e}")
    OPTIMIZATION_AVAILABLE = False
    st.session_state["status"] = "fallback activated"

# Import AgentHub for unified agent routing
try:
    from core.agent_hub import AgentHub

    AGENT_HUB_AVAILABLE = True
except ImportError as e:
    logger.warning(f"AgentHub import failed: {e}")
    AGENT_HUB_AVAILABLE = False

try:
    from trading.agents.strategy_switcher import StrategySwitcher
except ImportError as e:
    logger.warning(f"StrategySwitcher import failed: {e}")
    StrategySwitcher = None

try:
    from trading.utils.memory_logger import MemoryLogger
except ImportError as e:
    logger.warning(f"MemoryLogger import failed: {e}")
    MemoryLogger = None

# Import data providers
try:
    from trading.data.providers import load_data

    DATA_PROVIDERS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Data providers import failed: {e}")
    DATA_PROVIDERS_AVAILABLE = False


def get_available_symbols() -> list:
    """Get available symbols from configuration or user input."""
    if CONFIG_AVAILABLE:
        # In production, this would come from a database or API
        # For now, we'll use a dynamic approach
        return []
    return []


def get_user_symbols() -> list:
    """Get symbols from user input or session state."""
    if "user_symbols" in st.session_state:
        return st.session_state["user_symbols"]
    return []


def load_strategy_data(symbol: Optional[str] = None) -> pd.DataFrame:
    """Load strategy performance data for optimization.

    Args:
        symbol: Optional symbol to load data for

    Returns:
        DataFrame with OHLCV data for strategy optimization

    Raises:
        ValueError: If no symbol provided or data unavailable
    """
    if not symbol:
        raise ValueError("Symbol is required for data loading")

    if not DATA_PROVIDERS_AVAILABLE:
        raise RuntimeError(
            "Data providers not available. Please check data source configuration."
        )

    try:
        # Get date range from configuration
        if CONFIG_AVAILABLE:
            start_date, end_date = config.get_date_range()
            data_source = config.get("data.default_source", "auto")
            interval = config.get("data.default_interval", "1d")
        else:
            # Fallback to dynamic defaults
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            start_date = start_date.strftime("%Y-%m-%d")
            end_date = end_date.strftime("%Y-%m-%d")
            data_source = "auto"
            interval = "1d"

        # Load data using the data providers
        data = load_data(
            symbol=symbol,
            source=data_source,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )

        if data.empty:
            raise ValueError(f"No data available for symbol: {symbol}")

        # Normalize column names
        data = _normalize_column_names(data)

        logger.info(f"Loaded {len(data)} rows of data for {symbol}")
        return data

    except Exception as e:
        logger.error(f"Error loading strategy data: {str(e)}")
        raise RuntimeError(f"Failed to load data for {symbol}: {str(e)}")


def _normalize_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase for consistency.

    Args:
        data: DataFrame with market data

    Returns:
        DataFrame with normalized column names
    """
    try:
        # Create a copy to avoid modifying the original
        normalized_data = data.copy()

        # Define column name mappings
        column_mappings = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj_close",
            "Date": "date",
            "Datetime": "datetime",
        }

        # Rename columns if they exist
        for old_name, new_name in column_mappings.items():
            if old_name in normalized_data.columns:
                normalized_data = normalized_data.rename(columns={old_name: new_name})

        # Also handle any remaining uppercase columns
        normalized_data.columns = [col.lower() for col in normalized_data.columns]

        logger.info(f"Normalized column names: {list(normalized_data.columns)}")
        return normalized_data

    except Exception as e:
        logger.error(f"Error normalizing column names: {str(e)}")
        return data


def main():
    st.title("Strategy Optimizer")

    # Initialize configuration
    if not CONFIG_AVAILABLE:
        st.warning("Configuration system not available. Using fallback settings.")

    # Initialize AgentHub if available
    if AGENT_HUB_AVAILABLE and "agent_hub" not in st.session_state:
        st.session_state["agent_hub"] = AgentHub()

    # Natural Language Input Section
    if AGENT_HUB_AVAILABLE:
        st.subheader("🤖 AI Agent Interface")
        st.markdown("Ask for optimization help or request specific optimizations:")

        user_prompt = st.text_area(
            "What optimization would you like to perform?",
            placeholder="e.g., 'Optimize RSI strategy for AAPL' or 'Find best parameters for MACD strategy'",
            height=100,
        )

        if st.button("🚀 Process with AI Agent"):
            if user_prompt:
                with st.spinner("Processing optimization request..."):
                    try:
                        agent_hub = st.session_state["agent_hub"]
                        response = agent_hub.route(user_prompt)

                        st.subheader("🤖 AI Response")
                        st.write(response["content"])

                        if response["type"] == "fallback":
                            st.warning("Using fallback optimization interface")

                    except Exception as e:
                        st.error(f"Failed to process request: {e}")
                        logger.error(f"AgentHub error: {e}")
            else:
                st.warning("Please enter a prompt to process.")

        st.divider()

    # Check if optimization is available
    if not OPTIMIZATION_AVAILABLE:
        st.error("Optimization module not available")
        st.info(
            "Please check that the trading.optimization module is properly installed."
        )
        return

    # Initialize components
    try:
        StrategySwitcher()
        MemoryLogger()

        # Get optimizer config from configuration
        if CONFIG_AVAILABLE:
            opt_settings = config.get_optimization_settings()
            optimizer_config = {
                "name": "strategy_optimizer",
                "optimizer_type": opt_settings["default_optimizer"],
                "n_initial_points": opt_settings["initial_points"],
                "n_iterations": opt_settings["max_iterations"],
                "primary_metric": opt_settings["primary_metric"],
            }
        else:
            optimizer_config = {
                "name": "strategy_optimizer",
                "optimizer_type": "bayesian",
                "n_initial_points": 10,
                "n_iterations": 50,
                "primary_metric": "sharpe_ratio",
            }

        optimizer = StrategyOptimizer(optimizer_config)
        logger.info("StrategyOptimizer initialized successfully")

    except Exception as e:
        st.error(f"Failed to initialize optimization components: {e}")
        logger.error(f"Optimization initialization error: {e}")
        st.session_state["status"] = "fallback activated"
        return

    # Sidebar configuration
    st.sidebar.header("Optimization Settings")

    # Data selection
    st.sidebar.subheader("📊 Data Selection")

    # Symbol input (no hardcoded values)
    symbol_input = st.sidebar.text_input(
        "Enter Symbol",
        placeholder="e.g., AAPL, MSFT, GOOGL",
        help="Enter any valid stock symbol",
    )

    # Data source selection
    if DATA_PROVIDERS_AVAILABLE:
        if CONFIG_AVAILABLE:
            available_sources = config.get(
                "data.available_sources", ["auto", "yfinance", "alpha_vantage"]
            )
            default_source = config.get("data.default_source", "auto")
        else:
            available_sources = ["auto", "yfinance", "alpha_vantage"]
            default_source = "auto"

        data_source = st.sidebar.selectbox(
            "Data Source",
            available_sources,
            index=available_sources.index(default_source),
        )
    else:
        st.sidebar.error("Data providers not available. Please configure data sources.")
        return

    # Date range selection (dynamic)
    if CONFIG_AVAILABLE:
        lookback_days = config.get("data.default_lookback_days", 365)
    else:
        lookback_days = 365

    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date_input = st.date_input(
            "Start Date", value=start_date.date(), max_value=datetime.now().date()
        )
    with col2:
        end_date_input = st.date_input(
            "End Date", value=end_date.date(), max_value=datetime.now().date()
        )

    # Load data button
    if st.sidebar.button("🔄 Load Data"):
        if not symbol_input:
            st.sidebar.error("Please enter a symbol to load data.")
            return

        with st.spinner("Loading market data..."):
            try:
                data = load_data(
                    symbol=symbol_input,
                    source=data_source,
                    start_date=start_date_input.strftime("%Y-%m-%d"),
                    end_date=end_date_input.strftime("%Y-%m-%d"),
                    interval=(
                        config.get("data.default_interval", "1d")
                        if CONFIG_AVAILABLE
                        else "1d"
                    ),
                )
                if not data.empty:
                    # Normalize column names
                    data = _normalize_column_names(data)
                    st.session_state["optimization_data"] = data
                    st.session_state["current_symbol"] = symbol_input
                    st.success(f"Loaded {len(data)} rows for {symbol_input}")
                else:
                    st.error(f"No data found for {symbol_input}")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                logger.error(f"Data loading error: {e}")

    # Strategy selection
    st.sidebar.subheader("📈 Strategy Selection")
    strategy = st.sidebar.selectbox(
        "Select Strategy", ["RSI", "MACD", "Bollinger", "SMA"]
    )

    # Optimizer selection
    optimizer_type = st.sidebar.selectbox(
        "Select Optimizer", optimizer.get_available_optimizers()
    )

    # Parameter space configuration
    st.sidebar.subheader("⚙️ Parameter Space")
    param_space = optimizer.get_strategy_param_space(strategy)

    # Display parameter ranges
    for param, range_info in param_space.items():
        st.sidebar.text(f"{param}: {range_info}")

    # Optimization settings (dynamic from config)
    st.sidebar.subheader("🔧 Optimization Settings")

    if optimizer_type == "Grid":
        n_jobs = st.sidebar.slider("Number of Jobs", 1, 8, 4)
        settings = {"n_jobs": n_jobs}

    elif optimizer_type == "Bayesian":
        if CONFIG_AVAILABLE:
            opt_settings = config.get_optimization_settings()
            default_initial = opt_settings["initial_points"]
            default_iterations = opt_settings["max_iterations"]
        else:
            default_initial = 10
            default_iterations = 50

        n_initial_points = st.sidebar.slider(
            "Initial Random Points", 5, 20, default_initial
        )
        n_iterations = st.sidebar.slider(
            "Optimization Iterations", 10, 100, default_iterations
        )
        settings = {"n_initial_points": n_initial_points, "n_iterations": n_iterations}

    elif optimizer_type == "Genetic":
        population_size = st.sidebar.slider("Population Size", 20, 200, 100)
        n_generations = st.sidebar.slider("Number of Generations", 10, 100, 50)
        mutation_prob = st.sidebar.slider("Mutation Probability", 0.0, 1.0, 0.2)
        crossover_prob = st.sidebar.slider("Crossover Probability", 0.0, 1.0, 0.8)
        settings = {
            "population_size": population_size,
            "n_generations": n_generations,
            "mutation_prob": mutation_prob,
            "crossover_prob": crossover_prob,
        }

    # Check if data is loaded
    if (
        "optimization_data" not in st.session_state
        or st.session_state["optimization_data"].empty
    ):
        st.sidebar.warning("No data loaded. Please load market data first.")
        data = None
        current_symbol = None
    else:
        data = st.session_state["optimization_data"]
        current_symbol = st.session_state.get("current_symbol", "UNKNOWN")

    # Display data info
    if data is not None and not data.empty:
        st.sidebar.subheader("📋 Data Info")
        st.sidebar.text(f"Symbol: {current_symbol}")
        st.sidebar.text(f"Rows: {len(data)}")
        st.sidebar.text(
            f"Date Range: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}"
        )

        # Safely get latest close price
        if "close" in data.columns:
            latest_close = data["close"].iloc[-1]
            st.sidebar.text(f"Latest Close: ${latest_close:.2f}")
        elif "Close" in data.columns:
            latest_close = data["Close"].iloc[-1]
            st.sidebar.text(f"Latest Close: ${latest_close:.2f}")
        else:
            st.sidebar.text("Latest Close: N/A")

    # Run optimization
    if st.sidebar.button("🚀 Start Optimization"):
        if data is None or data.empty:
            st.sidebar.error("Please load market data before running optimization.")

        with st.spinner("Running optimization..."):
            try:
                results = optimizer.optimize_strategy(
                    strategy=strategy,
                    optimizer_type=optimizer_type,
                    param_space=param_space,
                    training_data=data,
                    **settings,
                )

                # Display results
                OptimizationVisualizer.display_optimization_summary(results)

                # Save results
                if st.sidebar.button("💾 Save Results"):
                    save_path = f"optimization_results/{strategy}_{optimizer_type}.json"
                    optimizer.save_optimization_results(results, save_path)
                    st.success(f"Results saved to {save_path}")
            except Exception as e:
                st.error(f"Optimization failed: {str(e)}")
                logger.error(f"Optimization error: {e}")

    # Load previous results
    st.sidebar.subheader("📁 Load Previous Results")
    result_files = [
        f for f in os.listdir("optimization_results") if f.endswith(".json")
    ]

    if result_files:
        selected_file = st.sidebar.selectbox("Select Results File", result_files)

        if st.sidebar.button("📂 Load Results"):
            try:
                results = optimizer.load_optimization_results(
                    f"optimization_results/{selected_file}"
                )
                OptimizationVisualizer.display_optimization_summary(results)
            except Exception as e:
                st.error(f"Failed to load results: {str(e)}")

    # Main content area
    st.subheader("📊 Data Preview")

    if data is not None and not data.empty:
        # Get display settings from configuration
        if CONFIG_AVAILABLE:
            display_settings = config.get_display_settings()
            chart_days = display_settings["chart_days"]
            table_rows = display_settings["table_rows"]
            show_volatility = display_settings["show_volatility"]
            trading_days = config.get("trading.trading_days_per_year", 252)
        else:
            chart_days = 100
            table_rows = 20
            show_volatility = True
            trading_days = 252

        # Display data statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Data Points", len(data))
        with col2:
            st.metric(
                "Date Range",
                f"{data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}",
            )
        with col3:
            # Safely get latest close price
            if "close" in data.columns:
                latest_close = data["close"].iloc[-1]
                st.metric("Latest Close", f"${latest_close:.2f}")
            elif "Close" in data.columns:
                latest_close = data["Close"].iloc[-1]
                st.metric("Latest Close", f"${latest_close:.2f}")
            else:
                st.metric("Latest Close", "N/A")
        with col4:
            # Safely calculate volatility
            close_col = None
            if "close" in data.columns:
                close_col = "close"
            elif "Close" in data.columns:
                close_col = "Close"

            if close_col and show_volatility:
                returns = data[close_col].pct_change().dropna()
                volatility = returns.std() * np.sqrt(trading_days) * 100
                st.metric("Annual Volatility", f"{volatility:.1f}%")
            else:
                st.metric("Annual Volatility", "N/A")

        # Display price chart
        st.subheader("📈 Price Chart")
        if "close" in data.columns:
            chart_data = data[["open", "high", "low", "close"]].tail(chart_days)
            st.line_chart(chart_data["close"])
        elif "Close" in data.columns:
            chart_data = data[["Open", "High", "Low", "Close"]].tail(chart_days)
            st.line_chart(chart_data["Close"])
        else:
            st.warning("No close price data available for charting")

        # Display data table
        st.subheader("📋 Data Table")
        st.dataframe(data.tail(table_rows))

        # Display optimization status
        st.subheader("🎯 Optimization Status")
        if st.button("🔄 Run Quick Test"):
            with st.spinner("Running quick strategy test..."):
                try:
                    # Run a quick test with default parameters
                    test_results = optimizer.optimize_strategy(
                        strategy=strategy,
                        optimizer_type="Grid",
                        param_space=param_space,
                        training_data=data,
                        n_jobs=1,
                    )

                    st.success("Quick test completed successfully!")
                    st.write(
                        "Best parameters found:", test_results.get("best_params", "N/A")
                    )
                    st.write("Best score:", test_results.get("best_score", "N/A"))

                except Exception as e:
                    st.error(f"Quick test failed: {str(e)}")
                    logger.error(f"Quick test error: {e}")
    else:
        st.info("No data loaded. Please use the sidebar to load market data.")
        st.info(
            "Enter a symbol (e.g., AAPL, MSFT, GOOGL) and click 'Load Data' to begin."
        )


if __name__ == "__main__":
    main()
