"""Base UI components for the trading system.

This module provides reusable UI components that can be used across different pages
and can be integrated with agentic systems for monitoring and control.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from trading.backtesting.edge_case_handler import EdgeCaseHandler

from .config.registry import registry

logger = logging.getLogger(__name__)


def create_date_range_selector(
    default_days: int = 30,
    min_days: int = 1,
    max_days: int = 365,
    key: str = "date_range",
) -> Tuple[datetime, datetime]:
    """Create a date range selector with validation.

    Args:
        default_days: Default number of days to look back
        min_days: Minimum allowed days
        max_days: Maximum allowed days
        key: Unique key for the Streamlit component

    Returns:
        Tuple of (start_date, end_date)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=default_days)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=start_date,
            min_value=end_date - timedelta(days=max_days),
            max_value=end_date,
            key=f"{key}_start",
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=end_date,
            min_value=start_date,
            max_value=datetime.now(),
            key=f"{key}_end",
        )

    # Validate date range
    if (end_date - start_date).days < min_days:
        st.warning(f"Date range must be at least {min_days} days")

    return start_date, end_date


def create_model_selector(
    category: Optional[str] = None,
    default_model: Optional[str] = None,
    key: str = "model_selector",
) -> Optional[str]:
    """Create a model selector with dynamic options from registry.

    Args:
        category: Optional category to filter models
        default_model: Optional default model to select
        key: Unique key for the Streamlit component

    Returns:
        Selected model name or None if no selection
    """
    models = registry.get_available_models(category)
    if not models:
        st.warning("No models available for the selected category")

    model_names = [model.name for model in models]
    selected_model = st.selectbox(
        "Select Model",
        options=model_names,
        index=model_names.index(default_model) if default_model in model_names else 0,
        key=key,
    )

    # Log selection for agentic monitoring
    logger.info(f"Model selected: {selected_model}")

    return selected_model


def create_strategy_selector(
    category: Optional[str] = None,
    default_strategy: Optional[str] = None,
    key: str = "strategy_selector",
) -> Optional[str]:
    """Create a strategy selector with dynamic options from registry.

    Args:
        category: Optional category to filter strategies
        default_strategy: Optional default strategy to select
        key: Unique key for the Streamlit component

    Returns:
        Selected strategy name or None if no selection
    """
    strategies = registry.get_available_strategies(category)
    if not strategies:
        st.warning("No strategies available for the selected category")

    strategy_names = [strategy.name for strategy in strategies]
    selected_strategy = st.selectbox(
        "Select Strategy",
        options=strategy_names,
        index=strategy_names.index(default_strategy)
        if default_strategy in strategy_names
        else 0,
        key=key,
    )

    # Log selection for agentic monitoring
    logger.info(f"Strategy selected: {selected_strategy}")

    return selected_strategy


def create_parameter_inputs(
    parameters: Dict[str, Union[str, float, int, bool]], key_prefix: str = "param"
) -> Dict[str, Any]:
    """Create input fields for model or strategy parameters.

    Args:
        parameters: Dictionary of parameter names and default values
        key_prefix: Prefix for Streamlit component keys

    Returns:
        Dictionary of parameter values
    """
    values = {}
    for param_name, default_value in parameters.items():
        key = f"{key_prefix}_{param_name}"

        if isinstance(default_value, bool):
            values[param_name] = st.checkbox(param_name, value=default_value, key=key)
        elif isinstance(default_value, int):
            values[param_name] = st.number_input(
                param_name, value=default_value, step=1, key=key
            )
        elif isinstance(default_value, float):
            values[param_name] = st.number_input(
                param_name, value=default_value, step=0.01, key=key
            )
        else:
            values[param_name] = st.text_input(
                param_name, value=str(default_value), key=key
            )

    # Log parameter changes for agentic monitoring
    logger.info(f"Parameters updated: {values}")

    return values


def create_asset_selector(
    assets: List[str], default_asset: Optional[str] = None, key: str = "asset_selector"
) -> Optional[str]:
    """Create an asset selector with validation.

    Args:
        assets: List of available assets
        default_asset: Optional default asset to select
        key: Unique key for the Streamlit component

    Returns:
        Selected asset symbol or None if no selection
    """
    if not assets:
        st.warning("No assets available")

    selected_asset = st.selectbox(
        "Select Asset",
        options=assets,
        index=assets.index(default_asset) if default_asset in assets else 0,
        key=key,
    )

    # Log selection for agentic monitoring
    logger.info(f"Asset selected: {selected_asset}")

    return selected_asset


def create_timeframe_selector(
    timeframes: List[str],
    default_timeframe: Optional[str] = None,
    key: str = "timeframe_selector",
) -> Optional[str]:
    """Create a timeframe selector with validation.

    Args:
        timeframes: List of available timeframes
        default_timeframe: Optional default timeframe to select
        key: Unique key for the Streamlit component

    Returns:
        Selected timeframe or None if no selection
    """
    if not timeframes:
        st.warning("No timeframes available")

    selected_timeframe = st.selectbox(
        "Select Timeframe",
        options=timeframes,
        index=timeframes.index(default_timeframe)
        if default_timeframe in timeframes
        else 0,
        key=key,
    )

    # Log selection for agentic monitoring
    logger.info(f"Timeframe selected: {selected_timeframe}")

    return selected_timeframe


def create_confidence_interval(
    data: pd.DataFrame, confidence_level: float = 0.95
) -> Tuple[pd.Series, pd.Series]:
    """Calculate confidence intervals for predictions.

    Args:
        data: DataFrame containing predictions and standard errors
        confidence_level: Confidence level for the interval

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    z_score = 1.96  # 95% confidence interval
    lower_bound = data["prediction"] - z_score * data["std_error"]
    upper_bound = data["prediction"] + z_score * data["std_error"]
    return (lower_bound, upper_bound)


def create_benchmark_overlay(
    data: pd.DataFrame, benchmark_column: str, prediction_column: str
) -> go.Figure:
    """Create a chart with benchmark and prediction overlay.

    Args:
        data: DataFrame containing benchmark and prediction data
        benchmark_column: Name of benchmark column
        prediction_column: Name of prediction column

    Returns:
        Plotly figure with overlay
    """
    fig = go.Figure()

    # Add benchmark line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data[benchmark_column],
            mode="lines",
            name="Benchmark",
            line=dict(color="blue", width=2),
        )
    )

    # Add prediction line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data[prediction_column],
            mode="lines",
            name="Prediction",
            line=dict(color="red", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title="Benchmark vs Prediction",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
    )

    return fig


def create_prompt_input() -> Dict[str, Any]:
    """Create a prompt input component for LLM interactions.

    Returns:
        Dictionary containing prompt and settings
    """
    st.subheader("LLM Prompt Configuration")

    # Prompt input
    prompt = st.text_area(
        "Enter your prompt",
        height=150,
        placeholder="Describe what you want to analyze or predict...",
        key="llm_prompt",
    )

    # Model selection
    model_options = ["gpt-4", "gpt-3.5-turbo", "claude-3", "local"]
    selected_model = st.selectbox("Select LLM Model", model_options, key="llm_model")

    # Temperature setting
    temperature = st.slider(
        "Temperature (Creativity)",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        key="llm_temperature",
    )

    # Max tokens
    max_tokens = st.number_input(
        "Max Tokens",
        min_value=100,
        max_value=4000,
        value=1000,
        step=100,
        key="llm_max_tokens",
    )

    # Execute button
    execute = st.button("Execute LLM Analysis", key="llm_execute")

    return {
        "prompt": prompt,
        "model": selected_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "execute": execute,
    }


def create_sidebar() -> Dict[str, Any]:
    """Create a comprehensive sidebar with all configuration options.

    Returns:
        Dictionary containing all sidebar selections
    """
    st.sidebar.title("Configuration")

    # Data source selection
    st.sidebar.subheader("Data Source")
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Yahoo Finance", "Alpha Vantage", "Polygon", "Local CSV"],
        key="data_source",
    )

    # Asset selection
    st.sidebar.subheader("Asset")
    asset = st.sidebar.text_input("Asset Symbol", value="AAPL", key="asset_symbol")

    # Timeframe selection
    st.sidebar.subheader("Timeframe")
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"],
        index=6,
        key="timeframe",
    )

    # Date range
    st.sidebar.subheader("Date Range")
    days_back = st.sidebar.slider(
        "Days Back", min_value=1, max_value=365, value=30, key="days_back"
    )

    # Model selection
    st.sidebar.subheader("Model")
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["LSTM", "XGBoost", "Ensemble", "TCN", "Hybrid"],
        key="model_type",
    )

    # Strategy selection
    st.sidebar.subheader("Strategy")
    strategy_type = st.sidebar.selectbox(
        "Strategy Type",
        ["RSI", "MACD", "Bollinger Bands", "SMA", "Custom"],
        key="strategy_type",
    )

    # Advanced options
    st.sidebar.subheader("Advanced Options")
    show_confidence = st.sidebar.checkbox("Show Confidence Intervals", value=True)
    enable_logging = st.sidebar.checkbox("Enable Detailed Logging", value=False)
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=False)

    # Execute button
    execute = st.sidebar.button("Run Analysis", type="primary")

    return {
        "data_source": data_source,
        "asset": asset,
        "timeframe": timeframe,
        "days_back": days_back,
        "model_type": model_type,
        "strategy_type": strategy_type,
        "show_confidence": show_confidence,
        "enable_logging": enable_logging,
        "auto_refresh": auto_refresh,
        "execute": execute,
    }


def create_forecast_chart(
    historical_data: pd.DataFrame,
    forecast_data: pd.DataFrame,
    title: str = "Price Forecast",
) -> go.Figure:
    """Create a comprehensive forecast chart with historical and predicted data.

    Args:
        historical_data: DataFrame with historical price data
        forecast_data: DataFrame with forecast data
        title: Chart title

    Returns:
        Plotly figure with forecast visualization
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(title, "Volume"),
        row_heights=[0.7, 0.3],
    )

    # Historical candlestick
    fig.add_trace(
        go.Candlestick(
            x=historical_data.index,
            open=historical_data["open"],
            high=historical_data["high"],
            low=historical_data["low"],
            close=historical_data["close"],
            name="Historical",
            increasing_line_color="green",
            decreasing_line_color="red",
        ),
        row=1,
        col=1,
    )

    # Forecast line
    if "forecast" in forecast_data.columns:
        fig.add_trace(
            go.Scatter(
                x=forecast_data.index,
                y=forecast_data["forecast"],
                mode="lines",
                name="Forecast",
                line=dict(color="blue", width=2, dash="dash"),
            ),
            row=1,
            col=1,
        )

    # Confidence intervals
    if "lower_bound" in forecast_data.columns and "upper_bound" in forecast_data.columns:
        fig.add_trace(
            go.Scatter(
                x=forecast_data.index,
                y=forecast_data["upper_bound"],
                mode="lines",
                name="Upper Bound",
                line=dict(color="lightblue", width=1),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_data.index,
                y=forecast_data["lower_bound"],
                mode="lines",
                fill="tonexty",
                name="Confidence Interval",
                line=dict(color="lightblue", width=1),
            ),
            row=1,
            col=1,
        )

    # Volume
    if "volume" in historical_data.columns:
        fig.add_trace(
            go.Bar(
                x=historical_data.index,
                y=historical_data["volume"],
                name="Volume",
                marker_color="gray",
                opacity=0.5,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        height=600,
        showlegend=True,
    )

    return fig


def create_strategy_chart(
    data: pd.DataFrame, signals: pd.DataFrame, title: str = "Strategy Signals"
) -> go.Figure:
    """Create a chart showing strategy signals and performance.

    Args:
        data: Market data DataFrame
        signals: Signals DataFrame
        title: Chart title

    Returns:
        Plotly figure with strategy visualization
    """
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Price & Signals", "Strategy Returns", "Cumulative Returns"),
        row_heights=[0.5, 0.25, 0.25],
    )

    # Price and signals
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["close"],
            mode="lines",
            name="Price",
            line=dict(color="black", width=1),
        ),
        row=1,
        col=1,
    )

    # Buy signals
    buy_signals = signals[signals["signal"] == 1]
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals["close"],
                mode="markers",
                name="Buy Signal",
                marker=dict(color="green", size=8, symbol="triangle-up"),
            ),
            row=1,
            col=1,
        )

    # Sell signals
    sell_signals = signals[signals["signal"] == -1]
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals["close"],
                mode="markers",
                name="Sell Signal",
                marker=dict(color="red", size=8, symbol="triangle-down"),
            ),
            row=1,
            col=1,
        )

    # Strategy returns
    if "strategy_returns" in signals.columns:
        fig.add_trace(
            go.Scatter(
                x=signals.index,
                y=signals["strategy_returns"],
                mode="lines",
                name="Strategy Returns",
                line=dict(color="blue", width=1),
            ),
            row=2,
            col=1,
        )

    # Cumulative returns
    if "cumulative_returns" in signals.columns:
        fig.add_trace(
            go.Scatter(
                x=signals.index,
                y=signals["cumulative_returns"],
                mode="lines",
                name="Cumulative Returns",
                line=dict(color="green", width=2),
            ),
            row=3,
            col=1,
        )

    fig.update_layout(
        title=title,
        height=600,
        showlegend=True,
    )

    return fig


def create_performance_report(
    results: Dict[str, Any], title: str = "Performance Report"
) -> Dict[str, Any]:
    """Create a comprehensive performance report.

    Args:
        results: Dictionary containing performance metrics
        title: Report title

    Returns:
        Dictionary containing report components
    """
    st.subheader(title)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Return",
            f"{results.get('total_return', 0):.2%}",
            delta=f"{results.get('daily_return', 0):.2%}",
        )

    with col2:
        st.metric(
            "Sharpe Ratio",
            f"{results.get('sharpe_ratio', 0):.2f}",
        )

    with col3:
        st.metric(
            "Max Drawdown",
            f"{results.get('max_drawdown', 0):.2%}",
        )

    with col4:
        st.metric(
            "Win Rate",
            f"{results.get('win_rate', 0):.2%}",
        )

    # Detailed metrics table
    st.subheader("Detailed Metrics")
    metrics_df = pd.DataFrame(
        [
            ["Total Return", f"{results.get('total_return', 0):.4f}"],
            ["Annualized Return", f"{results.get('annualized_return', 0):.4f}"],
            ["Volatility", f"{results.get('volatility', 0):.4f}"],
            ["Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.4f}"],
            ["Sortino Ratio", f"{results.get('sortino_ratio', 0):.4f}"],
            ["Max Drawdown", f"{results.get('max_drawdown', 0):.4f}"],
            ["Calmar Ratio", f"{results.get('calmar_ratio', 0):.4f}"],
            ["Win Rate", f"{results.get('win_rate', 0):.4f}"],
            ["Profit Factor", f"{results.get('profit_factor', 0):.4f}"],
            ["Total Trades", f"{results.get('total_trades', 0)}"],
        ],
        columns=["Metric", "Value"],
    )

    st.table(metrics_df)

    # Risk metrics
    st.subheader("Risk Analysis")
    risk_col1, risk_col2 = st.columns(2)

    with risk_col1:
        st.write("**Value at Risk (VaR)**")
        st.write(f"95% VaR: {results.get('var_95', 0):.4f}")
        st.write(f"99% VaR: {results.get('var_99', 0):.4f}")

    with risk_col2:
        st.write("**Expected Shortfall**")
        st.write(f"95% ES: {results.get('es_95', 0):.4f}")
        st.write(f"99% ES: {results.get('es_99', 0):.4f}")

    return {
        "metrics": results,
        "displayed": True,
    }


def create_error_block(message: str) -> Dict[str, Any]:
    """Create an error display block.

    Args:
        message: Error message to display

    Returns:
        Dictionary containing error information
    """
    st.error(f"âŒ Error: {message}")
    logger.error(f"UI Error: {message}")

    return {
        "error": True,
        "message": message,
        "timestamp": datetime.now().isoformat(),
    }


def create_loading_spinner(message: str = "Processing...") -> Dict[str, Any]:
    """Create a loading spinner with message.

    Args:
        message: Loading message to display

    Returns:
        Dictionary containing loading state
    """
    with st.spinner(message):
        st.info(f"â³ {message}")
        logger.info(f"Loading: {message}")

    return {
        "loading": True,
        "message": message,
        "timestamp": datetime.now().isoformat(),
    }


def create_forecast_metrics(forecast_results: Dict[str, Any]) -> Dict[str, Any]:
    """Create forecast accuracy metrics display.

    Args:
        forecast_results: Dictionary containing forecast results and metrics

    Returns:
        Dictionary containing metrics display
    """
    st.subheader("Forecast Accuracy Metrics")

    metrics = forecast_results.get("metrics", {})
    if not metrics:
        st.warning("No forecast metrics available")
        return {"displayed": False}

    # Accuracy metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "RMSE",
            f"{metrics.get('rmse', 0):.4f}",
        )

    with col2:
        st.metric(
            "MAE",
            f"{metrics.get('mae', 0):.4f}",
        )

    with col3:
        st.metric(
            "MAPE",
            f"{metrics.get('mape', 0):.2f}%",
        )

    with col4:
        st.metric(
            "RÂ² Score",
            f"{metrics.get('r2_score', 0):.4f}",
        )

    # Detailed metrics table
    st.subheader("Detailed Forecast Metrics")
    forecast_metrics_df = pd.DataFrame(
        [
            ["RMSE", f"{metrics.get('rmse', 0):.6f}"],
            ["MAE", f"{metrics.get('mae', 0):.6f}"],
            ["MAPE", f"{metrics.get('mape', 0):.4f}%"],
            ["RÂ² Score", f"{metrics.get('r2_score', 0):.6f}"],
            ["Explained Variance", f"{metrics.get('explained_variance', 0):.6f}"],
            ["Mean Absolute Error", f"{metrics.get('mean_absolute_error', 0):.6f}"],
            ["Median Absolute Error", f"{metrics.get('median_absolute_error', 0):.6f}"],
            ["Max Error", f"{metrics.get('max_error', 0):.6f}"],
        ],
        columns=["Metric", "Value"],
    )

    st.table(forecast_metrics_df)

    return {
        "metrics": metrics,
        "displayed": True,
    }


def create_forecast_table(forecast_results: Dict[str, Any]) -> Dict[str, Any]:
    """Create a forecast results table.

    Args:
        forecast_results: Dictionary containing forecast results

    Returns:
        Dictionary containing table display
    """
    st.subheader("Forecast Results")

    forecast_data = forecast_results.get("forecast_data")
    if forecast_data is None or forecast_data.empty:
        st.warning("No forecast data available")
        return {"displayed": False}

    # Display forecast table
    st.dataframe(
        forecast_data,
        use_container_width=True,
        height=400,
    )

    # Download button
    csv = forecast_data.to_csv(index=True)
    st.download_button(
        label="Download Forecast Data",
        data=csv,
        file_name="forecast_results.csv",
        mime="text/csv",
    )

    return {
        "data": forecast_data,
        "displayed": True,
    }


def create_system_metrics_panel(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Create a system metrics monitoring panel.

    Args:
        metrics: Dictionary containing system metrics

    Returns:
        Dictionary containing metrics display
    """
    st.subheader("System Metrics")

    if not metrics:
        st.warning("No system metrics available")
        return {"displayed": False}

    # System health indicators
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        cpu_usage = metrics.get("cpu_usage", 0)
        st.metric(
            "CPU Usage",
            f"{cpu_usage:.1f}%",
            delta=f"{cpu_usage - 50:.1f}%" if cpu_usage > 50 else None,
        )

    with col2:
        memory_usage = metrics.get("memory_usage", 0)
        st.metric(
            "Memory Usage",
            f"{memory_usage:.1f}%",
            delta=f"{memory_usage - 70:.1f}%" if memory_usage > 70 else None,
        )

    with col3:
        disk_usage = metrics.get("disk_usage", 0)
        st.metric(
            "Disk Usage",
            f"{disk_usage:.1f}%",
            delta=f"{disk_usage - 80:.1f}%" if disk_usage > 80 else None,
        )

    with col4:
        network_latency = metrics.get("network_latency", 0)
        st.metric(
            "Network Latency",
            f"{network_latency:.1f}ms",
        )

    # Performance metrics
    st.subheader("Performance Metrics")
    perf_col1, perf_col2 = st.columns(2)

    with perf_col1:
        st.write("**Model Performance**")
        st.write(f"Average Prediction Time: {metrics.get('avg_prediction_time', 0):.3f}s")
        st.write(f"Model Accuracy: {metrics.get('model_accuracy', 0):.2%}")
        st.write(f"Cache Hit Rate: {metrics.get('cache_hit_rate', 0):.2%}")

    with perf_col2:
        st.write("**System Performance**")
        st.write(f"Active Connections: {metrics.get('active_connections', 0)}")
        st.write(f"Request Rate: {metrics.get('request_rate', 0):.1f} req/s")
        st.write(f"Error Rate: {metrics.get('error_rate', 0):.2%}")

    # System status
    st.subheader("System Status")
    status_col1, status_col2, status_col3 = st.columns(3)

    with status_col1:
        if metrics.get("system_healthy", True):
            st.success("âœ… System Healthy")
        else:
            st.error("âŒ System Issues Detected")

    with status_col2:
        if metrics.get("models_loaded", True):
            st.success("âœ… Models Loaded")
        else:
            st.error("âŒ Model Loading Issues")

    with status_col3:
        if metrics.get("data_feed_active", True):
            st.success("âœ… Data Feed Active")
        else:
            st.error("âŒ Data Feed Issues")

    return {
        "metrics": metrics,
        "displayed": True,
    }


def create_strategy_pipeline_selector(
    key: str = "strategy_pipeline_selector",
    allow_combos: bool = True,
) -> Dict[str, Any]:
    """Create a strategy pipeline selector with multiple strategy options.

    Args:
        key: Unique key for the Streamlit component
        allow_combos: Whether to allow strategy combinations

    Returns:
        Dictionary containing selected strategies and configuration
    """
    try:
        from strategies.strategy_pipeline import (
            AVAILABLE_STRATEGIES,
            COMBINE_MODES,
        )

        st.subheader("Strategy Pipeline Configuration")

        if allow_combos:
            # Multiple strategy selection
            selected_strategies = st.multiselect(
                "Select Strategies",
                options=AVAILABLE_STRATEGIES,
                default=AVAILABLE_STRATEGIES[:2],
                key=f"{key}_strategies",
            )

            if len(selected_strategies) > 1:
                # Combination mode selection
                combine_mode = st.selectbox(
                    "Combination Mode",
                    options=COMBINE_MODES,
                    key=f"{key}_combine_mode",
                )

                # Weights for weighted combination
                if combine_mode == "weighted":
                    st.write("Strategy Weights:")
                    weights = {}
                    for strategy in selected_strategies:
                        weight = st.slider(
                            f"{strategy} Weight",
                            min_value=0.0,
                            max_value=1.0,
                            value=1.0 / len(selected_strategies),
                            step=0.1,
                            key=f"{key}_weight_{strategy}",
                        )
                        weights[strategy] = weight

                    # Normalize weights
                    total_weight = sum(weights.values())
                    if total_weight > 0:
                        weights = {k: v / total_weight for k, v in weights.items()}
                else:
                    weights = None

                result = {
                    "strategies": selected_strategies,
                    "combine_mode": combine_mode,
                    "weights": weights,
                    "parameters": {},
                    "is_combo": True,
                }
            else:
                result = {
                    "strategies": selected_strategies,
                    "combine_mode": None,
                    "weights": None,
                    "parameters": {},
                    "is_combo": False,
                }
        else:
            # Single strategy selection only
            selected_strategy = st.selectbox(
                "Select Strategy",
                options=AVAILABLE_STRATEGIES,
                key=key,
            )

            result = {
                "strategies": [selected_strategy] if selected_strategy else [],
                "combine_mode": None,
                "weights": [],
                "parameters": {},
                "is_combo": False,
            }

        # Log selection for agentic monitoring
        logger.info(f"Strategy pipeline selection: {result}")

        return result

    except ImportError:
        st.error("Strategy pipeline module not available")
        return {}
    except Exception as e:
        logger.error(f"Error in strategy pipeline selector: {e}")
        st.error(f"Error creating strategy selector: {str(e)}")
        return {}


def execute_strategy_pipeline(
    strategy_config: Dict[str, Any],
    market_data: pd.DataFrame,
    key: str = "strategy_execution",
) -> Dict[str, Any]:
    """Execute a strategy pipeline with given configuration and market data.

    Args:
        strategy_config: Configuration from create_strategy_pipeline_selector
        market_data: Market data DataFrame with OHLCV columns
        key: Unique key for the Streamlit component

    Returns:
        Dictionary containing signals, performance metrics, and execution info
    """
    try:
        from strategies.strategy_pipeline import (
            rsi_strategy,
            macd_strategy,
            bollinger_strategy,
            sma_strategy,
            combine_signals,
        )

        if not strategy_config or not strategy_config.get("strategies"):
            st.error("No strategies selected")
            return {}

        strategies = strategy_config["strategies"]
        parameters = strategy_config.get("parameters", {})
        combine_mode = strategy_config.get("combine_mode")
        weights = strategy_config.get("weights")

        # Generate individual strategy signals
        signals_list = []
        strategy_names = []

        for strategy in strategies:
            strategy_params = parameters.get(strategy, {})
            if strategy == "RSI":
                signal = rsi_strategy(
                    market_data,
                    window=strategy_params.get("window", 14),
                    overbought=strategy_params.get("overbought", 70),
                    oversold=strategy_params.get("oversold", 30),
                )
            elif strategy == "MACD":
                signal = macd_strategy(
                    market_data,
                    fast=strategy_params.get("fast", 12),
                    slow=strategy_params.get("slow", 26),
                    signal=strategy_params.get("signal", 9),
                )
            elif strategy == "Bollinger":
                signal = bollinger_strategy(
                    market_data,
                    window=strategy_params.get("window", 20),
                    num_std=strategy_params.get("num_std", 2.0),
                )
            elif strategy == "SMA":
                signal = sma_strategy(
                    market_data,
                    window=strategy_params.get("window", 20),
                )
            else:
                st.warning(f"Unknown strategy: {strategy}")
                continue

            signals_list.append(signal)
            strategy_names.append(strategy)

        if not signals_list:
            st.error("No valid signals generated")
            return {}

        # Combine signals if multiple strategies
        if len(signals_list) > 1 and combine_mode:
            combined_signal = combine_signals(
                signals_list, mode=combine_mode, weights=weights
            )
            final_signal = combined_signal
            signal_type = f"Combined ({combine_mode})"
        else:
            final_signal = signals_list[0]
            signal_type = "Single"

        # Calculate basic performance metrics
        performance_metrics = calculate_signal_performance(
            market_data, final_signal, strategy_names
        )

        # Create result dictionary
        result = {
            "signal": final_signal,
            "individual_signals": dict(zip(strategy_names, signals_list)),
            "signal_type": signal_type,
            "strategies_used": strategies,
            "combine_mode": combine_mode,
            "weights": weights,
            "performance": performance_metrics,
            "execution_time": datetime.now().isoformat(),
        }

        # Log execution for agentic monitoring
        logger.info(
            f"Strategy pipeline executed: {len(strategies)} strategies, mode: {combine_mode}"
        )

        return result

    except ImportError as e:
        st.error(f"Strategy pipeline module not available: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error executing strategy pipeline: {e}")
        st.error(f"Error executing strategy pipeline: {str(e)}")
        return {}


def calculate_signal_performance(
    market_data: pd.DataFrame,
    signals: pd.Series,
    strategy_names: List[str],
) -> Dict[str, Any]:
    """Calculate basic performance metrics for strategy signals.

    Args:
        market_data: Market data DataFrame
        signals: Signal series (1=buy, -1=sell, 0=hold)
        strategy_names: List of strategy names used

    Returns:
        Dictionary of performance metrics
    """
    try:
        # Calculate returns
        price_returns = market_data["close"].pct_change()

        # Calculate strategy returns (assuming 100% position size)
        strategy_returns = price_returns * signals.shift(1)

        # Basic metrics
        total_return = strategy_returns.sum()
        sharpe_ratio = (
            strategy_returns.mean() / (strategy_returns.std() + 1e-9) * np.sqrt(252)
        )
        max_drawdown = (
            strategy_returns.cumsum() - strategy_returns.cumsum().expanding().max()
        ).min()

        # Signal statistics
        buy_signals = (signals == 1).sum()
        sell_signals = (signals == -1).sum()
        hold_signals = (signals == 0).sum()
        total_signals = len(signals)

        # Win rate (simplified)
        positive_returns = (strategy_returns > 0).sum()
        total_trades = (strategy_returns != 0).sum()
        win_rate = positive_returns / (total_trades + 1e-9)

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "hold_signals": hold_signals,
            "total_signals": total_signals,
            "win_rate": win_rate,
            "strategy_names": strategy_names,
        }

    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}")
        return {
            "error": str(e),
            "strategy_names": strategy_names,
        }
