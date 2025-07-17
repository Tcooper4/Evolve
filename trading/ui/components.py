"""Base UI components for the trading system.

This module provides reusable UI components that can be used across different pages
and can be integrated with agentic systems for monitoring and control.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

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
    """Create a plot with benchmark overlay.

    Args:
        data: DataFrame containing predictions and benchmark data
        benchmark_column: Name of the benchmark column
        prediction_column: Name of the prediction column

    Returns:
        Plotly figure with benchmark overlay
    """
    fig = go.Figure()

    # Add prediction line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data[prediction_column],
            name="Prediction",
            line=dict(color="blue"),
        )
    )

    # Add benchmark line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data[benchmark_column],
            name="Benchmark",
            line=dict(color="gray", dash="dash"),
        )
    )

    # Add confidence interval if available
    if "std_error" in data.columns:
        lower, upper = create_confidence_interval(data)
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=upper,
                fill=None,
                mode="lines",
                line_color="rgba(0,100,80,0.2)",
                name="Upper Bound",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=lower,
                fill="tonexty",
                mode="lines",
                line_color="rgba(0,100,80,0.2)",
                name="Lower Bound",
            )
        )

    fig.update_layout(
        title="Prediction with Benchmark Overlay",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
    )

    return fig


def create_prompt_input() -> Dict[str, Any]:
    """Create a prompt input component.

    Returns:
        Dictionary with prompt text and metadata
    """
    try:
        prompt = st.text_area(
            "Enter your prompt",
            placeholder="Describe what you want to analyze or predict...",
            height=100,
            key=os.getenv("KEY", ""),
        )

        if prompt:
            return {
                "success": True,
                "prompt": prompt,
                "length": len(prompt),
                "timestamp": datetime.now().isoformat(),
            }
        else:
            return {
                "success": False,
                "prompt": "",
                "message": "No prompt entered",
                "timestamp": datetime.now().isoformat(),
            }

    except Exception as e:
        logger.error(f"Error creating prompt input: {e}")
        return {
            "success": False,
            "prompt": "",
            "message": f"Error creating prompt input: {str(e)}",
            "timestamp": datetime.now().isoformat(),
        }


def create_sidebar() -> Dict[str, Any]:
    """Create the main sidebar with navigation and settings.

    Returns:
        Dictionary with sidebar configuration and selections
    """
    try:
        with st.sidebar:
            st.title("Evolve Trading")

            # Navigation
            page = st.selectbox(
                "Navigation",
                ["Dashboard", "Forecast", "Strategy", "Analysis", "Settings"],
                key=os.getenv("KEY", ""),
            )

            # Model settings
            st.subheader("Model Settings")
            model_type = st.selectbox(
                "Model Type",
                ["LSTM", "Transformer", "Ensemble", "Custom"],
                key=os.getenv("KEY", ""),
            )

            # Strategy settings
            st.subheader("Strategy Settings")
            strategy_type = st.selectbox(
                "Strategy Type",
                ["Momentum", "Mean Reversion", "Breakout", "Custom"],
                key=os.getenv("KEY", ""),
            )

            # Risk settings
            st.subheader("Risk Settings")
            max_position_size = st.slider(
                "Max Position Size (%)",
                min_value=1,
                max_value=100,
                value=10,
                key=os.getenv("KEY", ""),
            )

            stop_loss = st.slider(
                "Stop Loss (%)",
                min_value=1,
                max_value=50,
                value=5,
                key=os.getenv("KEY", ""),
            )

            # System settings
            st.subheader("System Settings")
            auto_refresh = st.checkbox(
                "Auto Refresh", value=True, key=os.getenv("KEY", "")
            )
            debug_mode = st.checkbox(
                "Debug Mode", value=False, key=os.getenv("KEY", "")
            )

            return {
                "success": True,
                "page": page,
                "model_type": model_type,
                "strategy_type": strategy_type,
                "max_position_size": max_position_size,
                "stop_loss": stop_loss,
                "auto_refresh": auto_refresh,
                "debug_mode": debug_mode,
                "timestamp": datetime.now().isoformat(),
            }

    except Exception as e:
        logger.error(f"Error creating sidebar: {e}")
        return {
            "success": False,
            "message": f"Error creating sidebar: {str(e)}",
            "page": "Dashboard",
            "timestamp": datetime.now().isoformat(),
        }


def create_forecast_chart(
    historical_data: pd.DataFrame,
    forecast_data: pd.DataFrame,
    title: str = "Price Forecast",
) -> go.Figure:
    """Create an interactive forecast chart.

    Args:
        historical_data: Historical price data
        forecast_data: Forecast data with confidence intervals
        title: Chart title

    Returns:
        Plotly figure object
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=historical_data.index,
            y=historical_data["close"],
            name="Historical",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    # Add forecast
    fig.add_trace(
        go.Scatter(
            x=forecast_data.index,
            y=forecast_data["forecast"],
            name="Forecast",
            line=dict(color="red", dash="dash"),
        ),
        row=1,
        col=1,
    )

    # Add confidence intervals
    fig.add_trace(
        go.Scatter(
            x=forecast_data.index,
            y=forecast_data["upper"],
            name="Upper Bound",
            line=dict(color="gray", dash="dot"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_data.index,
            y=forecast_data["lower"],
            name="Lower Bound",
            line=dict(color="gray", dash="dot"),
            fill="tonexty",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        showlegend=True,
        height=600,
    )

    return fig


def create_strategy_chart(
    data: pd.DataFrame, signals: pd.DataFrame, title: str = "Strategy Signals"
) -> go.Figure:
    """Create an interactive strategy chart.

    Args:
        data: Price data
        signals: Trading signals
        title: Chart title

    Returns:
        Plotly figure object
    """
    edge_handler = EdgeCaseHandler()
    # Validate signals with edge handler
    validation = edge_handler.validate_signals(
        signals["signal"]
        if isinstance(signals, pd.DataFrame) and "signal" in signals.columns
        else signals
    )
    if validation["status"] != "success":
        st.warning(validation["message"])
        fallback = edge_handler.create_fallback_chart(data, chart_type="equity")
        if fallback["chart_data"]:
            st.write(fallback["message"])
        return go.Figure()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

    # Add price data
    fig.add_trace(
        go.Scatter(
            x=data.index, y=data["close"], name="Price", line=dict(color="blue")
        ),
        row=1,
        col=1,
    )

    # Add signals
    fig.add_trace(
        go.Scatter(
            x=signals.index, y=signals["signal"], name="Signal", line=dict(color="red")
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        showlegend=True,
        height=600,
    )

    return fig


def create_performance_report(
    results: Dict[str, Any], title: str = "Performance Report"
) -> Dict[str, Any]:
    """Create an expandable performance report section.

    Args:
        results: Dictionary containing performance metrics
        title: Section title

    Returns:
        Dictionary with report creation status and metrics summary
    """
    edge_handler = EdgeCaseHandler()
    # Validate results for performance metrics
    if not results or not results.get("total_return"):
        st.warning("No performance data available.")
        fallback = edge_handler.create_fallback_chart(
            pd.DataFrame(), chart_type="performance"
        )
        st.write(
            fallback["chart_data"]["message"] if fallback["chart_data"] else "No data."
        )
        return {"status": "warning", "message": fallback["message"]}

    with st.expander(title, expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Return", f"{results.get('total_return', 0):.2%}")
            st.metric("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}")
            st.metric("Max Drawdown", f"{results.get('max_drawdown', 0):.2%}")

        with col2:
            st.metric("Win Rate", f"{results.get('win_rate', 0):.2%}")
            st.metric("Avg Trade", f"{results.get('avg_trade', 0):.2%}")
            st.metric("Profit Factor", f"{results.get('profit_factor', 0):.2f}")

        with col3:
            st.metric("CAGR", f"{results.get('cagr', 0):.2%}")
            st.metric("Volatility", f"{results.get('volatility', 0):.2%}")
            st.metric("Calmar Ratio", f"{results.get('calmar_ratio', 0):.2f}")

        # Create equity curve
        if "equity_curve" in results:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=results["equity_curve"].index,
                    y=results["equity_curve"].values,
                    name="Equity",
                    line=dict(color="blue"),
                )
            )
            fig.update_layout(
                title="Equity Curve",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Create drawdown chart
        if "drawdowns" in results:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=results["drawdowns"].index,
                    y=results["drawdowns"].values,
                    name="Drawdown",
                    fill="tozeroy",
                    line=dict(color="red"),
                )
            )
            fig.update_layout(
                title="Drawdowns",
                xaxis_title="Date",
                yaxis_title="Drawdown",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Log report creation for agentic monitoring
    logger.info(f"Performance report created: {title}")

    return {
        "success": True,
        "title": title,
        "metrics_count": len(results),
        "has_equity_curve": "equity_curve" in results,
        "has_drawdowns": "drawdowns" in results,
        "key_metrics": {
            "total_return": results.get("total_return", 0),
            "sharpe_ratio": results.get("sharpe_ratio", 0),
            "max_drawdown": results.get("max_drawdown", 0),
        },
    }


def create_error_block(message: str) -> Dict[str, Any]:
    """Create an error message block.

    Args:
        message: Error message to display

    Returns:
        Dictionary with error display status
    """
    st.error(message)
    st.info("Try adjusting your parameters or selecting a different strategy.")

    # Log error for agentic monitoring
    logger.error(f"Error block displayed: {message}")

    return {
        "success": True,
        "error_displayed": True,
        "message": message,
        "timestamp": datetime.now().isoformat(),
    }


def create_loading_spinner(message: str = "Processing...") -> Dict[str, Any]:
    """Create a loading spinner with message.

    Args:
        message: Message to display during loading

    Returns:
        Dictionary with spinner creation status
    """
    spinner = st.spinner(message)

    # Log spinner creation for agentic monitoring
    logger.info(f"Loading spinner created: {message}")

    return {
        "success": True,
        "spinner_created": True,
        "message": message,
        "spinner_object": spinner,
    }


def create_forecast_metrics(forecast_results: Dict[str, Any]) -> Dict[str, Any]:
    """Create forecast metrics display from forecast results.

    Args:
        forecast_results: Dictionary containing forecast results and metrics

    Returns:
        Dictionary with metrics display status
    """
    st.subheader("Forecast Metrics")

    metrics = forecast_results.get("metrics", {})

    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")

    with col2:
        st.metric("MSE", f"{metrics.get('mse', 0):.4f}")

    with col3:
        st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")

    with col4:
        st.metric("MAE", f"{metrics.get('mae', 0):.4f}")

    # Log metrics for agentic monitoring
    logger.info(
        f"Forecast metrics displayed for {forecast_results.get('ticker', 'unknown')}"
    )

    return {
        "success": True,
        "metrics_displayed": True,
        "ticker": forecast_results.get("ticker", "unknown"),
        "metrics_count": len(metrics),
        "key_metrics": {
            "accuracy": metrics.get("accuracy", 0),
            "mse": metrics.get("mse", 0),
            "rmse": metrics.get("rmse", 0),
            "mae": metrics.get("mae", 0),
        },
    }


def create_forecast_table(forecast_results: Dict[str, Any]) -> Dict[str, Any]:
    """Create forecast table display from forecast results.

    Args:
        forecast_results: Dictionary containing forecast results

    Returns:
        Dictionary with table creation status
    """
    st.subheader("Forecast Results")

    # Create a simple table with forecast information
    forecast_data = {
        "Metric": ["Ticker", "Model", "Strategy", "Prediction"],
        "Value": [
            forecast_results.get("ticker", "N/A"),
            forecast_results.get("model", "N/A"),
            forecast_results.get("strategy", "N/A"),
            forecast_results.get("prediction", "N/A"),
        ],
    }

    df = pd.DataFrame(forecast_data)
    st.table(df)

    # Log table creation for agentic monitoring
    logger.info(
        f"Forecast table displayed for {forecast_results.get('ticker', 'unknown')}"
    )

    return {
        "success": True,
        "table_created": True,
        "ticker": forecast_results.get("ticker", "unknown"),
        "table_rows": len(forecast_data["Metric"]),
        "forecast_data": forecast_data,
    }


def create_system_metrics_panel(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Create a system-wide metrics panel showing key performance indicators.

    Args:
        metrics: Dictionary containing performance metrics

    Returns:
        Dictionary with panel creation status and metrics summary
    """
    st.subheader("📊 System Performance Metrics")

    # Create columns for metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        sharpe = metrics.get("sharpe_ratio", 0.0)
        if sharpe >= 1.5:
            st.metric(
                "Sharpe Ratio", f"{sharpe:.2f}", delta="Excellent", delta_color="normal"
            )
        elif sharpe >= 1.0:
            st.metric(
                "Sharpe Ratio", f"{sharpe:.2f}", delta="Good", delta_color="normal"
            )
        elif sharpe >= 0.5:
            st.metric("Sharpe Ratio", f"{sharpe:.2f}", delta="Fair", delta_color="off")
        else:
            st.metric(
                "Sharpe Ratio", f"{sharpe:.2f}", delta="Poor", delta_color="inverse"
            )

    with col2:
        total_return = metrics.get("total_return", 0.0)
        if total_return >= 0.20:
            st.metric(
                "Total Return",
                f"{total_return:.1%}",
                delta="Strong",
                delta_color="normal",
            )
        elif total_return >= 0.10:
            st.metric(
                "Total Return",
                f"{total_return:.1%}",
                delta="Good",
                delta_color="normal",
            )
        elif total_return >= 0.05:
            st.metric(
                "Total Return",
                f"{total_return:.1%}",
                delta="Moderate",
                delta_color="off",
            )
        else:
            st.metric(
                "Total Return",
                f"{total_return:.1%}",
                delta="Weak",
                delta_color="inverse",
            )

    with col3:
        max_dd = metrics.get("max_drawdown", 0.0)
        if max_dd <= 0.10:
            st.metric(
                "Max Drawdown", f"{max_dd:.1%}", delta="Low Risk", delta_color="normal"
            )
        elif max_dd <= 0.20:
            st.metric(
                "Max Drawdown", f"{max_dd:.1%}", delta="Moderate", delta_color="off"
            )
        elif max_dd <= 0.30:
            st.metric(
                "Max Drawdown", f"{max_dd:.1%}", delta="High Risk", delta_color="off"
            )
        else:
            st.metric(
                "Max Drawdown",
                f"{max_dd:.1%}",
                delta="Very High Risk",
                delta_color="inverse",
            )

    with col4:
        win_rate = metrics.get("win_rate", 0.0)
        if win_rate >= 0.60:
            st.metric(
                "Win Rate", f"{win_rate:.1%}", delta="Excellent", delta_color="normal"
            )
        elif win_rate >= 0.50:
            st.metric("Win Rate", f"{win_rate:.1%}", delta="Good", delta_color="normal")
        elif win_rate >= 0.40:
            st.metric("Win Rate", f"{win_rate:.1%}", delta="Fair", delta_color="off")
        else:
            st.metric(
                "Win Rate", f"{win_rate:.1%}", delta="Poor", delta_color="inverse"
            )

    with col5:
        pnl = metrics.get("total_pnl", 0.0)
        if pnl >= 10000:
            st.metric("Total PnL", f"${pnl:,.0f}", delta="Strong", delta_color="normal")
        elif pnl >= 5000:
            st.metric("Total PnL", f"${pnl:,.0f}", delta="Good", delta_color="normal")
        elif pnl >= 1000:
            st.metric("Total PnL", f"${pnl:,.0f}", delta="Moderate", delta_color="off")
        else:
            st.metric("Total PnL", f"${pnl:,.0f}", delta="Weak", delta_color="inverse")

    # Log panel creation for agentic monitoring
    logger.info(f"System metrics panel created with {len(metrics)} metrics")

    return {
        "success": True,
        "panel_created": True,
        "metrics_count": len(metrics),
        "key_metrics": {
            "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
            "total_return": metrics.get("total_return", 0.0),
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "win_rate": metrics.get("win_rate", 0.0),
            "total_pnl": metrics.get("total_pnl", 0.0),
        },
    }


def create_strategy_pipeline_selector(
    key: str = "strategy_pipeline_selector",
    allow_combos: bool = True,
) -> Dict[str, Any]:
    """Create a strategy pipeline selector with combo functionality.

    Args:
        key: Unique key for the Streamlit component
        allow_combos: Whether to allow strategy combinations

    Returns:
        Dictionary containing selected strategies and combo settings
    """
    try:
        from strategies.strategy_pipeline import get_strategy_names, get_combine_modes
        
        # Get available strategies from pipeline
        strategy_names = get_strategy_names()
        combine_modes = get_combine_modes()
        
        if not strategy_names:
            st.warning("No strategies available in pipeline")
            return {}
        
        # Strategy selection
        if allow_combos:
            st.subheader("🎯 Strategy Selection")
            selected_strategies = st.multiselect(
                "Select Strategies",           options=strategy_names,
                default=[strategy_names[0]] if strategy_names else [],
                help="Select one or more strategies to combine"
            )
            
            if not selected_strategies:
                st.warning("Please select at least one strategy")
                return {}
            
            # Show selected strategies
            if len(selected_strategies) > 1:
                st.info(f"Selected strategies: {', '.join(selected_strategies)}")
                
                # Combination mode selection
                st.subheader("🔗 Combination Mode")
                combine_mode = st.selectbox(
                 "How to combine signals",
                    options=combine_modes,
                    index=1,  # Default to intersection
                    help="union: signal if any strategy signals, intersection: signal only if all strategies agree, weighted: weighted sum of signals"
                )
                
                # Weights for weighted mode
                weights = None
                if combine_mode == 'weighted':
                    st.subheader("⚖️ Strategy Weights")
                    weights = []
                    for i, strategy in enumerate(selected_strategies):
                        weight = st.slider(
                            f"Weight for {strategy}",
                            min_value=0.0,
                            max_value=2.0,
                            value=1.0,
                            step=0.1,
                            key=f"{key}_weight_{i}"
                        )
                        weights.append(weight)
                    
                    # Normalize weights
                    if weights:
                        total_weight = sum(weights)
                        if total_weight > 0:
                            weights = [w / total_weight for w in weights]
                        st.info(f"Normalized weights: {[f'{w:.2f}' for w in weights]}")
                
                # Strategy parameters
                st.subheader("⚙️ Strategy Parameters")
                strategy_params = {}
                
                for strategy in selected_strategies:
                    with st.expander(f"Parameters for {strategy}"):
                        if strategy == "RSI":
                            strategy_params[strategy] = {
                             'window': st.slider(f"{strategy} Window", 5, 14, key=f"{key}_rsi_window"),
                                'overbought': st.slider(f"{strategy} Overbought", 60, 90, key=f"{key}_rsi_overbought"),
                               'oversold': st.slider(f"{strategy} Oversold", 10, 40, key=f"{key}_rsi_oversold")
                            }
                        elif strategy == "MACD":
                            strategy_params[strategy] = {
                               'fast': st.slider(f"{strategy} Fast Period", 8, 20, key=f"{key}_macd_fast"),
                               'slow': st.slider(f"{strategy} Slow Period", 20, 40, key=f"{key}_macd_slow"),
                             'signal': st.slider(f"{strategy} Signal Period", 5, 15, key=f"{key}_macd_signal")
                            }
                        elif strategy == "Bollinger":
                            strategy_params[strategy] = {
                             'window': st.slider(f"{strategy} Window", 10, 20, key=f"{key}_bollinger_window"),
                              'num_std': st.slider(f"{strategy} Standard Deviations", 1.0, 3.0, step=0.1, key=f"{key}_bollinger_std")
                            }
                        elif strategy == "SMA":
                            strategy_params[strategy] = {
                             'window': st.slider(f"{strategy} Window", 5, 20, key=f"{key}_sma_window")
                            }
                
                result = {
               'strategies': selected_strategies,
                 'combine_mode': combine_mode,
                    'weights': weights,
               'parameters': strategy_params,
                  'is_combo': len(selected_strategies) > 1
                }
                
            else:
                # Single strategy selected
                strategy = selected_strategies[0]
                st.info(f"Single strategy selected: {strategy}")
                
                # Strategy parameters for single strategy
                st.subheader("⚙️ Strategy Parameters")
                strategy_params = {}
                
                with st.expander(f"Parameters for {strategy}"):
                    if strategy == "RSI":
                        strategy_params[strategy] = {
                         'window': st.slider(f"{strategy} Window", 5, 14, key=f"{key}_rsi_window"),
                            'overbought': st.slider(f"{strategy} Overbought", 60, 90, key=f"{key}_rsi_overbought"),
                           'oversold': st.slider(f"{strategy} Oversold", 10, 40, key=f"{key}_rsi_oversold")
                        }
                    elif strategy == "MACD":
                        strategy_params[strategy] = {
                           'fast': st.slider(f"{strategy} Fast Period", 8, 20, key=f"{key}_macd_fast"),
                           'slow': st.slider(f"{strategy} Slow Period", 20, 40, key=f"{key}_macd_slow"),
                         'signal': st.slider(f"{strategy} Signal Period", 5, 15, key=f"{key}_macd_signal")
                        }
                    elif strategy == "Bollinger":
                        strategy_params[strategy] = {
                         'window': st.slider(f"{strategy} Window", 10, 20, key=f"{key}_bollinger_window"),
                          'num_std': st.slider(f"{strategy} Standard Deviations", 1.0, 3.0, step=0.1, key=f"{key}_bollinger_std")
                        }
                    elif strategy == "SMA":
                        strategy_params[strategy] = {
                         'window': st.slider(f"{strategy} Window", 5, 20, key=f"{key}_sma_window")
                        }
                
                result = {
               'strategies': selected_strategies,
                 'combine_mode': None,
                   'weights': [],
                'parameters': strategy_params,
                    'is_combo': False
                }
        else:
            # Single strategy selection only
            selected_strategy = st.selectbox(
              "Select Strategy",           options=strategy_names,
                key=key
            )
            
            result = {
                'strategies': [selected_strategy] if selected_strategy else [],
                'combine_mode': None,
                'weights': [],
                'parameters': {},
                'is_combo': False
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
    key: str = "strategy_execution"
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
            rsi_strategy, macd_strategy, bollinger_strategy, sma_strategy,
            combine_signals
        )
        
        if not strategy_config or not strategy_config.get('strategies'):
            st.error("No strategies selected")
            return {}
        
        strategies = strategy_config['strategies']
        parameters = strategy_config.get('parameters', {})
        combine_mode = strategy_config.get('combine_mode')
        weights = strategy_config.get('weights')
        
        # Generate individual strategy signals
        signals_list = []
        strategy_names = []
        
        for strategy in strategies:
            strategy_params = parameters.get(strategy, {})
            if strategy == "RSI":
                signal = rsi_strategy(
                    market_data,
                    window=strategy_params.get('window', 14),
                    overbought=strategy_params.get('overbought', 70),
                    oversold=strategy_params.get('oversold', 30)
                )
            elif strategy == "MACD":
                signal = macd_strategy(
                    market_data,
                    fast=strategy_params.get('fast', 12),
                    slow=strategy_params.get('slow', 26),
                    signal=strategy_params.get('signal', 9)
                )
            elif strategy == "Bollinger":
                signal = bollinger_strategy(
                    market_data,
                    window=strategy_params.get('window', 20),
                    num_std=strategy_params.get('num_std', 2.0)
                )
            elif strategy == "SMA":
                signal = sma_strategy(
                    market_data,
                    window=strategy_params.get('window', 20)
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
                signals_list,
                mode=combine_mode,
                weights=weights
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
           'signal': final_signal,
          'individual_signals': dict(zip(strategy_names, signals_list)),
        'signal_type': signal_type,
            'strategies_used': strategies,
         'combine_mode': combine_mode,
            'weights': weights,
        'performance': performance_metrics,
           'execution_time':datetime.now().isoformat()
        }
        
        # Log execution for agentic monitoring
        logger.info(f"Strategy pipeline executed: {len(strategies)} strategies, mode: {combine_mode}")
        
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
    strategy_names: List[str]
) -> Dict[str, Any]:
    """Calculate basic performance metrics for strategy signals.

    Args:
        market_data: Market data DataFrame
        signals: Signal series (1y,-1l, 0=hold)
        strategy_names: List of strategy names used

    Returns:
        Dictionary of performance metrics
    """
    try:
        # Calculate returns
        price_returns = market_data['close'].pct_change()
        
        # Calculate strategy returns (assuming 100% position size)
        strategy_returns = price_returns * signals.shift(1)
        
        # Basic metrics
        total_return = strategy_returns.sum()
        sharpe_ratio = strategy_returns.mean() / (strategy_returns.std() + 1e-9) * np.sqrt(252)
        max_drawdown = (strategy_returns.cumsum() - strategy_returns.cumsum().expanding().max()).min()
        
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
         'total_return': total_return,
         'sharpe_ratio': sharpe_ratio,
         'max_drawdown': max_drawdown,
            'buy_signals': buy_signals,
         'sell_signals': sell_signals,
         'hold_signals': hold_signals,
          'total_signals': total_signals,
         'win_rate': win_rate,
           'strategy_names': strategy_names
        }
        
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}")
        return {
           'error': str(e),
           'strategy_names': strategy_names
        }
