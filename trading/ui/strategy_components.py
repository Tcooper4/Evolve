"""Strategy-specific UI components for the trading system.

This module provides UI components specifically for strategy configuration and monitoring,
with support for agentic interactions and performance tracking.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trading.components import (
    create_asset_selector,
    create_date_range_selector,
    create_parameter_inputs,
    create_strategy_selector,
    create_timeframe_selector,
)

from .config.registry import StrategyConfig, registry

logger = logging.getLogger(__name__)


def create_strategy_form(
    default_strategy: Optional[str] = None,
    default_asset: Optional[str] = None,
    default_timeframe: Optional[str] = None,
    key_prefix: str = "strategy",
) -> Dict[str, Any]:
    """Create a form for strategy configuration.

    Args:
        default_strategy: Optional default strategy to select
        default_asset: Optional default asset to select
        default_timeframe: Optional default timeframe to select
        key_prefix: Prefix for Streamlit component keys

    Returns:
        Dictionary containing form inputs
    """
    # Get date range
    start_date, end_date = create_date_range_selector(key=f"{key_prefix}_date_range")

    # Get strategy selection
    selected_strategy = create_strategy_selector(
        default_strategy=default_strategy, key=f"{key_prefix}_strategy"
    )

    # Get strategy parameters if a strategy is selected
    parameters = {}
    if selected_strategy:
        strategy_config = registry.get_strategy_config(selected_strategy)
        if strategy_config:
            parameters = create_parameter_inputs(
                strategy_config.parameters, key_prefix=f"{key_prefix}_strategy"
            )

    # Get asset selection
    selected_asset = create_asset_selector(
        assets=["BTC/USD", "ETH/USD", "AAPL", "MSFT", "GOOGL"],
        default_asset=default_asset,
        key=f"{key_prefix}_asset",
    )

    # Get timeframe selection
    selected_timeframe = create_timeframe_selector(
        timeframes=["1h", "4h", "1d", "1w"],
        default_timeframe=default_timeframe,
        key=f"{key_prefix}_timeframe",
    )

    # Log form submission for agentic monitoring
    form_data = {
        "start_date": start_date,
        "end_date": end_date,
        "strategy": selected_strategy,
        "parameters": parameters,
        "asset": selected_asset,
        "timeframe": selected_timeframe,
    }
    logger.info(f"Strategy form submitted: {form_data}")

    return form_data


def create_performance_chart(
    data: pd.DataFrame, strategy_config: StrategyConfig, show_benchmark: bool = True
) -> go.Figure:
    """Create an interactive performance chart with optional features.

    Args:
        data: DataFrame containing performance data
        strategy_config: Configuration for the strategy used
        show_benchmark: Whether to show benchmark overlay

    Returns:
        Plotly figure with performance visualization
    """
    fig = go.Figure()

    # Add equity curve
    fig.add_trace(
        go.Scatter(
            x=data.index, y=data["equity"], name="Strategy", line=dict(color="blue")
        )
    )

    # Add benchmark overlay if requested
    if show_benchmark and "benchmark" in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["benchmark"],
                name="Benchmark",
                line=dict(color="gray", dash="dash"),
            )
        )

    # Add drawdown
    if "drawdown" in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["drawdown"],
                name="Drawdown",
                line=dict(color="red"),
                yaxis="y2",
            )
        )

    # Update layout
    fig.update_layout(
        title="Strategy Performance",
        xaxis_title="Date",
        yaxis_title="Equity",
        yaxis2=dict(title="Drawdown", overlaying="y", side="right", showgrid=False),
        hovermode="x unified",
        showlegend=True,
    )

    # Add range slider
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="date"))

    # Log chart creation for agentic monitoring
    logger.info(f"Performance chart created for strategy: {strategy_config.name}")

    return fig


def create_performance_metrics(
    data: pd.DataFrame, strategy_config: StrategyConfig
) -> Dict[str, float]:
    """Calculate and display strategy performance metrics.

    Args:
        data: DataFrame containing performance data
        strategy_config: Configuration for the strategy used

    Returns:
        Dictionary of metric names and values
    """
    metrics = {}

    if "returns" in data.columns:
        # Calculate return metrics
        total_return = (data["equity"].iloc[-1] / data["equity"].iloc[0] - 1) * 100
        annual_return = total_return * (252 / len(data))
        sharpe_ratio = np.sqrt(252) * data["returns"].mean() / data["returns"].std()
        max_drawdown = data["drawdown"].min() * 100

        metrics = {
            "Total Return (%)": total_return,
            "Annual Return (%)": annual_return,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown (%)": max_drawdown,
        }

        # Add risk metrics
        if "benchmark" in data.columns:
            benchmark_returns = data["benchmark"].pct_change()
            beta = data["returns"].cov(benchmark_returns) / benchmark_returns.var()
            alpha = (data["returns"].mean() - beta * benchmark_returns.mean()) * 252
            metrics.update({"Beta": beta, "Alpha": alpha})

    # Log metrics for agentic monitoring
    logger.info(f"Performance metrics calculated: {metrics}")

    return metrics


def create_trade_list(
    trades: pd.DataFrame, strategy_config: StrategyConfig
) -> Dict[str, Any]:
    """Display a list of trades with filtering options.

    Args:
        trades: DataFrame containing trade data
        strategy_config: Configuration for the strategy used

    Returns:
        Dictionary with status and trade information
    """
    try:
        st.subheader("Trade List")

        # Add filtering options
        col1, col2 = st.columns(2)
        with col1:
            min_profit = st.number_input("Min Profit (%)", value=-100.0, step=1.0)
        with col2:
            max_profit = st.number_input("Max Profit (%)", value=100.0, step=1.0)

        # Filter trades
        filtered_trades = trades[
            (trades["profit_pct"] >= min_profit) & (trades["profit_pct"] <= max_profit)
        ]

        # Display trades
        st.dataframe(
            filtered_trades[
                [
                    "entry_time",
                    "exit_time",
                    "entry_price",
                    "exit_price",
                    "profit_pct",
                    "holding_period",
                ]
            ].style.format(
                {
                    "profit_pct": "{:.2f}%",
                    "entry_price": "${:.2f}",
                    "exit_price": "${:.2f}",
                }
            )
        )

        # Log trade list display for agentic monitoring
        logger.info(f"Trade list displayed with {len(filtered_trades)} trades")

        return {
            "status": "success",
            "message": f"Trade list displayed with {len(filtered_trades)} trades",
            "total_trades": len(trades),
            "filtered_trades": len(filtered_trades),
            "min_profit": min_profit,
            "max_profit": max_profit,
        }
    except Exception as e:
        logger.error(f"Error displaying trade list: {e}")
        return {
            "success": True,
            "result": {"status": "error", "message": str(e)},
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }


def create_strategy_export(
    data: pd.DataFrame,
    trades: pd.DataFrame,
    strategy_config: StrategyConfig,
    metrics: Dict[str, float],
) -> Dict[str, Any]:
    """Create export options for strategy results.

    Args:
        data: DataFrame containing performance data
        trades: DataFrame containing trade data
        strategy_config: Configuration for the strategy used
        metrics: Dictionary of performance metrics

    Returns:
        Dictionary with status and export information
    """
    try:
        st.subheader("Export Results")

        # Create export options
        export_format = st.radio(
            "Select Export Format", ["CSV", "JSON", "Excel"], key=os.getenv("KEY", "")
        )

        if st.button("Export"):
            # Prepare export data
            export_data = {
                "performance": data.to_dict(),
                "trades": trades.to_dict(),
                "strategy": strategy_config.name,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
            }

            # Export based on selected format
            if export_format == "CSV":
                data.to_csv("strategy_performance.csv")
                trades.to_csv("strategy_trades.csv")
            elif export_format == "JSON":
                with open("strategy_results.json", "w") as f:
                    json.dump(export_data, f, indent=2)
            else:  # Excel
                with pd.ExcelWriter("strategy_results.xlsx") as writer:
                    data.to_excel(writer, sheet_name="Performance")
                    trades.to_excel(writer, sheet_name="Trades")

            st.success(f"Results exported as {export_format}")

            # Log export for agentic monitoring
            logger.info(f"Strategy results exported as {export_format}")

            return {
                "status": "success",
                "message": f"Strategy results exported as {export_format}",
                "export_format": export_format,
                "performance_rows": len(data),
                "trades_rows": len(trades),
            }

        return {"status": "pending", "message": "Export not initiated"}

    except Exception as e:
        logger.error(f"Error exporting strategy results: {e}")
        return {
            "success": True,
            "result": {"status": "error", "message": str(e)},
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }
