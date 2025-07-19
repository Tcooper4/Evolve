"""
Trading Cost Configuration Component

This module provides a Streamlit sidebar component for configuring trading costs
including commission, slippage, spread, and cash drag parameters.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
from dataclasses import dataclass

from trading.backtesting.performance_analysis import CostParameters


@dataclass
class CostConfigUI:
    """UI configuration for cost parameters."""
    show_advanced: bool = False
    show_presets: bool = True
    show_cost_breakdown: bool = True
    show_validation: bool = True


def render_cost_config_sidebar(
    config_ui: Optional[CostConfigUI] = None,
    default_params: Optional[CostParameters] = None
) -> CostParameters:
    """
    Render cost configuration in Streamlit sidebar.

    Args:
        config_ui: UI configuration options
        default_params: Default cost parameters

    Returns:
        Configured CostParameters object
    """
    if config_ui is None:
        config_ui = CostConfigUI()

    if default_params is None:
        default_params = CostParameters()

    st.sidebar.header("💰 Trading Costs")

    # Enable/disable cost adjustment
    enable_cost_adjustment = st.sidebar.checkbox(
        "Enable Cost Adjustment",
        value=default_params.enable_cost_adjustment,
        help="Enable realistic trading costs in backtesting"
    )

    if not enable_cost_adjustment:
        return CostParameters(enable_cost_adjustment=False)

    # Preset configurations
    if config_ui.show_presets:
        st.sidebar.subheader("📋 Cost Presets")
        preset = st.sidebar.selectbox(
            "Select Cost Preset:",
            options=[
                "Custom",
                "Retail Trading",
                "Institutional",
                "High Frequency",
                "Crypto Trading",
                "Low Cost"
            ],
            help="Choose from predefined cost configurations"
        )

        # Apply preset
        if preset != "Custom":
            cost_params = _get_preset_params(preset)
            st.sidebar.info(f"Applied {preset} preset")
        else:
            cost_params = default_params
    else:
        cost_params = default_params

    # Basic cost parameters
    st.sidebar.subheader("📊 Basic Costs")

    commission_rate = st.sidebar.slider(
        "Commission Rate (%)",
        min_value=0.0,
        max_value=1.0,
        value=cost_params.commission_rate * 100,
        step=0.01,
        help="Commission as percentage of trade value"
    ) / 100

    slippage_rate = st.sidebar.slider(
        "Slippage Rate (%)",
        min_value=0.0,
        max_value=1.0,
        value=cost_params.slippage_rate * 100,
        step=0.01,
        help="Slippage as percentage of trade value"
    ) / 100

    spread_rate = st.sidebar.slider(
        "Bid-Ask Spread (%)",
        min_value=0.0,
        max_value=1.0,
        value=cost_params.spread_rate * 100,
        step=0.01,
        help="Bid-ask spread as percentage of price"
    ) / 100

    # Advanced parameters
    if config_ui.show_advanced:
        st.sidebar.subheader("⚙️ Advanced Settings")

        col1, col2 = st.sidebar.columns(2)

        with col1:
            min_commission = st.number_input(
                "Min Commission ($)",
                min_value=0.0,
                max_value=100.0,
                value=cost_params.min_commission,
                step=0.5,
                help="Minimum commission per trade"
            )

        with col2:
            max_commission = st.number_input(
                "Max Commission ($)",
                min_value=0.0,
                max_value=10000.0,
                value=cost_params.max_commission,
                step=10.0,
                help="Maximum commission per trade"
            )

        cash_drag_rate = st.sidebar.slider(
            "Cash Drag Rate (%/year)",
            min_value=0.0,
            max_value=10.0,
            value=cost_params.cash_drag_rate * 100,
            step=0.1,
            help="Annual opportunity cost of holding cash"
        ) / 100

    # Cost breakdown preview
    if config_ui.show_cost_breakdown:
        st.sidebar.subheader("📈 Cost Impact Preview")

        # Sample trade calculation
        sample_trade_value = st.sidebar.number_input(
            "Sample Trade Value ($)",
            min_value=100.0,
            max_value=100000.0,
            value=10000.0,
            step=100.0,
            help="Calculate costs for a sample trade"
        )

        if sample_trade_value > 0:
            _show_cost_breakdown_preview(
                sample_trade_value,
                commission_rate,
                slippage_rate,
                spread_rate,
                min_commission if config_ui.show_advanced else cost_params.min_commission,
                max_commission if config_ui.show_advanced else cost_params.max_commission
            )

    # Validation
    if config_ui.show_validation:
        _validate_cost_parameters(
            commission_rate,
            slippage_rate,
            spread_rate,
            min_commission if config_ui.show_advanced else cost_params.min_commission,
            max_commission if config_ui.show_advanced else cost_params.max_commission
        )

    # Create and return CostParameters object
    return CostParameters(
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        spread_rate=spread_rate,
        cash_drag_rate=cash_drag_rate if config_ui.show_advanced else cost_params.cash_drag_rate,
        min_commission=min_commission if config_ui.show_advanced else cost_params.min_commission,
        max_commission=max_commission if config_ui.show_advanced else cost_params.max_commission,
        enable_cost_adjustment=enable_cost_adjustment
    )


def _get_preset_params(preset: str) -> CostParameters:
    """Get preset cost parameters."""
    presets = {
        "Retail Trading": CostParameters(
            commission_rate=0.001,  # 0.1%
            slippage_rate=0.002,    # 0.2%
            spread_rate=0.0005,     # 0.05%
            cash_drag_rate=0.02,    # 2%
            min_commission=1.0,
            max_commission=1000.0
        ),
        "Institutional": CostParameters(
            commission_rate=0.0005,  # 0.05%
            slippage_rate=0.001,     # 0.1%
            spread_rate=0.0002,      # 0.02%
            cash_drag_rate=0.015,    # 1.5%
            min_commission=5.0,
            max_commission=5000.0
        ),
        "High Frequency": CostParameters(
            commission_rate=0.0001,  # 0.01%
            slippage_rate=0.0005,    # 0.05%
            spread_rate=0.0001,      # 0.01%
            cash_drag_rate=0.01,     # 1%
            min_commission=0.1,
            max_commission=100.0
        ),
        "Crypto Trading": CostParameters(
            commission_rate=0.002,   # 0.2%
            slippage_rate=0.003,     # 0.3%
            spread_rate=0.001,       # 0.1%
            cash_drag_rate=0.025,    # 2.5%
            min_commission=0.5,
            max_commission=500.0
        ),
        "Low Cost": CostParameters(
            commission_rate=0.0002,  # 0.02%
            slippage_rate=0.0005,    # 0.05%
            spread_rate=0.0001,      # 0.01%
            cash_drag_rate=0.01,     # 1%
            min_commission=0.5,
            max_commission=100.0
        )
    }

    return presets.get(preset, CostParameters())


def _show_cost_breakdown_preview(
    trade_value: float,
    commission_rate: float,
    slippage_rate: float,
    spread_rate: float,
    min_commission: float,
    max_commission: float
):
    """Show cost breakdown preview for a sample trade."""

    # Calculate costs
    commission = max(
        min_commission,
        min(trade_value * commission_rate, max_commission)
    )
    slippage = trade_value * slippage_rate
    spread = trade_value * spread_rate
    total_cost = commission + slippage + spread

    # Display breakdown
    col1, col2 = st.sidebar.columns(2)

    with col1:
        st.metric("Commission", f"${commission:.2f}")
        st.metric("Slippage", f"${slippage:.2f}")

    with col2:
        st.metric("Spread", f"${spread:.2f}")
        st.metric("Total Cost", f"${total_cost:.2f}")

    # Cost percentage
    cost_percentage = (total_cost / trade_value) * 100
    st.sidebar.metric("Cost %", f"{cost_percentage:.3f}%")

    # Color coding based on cost level
    if cost_percentage < 0.1:
        st.sidebar.success("✅ Low cost")
    elif cost_percentage < 0.3:
        st.sidebar.info("ℹ️ Moderate cost")
    else:
        st.sidebar.warning("⚠️ High cost")


def _validate_cost_parameters(
    commission_rate: float,
    slippage_rate: float,
    spread_rate: float,
    min_commission: float,
    max_commission: float
):
    """Validate cost parameters and show warnings."""
    warnings = []

    # Check for reasonable ranges
    if commission_rate > 0.01:  # > 1%
        warnings.append("Commission rate seems high (>1%)")

    if slippage_rate > 0.01:  # > 1%
        warnings.append("Slippage rate seems high (>1%)")

    if spread_rate > 0.005:  # > 0.5%
        warnings.append("Spread rate seems high (>0.5%)")

    if min_commission > max_commission:
        warnings.append("Min commission > Max commission")

    total_rate = commission_rate + slippage_rate + spread_rate
    if total_rate > 0.02:  # > 2%
        warnings.append("Total cost rate seems high (>2%)")

    # Display warnings
    for warning in warnings:
        st.sidebar.warning(warning)


def render_cost_summary(metrics: Dict[str, Any]) -> None:
    """
    Render cost summary from performance metrics.

    Args:
        metrics: Performance metrics dictionary
    """
    if not metrics:
        return

    st.subheader("💰 Trading Cost Summary")

    # Check if cost metrics are available
    cost_metrics = [
        "total_trading_costs",
        "cost_per_trade",
        "cost_impact",
        "total_commission",
        "total_slippage",
        "total_spread",
        "cost_percentage"
    ]

    if not any(metric in metrics for metric in cost_metrics):
        st.info("No cost metrics available. Enable cost adjustment in settings.")
        return

    # Create columns for metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        if "total_trading_costs" in metrics:
            st.metric(
                "Total Trading Costs",
                f"${metrics['total_trading_costs']:,.2f}"
            )

        if "cost_per_trade" in metrics:
            st.metric(
                "Cost per Trade",
                f"${metrics['cost_per_trade']:.2f}"
            )

    with col2:
        if "cost_impact" in metrics:
            st.metric(
                "Cost Impact",
                f"{metrics['cost_impact']:.2f}%",
                delta=f"-{metrics['cost_impact']:.2f}%"
            )

        if "cost_percentage" in metrics:
            st.metric(
                "Cost % of Volume",
                f"{metrics['cost_percentage']:.3f}%"
            )

    with col3:
        if "num_trades" in metrics:
            st.metric(
                "Number of Trades",
                f"{metrics['num_trades']:,}"
            )

        if "turnover_ratio" in metrics and not pd.isna(metrics['turnover_ratio']):
            st.metric(
                "Turnover Ratio",
                f"{metrics['turnover_ratio']:.2f}"
            )

    # Cost breakdown chart
    if all(metric in metrics for metric in ["total_commission", "total_slippage", "total_spread"]):
        st.subheader("📊 Cost Breakdown")

        cost_data = {
            "Commission": metrics["total_commission"],
            "Slippage": metrics["total_slippage"],
            "Spread": metrics["total_spread"]
        }

        # Create pie chart
        import plotly.express as px

        fig = px.pie(
            values=list(cost_data.values()),
            names=list(cost_data.keys()),
            title="Trading Cost Breakdown"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Cash efficiency metrics
    if any(metric in metrics for metric in ["avg_cash_utilization", "cash_drag_cost", "cash_drag_percentage"]):
        st.subheader("💵 Cash Efficiency")

        col1, col2, col3 = st.columns(3)

        with col1:
            if "avg_cash_utilization" in metrics and not pd.isna(metrics['avg_cash_utilization']):
                st.metric(
                    "Avg Cash Utilization",
                    f"{metrics['avg_cash_utilization']*100:.1f}%"
                )

        with col2:
            if "cash_drag_cost" in metrics:
                st.metric(
                    "Cash Drag Cost",
                    f"${metrics['cash_drag_cost']:,.2f}"
                )

        with col3:
            if "cash_drag_percentage" in metrics:
                st.metric(
                    "Cash Drag %",
                    f"{metrics['cash_drag_percentage']:.2f}%"
                )


def get_cost_config_from_session() -> CostParameters:
    """Get cost configuration from Streamlit session state."""
    if "cost_params" not in st.session_state:
        st.session_state.cost_params = CostParameters()

    return st.session_state.cost_params


def save_cost_config_to_session(cost_params: CostParameters) -> None:
    """Save cost configuration to Streamlit session state."""
    st.session_state.cost_params = cost_params 