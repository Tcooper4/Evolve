"""
Hybrid Model Configuration UI Component

This module provides a Streamlit sidebar component for configuring the hybrid model's
risk-aware weighting parameters and ensemble methods.
"""

import streamlit as st
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass

from trading.forecasting.hybrid_model import HybridModel


@dataclass
class HybridModelConfigUI:
    """UI configuration for hybrid model parameters."""
    show_advanced: bool = False
    show_performance_summary: bool = True
    show_weighting_info: bool = True
    show_validation: bool = True


def render_hybrid_model_config_sidebar(
    hybrid_model: HybridModel,
    config_ui: Optional[HybridModelConfigUI] = None
) -> Dict[str, Any]:
    """
    Render hybrid model configuration in Streamlit sidebar.

    Args:
        hybrid_model: HybridModel instance to configure
        config_ui: UI configuration options

    Returns:
        Dictionary with updated configuration
    """
    if config_ui is None:
        config_ui = HybridModelConfigUI()

    st.sidebar.header("ðŸ¤– Hybrid Model Configuration")

    # Get current configuration
    current_config = hybrid_model.scoring_config
    weighting_info = hybrid_model.get_weighting_metric_info()

    # Ensemble Method Selection
    st.sidebar.subheader("ðŸ“Š Ensemble Method")
    method = st.sidebar.selectbox(
        "Select Ensemble Method:",
        options=list(weighting_info["available_methods"].keys()),
        index=list(weighting_info["available_methods"].keys()).index(current_config["method"]),
        help="Choose how to weight the ensemble models"
    )

    # Weighting Metric Selection (for risk-aware method)
    if method == "risk_aware":
        st.sidebar.subheader("ðŸŽ¯ Risk-Aware Weighting")

        metric = st.sidebar.selectbox(
            "Select Weighting Metric:",
            options=list(weighting_info["available_metrics"].keys()),
            index=list(weighting_info["available_metrics"].keys()).index(current_config["weighting_metric"]),
            help="Choose the primary metric for risk-aware weighting"
        )

        # Show metric description
        if config_ui.show_weighting_info:
            metric_info = weighting_info["available_metrics"][metric]
            st.sidebar.info(f"**{metric.title()} Weighting**: {metric_info['description']}")

        # Metric-specific parameters
        if metric == "sharpe":
            sharpe_floor = st.sidebar.slider(
                "Sharpe Floor",
                min_value=-2.0,
                max_value=2.0,
                value=current_config["sharpe_floor"],
                step=0.1,
                help="Minimum Sharpe ratio to avoid negative weights"
            )
            current_config["sharpe_floor"] = sharpe_floor

        elif metric == "drawdown":
            drawdown_ceiling = st.sidebar.slider(
                "Drawdown Ceiling",
                min_value=-1.0,
                max_value=0.0,
                value=current_config["drawdown_ceiling"],
                step=0.05,
                help="Maximum drawdown threshold (negative values)"
            )
            current_config["drawdown_ceiling"] = drawdown_ceiling

        elif metric == "mse":
            mse_ceiling = st.sidebar.slider(
                "MSE Ceiling",
                min_value=100.0,
                max_value=10000.0,
                value=current_config["mse_ceiling"],
                step=100.0,
                help="Maximum MSE threshold"
            )
            current_config["mse_ceiling"] = mse_ceiling

    # Advanced Settings
    if config_ui.show_advanced:
        st.sidebar.subheader("âš™ï¸ Advanced Settings")

        min_performance_threshold = st.sidebar.slider(
            "Min Performance Threshold",
            min_value=0.0,
            max_value=1.0,
            value=current_config["min_performance_threshold"],
            step=0.05,
            help="Minimum performance to avoid zero weights"
        )
        current_config["min_performance_threshold"] = min_performance_threshold

        recency_weight = st.sidebar.slider(
            "Recency Weight",
            min_value=0.0,
            max_value=1.0,
            value=current_config["recency_weight"],
            step=0.1,
            help="Weight for recent vs historical performance"
        )
        current_config["recency_weight"] = recency_weight

        risk_free_rate = st.sidebar.slider(
            "Risk-Free Rate",
            min_value=0.0,
            max_value=0.1,
            value=current_config["risk_free_rate"],
            step=0.001,
            help="Risk-free rate for Sharpe calculations"
        )
        current_config["risk_free_rate"] = risk_free_rate

    # Update configuration
    current_config["method"] = method
    if method == "risk_aware":
        current_config["weighting_metric"] = metric

    # Apply configuration
    if st.sidebar.button("Apply Configuration"):
        hybrid_model.set_scoring_config(current_config)
        st.sidebar.success("âœ… Configuration applied!")

    # Performance Summary
    if config_ui.show_performance_summary:
        st.sidebar.subheader("ðŸ“ˆ Model Performance")

        summary = hybrid_model.get_model_performance_summary()

        for model_name, model_info in summary.items():
            if model_info["status"] == "active":
                col1, col2 = st.sidebar.columns(2)

                with col1:
                    st.metric(
                        f"{model_name} Weight",
                        f"{model_info['current_weight']:.2%}"
                    )

                with col2:
                    avg_sharpe = model_info["avg_metrics"].get("sharpe_ratio", 0)
                    st.metric(
                        "Avg Sharpe",
                        f"{avg_sharpe:.2f}"
                    )

    # Validation
    if config_ui.show_validation:
        _validate_hybrid_config(current_config, method)

    return current_config


def render_weighting_metric_comparison(hybrid_model: HybridModel):
    """
    Render a comparison of different weighting metrics.

    Args:
        hybrid_model: HybridModel instance
    """
    st.subheader("ðŸ“Š Weighting Metric Comparison")

    # Get current configuration
    original_config = hybrid_model.scoring_config.copy()
    original_weights = hybrid_model.weights.copy()

    # Test different weighting metrics
    metrics = ["sharpe", "drawdown", "mse"]
    metric_results = {}

    for metric in metrics:
        # Temporarily set the metric
        hybrid_model.set_weighting_metric(metric)
        metric_results[metric] = hybrid_model.weights.copy()

    # Restore original configuration
    hybrid_model.scoring_config = original_config
    hybrid_model.weights = original_weights

    # Display comparison
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Sharpe Ratio Weighting")
        _display_weights(metric_results["sharpe"])

    with col2:
        st.subheader("Drawdown Weighting")
        _display_weights(metric_results["drawdown"])

    with col3:
        st.subheader("MSE Weighting")
        _display_weights(metric_results["mse"])

    # Create comparison chart
    _create_weighting_comparison_chart(metric_results)


def _display_weights(weights: Dict[str, float]):
    """Display model weights in a formatted way."""
    for model_name, weight in weights.items():
        st.metric(
            model_name,
            f"{weight:.2%}"
        )


def _create_weighting_comparison_chart(metric_results: Dict[str, Dict[str, float]]):
    """Create a comparison chart of different weighting methods."""
    import plotly.express as px
    import pandas as pd

    # Prepare data for plotting
    data = []
    for metric, weights in metric_results.items():
        for model_name, weight in weights.items():
            data.append({
                "Metric": metric.title(),
                "Model": model_name,
                "Weight": weight
            })

    df = pd.DataFrame(data)

    # Create bar chart
    fig = px.bar(
        df,
        x="Model",
        y="Weight",
        color="Metric",
        title="Model Weights by Weighting Metric",
        barmode="group"
    )

    st.plotly_chart(fig, use_container_width=True)


def render_model_performance_dashboard(hybrid_model: HybridModel):
    """
    Render a comprehensive model performance dashboard.

    Args:
        hybrid_model: HybridModel instance
    """
    st.subheader("ðŸ“Š Model Performance Dashboard")

    summary = hybrid_model.get_model_performance_summary()

    # Create performance metrics table
    performance_data = []

    for model_name, model_info in summary.items():
        if model_info["status"] == "active":
            avg_metrics = model_info["avg_metrics"]
            performance_data.append({
                "Model": model_name,
                "Weight": f"{model_info['current_weight']:.2%}",
                "Sharpe": f"{avg_metrics.get('sharpe_ratio', 0):.3f}",
                "Win Rate": f"{avg_metrics.get('win_rate', 0):.2%}",
                "Max DD": f"{avg_metrics.get('max_drawdown', 0):.2%}",
                "MSE": f"{avg_metrics.get('mse', 0):.2f}",
                "Total Return": f"{avg_metrics.get('total_return', 0):.2%}",
                "Performance Count": model_info["performance_count"]
            })

    if performance_data:
        import pandas as pd
        df = pd.DataFrame(performance_data)
        st.dataframe(df, use_container_width=True)

        # Create performance visualization
        _create_performance_visualization(summary)
    else:
        st.info("No performance data available. Run the hybrid model first.")


def _create_performance_visualization(summary: Dict[str, Any]):
    """Create performance visualization charts."""
    import plotly.express as px
    import pandas as pd

    # Prepare data for visualization
    viz_data = []

    for model_name, model_info in summary.items():
        if model_info["status"] == "active":
            avg_metrics = model_info["avg_metrics"]
            viz_data.append({
                "Model": model_name,
                "Sharpe Ratio": avg_metrics.get("sharpe_ratio", 0),
                "Win Rate": avg_metrics.get("win_rate", 0),
                "Max Drawdown": abs(avg_metrics.get("max_drawdown", 0)),  # Use absolute value for visualization
                "Weight": model_info["current_weight"]
            })

    if not viz_data:
        return

    df = pd.DataFrame(viz_data)

    # Create radar chart for performance metrics
    fig = px.line_polar(
        df,
        r=["Sharpe Ratio", "Win Rate", "Max Drawdown"],
        theta=["Sharpe Ratio", "Win Rate", "Max Drawdown"],
        color="Model",
        line_close=True,
        title="Model Performance Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Create weight distribution chart
    fig2 = px.pie(
        df,
        values="Weight",
        names="Model",
        title="Current Model Weights"
    )

    st.plotly_chart(fig2, use_container_width=True)


def _validate_hybrid_config(config: Dict[str, Any], method: str):
    """Validate hybrid model configuration and show warnings."""
    warnings = []

    # Check method-specific parameters
    if method == "risk_aware":
        metric = config.get("weighting_metric", "sharpe")

        if metric == "sharpe":
            sharpe_floor = config.get("sharpe_floor", 0.0)
            if sharpe_floor < -1.0:
                warnings.append("Sharpe floor seems too low (< -1.0)")
            elif sharpe_floor > 1.0:
                warnings.append("Sharpe floor seems too high (> 1.0)")

        elif metric == "drawdown":
            drawdown_ceiling = config.get("drawdown_ceiling", -0.5)
            if drawdown_ceiling > -0.1:
                warnings.append("Drawdown ceiling seems too high (> -0.1)")
            elif drawdown_ceiling < -0.9:
                warnings.append("Drawdown ceiling seems too low (< -0.9)")

        elif metric == "mse":
            mse_ceiling = config.get("mse_ceiling", 1000.0)
            if mse_ceiling < 100.0:
                warnings.append("MSE ceiling seems too low (< 100)")
            elif mse_ceiling > 10000.0:
                warnings.append("MSE ceiling seems too high (> 10000)")

    # Check general parameters
    min_threshold = config.get("min_performance_threshold", 0.1)
    if min_threshold > 0.5:
        warnings.append("Min performance threshold seems high (> 0.5)")

    recency_weight = config.get("recency_weight", 0.7)
    if recency_weight > 0.9:
        warnings.append("Recency weight seems too high (> 0.9)")

    # Display warnings
    for warning in warnings:
        st.sidebar.warning(warning)


def get_hybrid_model_recommendations(hybrid_model: HybridModel) -> Dict[str, Any]:
    """
    Get recommendations for hybrid model configuration based on current performance.

    Args:
        hybrid_model: HybridModel instance

    Returns:
        Dictionary with recommendations
    """
    summary = hybrid_model.get_model_performance_summary()

    recommendations = {
        "best_weighting_metric": None,
        "model_improvements": [],
        "configuration_suggestions": []
    }

    # Analyze model performance
    model_performances = []
    for model_name, model_info in summary.items():
        if model_info["status"] == "active":
            avg_metrics = model_info["avg_metrics"]
            model_performances.append({
                "name": model_name,
                "sharpe": avg_metrics.get("sharpe_ratio", 0),
                "win_rate": avg_metrics.get("win_rate", 0),
                "drawdown": avg_metrics.get("max_drawdown", 0),
                "mse": avg_metrics.get("mse", 0)
            })

    if not model_performances:
        return recommendations

    # Determine best weighting metric
    sharpe_variance = np.var([m["sharpe"] for m in model_performances])
    drawdown_variance = np.var([abs(m["drawdown"]) for m in model_performances])
    mse_variance = np.var([m["mse"] for m in model_performances])

    variances = {
        "sharpe": sharpe_variance,
        "drawdown": drawdown_variance,
        "mse": mse_variance
    }

    best_metric = max(variances, key=variances.get)
    recommendations["best_weighting_metric"] = best_metric

    # Generate model improvement suggestions
    for model in model_performances:
        if model["sharpe"] < 0.5:
            recommendations["model_improvements"].append(
                f"{model['name']}: Low Sharpe ratio ({model['sharpe']:.2f}), consider retraining"
            )

        if model["win_rate"] < 0.4:
            recommendations["model_improvements"].append(
                f"{model['name']}: Low win rate ({model['win_rate']:.1%}), consider feature engineering"
            )

        if abs(model["drawdown"]) > 0.3:
            recommendations["model_improvements"].append(
                f"{model['name']}: High drawdown ({model['drawdown']:.1%}), consider risk management"
            )

    # Configuration suggestions
    if best_metric != hybrid_model.scoring_config["weighting_metric"]:
        recommendations["configuration_suggestions"].append(
            f"Consider switching to {best_metric} weighting for better model differentiation"
        )

    return recommendations


def render_recommendations(hybrid_model: HybridModel):
    """Render hybrid model recommendations."""
    st.subheader("ðŸ’¡ Recommendations")

    recommendations = get_hybrid_model_recommendations(hybrid_model)

    if recommendations["best_weighting_metric"]:
        st.info(f"**Recommended Weighting Metric**: {recommendations['best_weighting_metric'].title()}")

    if recommendations["model_improvements"]:
        st.subheader("ðŸ”§ Model Improvements")
        for improvement in recommendations["model_improvements"]:
            st.write(f"â€¢ {improvement}")

    if recommendations["configuration_suggestions"]:
        st.subheader("âš™ï¸ Configuration Suggestions")
        for suggestion in recommendations["configuration_suggestions"]:
            st.write(f"â€¢ {suggestion}")


def get_hybrid_config_from_session() -> Dict[str, Any]:
    """Get hybrid model configuration from Streamlit session state."""
    if "hybrid_config" not in st.session_state:
        st.session_state.hybrid_config = {
            "method": "risk_aware",
            "weighting_metric": "sharpe",
            "min_performance_threshold": 0.1,
            "recency_weight": 0.7,
            "risk_free_rate": 0.02,
            "sharpe_floor": 0.0,
            "drawdown_ceiling": -0.5,
            "mse_ceiling": 1000.0
        }

    return st.session_state.hybrid_config


def save_hybrid_config_to_session(config: Dict[str, Any]) -> None:
    """Save hybrid model configuration to Streamlit session state."""
    st.session_state.hybrid_config = config
