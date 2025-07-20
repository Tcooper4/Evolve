"""
Visualization utilities for trading strategies and models.

This module provides plotting functions for forecasts, backtest results,
model components, and other trading-related visualizations.
"""

import logging
import time
from functools import wraps
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Try to import Plotly
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    make_subplots = None

logger = logging.getLogger(__name__)


def _log_rendering_time(func):
    """Decorator to log rendering time for visualization functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} rendered in {end_time - start_time:.2f}s")
        return result
    return wrapper


def _create_fallback_figure(
    title: str, error_message: str = "Visualization not available"
) -> Union[go.Figure, "Figure"]:
    """Create a fallback figure when visualization fails."""
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_annotation(
            text=error_message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title=title)
        return fig
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, error_message, ha='center', va='center',
                transform=ax.transAxes, fontsize=16)
        ax.set_title(title)
        plt.tight_layout()
        return fig


@_log_rendering_time
def plot_forecast(
    data: pd.DataFrame,
    predictions: np.ndarray,
    show_confidence: bool = False,
    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Union[go.Figure, "Figure"]:
    """Plot forecast results.

    Args:
        data: Historical data
        predictions: Forecast predictions
        show_confidence: Whether to show confidence intervals
        confidence_intervals: Tuple of (lower, upper) confidence bounds

    Returns:
        Plotly figure or Matplotlib figure
    """
    if PLOTLY_AVAILABLE:
        return _plot_forecast_plotly(
            data, predictions, show_confidence, confidence_intervals
        )
    else:
        return _plot_forecast_matplotlib(
            data, predictions, show_confidence, confidence_intervals
        )


def _plot_forecast_plotly(
    data: pd.DataFrame,
    predictions: np.ndarray,
    show_confidence: bool = False,
    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> go.Figure:
    """Plot forecast using Plotly."""
    fig = go.Figure()

    # Plot historical data
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data.iloc[:, 0],  # Assume first column is target
        mode='lines',
        name='Historical',
        line=dict(color='blue')
    ))

    # Create forecast index
    forecast_index = pd.date_range(
        start=data.index[-1],
        periods=len(predictions) + 1,
        freq='D'
    )[1:]

    # Plot predictions
    fig.add_trace(go.Scatter(
        x=forecast_index,
        y=predictions,
        mode='lines',
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))

    # Plot confidence intervals if available
    if show_confidence and confidence_intervals is not None:
        lower, upper = confidence_intervals
        fig.add_trace(go.Scatter(
            x=forecast_index,
            y=upper,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_index,
            y=lower,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.2)',
            name='Confidence Interval'
        ))

    fig.update_layout(
        title="Forecast Results",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified'
    )

    return fig


def _plot_forecast_matplotlib(
    data: pd.DataFrame,
    predictions: np.ndarray,
    show_confidence: bool = False,
    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> "Figure":
    """Plot forecast using Matplotlib."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot historical data
    ax.plot(data.index, data.iloc[:, 0], label='Historical', color='blue')

    # Create forecast index
    forecast_index = pd.date_range(
        start=data.index[-1],
        periods=len(predictions) + 1,
        freq='D'
    )[1:]

    # Plot predictions
    ax.plot(forecast_index, predictions, label='Forecast', color='red', linestyle='--')

    # Plot confidence intervals if available
    if show_confidence and confidence_intervals is not None:
        lower, upper = confidence_intervals
        ax.fill_between(forecast_index, lower, upper, alpha=0.2, color='red',
                       label='Confidence Interval')

    ax.set_title("Forecast Results")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


@_log_rendering_time
def plot_attention_heatmap(model: Any, data: pd.DataFrame) -> Union[go.Figure, "Figure"]:
    """Plot attention heatmap for transformer models.

    Args:
        model: Model with attention weights
        data: Input data

    Returns:
        Plotly figure or Matplotlib figure
    """
    try:
        attention_weights = model.get_attention_weights(data)
        if PLOTLY_AVAILABLE:
            return _plot_attention_heatmap_plotly(attention_weights, data)
        else:
            return _plot_attention_heatmap_matplotlib(attention_weights, data)
    except Exception as e:
        logger.error(f"Error plotting attention heatmap: {e}")
        return _create_fallback_figure(
            "Attention Heatmap", f"Error: {str(e)}"
        )


def _plot_attention_heatmap_plotly(
    attention_weights: np.ndarray, data: pd.DataFrame
) -> go.Figure:
    """Plot attention heatmap using Plotly."""
    fig = go.Figure(data=go.Heatmap(
        z=attention_weights,
        x=data.columns,
        y=data.index,
        colorscale='Viridis'
    ))

    fig.update_layout(
        title="Attention Heatmap",
        xaxis_title="Features",
        yaxis_title="Time Steps"
    )

    return fig


def _plot_attention_heatmap_matplotlib(
    attention_weights: np.ndarray, data: pd.DataFrame
) -> "Figure":
    """Plot attention heatmap using Matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(attention_weights, cmap='viridis', aspect='auto')
    ax.set_xlabel('Features')
    ax.set_ylabel('Time Steps')
    ax.set_title('Attention Heatmap')

    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    return fig


@_log_rendering_time
def plot_shap_values(model: Any, data: pd.DataFrame) -> Union[go.Figure, "Figure"]:
    """Generate SHAP values visualization.

    Args:
        model: Model with SHAP explainer
        data: Input data

    Returns:
        Plotly figure or Matplotlib figure
    """
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)

        if PLOTLY_AVAILABLE:
            return _plot_shap_values_plotly(shap_values, data)
        else:
            return _plot_shap_values_matplotlib(shap_values, data)
    except ImportError:
        return _create_fallback_figure(
            "SHAP Values", "SHAP library not available"
        )
    except Exception as e:
        logger.error(f"Error generating SHAP values: {e}")
        return _create_fallback_figure(
            "SHAP Values", f"Error: {str(e)}"
        )


def _plot_shap_values_plotly(shap_values: np.ndarray, data: pd.DataFrame) -> go.Figure:
    """Plot SHAP values using Plotly."""
    # Calculate feature importance
    feature_importance = np.abs(shap_values).mean(axis=0)

    fig = go.Figure(data=go.Bar(
        x=data.columns,
        y=feature_importance,
        marker_color='lightblue'
    ))

    fig.update_layout(
        title="SHAP Feature Importance",
        xaxis_title="Features",
        yaxis_title="Mean |SHAP Value|"
    )

    return fig


def _plot_shap_values_matplotlib(shap_values: np.ndarray, data: pd.DataFrame) -> "Figure":
    """Plot SHAP values using Matplotlib."""
    # Calculate feature importance
    feature_importance = np.abs(shap_values).mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(data.columns, feature_importance, color='lightblue')
    ax.set_title("SHAP Feature Importance")
    ax.set_xlabel("Features")
    ax.set_ylabel("Mean |SHAP Value|")
    plt.xticks(rotation=45)

    plt.tight_layout()
    return fig


@_log_rendering_time
def plot_backtest_results(results: pd.DataFrame) -> Union[go.Figure, "Figure"]:
    """Plot backtest results.

    Args:
        results: DataFrame with backtest results

    Returns:
        Plotly figure or Matplotlib figure
    """
    if PLOTLY_AVAILABLE:
        return _plot_backtest_results_plotly(results)
    else:
        return _plot_backtest_results_matplotlib(results)


def _plot_backtest_results_plotly(results: pd.DataFrame) -> go.Figure:
    """Plot backtest results using Plotly."""
    fig = go.Figure()

    # Plot equity curve
    fig.add_trace(go.Scatter(
        x=results.index,
        y=results['equity_curve'],
        mode='lines',
        name='Equity Curve',
        line=dict(color='blue')
    ))

    # Plot benchmark if available
    if 'benchmark' in results.columns:
        fig.add_trace(go.Scatter(
            x=results.index,
            y=results['benchmark'],
            mode='lines',
            name='Benchmark',
            line=dict(color='red', dash='dash')
        ))

    fig.update_layout(
        title="Backtest Results",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        hovermode='x unified'
    )

    return fig


def _plot_backtest_results_matplotlib(results: pd.DataFrame) -> "Figure":
    """Plot backtest results using Matplotlib."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot equity curve
    ax.plot(results.index, results['equity_curve'], label='Equity Curve', color='blue')

    # Plot benchmark if available
    if 'benchmark' in results.columns:
        ax.plot(results.index, results['benchmark'], label='Benchmark', color='red', linestyle='--')

    ax.set_title("Backtest Results")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


@_log_rendering_time
def plot_model_components(model: Any, data: pd.DataFrame) -> Union[go.Figure, "Figure"]:
    """Plot model components (trend, seasonal, residual).

    Args:
        model: Model with components
        data: Input data

    Returns:
        Plotly figure or Matplotlib figure
    """
    try:
        components = model.components(data)
        if PLOTLY_AVAILABLE:
            return _plot_model_components_plotly(components, data)
        else:
            return _plot_model_components_matplotlib(components, data)
    except Exception as e:
        logger.error(f"Error plotting model components: {e}")
        return _create_fallback_figure(
            "Model Components", f"Error: {str(e)}"
        )


def _plot_model_components_plotly(
    components: Dict[str, np.ndarray], data: pd.DataFrame
) -> go.Figure:
    """Plot model components using Plotly."""
    fig = make_subplots(
        rows=len(components),
        cols=1,
        subplot_titles=list(components.keys()),
        vertical_spacing=0.1
    )

    for i, (component_name, component_data) in enumerate(components.items(), 1):
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=component_data,
                mode='lines',
                name=component_name
            ),
            row=i, col=1
        )

    fig.update_layout(
        title="Model Components",
        height=200 * len(components),
        showlegend=False
    )

    return fig


def _plot_model_components_matplotlib(
    components: Dict[str, np.ndarray], data: pd.DataFrame
) -> "Figure":
    """Plot model components using Matplotlib."""
    fig, axes = plt.subplots(
        len(components), 1,
        figsize=(12, 4 * len(components)),
        sharex=True
    )

    if len(components) == 1:
        axes = [axes]

    for ax, (component_name, component_data) in zip(axes, components.items()):
        ax.plot(data.index, component_data)
        ax.set_title(component_name)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


@_log_rendering_time
def plot_performance_over_time(
    performance_data: pd.DataFrame,
) -> Union[go.Figure, "Figure"]:
    """Plot performance metrics over time.

    Args:
        performance_data: DataFrame with performance metrics

    Returns:
        Plotly figure or Matplotlib figure
    """
    if PLOTLY_AVAILABLE:
        return _plot_performance_over_time_plotly(performance_data)
    else:
        return _plot_performance_over_time_matplotlib(performance_data)


def _plot_performance_over_time_plotly(performance_data: pd.DataFrame) -> go.Figure:
    """Plot performance over time using Plotly."""
    fig = go.Figure()

    for column in performance_data.columns:
        fig.add_trace(go.Scatter(
            x=performance_data.index,
            y=performance_data[column],
            mode='lines',
            name=column
        ))

    fig.update_layout(
        title="Performance Over Time",
        xaxis_title="Date",
        yaxis_title="Metric Value",
        hovermode='x unified'
    )

    return fig


def _plot_performance_over_time_matplotlib(performance_data: pd.DataFrame) -> "Figure":
    """Plot performance over time using Matplotlib."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for column in performance_data.columns:
        ax.plot(performance_data.index, performance_data[column], label=column)

    ax.set_title("Performance Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Metric Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


@_log_rendering_time
def plot_model_comparison(metrics: pd.DataFrame) -> Union[go.Figure, "Figure"]:
    """Plot model comparison metrics.

    Args:
        metrics: DataFrame with model comparison metrics

    Returns:
        Plotly figure or Matplotlib figure
    """
    if PLOTLY_AVAILABLE:
        return _plot_model_comparison_plotly(metrics)
    else:
        return _plot_model_comparison_matplotlib(metrics)


def _plot_model_comparison_plotly(metrics: pd.DataFrame) -> go.Figure:
    """Plot model comparison using Plotly."""
    fig = go.Figure(data=go.Bar(
        x=metrics.index,
        y=metrics.iloc[:, 0],  # First metric
        marker_color='lightblue'
    ))

    fig.update_layout(
        title="Model Comparison",
        xaxis_title="Models",
        yaxis_title=metrics.columns[0]
    )

    return fig


def _plot_model_comparison_matplotlib(metrics: pd.DataFrame) -> "Figure":
    """Plot model comparison using Matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(metrics.index, metrics.iloc[:, 0], color='lightblue')
    ax.set_title("Model Comparison")
    ax.set_xlabel("Models")
    ax.set_ylabel(metrics.columns[0])
    plt.xticks(rotation=45)

    plt.tight_layout()
    return fig


def save_figure(
    fig: Union[go.Figure, "Figure"], filepath: str, format: str = "png"
) -> bool:
    """Save figure to file.

    Args:
        fig: Figure to save
        filepath: Output file path
        format: Output format (png, jpg, svg, pdf)

    Returns:
        True if successful, False otherwise
    """
    try:
        if PLOTLY_AVAILABLE and isinstance(fig, go.Figure):
            fig.write_image(filepath)
        else:
            fig.savefig(filepath, format=format, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving figure: {e}")
        return False 