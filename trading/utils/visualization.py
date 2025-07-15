"""Utility functions for visualizing forecasts and model interpretability."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import Plotly, fallback to Matplotlib
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
    logger.info("Plotly visualization backend available")
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available, falling back to Matplotlib")

# Fallback to Matplotlib
if not PLOTLY_AVAILABLE:
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure

        MATPLOTLIB_AVAILABLE = True
        logger.info("Matplotlib visualization backend available")
    except ImportError:
        MATPLOTLIB_AVAILABLE = False
        logger.error("No visualization backend available")


def _log_rendering_time(func):
    """Decorator to log chart rendering time."""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            rendering_time = time.time() - start_time
            logger.debug(
                f"Chart rendering completed in {rendering_time:.3f}s: {func.__name__}"
            )
            return result
        except Exception as e:
            rendering_time = time.time() - start_time
            logger.error(
                f"Chart rendering failed after {rendering_time:.3f}s: {func.__name__} - {str(e)}"
            )
            raise

    return wrapper


def _create_fallback_figure(
    title: str, error_message: str = "Visualization not available"
) -> Union[go.Figure, "Figure"]:
    """Create a fallback figure when visualization backend is not available.

    Args:
        title: Figure title
        error_message: Error message to display

    Returns:
        Fallback figure object
    """
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_annotation(
            text=error_message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="red"),
        )
        fig.update_layout(title=title, height=400, showlegend=False)
        return fig
    elif MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            error_message,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=16,
            color="red",
        )
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        return fig
    else:
        raise RuntimeError("No visualization backend available")


@_log_rendering_time
def plot_forecast(
    data: pd.DataFrame,
    predictions: np.ndarray,
    show_confidence: bool = False,
    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Union[go.Figure, "Figure"]:
    """Plot actual values and predictions with optional confidence intervals.

    Args:
        data: DataFrame with actual values
        predictions: Array of predicted values
        show_confidence: Whether to show confidence intervals
        confidence_intervals: Optional tuple of (lower, upper) confidence bounds

    Returns:
        Plotly figure or Matplotlib figure
    """
    if not PLOTLY_AVAILABLE and not MATPLOTLIB_AVAILABLE:
        return _create_fallback_figure(
            "Price Forecast", "No visualization backend available"
        )

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

    # Plot actual values
    fig.add_trace(
        go.Scatter(
            x=data.index, y=data["close"], name="Actual", line=dict(color="blue")
        )
    )

    # Plot predictions
    fig.add_trace(
        go.Scatter(
            x=data.index[-len(predictions) :],
            y=predictions,
            name="Predicted",
            line=dict(color="red", dash="dash"),
        )
    )

    # Add confidence intervals if requested
    if show_confidence and confidence_intervals is not None:
        lower, upper = confidence_intervals
        fig.add_trace(
            go.Scatter(
                x=data.index[-len(predictions) :],
                y=upper,
                fill=None,
                mode="lines",
                line_color="rgba(0,100,80,0.2)",
                name="Upper Bound",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data.index[-len(predictions) :],
                y=lower,
                fill="tonexty",
                mode="lines",
                line_color="rgba(0,100,80,0.2)",
                name="Lower Bound",
            )
        )

    fig.update_layout(
        title="Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
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

    # Plot actual values
    ax.plot(data.index, data["close"], label="Actual", color="blue")

    # Plot predictions
    ax.plot(
        data.index[-len(predictions) :],
        predictions,
        label="Predicted",
        color="red",
        linestyle="--",
    )

    # Add confidence intervals if requested
    if show_confidence and confidence_intervals is not None:
        lower, upper = confidence_intervals
        ax.fill_between(
            data.index[-len(predictions) :],
            lower,
            upper,
            alpha=0.3,
            color="green",
            label="Confidence Interval",
        )

    ax.set_title("Price Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    return fig


@_log_rendering_time
def plot_attention_heatmap(model: Any, data: pd.DataFrame) -> Union[go.Figure, "Figure"]:
    """Generate attention heatmap for transformer-based models.

    Args:
        model: Model with attention_heatmap method
        data: Input data

    Returns:
        Plotly figure or Matplotlib figure
    """
    if not hasattr(model, "attention_heatmap"):
        return _create_fallback_figure(
            "Attention Heatmap", "Model does not support attention heatmap"
        )

    try:
        attention_weights = model.attention_heatmap(data)
        if PLOTLY_AVAILABLE:
            return _plot_attention_heatmap_plotly(attention_weights, data)
        else:
            return _plot_attention_heatmap_matplotlib(attention_weights, data)
    except Exception as e:
        logger.error(f"Error generating attention heatmap: {e}")
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
        rows=len(components), cols=1,
        subplot_titles=list(components.keys()),
        shared_xaxes=True
    )
    
    for i, (component_name, component_values) in enumerate(components.items(), 1):
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=component_values,
                mode='lines',
                name=component_name
            ),
            row=i, col=1
        )
    
    fig.update_layout(height=200 * len(components), showlegend=False)
    return fig


def _plot_model_components_matplotlib(
    components: Dict[str, np.ndarray], data: pd.DataFrame
) -> "Figure":
    """Plot model components using Matplotlib."""
    fig, axes = plt.subplots(len(components), 1, figsize=(12, 4 * len(components)))
    
    if len(components) == 1:
        axes = [axes]
    
    for ax, (component_name, component_values) in zip(axes, components.items()):
        ax.plot(data.index, component_values)
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
        if column != 'date':
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
        if column != 'date':
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
        metrics: DataFrame with model metrics

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
        y=metrics['score'],
        text=metrics['score'].round(3),
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Model Comparison",
        xaxis_title="Models",
        yaxis_title="Score"
    )
    
    return fig


def _plot_model_comparison_matplotlib(metrics: pd.DataFrame) -> "Figure":
    """Plot model comparison using Matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(metrics.index, metrics['score'])
    ax.set_title("Model Comparison")
    ax.set_xlabel("Models")
    ax.set_ylabel("Score")
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
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
        True if save successful
    """
    try:
        if isinstance(fig, go.Figure):
            fig.write_image(filepath)
        else:
            fig.savefig(filepath, format=format, dpi=300, bbox_inches='tight')
        return True
    except Exception as e:
        logger.error(f"Error saving figure: {e}")
        return False
