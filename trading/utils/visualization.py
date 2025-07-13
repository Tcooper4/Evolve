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
            logger.debug(f"Chart rendering completed in {rendering_time:.3f}s: {func.__name__}")
            return result
        except Exception as e:
            rendering_time = time.time() - start_time
            logger.error(f"Chart rendering failed after {rendering_time:.3f}s: {func.__name__} - {str(e)}")
            raise

    return wrapper


def _create_fallback_figure(title: str, error_message: str = "Visualization not available") -> Union[go.Figure, Figure]:
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
        ax.text(0.5, 0.5, error_message, ha="center", va="center", transform=ax.transAxes, fontsize=16, color="red")
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
) -> Union[go.Figure, Figure]:
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
        return _create_fallback_figure("Price Forecast", "No visualization backend available")

    if PLOTLY_AVAILABLE:
        return _plot_forecast_plotly(data, predictions, show_confidence, confidence_intervals)
    else:
        return _plot_forecast_matplotlib(data, predictions, show_confidence, confidence_intervals)


def _plot_forecast_plotly(
    data: pd.DataFrame,
    predictions: np.ndarray,
    show_confidence: bool = False,
    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> go.Figure:
    """Plot forecast using Plotly."""
    fig = go.Figure()

    # Plot actual values
    fig.add_trace(go.Scatter(x=data.index, y=data["close"], name="Actual", line=dict(color="blue")))

    # Plot predictions
    fig.add_trace(
        go.Scatter(
            x=data.index[-len(predictions) :], y=predictions, name="Predicted", line=dict(color="red", dash="dash")
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

    fig.update_layout(title="Price Forecast", xaxis_title="Date", yaxis_title="Price", hovermode="x unified")

    return fig


def _plot_forecast_matplotlib(
    data: pd.DataFrame,
    predictions: np.ndarray,
    show_confidence: bool = False,
    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Figure:
    """Plot forecast using Matplotlib."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot actual values
    ax.plot(data.index, data["close"], label="Actual", color="blue")

    # Plot predictions
    ax.plot(data.index[-len(predictions) :], predictions, label="Predicted", color="red", linestyle="--")

    # Add confidence intervals if requested
    if show_confidence and confidence_intervals is not None:
        lower, upper = confidence_intervals
        ax.fill_between(
            data.index[-len(predictions) :], lower, upper, alpha=0.3, color="green", label="Confidence Interval"
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
def plot_attention_heatmap(model: Any, data: pd.DataFrame) -> Union[go.Figure, Figure]:
    """Generate attention heatmap for transformer-based models.

    Args:
        model: Model with attention_heatmap method
        data: Input data

    Returns:
        Plotly figure or Matplotlib figure
    """
    if not hasattr(model, "attention_heatmap"):
        return _create_fallback_figure("Attention Weights Heatmap", "Model does not support attention visualization")

    try:
        attention_weights = model.attention_heatmap(data)

        if PLOTLY_AVAILABLE:
            return _plot_attention_heatmap_plotly(attention_weights, data)
        elif MATPLOTLIB_AVAILABLE:
            return _plot_attention_heatmap_matplotlib(attention_weights, data)
        else:
            return _create_fallback_figure("Attention Weights Heatmap", "No visualization backend available")
    except Exception as e:
        logger.error(f"Failed to generate attention heatmap: {str(e)}")
        return _create_fallback_figure("Attention Weights Heatmap", f"Error: {str(e)}")


def _plot_attention_heatmap_plotly(attention_weights: np.ndarray, data: pd.DataFrame) -> go.Figure:
    """Plot attention heatmap using Plotly."""
    fig = go.Figure(
        data=go.Heatmap(
            z=attention_weights,
            x=data.index[-attention_weights.shape[1] :],
            y=range(attention_weights.shape[0]),
            colorscale="Viridis",
        )
    )

    fig.update_layout(title="Attention Weights Heatmap", xaxis_title="Time Steps", yaxis_title="Attention Heads")

    return fig


def _plot_attention_heatmap_matplotlib(attention_weights: np.ndarray, data: pd.DataFrame) -> Figure:
    """Plot attention heatmap using Matplotlib."""
    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(attention_weights, cmap="viridis", aspect="auto")
    ax.set_title("Attention Weights Heatmap")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Attention Heads")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Attention Weight")

    plt.tight_layout()
    return fig


@_log_rendering_time
def plot_shap_values(model: Any, data: pd.DataFrame) -> Union[go.Figure, Figure]:
    """Visualize SHAP values for model interpretability.

    Args:
        model: Model with shap_interpret method
        data: Input data

    Returns:
        Plotly figure or Matplotlib figure
    """
    # Check if SHAP is available
    try:
        pass

        SHAP_AVAILABLE = True
    except ImportError:
        SHAP_AVAILABLE = False

    if not SHAP_AVAILABLE:
        return _create_fallback_figure("SHAP Values Analysis", "SHAP not available. Install with: pip install shap")

    if not hasattr(model, "shap_interpret"):
        return _create_fallback_figure("SHAP Values Analysis", "Model does not support SHAP interpretation")

    try:
        shap_values = model.shap_interpret(data)

        if PLOTLY_AVAILABLE:
            return _plot_shap_values_plotly(shap_values, data)
        elif MATPLOTLIB_AVAILABLE:
            return _plot_shap_values_matplotlib(shap_values, data)
        else:
            return _create_fallback_figure("SHAP Values Analysis", "No visualization backend available")

    except Exception as e:
        logger.error(f"SHAP calculation failed: {str(e)}")
        return _create_fallback_figure("SHAP Values Analysis", f"SHAP calculation failed: {str(e)}")


def _plot_shap_values_plotly(shap_values: np.ndarray, data: pd.DataFrame) -> go.Figure:
    """Plot SHAP values using Plotly."""
    # Create subplot for SHAP summary plot
    fig = make_subplots(rows=2, cols=1, subplot_titles=("SHAP Summary Plot", "Feature Importance"))

    # Add SHAP summary plot
    for i, feature in enumerate(data.columns):
        fig.add_trace(
            go.Scatter(
                x=shap_values[:, i],
                y=[feature] * len(shap_values),
                mode="markers",
                name=feature,
                marker=dict(size=8, color=data[feature], colorscale="Viridis", showscale=True),
            ),
            row=1,
            col=1,
        )

    # Add feature importance bar plot
    feature_importance = np.abs(shap_values).mean(axis=0)
    fig.add_trace(go.Bar(x=data.columns, y=feature_importance, name="Feature Importance"), row=2, col=1)

    fig.update_layout(title="SHAP Values Analysis", height=800, showlegend=False)

    return fig


def _plot_shap_values_matplotlib(shap_values: np.ndarray, data: pd.DataFrame) -> Figure:
    """Plot SHAP values using Matplotlib."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # SHAP summary plot
    for i, feature in enumerate(data.columns):
        ax1.scatter(shap_values[:, i], [feature] * len(shap_values), c=data[feature], cmap="viridis", alpha=0.6)

    ax1.set_title("SHAP Summary Plot")
    ax1.set_xlabel("SHAP Value")
    ax1.set_ylabel("Feature")

    # Feature importance bar plot
    feature_importance = np.abs(shap_values).mean(axis=0)
    ax2.bar(data.columns, feature_importance)
    ax2.set_title("Feature Importance")
    ax2.set_xlabel("Feature")
    ax2.set_ylabel("Mean |SHAP Value|")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return fig


@_log_rendering_time
def plot_backtest_results(results: pd.DataFrame) -> Union[go.Figure, Figure]:
    """Plot backtest results including cumulative returns and drawdown.

    Args:
        results: DataFrame with backtest results

    Returns:
        Plotly figure or Matplotlib figure
    """
    if not PLOTLY_AVAILABLE and not MATPLOTLIB_AVAILABLE:
        return _create_fallback_figure("Backtest Results", "No visualization backend available")

    if PLOTLY_AVAILABLE:
        return _plot_backtest_results_plotly(results)
    else:
        return _plot_backtest_results_matplotlib(results)


def _plot_backtest_results_plotly(results: pd.DataFrame) -> go.Figure:
    """Plot backtest results using Plotly."""
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Cumulative Returns", "Drawdown"), vertical_spacing=0.1)

    # Plot cumulative returns
    fig.add_trace(
        go.Scatter(x=results.index, y=results["cumulative_returns"], name="Strategy Returns", line=dict(color="blue")),
        row=1,
        col=1,
    )

    if "benchmark_returns" in results.columns:
        fig.add_trace(
            go.Scatter(
                x=results.index,
                y=results["benchmark_returns"],
                name="Benchmark Returns",
                line=dict(color="gray", dash="dash"),
            ),
            row=1,
            col=1,
        )

    # Plot drawdown
    fig.add_trace(
        go.Scatter(x=results.index, y=results["drawdown"], name="Drawdown", line=dict(color="red"), fill="tonexty"),
        row=2,
        col=1,
    )

    fig.update_layout(title="Backtest Results", height=800, showlegend=True)

    return fig


def _plot_backtest_results_matplotlib(results: pd.DataFrame) -> Figure:
    """Plot backtest results using Matplotlib."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot cumulative returns
    ax1.plot(results.index, results["cumulative_returns"], label="Strategy Returns", color="blue")
    if "benchmark_returns" in results.columns:
        ax1.plot(results.index, results["benchmark_returns"], label="Benchmark Returns", color="gray", linestyle="--")
    ax1.set_title("Cumulative Returns")
    ax1.set_ylabel("Returns")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot drawdown
    ax2.fill_between(results.index, results["drawdown"], 0, color="red", alpha=0.3)
    ax2.plot(results.index, results["drawdown"], color="red")
    ax2.set_title("Drawdown")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Drawdown")
    ax2.grid(True, alpha=0.3)

    # Format x-axis dates
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    return fig


@_log_rendering_time
def plot_model_components(model: Any, data: pd.DataFrame) -> Union[go.Figure, Figure]:
    """Plot model components and decomposition.

    Args:
        model: Model with component analysis
        data: Input data

    Returns:
        Plotly figure or Matplotlib figure
    """
    if not hasattr(model, "get_components"):
        return _create_fallback_figure("Model Components", "Model does not support component analysis")

    try:
        components = model.get_components(data)

        if PLOTLY_AVAILABLE:
            return _plot_model_components_plotly(components, data)
        elif MATPLOTLIB_AVAILABLE:
            return _plot_model_components_matplotlib(components, data)
        else:
            return _create_fallback_figure("Model Components", "No visualization backend available")
    except Exception as e:
        logger.error(f"Failed to plot model components: {str(e)}")
        return _create_fallback_figure("Model Components", f"Error: {str(e)}")


def _plot_model_components_plotly(components: Dict[str, np.ndarray], data: pd.DataFrame) -> go.Figure:
    """Plot model components using Plotly."""
    fig = make_subplots(rows=len(components), cols=1, subplot_titles=list(components.keys()), vertical_spacing=0.05)

    for i, (name, values) in enumerate(components.items(), 1):
        fig.add_trace(
            go.Scatter(
                x=data.index, y=values, name=name, line=dict(color=f"rgb({50 + i*50}, {100 + i*30}, {150 + i*20})")
            ),
            row=i,
            col=1,
        )

    fig.update_layout(title="Model Components", height=200 * len(components), showlegend=False)

    return fig


def _plot_model_components_matplotlib(components: Dict[str, np.ndarray], data: pd.DataFrame) -> Figure:
    """Plot model components using Matplotlib."""
    fig, axes = plt.subplots(len(components), 1, figsize=(12, 3 * len(components)))
    if len(components) == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, components.items()):
        ax.plot(data.index, values, label=name)
        ax.set_title(name)
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    return fig


@_log_rendering_time
def plot_performance_over_time(performance_data: pd.DataFrame) -> Union[go.Figure, Figure]:
    """Plot performance metrics over time.

    Args:
        performance_data: DataFrame with performance metrics

    Returns:
        Plotly figure or Matplotlib figure
    """
    if not PLOTLY_AVAILABLE and not MATPLOTLIB_AVAILABLE:
        return _create_fallback_figure("Performance Over Time", "No visualization backend available")

    if PLOTLY_AVAILABLE:
        return _plot_performance_over_time_plotly(performance_data)
    else:
        return _plot_performance_over_time_matplotlib(performance_data)


def _plot_performance_over_time_plotly(performance_data: pd.DataFrame) -> go.Figure:
    """Plot performance over time using Plotly."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Sharpe Ratio", "Returns", "Volatility", "Drawdown"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}], [{"secondary_y": False}, {"secondary_y": False}]],
    )

    metrics = ["sharpe_ratio", "returns", "volatility", "drawdown"]
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for metric, pos in zip(metrics, positions):
        if metric in performance_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=performance_data.index,
                    y=performance_data[metric],
                    name=metric.replace("_", " ").title(),
                    line=dict(color=f"rgb({50 + len(metric)*20}, {100 + len(metric)*15}, {150 + len(metric)*10})"),
                ),
                row=pos[0],
                col=pos[1],
            )

    fig.update_layout(title="Performance Metrics Over Time", height=800, showlegend=True)

    return fig


def _plot_performance_over_time_matplotlib(performance_data: pd.DataFrame) -> Figure:
    """Plot performance over time using Matplotlib."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    metrics = ["sharpe_ratio", "returns", "volatility", "drawdown"]

    for i, metric in enumerate(metrics):
        if metric in performance_data.columns:
            axes[i].plot(performance_data.index, performance_data[metric])
            axes[i].set_title(metric.replace("_", " ").title())
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


@_log_rendering_time
def plot_model_comparison(metrics: pd.DataFrame) -> Union[go.Figure, Figure]:
    """Plot model comparison metrics.

    Args:
        metrics: DataFrame with model comparison metrics

    Returns:
        Plotly figure or Matplotlib figure
    """
    if not PLOTLY_AVAILABLE and not MATPLOTLIB_AVAILABLE:
        return _create_fallback_figure("Model Comparison", "No visualization backend available")

    if PLOTLY_AVAILABLE:
        return _plot_model_comparison_plotly(metrics)
    else:
        return _plot_model_comparison_matplotlib(metrics)


def _plot_model_comparison_plotly(metrics: pd.DataFrame) -> go.Figure:
    """Plot model comparison using Plotly."""
    fig = go.Figure()

    for model in metrics.index:
        fig.add_trace(
            go.Bar(
                name=model,
                x=metrics.columns,
                y=metrics.loc[model],
                text=metrics.loc[model].round(3),
                textposition="auto",
            )
        )

    fig.update_layout(title="Model Comparison", xaxis_title="Metrics", yaxis_title="Values", barmode="group")

    return fig


def _plot_model_comparison_matplotlib(metrics: pd.DataFrame) -> Figure:
    """Plot model comparison using Matplotlib."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(metrics.columns))
    width = 0.8 / len(metrics.index)

    for i, model in enumerate(metrics.index):
        ax.bar(x + i * width, metrics.loc[model], width, label=model)

    ax.set_title("Model Comparison")
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Values")
    ax.set_xticks(x + width * (len(metrics.index) - 1) / 2)
    ax.set_xticklabels(metrics.columns, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def save_figure(fig: Union[go.Figure, Figure], filepath: str, format: str = "png") -> bool:
    """Save figure to file.

    Args:
        fig: Figure to save
        filepath: Path to save the figure
        format: File format ('png', 'jpg', 'pdf', 'svg')

    Returns:
        Whether save was successful
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(fig, go.Figure):
            if PLOTLY_AVAILABLE:
                fig.write_image(str(filepath))
                return True
            else:
                logger.error("Cannot save Plotly figure without Plotly backend")
                return False
        elif isinstance(fig, Figure):
            if MATPLOTLIB_AVAILABLE:
                fig.savefig(str(filepath), format=format, dpi=300, bbox_inches="tight")
                return True
            else:
                logger.error("Cannot save Matplotlib figure without Matplotlib backend")
                return False
        else:
            logger.error(f"Unknown figure type: {type(fig)}")
            return False
    except Exception as e:
        logger.error(f"Failed to save figure: {str(e)}")
        return False
