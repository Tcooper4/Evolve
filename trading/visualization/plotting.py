"""Advanced plotting utilities for time series and performance visualization.

This module provides comprehensive plotting functionality for time series data,
performance metrics, and model predictions, with support for both Matplotlib
and Plotly backends.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose

logger = logging.getLogger(__name__)


class TimeSeriesPlotter:
    """Class for plotting time series data with performance metrics."""

    def __init__(
        self,
        style: str = "seaborn",
        backend: str = "matplotlib",
        figsize: tuple = (12, 6),
    ):
        """Initialize the plotter.

        Args:
            style: Plot style to use (for matplotlib)
            backend: Plotting backend ('matplotlib' or 'plotly')
            figsize: Default figure size
        """
        self.style = style
        self.backend = backend
        self.figsize = figsize

        if backend == "matplotlib":
            plt.style.use(style)

    def plot_time_series(
        self,
        data: pd.Series,
        title: str = "Time Series Plot",
        xlabel: str = "Time",
        ylabel: str = "Value",
        figsize: Optional[tuple] = None,
        show: bool = True,
        overlays: Optional[Dict[str, pd.Series]] = None,
        confidence_bands: Optional[Tuple[pd.Series, pd.Series]] = None,
        show_overlays: bool = True,
        show_confidence: bool = True,
    ) -> Union[plt.Figure, go.Figure]:
        """Plot a single time series with optional overlays and confidence bands.

        Args:
            data: Time series data
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size (matplotlib only)
            show: Whether to display the plot
            overlays: Dictionary of overlay series to plot
            confidence_bands: Tuple of (lower, upper) confidence bands
            show_overlays: Whether to show overlays
            show_confidence: Whether to show confidence bands

        Returns:
            Matplotlib or Plotly figure
        """
        # Check for empty series
        if data.empty:
            warnings.warn("Input data series is empty")
            return self._create_empty_plot(title, "No data available")

        if self.backend == "matplotlib":
            return self._plot_time_series_matplotlib(
                data,
                title,
                xlabel,
                ylabel,
                figsize,
                show,
                overlays,
                confidence_bands,
                show_overlays,
                show_confidence,
            )
        else:
            return self._plot_time_series_plotly(
                data,
                title,
                xlabel,
                ylabel,
                show,
                overlays,
                confidence_bands,
                show_overlays,
                show_confidence,
            )

    def _plot_time_series_matplotlib(
        self,
        data: pd.Series,
        title: str,
        xlabel: str,
        ylabel: str,
        figsize: Optional[tuple],
        show: bool,
        overlays: Optional[Dict[str, pd.Series]],
        confidence_bands: Optional[Tuple[pd.Series, pd.Series]],
        show_overlays: bool,
        show_confidence: bool,
    ) -> plt.Figure:
        """Plot time series using Matplotlib."""
        fig, ax = plt.subplots(figsize=figsize or self.figsize)

        # Plot main data
        data.plot(ax=ax, label="Main Series", linewidth=2)

        # Plot overlays if requested
        if show_overlays and overlays:
            for name, overlay_data in overlays.items():
                if not overlay_data.empty:
                    overlay_data.plot(ax=ax, label=name, alpha=0.7, linestyle="--")
                else:
                    logger.warning(f"Overlay '{name}' is empty, skipping")

        # Plot confidence bands if requested
        if show_confidence and confidence_bands:
            lower, upper = confidence_bands
            if not lower.empty and not upper.empty:
                ax.fill_between(
                    data.index,
                    lower,
                    upper,
                    alpha=0.3,
                    color="gray",
                    label="Confidence Band",
                )
            else:
                logger.warning("Confidence bands are empty, skipping")

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

        if show:
            plt.show()
        return fig

    def _plot_time_series_plotly(
        self,
        data: pd.Series,
        title: str,
        xlabel: str,
        ylabel: str,
        show: bool,
        overlays: Optional[Dict[str, pd.Series]],
        confidence_bands: Optional[Tuple[pd.Series, pd.Series]],
        show_overlays: bool,
        show_confidence: bool,
    ) -> go.Figure:
        """Plot time series using Plotly."""
        fig = go.Figure()

        # Plot main data
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data.values,
                mode="lines",
                name=data.name or "Main Series",
                line=dict(width=2),
            )
        )

        # Plot overlays if requested
        if show_overlays and overlays:
            for name, overlay_data in overlays.items():
                if not overlay_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=overlay_data.index,
                            y=overlay_data.values,
                            mode="lines",
                            name=name,
                            line=dict(dash="dash", width=1),
                            opacity=0.7,
                        )
                    )
                else:
                    logger.warning(f"Overlay '{name}' is empty, skipping")

        # Plot confidence bands if requested
        if show_confidence and confidence_bands:
            lower, upper = confidence_bands
            if not lower.empty and not upper.empty:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=upper,
                        fill=None,
                        mode="lines",
                        line_color="rgba(0,0,0,0)",
                        name="Upper Bound",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=lower,
                        fill="tonexty",
                        mode="lines",
                        line_color="rgba(0,0,0,0)",
                        name="Lower Bound",
                    )
                )
            else:
                logger.warning("Confidence bands are empty, skipping")

        fig.update_layout(
            title=title, xaxis_title=xlabel, yaxis_title=ylabel, showlegend=True
        )

        if show:
            fig.show()
        return fig

    def _create_empty_plot(
        self, title: str, message: str
    ) -> Union[plt.Figure, go.Figure]:
        """Create an empty plot with a message."""
        if self.backend == "matplotlib":
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            return fig
        else:
            fig = go.Figure()
            fig.add_annotation(
                text=message,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color="red"),
            )
            fig.update_layout(title=title, height=400, showlegend=False)
            return fig

    def plot_performance(
        self,
        returns: pd.Series,
        benchmark: Optional[pd.Series] = None,
        drawdown: Optional[pd.Series] = None,
        title: str = "Performance Analysis",
        figsize: Optional[tuple] = None,
        show: bool = True,
        show_benchmark: bool = True,
        show_drawdown: bool = True,
        confidence_bands: Optional[Tuple[pd.Series, pd.Series]] = None,
        show_confidence: bool = True,
    ) -> Union[plt.Figure, go.Figure]:
        """Plot performance metrics including returns, benchmark, and drawdown.

        Args:
            returns: Series of returns
            benchmark: Optional benchmark returns
            drawdown: Optional drawdown series
            title: Plot title
            figsize: Figure size (matplotlib only)
            show: Whether to display the plot
            show_benchmark: Whether to show benchmark
            show_drawdown: Whether to show drawdown
            confidence_bands: Optional confidence bands for returns
            show_confidence: Whether to show confidence bands

        Returns:
            Matplotlib or Plotly figure
        """
        # Check for empty series
        if returns.empty:
            warnings.warn("Returns series is empty")
            return self._create_empty_plot(title, "No returns data available")

        if self.backend == "matplotlib":
            return self._plot_performance_matplotlib(
                returns,
                benchmark,
                drawdown,
                title,
                figsize,
                show,
                show_benchmark,
                show_drawdown,
                confidence_bands,
                show_confidence,
            )
        else:
            return self._plot_performance_plotly(
                returns,
                benchmark,
                drawdown,
                title,
                show,
                show_benchmark,
                show_drawdown,
                confidence_bands,
                show_confidence,
            )

    def _plot_performance_matplotlib(
        self,
        returns: pd.Series,
        benchmark: Optional[pd.Series],
        drawdown: Optional[pd.Series],
        title: str,
        figsize: Optional[tuple],
        show: bool,
        show_benchmark: bool,
        show_drawdown: bool,
        confidence_bands: Optional[Tuple[pd.Series, pd.Series]],
        show_confidence: bool,
    ) -> plt.Figure:
        """Plot performance using Matplotlib."""
        num_subplots = 1 + (1 if show_drawdown and drawdown is not None else 0)
        fig, axes = plt.subplots(num_subplots, 1, figsize=figsize or self.figsize)
        if num_subplots == 1:
            axes = [axes]

        # Plot returns
        returns.cumsum().plot(ax=axes[0], label="Strategy", linewidth=2)

        if show_benchmark and benchmark is not None and not benchmark.empty:
            benchmark.cumsum().plot(
                ax=axes[0], label="Benchmark", linestyle="--", alpha=0.7
            )

        # Add confidence bands if requested
        if show_confidence and confidence_bands:
            lower, upper = confidence_bands
            if not lower.empty and not upper.empty:
                axes[0].fill_between(
                    returns.index,
                    lower.cumsum(),
                    upper.cumsum(),
                    alpha=0.3,
                    color="gray",
                    label="Confidence Band",
                )

        axes[0].set_title("Cumulative Returns")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Plot drawdown
        if show_drawdown and drawdown is not None and not drawdown.empty:
            drawdown.plot(ax=axes[1], color="red", linewidth=2)
            axes[1].set_title("Drawdown")
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def _plot_performance_plotly(
        self,
        returns: pd.Series,
        benchmark: Optional[pd.Series],
        drawdown: Optional[pd.Series],
        title: str,
        show: bool,
        show_benchmark: bool,
        show_drawdown: bool,
        confidence_bands: Optional[Tuple[pd.Series, pd.Series]],
        show_confidence: bool,
    ) -> go.Figure:
        """Plot performance using Plotly."""
        num_subplots = 1 + (1 if show_drawdown and drawdown is not None else 0)
        subplot_titles = ["Cumulative Returns"]
        if show_drawdown and drawdown is not None:
            subplot_titles.append("Drawdown")

        fig = make_subplots(rows=num_subplots, cols=1, subplot_titles=subplot_titles)

        # Plot returns
        fig.add_trace(
            go.Scatter(
                x=returns.index,
                y=returns.cumsum(),
                name="Strategy",
                line=dict(color="blue", width=2),
            ),
            row=1,
            col=1,
        )

        if show_benchmark and benchmark is not None and not benchmark.empty:
            fig.add_trace(
                go.Scatter(
                    x=benchmark.index,
                    y=benchmark.cumsum(),
                    name="Benchmark",
                    line=dict(color="gray", dash="dash", width=1),
                    opacity=0.7,
                ),
                row=1,
                col=1,
            )

        # Add confidence bands if requested
        if show_confidence and confidence_bands:
            lower, upper = confidence_bands
            if not lower.empty and not upper.empty:
                fig.add_trace(
                    go.Scatter(
                        x=returns.index,
                        y=upper.cumsum(),
                        fill=None,
                        mode="lines",
                        line_color="rgba(0,0,0,0)",
                        name="Upper Bound",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=returns.index,
                        y=lower.cumsum(),
                        fill="tonexty",
                        mode="lines",
                        line_color="rgba(0,0,0,0)",
                        name="Lower Bound",
                    ),
                    row=1,
                    col=1,
                )

        # Plot drawdown
        if show_drawdown and drawdown is not None and not drawdown.empty:
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown,
                    name="Drawdown",
                    line=dict(color="red", width=2),
                ),
                row=2,
                col=1,
            )

        fig.update_layout(title=title, height=400 * num_subplots, showlegend=True)

        if show:
            fig.show()
        return fig

    def plot_multiple_series(
        self,
        series_list: List[pd.Series],
        labels: Optional[List[str]] = None,
        title: str = "Multiple Time Series",
        xlabel: str = "Time",
        ylabel: str = "Value",
        figsize: Optional[tuple] = None,
        show: bool = True,
        overlays: Optional[Dict[str, pd.Series]] = None,
        show_overlays: bool = True,
    ) -> Union[plt.Figure, go.Figure]:
        """Plot multiple time series on the same plot.

        Args:
            series_list: List of time series to plot
            labels: Optional list of labels for each series
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size (matplotlib only)
            show: Whether to display the plot
            overlays: Dictionary of overlay series to plot
            show_overlays: Whether to show overlays

        Returns:
            Matplotlib or Plotly figure
        """
        # Check for empty series
        if not series_list:
            warnings.warn("No series provided for plotting")
            return self._create_empty_plot(title, "No series data available")

        empty_series = [i for i, series in enumerate(series_list) if series.empty]
        if empty_series:
            warnings.warn(f"Series at indices {empty_series} are empty")

        if self.backend == "matplotlib":
            return self._plot_multiple_series_matplotlib(
                series_list,
                labels,
                title,
                xlabel,
                ylabel,
                figsize,
                show,
                overlays,
                show_overlays,
            )
        else:
            return self._plot_multiple_series_plotly(
                series_list,
                labels,
                title,
                xlabel,
                ylabel,
                show,
                overlays,
                show_overlays,
            )

    def _plot_multiple_series_matplotlib(
        self,
        series_list: List[pd.Series],
        labels: Optional[List[str]],
        title: str,
        xlabel: str,
        ylabel: str,
        figsize: Optional[tuple],
        show: bool,
        overlays: Optional[Dict[str, pd.Series]],
        show_overlays: bool,
    ) -> plt.Figure:
        """Plot multiple series using Matplotlib."""
        fig, ax = plt.subplots(figsize=figsize or self.figsize)

        for i, series in enumerate(series_list):
            if not series.empty:
                label = labels[i] if labels and i < len(labels) else f"Series {i+1}"
                series.plot(ax=ax, label=label)

        # Plot overlays if requested
        if show_overlays and overlays:
            for name, overlay_data in overlays.items():
                if not overlay_data.empty:
                    overlay_data.plot(
                        ax=ax, label=f"{name} (Overlay)", alpha=0.7, linestyle="--"
                    )
                else:
                    logger.warning(f"Overlay '{name}' is empty, skipping")

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

        if show:
            plt.show()
        return fig

    def _plot_multiple_series_plotly(
        self,
        series_list: List[pd.Series],
        labels: Optional[List[str]],
        title: str,
        xlabel: str,
        ylabel: str,
        show: bool,
        overlays: Optional[Dict[str, pd.Series]],
        show_overlays: bool,
    ) -> go.Figure:
        """Plot multiple series using Plotly."""
        fig = go.Figure()

        for i, series in enumerate(series_list):
            if not series.empty:
                label = labels[i] if labels and i < len(labels) else f"Series {i+1}"
                fig.add_trace(
                    go.Scatter(
                        x=series.index, y=series.values, mode="lines", name=label
                    )
                )

        # Plot overlays if requested
        if show_overlays and overlays:
            for name, overlay_data in overlays.items():
                if not overlay_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=overlay_data.index,
                            y=overlay_data.values,
                            mode="lines",
                            name=f"{name} (Overlay)",
                            line=dict(dash="dash", width=1),
                            opacity=0.7,
                        )
                    )
                else:
                    logger.warning(f"Overlay '{name}' is empty, skipping")

        fig.update_layout(
            title=title, xaxis_title=xlabel, yaxis_title=ylabel, showlegend=True
        )

        if show:
            fig.show()
        return fig

    def plot_with_confidence(
        self,
        data: pd.Series,
        confidence_intervals: pd.DataFrame,
        title: str = "Time Series with Confidence Intervals",
        xlabel: str = "Time",
        ylabel: str = "Value",
        figsize: Optional[tuple] = None,
        show: bool = True,
    ) -> Union[plt.Figure, go.Figure]:
        """Plot time series with confidence intervals.

        Args:
            data: Time series data
            confidence_intervals: DataFrame with lower and upper confidence bounds
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size (matplotlib only)
            show: Whether to display the plot

        Returns:
            Matplotlib or Plotly figure
        """
        if self.backend == "matplotlib":
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
            data.plot(ax=ax, label="Actual")
            ax.fill_between(
                confidence_intervals.index,
                confidence_intervals["lower"],
                confidence_intervals["upper"],
                alpha=0.3,
                label="Confidence Interval",
            )
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True)
            ax.legend()

            if show:
                plt.show()
            return fig
        else:
            fig = go.Figure()

            # Add actual values
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data.values,
                    mode="lines",
                    name="Actual",
                    line=dict(color="blue"),
                )
            )

            # Add confidence intervals
            fig.add_trace(
                go.Scatter(
                    x=confidence_intervals.index,
                    y=confidence_intervals["upper"],
                    fill=None,
                    mode="lines",
                    line_color="rgba(0,100,80,0.2)",
                    name="Upper Bound",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=confidence_intervals.index,
                    y=confidence_intervals["lower"],
                    fill="tonexty",
                    mode="lines",
                    line_color="rgba(0,100,80,0.2)",
                    name="Lower Bound",
                )
            )

            fig.update_layout(
                title=title, xaxis_title=xlabel, yaxis_title=ylabel, showlegend=True
            )

            if show:
                fig.show()
            return fig

    def plot_seasonal_decomposition(
        self,
        data: pd.Series,
        period: int,
        title: str = "Seasonal Decomposition",
        figsize: Optional[tuple] = None,
        show: bool = True,
    ) -> Union[plt.Figure, go.Figure]:
        """Plot seasonal decomposition of time series.

        Args:
            data: Time series data
            period: Period for seasonal decomposition
            title: Plot title
            figsize: Figure size (matplotlib only)
            show: Whether to display the plot

        Returns:
            Matplotlib or Plotly figure
        """
        decomposition = seasonal_decompose(data, period=period)

        if self.backend == "matplotlib":
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(
                4, 1, figsize=figsize or self.figsize
            )

            decomposition.observed.plot(ax=ax1)
            ax1.set_title("Observed")
            decomposition.trend.plot(ax=ax2)
            ax2.set_title("Trend")
            decomposition.seasonal.plot(ax=ax3)
            ax3.set_title("Seasonal")
            decomposition.resid.plot(ax=ax4)
            ax4.set_title("Residual")

            plt.tight_layout()
            if show:
                plt.show()
            return fig
        else:
            fig = make_subplots(
                rows=4,
                cols=1,
                subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
            )

            # Add observed
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=decomposition.observed,
                    mode="lines",
                    name="Observed",
                ),
                row=1,
                col=1,
            )

            # Add trend
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=decomposition.trend, mode="lines", name="Trend"
                ),
                row=2,
                col=1,
            )

            # Add seasonal
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=decomposition.seasonal,
                    mode="lines",
                    name="Seasonal",
                ),
                row=3,
                col=1,
            )

            # Add residual
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=decomposition.resid, mode="lines", name="Residual"
                ),
                row=4,
                col=1,
            )

            fig.update_layout(title=title, height=1000, showlegend=False)

            if show:
                fig.show()
            return fig

    def plot_data(self, data: Union[pd.Series, np.ndarray, List[float]]) -> None:
        """Plot the given data.

        Args:
            data: Data to plot (pandas Series, numpy array, or list)
        """
        plt.plot(data)
        plt.show()

    def plot_histogram(self, data: Union[pd.Series, np.ndarray, List[float]]) -> None:
        """Plot a histogram of the given data.

        Args:
            data: Data to plot histogram for (pandas Series, numpy array, or list)
        """
        plt.hist(data)
        plt.show()


class PerformancePlotter:
    """Class for plotting performance metrics and analysis."""

    def __init__(self, backend: str = "matplotlib", figsize: tuple = (12, 8)):
        """Initialize the performance plotter.

        Args:
            backend: Plotting backend ('matplotlib' or 'plotly')
            figsize: Default figure size
        """
        self.backend = backend
        self.figsize = figsize

    def plot_performance_metrics(
        self,
        metrics: Dict[str, float],
        title: str = "Performance Metrics",
        show: bool = True,
    ) -> Union[plt.Figure, go.Figure]:
        """Plot performance metrics as a bar chart.

        Args:
            metrics: Dictionary of metric names and values
            title: Plot title
            show: Whether to display the plot

        Returns:
            Matplotlib or Plotly figure
        """
        if self.backend == "matplotlib":
            fig, ax = plt.subplots(figsize=self.figsize)

            metrics_names = list(metrics.keys())
            metrics_values = list(metrics.values())

            bars = ax.bar(metrics_names, metrics_values)
            ax.set_title(title)
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, value in zip(bars, metrics_values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                )

            plt.xticks(rotation=45)
            plt.tight_layout()

            if show:
                plt.show()
            return fig
        else:
            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=list(metrics.keys()),
                    y=list(metrics.values()),
                    text=[f"{v:.3f}" for v in metrics.values()],
                    textposition="auto",
                )
            )

            fig.update_layout(
                title=title,
                xaxis_title="Metrics",
                yaxis_title="Value",
                showlegend=False,
            )

            if show:
                fig.show()
            return fig

    def plot_rolling_performance(
        self,
        returns: pd.Series,
        window: int = 30,
        title: str = "Rolling Performance",
        show: bool = True,
    ) -> Union[plt.Figure, go.Figure]:
        """Plot rolling performance metrics.

        Args:
            returns: Series of returns
            window: Rolling window size
            title: Plot title
            show: Whether to display the plot

        Returns:
            Matplotlib or Plotly figure
        """
        # Calculate rolling metrics
        rolling_sharpe = (
            returns.rolling(window).mean()
            / returns.rolling(window).std()
            * np.sqrt(252)
        )
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        rolling_return = returns.rolling(window).mean() * 252

        if self.backend == "matplotlib":
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.figsize)

            # Rolling Sharpe Ratio
            rolling_sharpe.plot(ax=ax1, label=f"{window}-day Rolling Sharpe")
            ax1.set_title("Rolling Sharpe Ratio")
            ax1.grid(True)
            ax1.legend()

            # Rolling Volatility
            rolling_vol.plot(ax=ax2, label=f"{window}-day Rolling Volatility")
            ax2.set_title("Rolling Volatility")
            ax2.grid(True)
            ax2.legend()

            # Rolling Return
            rolling_return.plot(ax=ax3, label=f"{window}-day Rolling Return")
            ax3.set_title("Rolling Annualized Return")
            ax3.grid(True)
            ax3.legend()

            plt.tight_layout()
            if show:
                plt.show()
            return fig
        else:
            fig = make_subplots(
                rows=3,
                cols=1,
                subplot_titles=(
                    "Rolling Sharpe Ratio",
                    "Rolling Volatility",
                    "Rolling Return",
                ),
            )

            # Rolling Sharpe Ratio
            fig.add_trace(
                go.Scatter(
                    x=rolling_sharpe.index, y=rolling_sharpe.values, name="Sharpe Ratio"
                ),
                row=1,
                col=1,
            )

            # Rolling Volatility
            fig.add_trace(
                go.Scatter(
                    x=rolling_vol.index, y=rolling_vol.values, name="Volatility"
                ),
                row=2,
                col=1,
            )

            # Rolling Return
            fig.add_trace(
                go.Scatter(
                    x=rolling_return.index, y=rolling_return.values, name="Return"
                ),
                row=3,
                col=1,
            )

            fig.update_layout(title=title, height=800, showlegend=False)

            if show:
                fig.show()
            return fig


class FeatureImportancePlotter:
    """Class for plotting feature importance."""

    def __init__(self, backend: str = "matplotlib", figsize: tuple = (10, 6)):
        """Initialize the plotter.

        Args:
            backend: Plotting backend ('matplotlib' or 'plotly')
            figsize: Figure size (matplotlib only)
        """
        self.backend = backend
        self.figsize = figsize

    def plot_feature_importance(
        self,
        importance_scores: Dict[str, float],
        title: str = "Feature Importance",
        show: bool = True,
    ) -> Union[plt.Figure, go.Figure]:
        """Plot feature importance scores.

        Args:
            importance_scores: Dictionary of feature names and their importance scores
            title: Plot title
            show: Whether to display the plot

        Returns:
            Matplotlib or Plotly figure
        """
        features = list(importance_scores.keys())
        scores = list(importance_scores.values())

        # Sort features by importance
        sorted_idx = np.argsort(scores)
        features = [features[i] for i in sorted_idx]
        scores = [scores[i] for i in sorted_idx]

        if self.backend == "matplotlib":
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.barh(range(len(features)), scores)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_xlabel("Importance Score")
            ax.set_title(title)
            plt.tight_layout()

            if show:
                plt.show()
            return fig
        else:
            fig = go.Figure()
            fig.add_trace(go.Bar(y=features, x=scores, orientation="h"))
            fig.update_layout(
                title=title,
                xaxis_title="Importance Score",
                yaxis_title="Features",
                showlegend=False,
            )

            if show:
                fig.show()
            return fig

    def plot_feature_importance_comparison(
        self,
        importance_scores_list: List[Dict[str, float]],
        model_names: List[str],
        title: str = "Feature Importance Comparison",
        show: bool = True,
    ) -> Union[plt.Figure, go.Figure]:
        """Plot feature importance comparison across multiple models.

        Args:
            importance_scores_list: List of dictionaries containing feature importance scores
            model_names: List of model names
            title: Plot title
            show: Whether to display the plot

        Returns:
            Matplotlib or Plotly figure
        """
        # Get all unique features
        all_features = set()
        for scores in importance_scores_list:
            all_features.update(scores.keys())
        all_features = sorted(list(all_features))

        # Create comparison matrix
        comparison_matrix = np.zeros((len(all_features), len(model_names)))
        for i, scores in enumerate(importance_scores_list):
            for j, feature in enumerate(all_features):
                comparison_matrix[j, i] = scores.get(feature, 0)

        if self.backend == "matplotlib":
            fig, ax = plt.subplots(figsize=self.figsize)
            x = np.arange(len(all_features))
            width = 0.8 / len(model_names)

            for i, model_name in enumerate(model_names):
                ax.bar(
                    x + i * width - 0.4 + width / 2,
                    comparison_matrix[:, i],
                    width,
                    label=model_name,
                )

            ax.set_xlabel("Features")
            ax.set_ylabel("Importance Score")
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(all_features, rotation=45, ha="right")
            ax.legend()
            plt.tight_layout()

            if show:
                plt.show()
            return fig
        else:
            fig = go.Figure()

            for i, model_name in enumerate(model_names):
                fig.add_trace(
                    go.Bar(x=all_features, y=comparison_matrix[:, i], name=model_name)
                )

            fig.update_layout(
                title=title,
                xaxis_title="Features",
                yaxis_title="Importance Score",
                barmode="group",
                showlegend=True,
            )

            if show:
                fig.show()
            return fig


class PredictionPlotter:
    """Class for plotting model predictions."""

    def __init__(self, backend: str = "matplotlib", figsize: tuple = (12, 6)):
        """Initialize the plotter.

        Args:
            backend: Plotting backend ('matplotlib' or 'plotly')
            figsize: Figure size (matplotlib only)
        """
        self.backend = backend
        self.figsize = figsize

    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: Optional[np.ndarray] = None,
        title: str = "Model Predictions",
        xlabel: str = "Time",
        ylabel: str = "Value",
        show: bool = True,
    ) -> Union[plt.Figure, go.Figure]:
        """Plot actual values and predictions.

        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
            dates: Optional array of dates
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            show: Whether to display the plot

        Returns:
            Matplotlib or Plotly figure
        """
        if dates is None:
            dates = np.arange(len(y_true))

        if self.backend == "matplotlib":
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.plot(dates, y_true, label="Actual")
            ax.plot(dates, y_pred, label="Predicted", linestyle="--")
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True)
            ax.legend()

            if show:
                plt.show()
            return fig
        else:
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=y_true,
                    mode="lines",
                    name="Actual",
                    line=dict(color="blue"),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=y_pred,
                    mode="lines",
                    name="Predicted",
                    line=dict(color="red", dash="dash"),
                )
            )

            fig.update_layout(
                title=title, xaxis_title=xlabel, yaxis_title=ylabel, showlegend=True
            )

            if show:
                fig.show()
            return fig

    def plot_predictions_with_intervals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        dates: Optional[np.ndarray] = None,
        title: str = "Model Predictions with Confidence Intervals",
        xlabel: str = "Time",
        ylabel: str = "Value",
        show: bool = True,
    ) -> Union[plt.Figure, go.Figure]:
        """Plot predictions with confidence intervals.

        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
            lower_bound: Array of lower confidence bounds
            upper_bound: Array of upper confidence bounds
            dates: Optional array of dates
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            show: Whether to display the plot

        Returns:
            Matplotlib or Plotly figure
        """
        if dates is None:
            dates = np.arange(len(y_true))

        if self.backend == "matplotlib":
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.plot(dates, y_true, label="Actual")
            ax.plot(dates, y_pred, label="Predicted", linestyle="--")
            ax.fill_between(
                dates, lower_bound, upper_bound, alpha=0.3, label="Confidence Interval"
            )
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True)
            ax.legend()

            if show:
                plt.show()
            return fig
        else:
            fig = go.Figure()

            # Add actual values
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=y_true,
                    mode="lines",
                    name="Actual",
                    line=dict(color="blue"),
                )
            )

            # Add predictions
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=y_pred,
                    mode="lines",
                    name="Predicted",
                    line=dict(color="red", dash="dash"),
                )
            )

            # Add confidence intervals
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=upper_bound,
                    fill=None,
                    mode="lines",
                    line_color="rgba(0,100,80,0.2)",
                    name="Upper Bound",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=lower_bound,
                    fill="tonexty",
                    mode="lines",
                    line_color="rgba(0,100,80,0.2)",
                    name="Lower Bound",
                )
            )

            fig.update_layout(
                title=title, xaxis_title=xlabel, yaxis_title=ylabel, showlegend=True
            )

            if show:
                fig.show()
            return fig
