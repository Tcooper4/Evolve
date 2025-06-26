"""Advanced plotting utilities for time series and performance visualization.

This module provides comprehensive plotting functionality for time series data,
performance metrics, and model predictions, with support for both Matplotlib
and Plotly backends.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Optional, Tuple
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

class TimeSeriesPlotter:
    """Class for plotting time series data with performance metrics."""
    
    def __init__(
        self,
        style: str = 'seaborn',
        backend: str = 'matplotlib',
        figsize: tuple = (12, 6)
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
        
        if backend == 'matplotlib':
            plt.style.use(style)
    
    def plot_time_series(
        self,
        data: pd.Series,
        title: str = 'Time Series Plot',
        xlabel: str = 'Time',
        ylabel: str = 'Value',
        figsize: Optional[tuple] = None,
        show: bool = True
    ) -> Union[plt.Figure, go.Figure]:
        """Plot a single time series.
        
        Args:
            data: Time series data
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size (matplotlib only)
            show: Whether to display the plot
            
        Returns:
            Matplotlib or Plotly figure
        """
        if self.backend == 'matplotlib':
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
            data.plot(ax=ax)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True)
            if show:
                plt.show()
            return fig
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data.values,
                mode='lines',
                name=data.name or 'Value'
            ))
            fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                showlegend=True
            )
            if show:
                fig.show()
            return fig
    
    def plot_performance(
        self,
        returns: pd.Series,
        benchmark: Optional[pd.Series] = None,
        drawdown: Optional[pd.Series] = None,
        title: str = 'Performance Analysis',
        figsize: Optional[tuple] = None,
        show: bool = True
    ) -> Union[plt.Figure, go.Figure]:
        """Plot performance metrics including returns, benchmark, and drawdown.
        
        Args:
            returns: Series of returns
            benchmark: Optional benchmark returns
            drawdown: Optional drawdown series
            title: Plot title
            figsize: Figure size (matplotlib only)
            show: Whether to display the plot
            
        Returns:
            Matplotlib or Plotly figure
        """
        if self.backend == 'matplotlib':
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize or self.figsize)
            
            # Plot returns
            returns.cumsum().plot(ax=ax1, label='Strategy')
            if benchmark is not None:
                benchmark.cumsum().plot(ax=ax1, label='Benchmark', linestyle='--')
            ax1.set_title('Cumulative Returns')
            ax1.grid(True)
            ax1.legend()
            
            # Plot drawdown
            if drawdown is not None:
                drawdown.plot(ax=ax2, color='red')
                ax2.set_title('Drawdown')
                ax2.grid(True)
            
            plt.tight_layout()
            if show:
                plt.show()
            return fig
        else:
            fig = make_subplots(rows=2, cols=1, subplot_titles=('Cumulative Returns', 'Drawdown'))
            
            # Plot returns
            fig.add_trace(
                go.Scatter(
                    x=returns.index,
                    y=returns.cumsum(),
                    name='Strategy',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            if benchmark is not None:
                fig.add_trace(
                    go.Scatter(
                        x=benchmark.index,
                        y=benchmark.cumsum(),
                        name='Benchmark',
                        line=dict(color='gray', dash='dash')
                    ),
                    row=1, col=1
                )
            
            # Plot drawdown
            if drawdown is not None:
                fig.add_trace(
                    go.Scatter(
                        x=drawdown.index,
                        y=drawdown,
                        name='Drawdown',
                        line=dict(color='red')
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(
                title=title,
                height=800,
                showlegend=True
            )
            
            if show:
                fig.show()
            return fig
    
    def plot_multiple_series(
        self,
        series_list: List[pd.Series],
        labels: Optional[List[str]] = None,
        title: str = 'Multiple Time Series',
        xlabel: str = 'Time',
        ylabel: str = 'Value',
        figsize: Optional[tuple] = None,
        show: bool = True
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
            
        Returns:
            Matplotlib or Plotly figure
        """
        if self.backend == 'matplotlib':
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
            
            for i, series in enumerate(series_list):
                label = labels[i] if labels and i < len(labels) else f'Series {i+1}'
                series.plot(ax=ax, label=label)
            
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
            
            for i, series in enumerate(series_list):
                label = labels[i] if labels and i < len(labels) else f'Series {i+1}'
                fig.add_trace(go.Scatter(
                    x=series.index,
                    y=series.values,
                    mode='lines',
                    name=label
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                showlegend=True
            )
            
            if show:
                fig.show()
            return fig
    
    def plot_with_confidence(
        self,
        data: pd.Series,
        confidence_intervals: pd.DataFrame,
        title: str = 'Time Series with Confidence Intervals',
        xlabel: str = 'Time',
        ylabel: str = 'Value',
        figsize: Optional[tuple] = None,
        show: bool = True
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
        if self.backend == 'matplotlib':
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
            data.plot(ax=ax, label='Actual')
            ax.fill_between(
                confidence_intervals.index,
                confidence_intervals['lower'],
                confidence_intervals['upper'],
                alpha=0.3,
                label='Confidence Interval'
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
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data.values,
                mode='lines',
                name='Actual',
                line=dict(color='blue')
            ))
            
            # Add confidence intervals
            fig.add_trace(go.Scatter(
                x=confidence_intervals.index,
                y=confidence_intervals['upper'],
                fill=None,
                mode='lines',
                line_color='rgba(0,100,80,0.2)',
                name='Upper Bound'
            ))
            fig.add_trace(go.Scatter(
                x=confidence_intervals.index,
                y=confidence_intervals['lower'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,100,80,0.2)',
                name='Lower Bound'
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                showlegend=True
            )
            
            if show:
                fig.show()
            return fig
    
    def plot_seasonal_decomposition(
        self,
        data: pd.Series,
        period: int,
        title: str = 'Seasonal Decomposition',
        figsize: Optional[tuple] = None,
        show: bool = True
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
        
        if self.backend == 'matplotlib':
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=figsize or self.figsize)
            
            decomposition.observed.plot(ax=ax1)
            ax1.set_title('Observed')
            decomposition.trend.plot(ax=ax2)
            ax2.set_title('Trend')
            decomposition.seasonal.plot(ax=ax3)
            ax3.set_title('Seasonal')
            decomposition.resid.plot(ax=ax4)
            ax4.set_title('Residual')
            
            plt.tight_layout()
            if show:
                plt.show()
            return fig
        else:
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual')
            )
            
            # Add observed
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=decomposition.observed,
                    mode='lines',
                    name='Observed'
                ),
                row=1, col=1
            )
            
            # Add trend
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=decomposition.trend,
                    mode='lines',
                    name='Trend'
                ),
                row=2, col=1
            )
            
            # Add seasonal
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=decomposition.seasonal,
                    mode='lines',
                    name='Seasonal'
                ),
                row=3, col=1
            )
            
            # Add residual
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=decomposition.resid,
                    mode='lines',
                    name='Residual'
                ),
                row=4, col=1
            )
            
            fig.update_layout(
                title=title,
                height=1000,
                showlegend=False
            )
            
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

class FeatureImportancePlotter:
    """Class for plotting feature importance."""
    
    def __init__(
        self,
        backend: str = 'matplotlib',
        figsize: tuple = (10, 6)
    ):
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
        title: str = 'Feature Importance',
        show: bool = True
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
        
        if self.backend == 'matplotlib':
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.barh(range(len(features)), scores)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_xlabel('Importance Score')
            ax.set_title(title)
            plt.tight_layout()
            
            if show:
                plt.show()
            return fig
        else:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=features,
                x=scores,
                orientation='h'
            ))
            fig.update_layout(
                title=title,
                xaxis_title='Importance Score',
                yaxis_title='Features',
                showlegend=False
            )
            
            if show:
                fig.show()
            return fig
    
    def plot_feature_importance_comparison(
        self,
        importance_scores_list: List[Dict[str, float]],
        model_names: List[str],
        title: str = 'Feature Importance Comparison',
        show: bool = True
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
        
        if self.backend == 'matplotlib':
            fig, ax = plt.subplots(figsize=self.figsize)
            x = np.arange(len(all_features))
            width = 0.8 / len(model_names)
            
            for i, model_name in enumerate(model_names):
                ax.bar(
                    x + i * width - 0.4 + width/2,
                    comparison_matrix[:, i],
                    width,
                    label=model_name
                )
            
            ax.set_xlabel('Features')
            ax.set_ylabel('Importance Score')
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(all_features, rotation=45, ha='right')
            ax.legend()
            plt.tight_layout()
            
            if show:
                plt.show()
            return fig
        else:
            fig = go.Figure()
            
            for i, model_name in enumerate(model_names):
                fig.add_trace(go.Bar(
                    x=all_features,
                    y=comparison_matrix[:, i],
                    name=model_name
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Features',
                yaxis_title='Importance Score',
                barmode='group',
                showlegend=True
            )
            
            if show:
                fig.show()
            return fig

class PredictionPlotter:
    """Class for plotting model predictions."""
    
    def __init__(
        self,
        backend: str = 'matplotlib',
        figsize: tuple = (12, 6)
    ):
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
        title: str = 'Model Predictions',
        xlabel: str = 'Time',
        ylabel: str = 'Value',
        show: bool = True
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
        
        if self.backend == 'matplotlib':
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.plot(dates, y_true, label='Actual')
            ax.plot(dates, y_pred, label='Predicted', linestyle='--')
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
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=y_true,
                mode='lines',
                name='Actual',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=y_pred,
                mode='lines',
                name='Predicted',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                showlegend=True
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
        title: str = 'Model Predictions with Confidence Intervals',
        xlabel: str = 'Time',
        ylabel: str = 'Value',
        show: bool = True
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
        
        if self.backend == 'matplotlib':
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.plot(dates, y_true, label='Actual')
            ax.plot(dates, y_pred, label='Predicted', linestyle='--')
            ax.fill_between(
                dates,
                lower_bound,
                upper_bound,
                alpha=0.3,
                label='Confidence Interval'
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
            fig.add_trace(go.Scatter(
                x=dates,
                y=y_true,
                mode='lines',
                name='Actual',
                line=dict(color='blue')
            ))
            
            # Add predictions
            fig.add_trace(go.Scatter(
                x=dates,
                y=y_pred,
                mode='lines',
                name='Predicted',
                line=dict(color='red', dash='dash')
            ))
            
            # Add confidence intervals
            fig.add_trace(go.Scatter(
                x=dates,
                y=upper_bound,
                fill=None,
                mode='lines',
                line_color='rgba(0,100,80,0.2)',
                name='Upper Bound'
            ))
            fig.add_trace(go.Scatter(
                x=dates,
                y=lower_bound,
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,100,80,0.2)',
                name='Lower Bound'
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                showlegend=True
            )
            
            if show:
                fig.show()
            return fig 