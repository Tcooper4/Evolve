import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Optional
from statsmodels.tsa.seasonal import seasonal_decompose

class TimeSeriesPlotter:
    """Class for plotting time series data."""
    
    def __init__(self, style: str = 'seaborn'):
        """Initialize the plotter.
        
        Args:
            style: Plot style to use
        """
        self.style = style
        plt.style.use(style)
        
    def plot_time_series(self, data: pd.Series, title: str = 'Time Series Plot',
                        xlabel: str = 'Time', ylabel: str = 'Value',
                        figsize: tuple = (12, 6)) -> plt.Figure:
        """Plot a single time series.
        
        Args:
            data: Time series data
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        data.plot(ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        return fig
        
    def plot_multiple_series(self, series_list: List[pd.Series],
                           labels: Optional[List[str]] = None,
                           title: str = 'Multiple Time Series',
                           xlabel: str = 'Time', ylabel: str = 'Value',
                           figsize: tuple = (12, 6)) -> plt.Figure:
        """Plot multiple time series on the same plot.
        
        Args:
            series_list: List of time series to plot
            labels: Optional list of labels for each series
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, series in enumerate(series_list):
            label = labels[i] if labels and i < len(labels) else f'Series {i+1}'
            series.plot(ax=ax, label=label)
            
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()
        return fig
        
    def plot_with_confidence(self, data: pd.Series, confidence_intervals: pd.DataFrame,
                           title: str = 'Time Series with Confidence Intervals',
                           xlabel: str = 'Time', ylabel: str = 'Value',
                           figsize: tuple = (12, 6)) -> plt.Figure:
        """Plot time series with confidence intervals.
        
        Args:
            data: Time series data
            confidence_intervals: DataFrame with lower and upper confidence bounds
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        data.plot(ax=ax, label='Actual')
        ax.fill_between(confidence_intervals.index,
                       confidence_intervals['lower'],
                       confidence_intervals['upper'],
                       alpha=0.3, label='Confidence Interval')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()
        return fig
        
    def plot_seasonal_decomposition(self, data: pd.Series, period: int,
                                  title: str = 'Seasonal Decomposition',
                                  figsize: tuple = (12, 8)) -> plt.Figure:
        """Plot seasonal decomposition of time series.
        
        Args:
            data: Time series data
            period: Period for seasonal decomposition
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        decomposition = seasonal_decompose(data, period=period)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=figsize)
        
        decomposition.observed.plot(ax=ax1)
        ax1.set_title('Observed')
        decomposition.trend.plot(ax=ax2)
        ax2.set_title('Trend')
        decomposition.seasonal.plot(ax=ax3)
        ax3.set_title('Seasonal')
        decomposition.resid.plot(ax=ax4)
        ax4.set_title('Residual')
        
        plt.tight_layout()
        return fig

    def plot_data(self, data):
        """Plot the given data."""
        plt.plot(data)
        plt.show()

    def plot_histogram(self, data):
        """Plot a histogram of the given data."""
        plt.hist(data)
        plt.show()

class PerformancePlotter:
    def __init__(self):
        pass

    def plot_performance(self, data):
        """Plot the performance of the given data."""
        plt.plot(data)
        plt.title('Performance Over Time')
        plt.xlabel('Time')
        plt.ylabel('Performance')
        plt.grid(True)
        plt.show() 

class FeatureImportancePlotter:
    """Class for plotting feature importance."""
    
    def __init__(self, figsize: tuple = (10, 6)):
        """Initialize the plotter.
        
        Args:
            figsize: Figure size for plots
        """
        self.figsize = figsize
        
    def plot_feature_importance(self, importance_scores: Dict[str, float], 
                              title: str = 'Feature Importance') -> None:
        """Plot feature importance scores.
        
        Args:
            importance_scores: Dictionary of feature names and their importance scores
            title: Plot title
        """
        features = list(importance_scores.keys())
        scores = list(importance_scores.values())
        
        # Sort features by importance
        sorted_idx = np.argsort(scores)
        features = [features[i] for i in sorted_idx]
        scores = [scores[i] for i in sorted_idx]
        
        plt.figure(figsize=self.figsize)
        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title(title)
        plt.tight_layout()
        plt.show()
        
    def plot_feature_importance_comparison(self, 
                                         importance_scores_list: List[Dict[str, float]],
                                         model_names: List[str],
                                         title: str = 'Feature Importance Comparison') -> None:
        """Plot feature importance comparison across multiple models.
        
        Args:
            importance_scores_list: List of dictionaries containing feature importance scores
            model_names: List of model names
            title: Plot title
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
        
        # Plot
        plt.figure(figsize=self.figsize)
        x = np.arange(len(all_features))
        width = 0.8 / len(model_names)
        
        for i, model_name in enumerate(model_names):
            plt.bar(x + i * width - 0.4 + width/2, 
                   comparison_matrix[:, i], 
                   width, 
                   label=model_name)
        
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.title(title)
        plt.xticks(x, all_features, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def plot_feature_correlation(self, correlation_matrix: pd.DataFrame,
                               title: str = 'Feature Correlation Matrix') -> None:
        """Plot feature correlation matrix.
        
        Args:
            correlation_matrix: DataFrame containing correlation values
            title: Plot title
        """
        plt.figure(figsize=self.figsize)
        plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(correlation_matrix.columns)), 
                  correlation_matrix.columns, 
                  rotation=45, 
                  ha='right')
        plt.yticks(range(len(correlation_matrix.index)), 
                  correlation_matrix.index)
        plt.title(title)
        plt.tight_layout()
        plt.show()

class PredictionPlotter:
    """Class for plotting predictions and their confidence intervals."""
    
    def __init__(self, figsize: tuple = (12, 6)):
        """Initialize the plotter.
        
        Args:
            figsize: Figure size for plots
        """
        self.figsize = figsize
        
    def plot_predictions(self, 
                        y_true: np.ndarray,
                        y_pred: np.ndarray,
                        dates: Optional[np.ndarray] = None,
                        title: str = 'Model Predictions',
                        xlabel: str = 'Time',
                        ylabel: str = 'Value') -> None:
        """Plot true values and predictions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            dates: Optional array of dates for x-axis
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        plt.figure(figsize=self.figsize)
        
        if dates is not None:
            plt.plot(dates, y_true, label='True', color='blue')
            plt.plot(dates, y_pred, label='Predicted', color='red', linestyle='--')
            plt.xticks(rotation=45)
        else:
            plt.plot(y_true, label='True', color='blue')
            plt.plot(y_pred, label='Predicted', color='red', linestyle='--')
            
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def plot_predictions_with_intervals(self,
                                      y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      lower_bound: np.ndarray,
                                      upper_bound: np.ndarray,
                                      dates: Optional[np.ndarray] = None,
                                      title: str = 'Model Predictions with Confidence Intervals',
                                      xlabel: str = 'Time',
                                      ylabel: str = 'Value') -> None:
        """Plot true values, predictions, and confidence intervals.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            lower_bound: Lower bound of confidence interval
            upper_bound: Upper bound of confidence interval
            dates: Optional array of dates for x-axis
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        plt.figure(figsize=self.figsize)
        
        if dates is not None:
            plt.plot(dates, y_true, label='True', color='blue')
            plt.plot(dates, y_pred, label='Predicted', color='red', linestyle='--')
            plt.fill_between(dates, lower_bound, upper_bound, 
                           color='red', alpha=0.2, label='Confidence Interval')
            plt.xticks(rotation=45)
        else:
            plt.plot(y_true, label='True', color='blue')
            plt.plot(y_pred, label='Predicted', color='red', linestyle='--')
            plt.fill_between(range(len(y_true)), lower_bound, upper_bound,
                           color='red', alpha=0.2, label='Confidence Interval')
            
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def plot_residuals(self,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      dates: Optional[np.ndarray] = None,
                      title: str = 'Residuals Plot',
                      xlabel: str = 'Time',
                      ylabel: str = 'Residual') -> None:
        """Plot residuals (errors) between true and predicted values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            dates: Optional array of dates for x-axis
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        residuals = y_true - y_pred
        
        plt.figure(figsize=self.figsize)
        
        if dates is not None:
            plt.plot(dates, residuals, color='blue')
            plt.axhline(y=0, color='red', linestyle='--')
            plt.xticks(rotation=45)
        else:
            plt.plot(residuals, color='blue')
            plt.axhline(y=0, color='red', linestyle='--')
            
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.show() 