import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from trading.visualization.plotting import (
    TimeSeriesPlotter,
    PerformancePlotter,
    FeatureImportancePlotter,
    PredictionPlotter
)

class TestVisualization:
    """Test suite for visualization utilities."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        # Generate sample data
        dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='D')
        actual = pd.Series(np.random.randn(n_samples) * 10 + 100, index=dates)
        predicted = actual + np.random.randn(n_samples) * 2  # Add some noise
        
        return actual, predicted
    
    @pytest.fixture
    def time_series_plotter(self):
        return TimeSeriesPlotter()
    
    @pytest.fixture
    def performance_plotter(self):
        return PerformancePlotter()
    
    @pytest.fixture
    def feature_importance_plotter(self):
        return FeatureImportancePlotter()
    
    @pytest.fixture
    def prediction_plotter(self):
        return PredictionPlotter()
    
    def test_time_series_plotter(self, time_series_plotter, sample_data):
        """Test time series plotting."""
        actual, predicted = sample_data
        
        # Test basic plotting
        fig = time_series_plotter.plot_time_series(actual, predicted)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test with confidence intervals
        fig = time_series_plotter.plot_time_series_with_confidence(
            actual,
            predicted,
            lower_bound=predicted - 2,
            upper_bound=predicted + 2
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_performance_plotter(self, performance_plotter, sample_data):
        """Test performance plotting."""
        actual, predicted = sample_data
        
        # Calculate returns
        actual_returns = actual.pct_change().dropna()
        predicted_returns = predicted.pct_change().dropna()
        
        # Test cumulative returns plot
        fig = performance_plotter.plot_cumulative_returns(actual_returns, predicted_returns)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test drawdown plot
        fig = performance_plotter.plot_drawdown(actual_returns)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test rolling metrics plot
        fig = performance_plotter.plot_rolling_metrics(actual_returns, window=20)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_feature_importance_plotter(self, feature_importance_plotter):
        """Test feature importance plotting."""
        # Generate sample feature importance data
        features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
        importance = np.random.rand(5)
        
        # Test bar plot
        fig = feature_importance_plotter.plot_feature_importance(features, importance)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test horizontal bar plot
        fig = feature_importance_plotter.plot_feature_importance(
            features,
            importance,
            horizontal=True
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_prediction_plotter(self, prediction_plotter, sample_data):
        """Test prediction plotting."""
        actual, predicted = sample_data
        
        # Test scatter plot
        fig = prediction_plotter.plot_predictions_scatter(actual, predicted)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test residual plot
        fig = prediction_plotter.plot_residuals(actual, predicted)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test prediction intervals
        fig = prediction_plotter.plot_prediction_intervals(
            actual,
            predicted,
            lower_bound=predicted - 2,
            upper_bound=predicted + 2
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_customization(self, time_series_plotter, sample_data):
        """Test plot customization."""
        actual, predicted = sample_data
        
        # Test with custom style
        fig = time_series_plotter.plot_time_series(
            actual,
            predicted,
            style='dark_background',
            figsize=(12, 6),
            title='Custom Title',
            xlabel='Custom X Label',
            ylabel='Custom Y Label'
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_saving(self, time_series_plotter, sample_data, tmp_path):
        """Test plot saving."""
        actual, predicted = sample_data
        
        # Create and save plot
        fig = time_series_plotter.plot_time_series(actual, predicted)
        save_path = tmp_path / 'test_plot.png'
        fig.savefig(str(save_path))
        
        assert save_path.exists()
        plt.close(fig)
    
    def test_plot_interactivity(self, time_series_plotter, sample_data):
        """Test plot interactivity."""
        actual, predicted = sample_data
        
        # Test interactive plot
        fig = time_series_plotter.plot_time_series(
            actual,
            predicted,
            interactive=True
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_annotations(self, time_series_plotter, sample_data):
        """Test plot annotations."""
        actual, predicted = sample_data
        
        # Test plot with annotations
        annotations = {
            actual.index[10]: 'Important Point 1',
            actual.index[50]: 'Important Point 2'
        }
        
        fig = time_series_plotter.plot_time_series(
            actual,
            predicted,
            annotations=annotations
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_multiple_series(self, time_series_plotter):
        """Test plotting multiple series."""
        # Generate multiple series
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        series1 = pd.Series(np.random.randn(100), index=dates)
        series2 = pd.Series(np.random.randn(100), index=dates)
        series3 = pd.Series(np.random.randn(100), index=dates)
        
        # Test plotting multiple series
        fig = time_series_plotter.plot_multiple_series(
            [series1, series2, series3],
            labels=['Series 1', 'Series 2', 'Series 3']
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig) 