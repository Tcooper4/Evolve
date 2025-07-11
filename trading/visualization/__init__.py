"""
Trading Visualization Module

Core chart types and visualization utilities for trading analysis.
"""

from trading.visualization.plotting import (
    TimeSeriesPlotter,
    PerformancePlotter,
    FeatureImportancePlotter,
    PredictionPlotter
)

# Core chart types with consistent naming
from trading.utils.visualization import (
    plot_forecast,
    plot_backtest_results,
    plot_shap_values,
    plot_attention_heatmap,
    plot_model_components,
    plot_performance_over_time,
    plot_model_comparison
)

# Chart type aliases for consistency
line_chart = plot_forecast
candlestick_chart = plot_backtest_results  # For price data visualization
signal_overlay = plot_model_components  # For signal visualization

__all__ = [
    # Core plotters
    'TimeSeriesPlotter',
    'PerformancePlotter',
    'FeatureImportancePlotter',
    'PredictionPlotter',
    
    # Core chart functions
    'plot_forecast',
    'plot_backtest_results',
    'plot_shap_values',
    'plot_attention_heatmap',
    'plot_model_components',
    'plot_performance_over_time',
    'plot_model_comparison',
    
    # Chart type aliases
    'line_chart',
    'candlestick_chart',
    'signal_overlay'
] 