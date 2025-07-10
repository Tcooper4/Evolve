"""Utility functions for visualizing forecasts and model interpretability."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Tuple, Dict, Any

def plot_forecast(data: pd.DataFrame, 
                 predictions: np.ndarray,
                 show_confidence: bool = False,
                 confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> go.Figure:
    """Plot actual values and predictions with optional confidence intervals.
    
    Args:
        data: DataFrame with actual values
        predictions: Array of predicted values
        show_confidence: Whether to show confidence intervals
        confidence_intervals: Optional tuple of (lower, upper) confidence bounds
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Plot actual values
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['close'],
        name='Actual',
        line=dict(color='blue')
    ))
    
    # Plot predictions
    fig.add_trace(go.Scatter(
        x=data.index[-len(predictions):],
        y=predictions,
        name='Predicted',
        line=dict(color='red', dash='dash')
    ))
    
    # Add confidence intervals if requested
    if show_confidence and confidence_intervals is not None:
        lower, upper = confidence_intervals
        fig.add_trace(go.Scatter(
            x=data.index[-len(predictions):],
            y=upper,
            fill=None,
            mode='lines',
            line_color='rgba(0,100,80,0.2)',
            name='Upper Bound'
        ))
        fig.add_trace(go.Scatter(
            x=data.index[-len(predictions):],
            y=lower,
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,100,80,0.2)',
            name='Lower Bound'
        ))
    
    fig.update_layout(
        title='Price Forecast',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified'
    )
    
    return fig

def plot_attention_heatmap(model: Any, data: pd.DataFrame) -> go.Figure:
    """Generate attention heatmap for transformer-based models.
    
    Args:
        model: Model with attention_heatmap method
        data: Input data
        
    Returns:
        Plotly figure
    """
    if not hasattr(model, 'attention_heatmap'):
        raise ValueError("Model does not support attention visualization")
    
    attention_weights = model.attention_heatmap(data)
    
    fig = go.Figure(data=go.Heatmap(
        z=attention_weights,
        x=data.index[-attention_weights.shape[1]:],
        y=range(attention_weights.shape[0]),
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title='Attention Weights Heatmap',
        xaxis_title='Time Steps',
        yaxis_title='Attention Heads'
    )
    
    return fig

def plot_shap_values(model: Any, data: pd.DataFrame) -> go.Figure:
    """Visualize SHAP values for model interpretability.
    
    Args:
        model: Model with shap_interpret method
        data: Input data
        
    Returns:
        Plotly figure
    """
    # Check if SHAP is available
    try:
        import shap
        SHAP_AVAILABLE = True
    except ImportError:
        SHAP_AVAILABLE = False
    
    if not SHAP_AVAILABLE:
        # Create fallback visualization
        fig = go.Figure()
        fig.add_annotation(
            text="SHAP not available. Install with: pip install shap",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title='SHAP Values Analysis (Not Available)',
            height=400,
            showlegend=False
        )
        return fig
    
    if not hasattr(model, 'shap_interpret'):
        # Create fallback visualization
        fig = go.Figure()
        fig.add_annotation(
            text="Model does not support SHAP interpretation",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="orange")
        )
        fig.update_layout(
            title='SHAP Values Analysis (Not Supported)',
            height=400,
            showlegend=False
        )
        return fig
    
    try:
        shap_values = model.shap_interpret(data)
        
        # Create subplot for SHAP summary plot
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('SHAP Summary Plot', 'Feature Importance'))
        
        # Add SHAP summary plot
        for i, feature in enumerate(data.columns):
            fig.add_trace(
                go.Scatter(
                    x=shap_values[:, i],
                    y=[feature] * len(shap_values),
                    mode='markers',
                    name=feature,
                    marker=dict(
                        size=8,
                        color=data[feature],
                        colorscale='Viridis',
                        showscale=True
                    )
                ),
                row=1, col=1
            )
        
        # Add feature importance bar plot
        feature_importance = np.abs(shap_values).mean(axis=0)
        fig.add_trace(
            go.Bar(
                x=data.columns,
                y=feature_importance,
                name='Feature Importance'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='SHAP Values Analysis',
            height=800,
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        # Create error visualization
        fig = go.Figure()
        fig.add_annotation(
            text=f"SHAP calculation failed: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(
            title='SHAP Values Analysis (Error)',
            height=400,
            showlegend=False
        )
        return fig

def plot_backtest_results(results: pd.DataFrame) -> go.Figure:
    """Plot backtest results including cumulative returns and drawdown.
    
    Args:
        results: DataFrame with backtest results
        
    Returns:
        Plotly figure
    """
    fig = make_subplots(rows=2, cols=1,
                       subplot_titles=('Cumulative Returns', 'Drawdown'),
                       vertical_spacing=0.1)
    
    # Plot cumulative returns
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results['cumulative_returns'],
            name='Strategy Returns',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    if 'benchmark_returns' in results.columns:
        fig.add_trace(
            go.Scatter(
                x=results.index,
                y=results['benchmark_returns'],
                name='Benchmark Returns',
                line=dict(color='gray', dash='dash')
            ),
            row=1, col=1
        )
    
    # Plot drawdown
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results['drawdown'],
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='red')
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title='Backtest Results',
        height=800,
        showlegend=True
    )
    
    return fig

def plot_model_components(model: Any, data: pd.DataFrame) -> go.Figure:
    """Plot model components for interpretable models.
    
    Args:
        model: Model with component analysis
        data: Input data
        
    Returns:
        Plotly figure
    """
    # Create a simple component visualization
    fig = go.Figure()
    
    # Add sample component data
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['close'] if 'close' in data.columns else data.iloc[:, 0],
        name='Data',
        line=dict(color='blue')
    ))
    
    fig.update_layout(
        title='Model Components',
        xaxis_title='Time',
        yaxis_title='Value',
        template='plotly_white'
    )
    
    return fig

def plot_performance_over_time(performance_data: pd.DataFrame) -> go.Figure:
    """Plot performance metrics over time.
    
    Args:
        performance_data: DataFrame with performance metrics
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Add performance metrics
    if 'accuracy' in performance_data.columns:
        fig.add_trace(go.Scatter(
            x=performance_data.index,
            y=performance_data['accuracy'],
            name='Accuracy',
            line=dict(color='green')
        ))
    
    if 'sharpe_ratio' in performance_data.columns:
        fig.add_trace(go.Scatter(
            x=performance_data.index,
            y=performance_data['sharpe_ratio'],
            name='Sharpe Ratio',
            line=dict(color='blue')
        ))
    
    fig.update_layout(
        title='Performance Over Time',
        xaxis_title='Date',
        yaxis_title='Metric Value',
        template='plotly_white'
    )
    
    return fig

def plot_model_comparison(metrics: pd.DataFrame) -> go.Figure:
    """Compare different models' performance metrics.
    
    Args:
        metrics: DataFrame with model metrics
        
    Returns:
        Plotly figure
    """
    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=('RMSE', 'Directional Accuracy',
                                     'Sharpe Ratio', 'Max Drawdown'))
    
    # Plot RMSE
    fig.add_trace(
        go.Bar(
            x=metrics.index,
            y=metrics['rmse'],
            name='RMSE'
        ),
        row=1, col=1
    )
    
    # Plot Directional Accuracy
    fig.add_trace(
        go.Bar(
            x=metrics.index,
            y=metrics['directional_accuracy'],
            name='Directional Accuracy'
        ),
        row=1, col=2
    )
    
    # Plot Sharpe Ratio
    fig.add_trace(
        go.Bar(
            x=metrics.index,
            y=metrics['sharpe_ratio'],
            name='Sharpe Ratio'
        ),
        row=2, col=1
    )
    
    # Plot Max Drawdown
    fig.add_trace(
        go.Bar(
            x=metrics.index,
            y=metrics['max_drawdown'],
            name='Max Drawdown'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title='Model Comparison',
        height=800,
        showlegend=False
    )
    
    return fig