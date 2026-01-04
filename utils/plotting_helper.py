"""
Plotting helper that uses advanced plotting.py

Provides convenient wrapper functions for creating charts using the advanced plotting module.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# Try to import from trading.visualization.plotting
try:
    from trading.visualization.plotting import (
        TimeSeriesPlotter,
        PerformancePlotter,
        PredictionPlotter
    )
    PLOTTING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Advanced plotting module not available: {e}")
    PLOTTING_AVAILABLE = False
    TimeSeriesPlotter = None
    PerformancePlotter = None
    PredictionPlotter = None


def create_candlestick_chart(
    data: pd.DataFrame,
    title: str = "Candlestick Chart",
    show_volume: bool = True,
    show_ma: Optional[List[int]] = None,
    **kwargs
) -> go.Figure:
    """
    Create a candlestick chart with optional volume and moving averages.
    
    Args:
        data: DataFrame with OHLCV data (columns: open, high, low, close, volume)
        title: Chart title
        show_volume: Whether to show volume subplot
        show_ma: List of moving average periods to display (e.g., [20, 50, 200])
        **kwargs: Additional arguments
    
    Returns:
        Plotly figure
    """
    # Normalize column names
    data = data.copy()
    if 'Open' in data.columns:
        data['open'] = data['Open']
    if 'High' in data.columns:
        data['high'] = data['High']
    if 'Low' in data.columns:
        data['low'] = data['Low']
    if 'Close' in data.columns:
        data['close'] = data['Close']
    if 'Volume' in data.columns:
        data['volume'] = data['Volume']
    
    # Ensure we have required columns
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"Data must contain columns: {required_cols}")
    
    # Determine number of rows
    num_rows = 1
    if show_volume and 'volume' in data.columns:
        num_rows += 1
    
    # Create subplots
    row_heights = [0.7, 0.3] if show_volume else [1.0]
    fig = make_subplots(
        rows=num_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=row_heights,
        subplot_titles=(["Price"] if not show_volume else ["Price", "Volume"])
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name="Price"
        ),
        row=1,
        col=1
    )
    
    # Add moving averages
    if show_ma:
        for period in show_ma:
            if len(data) >= period:
                ma = data['close'].rolling(window=period).mean()
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=ma,
                        name=f'MA({period})',
                        line=dict(width=1, dash='dash')
                    ),
                    row=1,
                    col=1
                )
    
    # Volume subplot
    if show_volume and 'volume' in data.columns:
        colors = ['red' if data['close'].iloc[i] < data['open'].iloc[i] else 'green' 
                 for i in range(len(data))]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                name="Volume",
                marker_color=colors
            ),
            row=2,
            col=1
        )
        fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        height=600 if show_volume else 400,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig


def create_forecast_chart(
    historical: pd.DataFrame,
    forecast: Union[List, np.ndarray, pd.Series],
    forecast_dates: Optional[Union[List, pd.DatetimeIndex]] = None,
    confidence_intervals: Optional[Dict[str, Union[List, np.ndarray, pd.Series]]] = None,
    title: str = "Forecast Chart",
    **kwargs
) -> go.Figure:
    """
    Create a forecast chart with historical data and forecast.
    
    Args:
        historical: Historical data DataFrame
        forecast: Forecast values
        forecast_dates: Dates for forecast (if None, will be generated)
        confidence_intervals: Dict with 'lower' and 'upper' keys for confidence bands
        title: Chart title
        **kwargs: Additional arguments
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Get close price column
    close_col = None
    for col in ['close', 'Close', 'CLOSE']:
        if col in historical.columns:
            close_col = col
            break
    
    if close_col is None:
        close_col = historical.columns[0]
    
    # Historical data
    fig.add_trace(
        go.Scatter(
            x=historical.index,
            y=historical[close_col],
            mode='lines',
            name='Historical',
            line=dict(color='blue', width=2)
        )
    )
    
    # Generate forecast dates if not provided
    if forecast_dates is None:
        from datetime import timedelta
        last_date = historical.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=len(forecast),
            freq='D'
        )
    
    # Convert forecast to array
    if isinstance(forecast, (list, pd.Series)):
        forecast = np.array(forecast)
    
    # Confidence intervals
    if confidence_intervals:
        lower = confidence_intervals.get('lower')
        upper = confidence_intervals.get('upper')
        
        if lower is not None and upper is not None:
            # Convert to arrays
            if isinstance(lower, (list, pd.Series)):
                lower = np.array(lower)
            if isinstance(upper, (list, pd.Series)):
                upper = np.array(upper)
            
            # Upper bound
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=upper,
                    mode='lines',
                    name='Upper CI',
                    line=dict(width=0),
                    showlegend=False
                )
            )
            
            # Lower bound with fill
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=lower,
                    mode='lines',
                    name='Confidence Interval',
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    line=dict(width=0)
                )
            )
    
    # Forecast line
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=forecast,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=6)
        )
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode='x unified',
        height=500
    )
    
    return fig


def create_performance_chart(
    equity_curve: Union[pd.Series, np.ndarray],
    benchmark: Optional[Union[pd.Series, np.ndarray]] = None,
    title: str = "Performance Chart",
    **kwargs
) -> go.Figure:
    """
    Create a performance chart comparing equity curve with benchmark.
    
    Args:
        equity_curve: Equity curve values
        benchmark: Optional benchmark values
        title: Chart title
        **kwargs: Additional arguments
    
    Returns:
        Plotly figure
    """
    if PLOTTING_AVAILABLE and PerformancePlotter:
        plotter = PerformancePlotter(backend='plotly')
        
        # Convert to Series if needed
        if isinstance(equity_curve, np.ndarray):
            equity_curve = pd.Series(equity_curve)
        if benchmark is not None and isinstance(benchmark, np.ndarray):
            benchmark = pd.Series(benchmark)
        
        # Calculate returns from equity curve
        returns = equity_curve.pct_change().dropna()
        benchmark_returns = None
        if benchmark is not None:
            benchmark_returns = benchmark.pct_change().dropna()
        
        return plotter.plot_performance(
            returns=returns,
            benchmark=benchmark_returns,
            title=title,
            show=False
        )
    else:
        # Fallback implementation
        fig = go.Figure()
        
        # Convert to Series if needed
        if isinstance(equity_curve, np.ndarray):
            equity_curve = pd.Series(equity_curve)
        if benchmark is not None and isinstance(benchmark, np.ndarray):
            benchmark = pd.Series(benchmark)
        
        # Create index if needed
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            equity_curve.index = pd.date_range(start='2020-01-01', periods=len(equity_curve), freq='D')
        
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                mode='lines',
                name='Strategy',
                line=dict(color='blue', width=2)
            )
        )
        
        if benchmark is not None:
            if not isinstance(benchmark.index, pd.DatetimeIndex):
                benchmark.index = equity_curve.index[:len(benchmark)]
            fig.add_trace(
                go.Scatter(
                    x=benchmark.index,
                    y=benchmark.values,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='gray', width=1, dash='dash')
                )
            )
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode='x unified',
            height=400
        )
        
        return fig


def create_correlation_heatmap(
    data: pd.DataFrame,
    title: str = "Correlation Heatmap",
    **kwargs
) -> go.Figure:
    """
    Create a correlation heatmap.
    
    Args:
        data: DataFrame with numeric columns
        title: Chart title
        **kwargs: Additional arguments
    
    Returns:
        Plotly figure
    """
    import plotly.express as px
    
    # Calculate correlation
    corr = data.corr()
    
    fig = px.imshow(
        corr,
        labels=dict(color="Correlation"),
        x=corr.columns,
        y=corr.columns,
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title=title
    )
    
    fig.update_layout(height=600)
    
    return fig


def create_returns_distribution(
    returns: Union[pd.Series, np.ndarray],
    title: str = "Returns Distribution",
    **kwargs
) -> go.Figure:
    """
    Create a returns distribution histogram.
    
    Args:
        returns: Returns data
        title: Chart title
        **kwargs: Additional arguments
    
    Returns:
        Plotly figure
    """
    import plotly.express as px
    
    # Convert to Series if needed
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    
    # Remove NaN
    returns = returns.dropna()
    
    fig = px.histogram(
        x=returns.values,
        nbins=50,
        title=title,
        labels={'x': 'Returns', 'y': 'Frequency'}
    )
    
    # Add normal distribution overlay
    mean = returns.mean()
    std = returns.std()
    x_norm = np.linspace(returns.min(), returns.max(), 100)
    y_norm = np.exp(-0.5 * ((x_norm - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    y_norm = y_norm * len(returns) * (returns.max() - returns.min()) / 50
    
    fig.add_trace(
        go.Scatter(
            x=x_norm,
            y=y_norm,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='red', dash='dash')
        )
    )
    
    fig.update_layout(height=400)
    
    return fig


def create_drawdown_chart(
    equity_curve: Union[pd.Series, np.ndarray],
    title: str = "Drawdown Analysis",
    **kwargs
) -> go.Figure:
    """
    Create a drawdown chart.
    
    Args:
        equity_curve: Equity curve values
        title: Chart title
        **kwargs: Additional arguments
    
    Returns:
        Plotly figure
    """
    # Convert to Series if needed
    if isinstance(equity_curve, np.ndarray):
        equity_curve = pd.Series(equity_curve)
    
    # Create index if needed
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        equity_curve.index = pd.date_range(start='2020-01-01', periods=len(equity_curve), freq='D')
    
    # Calculate drawdown
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max * 100
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.3)',
            line=dict(color='red', width=2)
        )
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        height=400
    )
    
    return fig


def create_rolling_stats_chart(
    returns: Union[pd.Series, np.ndarray],
    window: int = 30,
    title: str = "Rolling Statistics",
    **kwargs
) -> go.Figure:
    """
    Create a rolling statistics chart.
    
    Args:
        returns: Returns data
        window: Rolling window size
        title: Chart title
        **kwargs: Additional arguments
    
    Returns:
        Plotly figure
    """
    if PLOTTING_AVAILABLE and PerformancePlotter:
        plotter = PerformancePlotter(backend='plotly')
        
        # Convert to Series if needed
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        return plotter.plot_rolling_performance(
            returns=returns,
            window=window,
            title=title,
            show=False
        )
    else:
        # Fallback implementation
        # Convert to Series if needed
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        # Calculate rolling metrics
        rolling_sharpe = (returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252))
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        rolling_return = returns.rolling(window).mean() * 252
        
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=("Rolling Sharpe Ratio", "Rolling Volatility", "Rolling Return")
        )
        
        fig.add_trace(
            go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, name="Sharpe Ratio"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=rolling_vol.index, y=rolling_vol.values, name="Volatility"),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=rolling_return.index, y=rolling_return.values, name="Return"),
            row=3, col=1
        )
        
        fig.update_layout(title=title, height=800, showlegend=False)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        return fig


# Re-export for easy use
__all__ = [
    'create_candlestick_chart',
    'create_forecast_chart',
    'create_performance_chart',
    'create_correlation_heatmap',
    'create_returns_distribution',
    'create_drawdown_chart',
    'create_rolling_stats_chart'
]

