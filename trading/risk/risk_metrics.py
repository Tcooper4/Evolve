"""Core risk metrics module with reusable functions and visualization support."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime

@dataclass
class RiskMetrics:
    """Container for risk metrics."""
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    volatility: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    tail_risk: float
    skewness: float
    kurtosis: float
    timestamp: datetime

def calculate_rolling_metrics(
    returns: pd.Series,
    window: int = 252,
    risk_free_rate: float = 0.02
) -> pd.DataFrame:
    """Calculate rolling risk metrics.
    
    Args:
        returns: Daily returns series
        window: Rolling window size in days
        risk_free_rate: Annual risk-free rate
        
    Returns:
        DataFrame with rolling metrics
    """
    metrics = pd.DataFrame(index=returns.index)
    
    # Rolling volatility
    metrics['volatility'] = returns.rolling(window).std() * np.sqrt(252)
    
    # Rolling Sharpe
    excess_returns = returns - risk_free_rate/252
    metrics['sharpe_ratio'] = (
        excess_returns.rolling(window).mean() / 
        returns.rolling(window).std()
    ) * np.sqrt(252)
    
    # Rolling Sortino
    downside_returns = returns[returns < 0]
    metrics['sortino_ratio'] = (
        excess_returns.rolling(window).mean() / 
        downside_returns.rolling(window).std()
    ) * np.sqrt(252)
    
    # Rolling Calmar
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.rolling(window).max()
    drawdown = (cum_returns - rolling_max) / rolling_max
    metrics['calmar_ratio'] = (
        excess_returns.rolling(window).mean() / 
        drawdown.rolling(window).min().abs()
    ) * np.sqrt(252)
    
    # Rolling max drawdown
    metrics['max_drawdown'] = drawdown.rolling(window).min()
    
    return metrics

def calculate_advanced_metrics(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """Calculate advanced risk metrics.
    
    Args:
        returns: Daily returns series
        confidence_level: VaR confidence level
        
    Returns:
        Dictionary of advanced metrics
    """
    # Value at Risk
    var = np.percentile(returns, (1 - confidence_level) * 100)
    
    # Conditional VaR
    cvar = returns[returns <= var].mean()
    
    # Tail Risk (probability of extreme returns)
    tail_threshold = 2 * returns.std()
    tail_risk = (returns.abs() > tail_threshold).mean()
    
    # Skewness and Kurtosis
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    
    return {
        'var_95': var,
        'cvar_95': cvar,
        'tail_risk': tail_risk,
        'skewness': skewness,
        'kurtosis': kurtosis
    }

def plot_risk_metrics(
    metrics: pd.DataFrame,
    title: str = "Risk Metrics Dashboard",
    height: int = 800
) -> go.Figure:
    """Create interactive risk metrics dashboard.
    
    Args:
        metrics: DataFrame with risk metrics
        title: Plot title
        height: Plot height
        
    Returns:
        Plotly figure object
    """
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Sharpe Ratio", "Volatility",
            "Sortino Ratio", "Max Drawdown",
            "Calmar Ratio", "Tail Risk"
        ),
        vertical_spacing=0.1
    )
    
    # Sharpe Ratio
    if 'sharpe_ratio' in metrics.columns:
        fig.add_trace(
            go.Scatter(
                x=metrics.index,
                y=metrics['sharpe_ratio'],
                name="Sharpe Ratio"
            ),
            row=1, col=1
        )
    
    # Volatility
    if 'volatility' in metrics.columns:
        fig.add_trace(
            go.Scatter(
                x=metrics.index,
                y=metrics['volatility'],
                name="Volatility"
            ),
            row=1, col=2
        )
    
    # Sortino Ratio
    if 'sortino_ratio' in metrics.columns:
        fig.add_trace(
            go.Scatter(
                x=metrics.index,
                y=metrics['sortino_ratio'],
                name="Sortino Ratio"
            ),
            row=2, col=1
        )
    
    # Max Drawdown
    if 'max_drawdown' in metrics.columns:
        fig.add_trace(
            go.Scatter(
                x=metrics.index,
                y=metrics['max_drawdown'],
                name="Max Drawdown",
                fill='tozeroy'
            ),
            row=2, col=2
        )
    
    # Calmar Ratio
    if 'calmar_ratio' in metrics.columns:
        fig.add_trace(
            go.Scatter(
                x=metrics.index,
                y=metrics['calmar_ratio'],
                name="Calmar Ratio"
            ),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=height,
        showlegend=True,
        template="plotly_white"
    )
    
    return fig

def plot_drawdown_heatmap(
    returns: pd.DataFrame,
    title: str = "Drawdown Heatmap",
    height: int = 600
) -> go.Figure:
    """Create drawdown heatmap for multiple assets.
    
    Args:
        returns: DataFrame of returns (columns = assets)
        title: Plot title
        height: Plot height
        
    Returns:
        Plotly figure object
    """
    # Calculate drawdowns
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=drawdowns.values.T,
        x=drawdowns.index,
        y=drawdowns.columns,
        colorscale='RdBu',
        zmid=0
    ))
    
    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Date",
        yaxis_title="Asset",
        template="plotly_white"
    )
    
    return fig

def calculate_regime_metrics(
    returns: pd.Series,
    window: int = 252
) -> Dict[str, float]:
    """Calculate regime-specific metrics.
    
    Args:
        returns: Daily returns series
        window: Rolling window size
        
    Returns:
        Dictionary of regime metrics
    """
    # Calculate rolling metrics
    rolling_metrics = calculate_rolling_metrics(returns, window)
    
    # Determine regime
    recent_sharpe = rolling_metrics['sharpe_ratio'].iloc[-1]
    recent_vol = rolling_metrics['volatility'].iloc[-1]
    recent_dd = rolling_metrics['max_drawdown'].iloc[-1]
    
    # Regime classification
    if recent_sharpe > 1.0 and recent_vol < 0.2:
        regime = "bull"
    elif recent_sharpe < 0 and recent_vol > 0.3:
        regime = "bear"
    else:
        regime = "neutral"
    
    return {
        'regime': regime,
        'sharpe_ratio': recent_sharpe,
        'volatility': recent_vol,
        'max_drawdown': recent_dd
    } 