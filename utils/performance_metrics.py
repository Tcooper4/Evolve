"""
Custom Performance Metrics Module

This module provides clean, custom implementations of financial performance metrics
to replace the empyrical dependency. All functions are designed to be compatible
with pandas Series and numpy arrays.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple


def sharpe_ratio(returns: Union[pd.Series, np.ndarray], 
                 risk_free: float = 0.0, 
                 period: int = 252) -> float:
    """
    Calculate the Sharpe ratio.
    
    Args:
        returns: Return series
        risk_free: Risk-free rate (annualized)
        period: Number of periods per year (default: 252 for daily data)
    
    Returns:
        Sharpe ratio
    """
    returns = np.array(returns)
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - (risk_free / period)
    if np.std(excess_returns) == 0:
        return 0.0
    
    return (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(period)


def sortino_ratio(returns: Union[pd.Series, np.ndarray], 
                  risk_free: float = 0.0, 
                  period: int = 252) -> float:
    """
    Calculate the Sortino ratio.
    
    Args:
        returns: Return series
        risk_free: Risk-free rate (annualized)
        period: Number of periods per year (default: 252 for daily data)
    
    Returns:
        Sortino ratio
    """
    returns = np.array(returns)
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - (risk_free / period)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0
    
    return (np.mean(excess_returns) / np.std(downside_returns)) * np.sqrt(period)


def max_drawdown(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate the maximum drawdown.
    
    Args:
        returns: Return series
    
    Returns:
        Maximum drawdown as a negative value
    """
    returns = np.array(returns)
    if len(returns) == 0:
        return 0.0
    
    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - peak) / peak
    return drawdowns.min()


def cumulative_return(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate the cumulative return.
    
    Args:
        returns: Return series
    
    Returns:
        Cumulative return
    """
    returns = np.array(returns)
    if len(returns) == 0:
        return 0.0
    
    return np.prod(1 + returns) - 1


def calmar_ratio(returns: Union[pd.Series, np.ndarray], 
                 risk_free: float = 0.0, 
                 period: int = 252) -> float:
    """
    Calculate the Calmar ratio.
    
    Args:
        returns: Return series
        risk_free: Risk-free rate (annualized)
        period: Number of periods per year (default: 252 for daily data)
    
    Returns:
        Calmar ratio
    """
    returns = np.array(returns)
    if len(returns) == 0:
        return 0.0
    
    max_dd = max_drawdown(returns)
    if max_dd == 0:
        return 0.0
    
    annual_return = (1 + np.mean(returns)) ** period - 1
    return annual_return / abs(max_dd)


def avg_drawdown(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate the average drawdown.
    
    Args:
        returns: Return series
    
    Returns:
        Average drawdown
    """
    returns = np.array(returns)
    if len(returns) == 0:
        return 0.0
    
    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - peak) / peak
    
    # Only consider negative drawdowns
    negative_drawdowns = drawdowns[drawdowns < 0]
    
    if len(negative_drawdowns) == 0:
        return 0.0
    
    return np.mean(negative_drawdowns)


def drawdown_details(returns: Union[pd.Series, np.ndarray]) -> pd.DataFrame:
    """
    Calculate detailed drawdown information.
    
    Args:
        returns: Return series
    
    Returns:
        DataFrame with drawdown details including start, end, duration, and depth
    """
    returns = np.array(returns)
    if len(returns) == 0:
        return pd.DataFrame()
    
    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - peak) / peak
    
    # Find drawdown periods
    in_drawdown = drawdowns < 0
    drawdown_starts = []
    drawdown_ends = []
    drawdown_depths = []
    
    i = 0
    while i < len(drawdowns):
        if in_drawdown[i]:
            start = i
            depth = drawdowns[i]
            
            # Find the end of this drawdown
            while i < len(drawdowns) and in_drawdown[i]:
                depth = min(depth, drawdowns[i])
                i += 1
            
            drawdown_starts.append(start)
            drawdown_ends.append(i - 1)
            drawdown_depths.append(depth)
        else:
            i += 1
    
    if not drawdown_starts:
        return pd.DataFrame()
    
    df = pd.DataFrame({
        'start': drawdown_starts,
        'end': drawdown_ends,
        'depth': drawdown_depths,
        'days': [end - start + 1 for start, end in zip(drawdown_starts, drawdown_ends)]
    })
    
    return df


def omega_ratio(returns: Union[pd.Series, np.ndarray], 
                threshold: float = 0.0) -> float:
    """
    Calculate the Omega ratio.
    
    Args:
        returns: Return series
        threshold: Return threshold (default: 0.0)
    
    Returns:
        Omega ratio
    """
    returns = np.array(returns)
    if len(returns) == 0:
        return 0.0
    
    gains = returns[returns > threshold]
    losses = returns[returns <= threshold]
    
    if len(losses) == 0:
        return np.inf if len(gains) > 0 else 0.0
    
    expected_gain = np.mean(gains) if len(gains) > 0 else 0.0
    expected_loss = abs(np.mean(losses))
    
    return expected_gain / expected_loss if expected_loss != 0 else 0.0


def information_ratio(returns: Union[pd.Series, np.ndarray], 
                     benchmark_returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate the Information ratio.
    
    Args:
        returns: Portfolio return series
        benchmark_returns: Benchmark return series
    
    Returns:
        Information ratio
    """
    returns = np.array(returns)
    benchmark_returns = np.array(benchmark_returns)
    
    if len(returns) != len(benchmark_returns) or len(returns) == 0:
        return 0.0
    
    active_returns = returns - benchmark_returns
    
    if np.std(active_returns) == 0:
        return 0.0
    
    return np.mean(active_returns) / np.std(active_returns)


def treynor_ratio(returns: Union[pd.Series, np.ndarray], 
                  benchmark_returns: Union[pd.Series, np.ndarray], 
                  risk_free: float = 0.0) -> float:
    """
    Calculate the Treynor ratio.
    
    Args:
        returns: Portfolio return series
        benchmark_returns: Benchmark return series
        risk_free: Risk-free rate
    
    Returns:
        Treynor ratio
    """
    returns = np.array(returns)
    benchmark_returns = np.array(benchmark_returns)
    
    if len(returns) != len(benchmark_returns) or len(returns) == 0:
        return 0.0
    
    # Calculate beta
    covariance = np.cov(returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns)
    
    if benchmark_variance == 0:
        return 0.0
    
    beta = covariance / benchmark_variance
    
    if beta == 0:
        return 0.0
    
    excess_return = np.mean(returns) - risk_free
    return excess_return / beta


def jensen_alpha(returns: Union[pd.Series, np.ndarray], 
                 benchmark_returns: Union[pd.Series, np.ndarray], 
                 risk_free: float = 0.0) -> float:
    """
    Calculate Jensen's Alpha.
    
    Args:
        returns: Portfolio return series
        benchmark_returns: Benchmark return series
        risk_free: Risk-free rate
    
    Returns:
        Jensen's Alpha
    """
    returns = np.array(returns)
    benchmark_returns = np.array(benchmark_returns)
    
    if len(returns) != len(benchmark_returns) or len(returns) == 0:
        return 0.0
    
    # Calculate beta
    covariance = np.cov(returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns)
    
    if benchmark_variance == 0:
        return 0.0
    
    beta = covariance / benchmark_variance
    
    portfolio_return = np.mean(returns)
    benchmark_return = np.mean(benchmark_returns)
    
    return portfolio_return - (risk_free + beta * (benchmark_return - risk_free))


def value_at_risk(returns: Union[pd.Series, np.ndarray], 
                  confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR).
    
    Args:
        returns: Return series
        confidence_level: Confidence level (default: 0.95 for 95% VaR)
    
    Returns:
        Value at Risk
    """
    returns = np.array(returns)
    if len(returns) == 0:
        return 0.0
    
    percentile = (1 - confidence_level) * 100
    return np.percentile(returns, percentile)


def conditional_value_at_risk(returns: Union[pd.Series, np.ndarray], 
                             confidence_level: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
    
    Args:
        returns: Return series
        confidence_level: Confidence level (default: 0.95 for 95% CVaR)
    
    Returns:
        Conditional Value at Risk
    """
    returns = np.array(returns)
    if len(returns) == 0:
        return 0.0
    
    var = value_at_risk(returns, confidence_level)
    tail_returns = returns[returns <= var]
    
    if len(tail_returns) == 0:
        return var
    
    return np.mean(tail_returns)


def downside_deviation(returns: Union[pd.Series, np.ndarray], 
                      threshold: float = 0.0) -> float:
    """
    Calculate downside deviation.
    
    Args:
        returns: Return series
        threshold: Return threshold (default: 0.0)
    
    Returns:
        Downside deviation
    """
    returns = np.array(returns)
    if len(returns) == 0:
        return 0.0
    
    downside_returns = returns[returns < threshold]
    
    if len(downside_returns) == 0:
        return 0.0
    
    return np.std(downside_returns)


def gain_loss_ratio(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate gain/loss ratio.
    
    Args:
        returns: Return series
    
    Returns:
        Gain/loss ratio
    """
    returns = np.array(returns)
    if len(returns) == 0:
        return 0.0
    
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    
    if len(losses) == 0:
        return np.inf if len(gains) > 0 else 0.0
    
    avg_gain = np.mean(gains) if len(gains) > 0 else 0.0
    avg_loss = abs(np.mean(losses))
    
    return avg_gain / avg_loss if avg_loss != 0 else 0.0


def profit_factor(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate profit factor.
    
    Args:
        returns: Return series
    
    Returns:
        Profit factor
    """
    returns = np.array(returns)
    if len(returns) == 0:
        return 0.0
    
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = abs(np.sum(returns[returns < 0]))
    
    return gross_profit / gross_loss if gross_loss != 0 else 0.0


def recovery_factor(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate recovery factor.
    
    Args:
        returns: Return series
    
    Returns:
        Recovery factor
    """
    returns = np.array(returns)
    if len(returns) == 0:
        return 0.0
    
    total_return = cumulative_return(returns)
    max_dd = abs(max_drawdown(returns))
    
    return total_return / max_dd if max_dd != 0 else 0.0


def risk_reward_ratio(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate risk/reward ratio.
    
    Args:
        returns: Return series
    
    Returns:
        Risk/reward ratio
    """
    returns = np.array(returns)
    if len(returns) == 0:
        return 0.0
    
    expected_return = np.mean(returns)
    risk = np.std(returns)
    
    return expected_return / risk if risk != 0 else 0.0 