"""
Strategy utility functions for testing and analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    """Calculate returns from price series.
    
    Args:
        prices: Price series
        method: Return calculation method ('simple' or 'log')
        
    Returns:
        Returns series
    """
    if method == 'log':
        return np.log(prices / prices.shift(1))
    else:
        return (prices - prices.shift(1)) / prices.shift(1)

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio.
    
    Args:
        returns: Returns series
        risk_free_rate: Risk-free rate
        
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0.0

def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown.
    
    Args:
        returns: Returns series
        
    Returns:
        Maximum drawdown
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def calculate_win_rate(returns: pd.Series) -> float:
    """Calculate win rate.
    
    Args:
        returns: Returns series
        
    Returns:
        Win rate (0-1)
    """
    return (returns > 0).mean()

def calculate_profit_factor(returns: pd.Series) -> float:
    """Calculate profit factor.
    
    Args:
        returns: Returns series
        
    Returns:
        Profit factor
    """
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    return gains / losses if losses > 0 else float('inf')

def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
    """Calculate volatility.
    
    Args:
        returns: Returns series
        annualize: Whether to annualize the volatility
        
    Returns:
        Volatility
    """
    vol = returns.std()
    if annualize:
        vol *= np.sqrt(252)  # Assuming daily data
    return vol

def calculate_beta(returns: pd.Series, market_returns: pd.Series) -> float:
    """Calculate beta relative to market.
    
    Args:
        returns: Strategy returns
        market_returns: Market returns
        
    Returns:
        Beta
    """
    covariance = returns.cov(market_returns)
    market_variance = market_returns.var()
    return covariance / market_variance if market_variance > 0 else 0.0

def calculate_alpha(returns: pd.Series, market_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate alpha.
    
    Args:
        returns: Strategy returns
        market_returns: Market returns
        risk_free_rate: Risk-free rate
        
    Returns:
        Alpha
    """
    beta = calculate_beta(returns, market_returns)
    return returns.mean() - (risk_free_rate + beta * (market_returns.mean() - risk_free_rate))

def calculate_information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate information ratio.
    
    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns
        
    Returns:
        Information ratio
    """
    excess_returns = returns - benchmark_returns
    return excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0.0

def calculate_calmar_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate Calmar ratio.
    
    Args:
        returns: Returns series
        risk_free_rate: Risk-free rate
        
    Returns:
        Calmar ratio
    """
    max_dd = calculate_max_drawdown(returns)
    excess_return = returns.mean() - risk_free_rate
    return excess_return / abs(max_dd) if max_dd < 0 else 0.0

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate Sortino ratio.
    
    Args:
        returns: Returns series
        risk_free_rate: Risk-free rate
        
    Returns:
        Sortino ratio
    """
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    downside_deviation = downside_returns.std()
    return excess_returns.mean() / downside_deviation if downside_deviation > 0 else 0.0

def calculate_ulcer_index(returns: pd.Series) -> float:
    """Calculate Ulcer Index.
    
    Args:
        returns: Returns series
        
    Returns:
        Ulcer Index
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return np.sqrt((drawdown ** 2).mean())

def calculate_gain_to_pain_ratio(returns: pd.Series) -> float:
    """Calculate gain-to-pain ratio.
    
    Args:
        returns: Returns series
        
    Returns:
        Gain-to-pain ratio
    """
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    return gains / losses if losses > 0 else float('inf')

def calculate_recovery_factor(returns: pd.Series) -> float:
    """Calculate recovery factor.
    
    Args:
        returns: Returns series
        
    Returns:
        Recovery factor
    """
    total_return = (1 + returns).prod() - 1
    max_dd = calculate_max_drawdown(returns)
    return total_return / abs(max_dd) if max_dd < 0 else 0.0

def calculate_risk_metrics(returns: pd.Series, market_returns: Optional[pd.Series] = None) -> Dict[str, float]:
    """Calculate comprehensive risk metrics.
    
    Args:
        returns: Returns series
        market_returns: Optional market returns for relative metrics
        
    Returns:
        Dictionary of risk metrics
    """
    metrics = {
        'total_return': (1 + returns).prod() - 1,
        'annualized_return': returns.mean() * 252,
        'volatility': calculate_volatility(returns),
        'sharpe_ratio': calculate_sharpe_ratio(returns),
        'sortino_ratio': calculate_sortino_ratio(returns),
        'max_drawdown': calculate_max_drawdown(returns),
        'win_rate': calculate_win_rate(returns),
        'profit_factor': calculate_profit_factor(returns),
        'calmar_ratio': calculate_calmar_ratio(returns),
        'ulcer_index': calculate_ulcer_index(returns),
        'gain_to_pain_ratio': calculate_gain_to_pain_ratio(returns),
        'recovery_factor': calculate_recovery_factor(returns)
    }
    
    if market_returns is not None:
        metrics.update({
            'beta': calculate_beta(returns, market_returns),
            'alpha': calculate_alpha(returns, market_returns),
            'information_ratio': calculate_information_ratio(returns, market_returns)
        })
    
    return metrics 