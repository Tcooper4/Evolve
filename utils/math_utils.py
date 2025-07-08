"""
Math Utilities for Financial Calculations

Provides mathematical and statistical functions commonly used in
financial analysis and trading system performance evaluation.
"""

from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def calculate_sharpe_ratio(
    returns: Union[pd.Series, np.ndarray, List[float]], 
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate the Sharpe ratio for a series of returns.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default: 2%)
        periods_per_year: Number of periods per year (default: 252 for daily)
        
    Returns:
        float: Sharpe ratio
    """
    try:
        if isinstance(returns, list):
            returns = pd.Series(returns)
        elif isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        if len(returns) == 0:
            logger.warning("Empty returns series provided")
            return 0.0
        
        # Remove NaN values
        returns = returns.dropna()
        
        if len(returns) == 0:
            logger.warning("No valid returns after removing NaN values")
            return 0.0
        
        # Calculate excess returns
        excess_returns = returns - (risk_free_rate / periods_per_year)
        
        # Calculate Sharpe ratio
        if excess_returns.std() == 0:
            logger.warning("Zero standard deviation in returns")
            return 0.0
        
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)
        
        logger.debug(f"Calculated Sharpe ratio: {sharpe_ratio:.4f}")
        return sharpe_ratio
        
    except Exception as e:
        logger.error(f"Error calculating Sharpe ratio: {e}")
        return 0.0

def calculate_max_drawdown(
    equity_curve: Union[pd.Series, np.ndarray, List[float]]
) -> Dict[str, float]:
    """
    Calculate maximum drawdown and related metrics.
    
    Args:
        equity_curve: Series of portfolio values or cumulative returns
        
    Returns:
        Dict[str, float]: Dictionary containing max drawdown and related metrics
    """
    try:
        if isinstance(equity_curve, list):
            equity_curve = pd.Series(equity_curve)
        elif isinstance(equity_curve, np.ndarray):
            equity_curve = pd.Series(equity_curve)
        
        if len(equity_curve) == 0:
            logger.warning("Empty equity curve provided")
            return {'max_drawdown': 0.0, 'drawdown_duration': 0, 'peak_value': 0.0}
        
        # Remove NaN values
        equity_curve = equity_curve.dropna()
        
        if len(equity_curve) == 0:
            logger.warning("No valid data after removing NaN values")
            return {'max_drawdown': 0.0, 'drawdown_duration': 0, 'peak_value': 0.0}
        
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        # Find maximum drawdown
        max_drawdown = drawdown.min()
        
        # Find peak value
        peak_value = running_max.max()
        
        # Calculate drawdown duration
        peak_idx = equity_curve.idxmax()
        bottom_idx = drawdown.idxmin()
        
        if peak_idx <= bottom_idx:
            drawdown_duration = (bottom_idx - peak_idx).days if hasattr(bottom_idx - peak_idx, 'days') else 0
        else:
            drawdown_duration = 0
        
        result = {
            'max_drawdown': max_drawdown,
            'drawdown_duration': drawdown_duration,
            'peak_value': peak_value,
            'current_value': equity_curve.iloc[-1]
        }
        
        logger.debug(f"Calculated max drawdown: {max_drawdown:.4f}")
        return result
        
    except Exception as e:
        logger.error(f"Error calculating max drawdown: {e}")
        return {'max_drawdown': 0.0, 'drawdown_duration': 0, 'peak_value': 0.0}

def calculate_win_rate(
    returns: Union[pd.Series, np.ndarray, List[float]], 
    threshold: float = 0.0
) -> Dict[str, float]:
    """
    Calculate win rate and related trading statistics.
    
    Args:
        returns: Series of returns
        threshold: Minimum return to consider a "win" (default: 0.0)
        
    Returns:
        Dict[str, float]: Dictionary containing win rate and related metrics
    """
    try:
        if isinstance(returns, list):
            returns = pd.Series(returns)
        elif isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        if len(returns) == 0:
            logger.warning("Empty returns series provided")
            return {'win_rate': 0.0, 'total_trades': 0, 'wins': 0, 'losses': 0}
        
        # Remove NaN values
        returns = returns.dropna()
        
        if len(returns) == 0:
            logger.warning("No valid returns after removing NaN values")
            return {'win_rate': 0.0, 'total_trades': 0, 'wins': 0, 'losses': 0}
        
        # Calculate wins and losses
        wins = (returns > threshold).sum()
        losses = (returns <= threshold).sum()
        total_trades = len(returns)
        
        # Calculate win rate
        win_rate = wins / total_trades if total_trades > 0 else 0.0
        
        # Calculate average win and loss
        winning_returns = returns[returns > threshold]
        losing_returns = returns[returns <= threshold]
        
        avg_win = winning_returns.mean() if len(winning_returns) > 0 else 0.0
        avg_loss = losing_returns.mean() if len(losing_returns) > 0 else 0.0
        
        result = {
            'win_rate': win_rate,
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        }
        
        logger.debug(f"Calculated win rate: {win_rate:.4f}")
        return result
        
    except Exception as e:
        logger.error(f"Error calculating win rate: {e}")
        return {'win_rate': 0.0, 'total_trades': 0, 'wins': 0, 'losses': 0}

def calculate_profit_factor(
    returns: Union[pd.Series, np.ndarray, List[float]]
) -> float:
    """
    Calculate the profit factor (gross profit / gross loss).
    
    Args:
        returns: Series of returns
        
    Returns:
        float: Profit factor
    """
    try:
        if isinstance(returns, list):
            returns = pd.Series(returns)
        elif isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        if len(returns) == 0:
            logger.warning("Empty returns series provided")
            return 0.0
        
        # Remove NaN values
        returns = returns.dropna()
        
        if len(returns) == 0:
            logger.warning("No valid returns after removing NaN values")
            return 0.0
        
        # Calculate gross profit and loss
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        
        # Calculate profit factor
        if gross_loss == 0:
            profit_factor = float('inf') if gross_profit > 0 else 0.0
        else:
            profit_factor = gross_profit / gross_loss
        
        logger.debug(f"Calculated profit factor: {profit_factor:.4f}")
        return profit_factor
        
    except Exception as e:
        logger.error(f"Error calculating profit factor: {e}")
        return 0.0

def calculate_calmar_ratio(
    returns: Union[pd.Series, np.ndarray, List[float]], 
    periods_per_year: int = 252
) -> float:
    """
    Calculate the Calmar ratio (annual return / max drawdown).
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods per year (default: 252 for daily)
        
    Returns:
        float: Calmar ratio
    """
    try:
        if isinstance(returns, list):
            returns = pd.Series(returns)
        elif isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        if len(returns) == 0:
            logger.warning("Empty returns series provided")
            return 0.0
        
        # Remove NaN values
        returns = returns.dropna()
        
        if len(returns) == 0:
            logger.warning("No valid returns after removing NaN values")
            return 0.0
        
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Calculate annual return
        total_return = cumulative_returns.iloc[-1] - 1
        years = len(returns) / periods_per_year
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
        
        # Calculate max drawdown
        max_dd_result = calculate_max_drawdown(cumulative_returns)
        max_drawdown = abs(max_dd_result['max_drawdown'])
        
        # Calculate Calmar ratio
        if max_drawdown == 0:
            calmar_ratio = float('inf') if annual_return > 0 else 0.0
        else:
            calmar_ratio = annual_return / max_drawdown
        
        logger.debug(f"Calculated Calmar ratio: {calmar_ratio:.4f}")
        return calmar_ratio
        
    except Exception as e:
        logger.error(f"Error calculating Calmar ratio: {e}")
        return 0.0

def calculate_volatility(
    returns: Union[pd.Series, np.ndarray, List[float]], 
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized volatility.
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods per year (default: 252 for daily)
        
    Returns:
        float: Annualized volatility
    """
    try:
        if isinstance(returns, list):
            returns = pd.Series(returns)
        elif isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        if len(returns) == 0:
            logger.warning("Empty returns series provided")
            return 0.0
        
        # Remove NaN values
        returns = returns.dropna()
        
        if len(returns) == 0:
            logger.warning("No valid returns after removing NaN values")
            return 0.0
        
        # Calculate annualized volatility
        volatility = returns.std() * np.sqrt(periods_per_year)
        
        logger.debug(f"Calculated volatility: {volatility:.4f}")
        return volatility
        
    except Exception as e:
        logger.error(f"Error calculating volatility: {e}")
        return 0.0

def calculate_beta(
    returns: Union[pd.Series, np.ndarray, List[float]], 
    market_returns: Union[pd.Series, np.ndarray, List[float]]
) -> float:
    """
    Calculate beta relative to market returns.
    
    Args:
        returns: Series of portfolio returns
        market_returns: Series of market returns
        
    Returns:
        float: Beta value
    """
    try:
        if isinstance(returns, list):
            returns = pd.Series(returns)
        elif isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        if isinstance(market_returns, list):
            market_returns = pd.Series(market_returns)
        elif isinstance(market_returns, np.ndarray):
            market_returns = pd.Series(market_returns)
        
        if len(returns) == 0 or len(market_returns) == 0:
            logger.warning("Empty returns series provided")
            return 0.0
        
        # Align series
        aligned_data = pd.concat([returns, market_returns], axis=1).dropna()
        
        if len(aligned_data) == 0:
            logger.warning("No aligned data after removing NaN values")
            return 0.0
        
        portfolio_returns = aligned_data.iloc[:, 0]
        market_returns_aligned = aligned_data.iloc[:, 1]
        
        # Calculate covariance and variance
        covariance = np.cov(portfolio_returns, market_returns_aligned)[0, 1]
        market_variance = np.var(market_returns_aligned)
        
        # Calculate beta
        if market_variance == 0:
            logger.warning("Zero market variance")
            return 0.0
        
        beta = covariance / market_variance
        
        logger.debug(f"Calculated beta: {beta:.4f}")
        return beta
        
    except Exception as e:
        logger.error(f"Error calculating beta: {e}")
        return 0.0

def calculate_alpha(
    returns: Union[pd.Series, np.ndarray, List[float]], 
    market_returns: Union[pd.Series, np.ndarray, List[float]], 
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate alpha (excess return relative to market).
    
    Args:
        returns: Series of portfolio returns
        market_returns: Series of market returns
        risk_free_rate: Annual risk-free rate (default: 2%)
        periods_per_year: Number of periods per year (default: 252 for daily)
        
    Returns:
        float: Alpha value
    """
    try:
        if isinstance(returns, list):
            returns = pd.Series(returns)
        elif isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        if isinstance(market_returns, list):
            market_returns = pd.Series(market_returns)
        elif isinstance(market_returns, np.ndarray):
            market_returns = pd.Series(market_returns)
        
        if len(returns) == 0 or len(market_returns) == 0:
            logger.warning("Empty returns series provided")
            return 0.0
        
        # Align series
        aligned_data = pd.concat([returns, market_returns], axis=1).dropna()
        
        if len(aligned_data) == 0:
            logger.warning("No aligned data after removing NaN values")
            return 0.0
        
        portfolio_returns = aligned_data.iloc[:, 0]
        market_returns_aligned = aligned_data.iloc[:, 1]
        
        # Calculate beta
        beta = calculate_beta(portfolio_returns, market_returns_aligned)
        
        # Calculate alpha
        portfolio_excess_return = portfolio_returns.mean() - (risk_free_rate / periods_per_year)
        market_excess_return = market_returns_aligned.mean() - (risk_free_rate / periods_per_year)
        
        alpha = portfolio_excess_return - (beta * market_excess_return)
        
        # Annualize alpha
        alpha_annualized = alpha * periods_per_year
        
        logger.debug(f"Calculated alpha: {alpha_annualized:.4f}")
        return alpha_annualized
        
    except Exception as e:
        logger.error(f"Error calculating alpha: {e}")
        return 0.0 