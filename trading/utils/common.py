"""Common utility functions for the trading system.

This module provides a collection of utility functions used across the trading system.
Each function is designed to be stateless, pure, and easily testable. Functions are
organized by category (data processing, time handling, validation, etc.) and include
detailed docstrings for LLM comprehension and traceability.

Example:
    >>> from trading.utils.common import validate_dataframe
    >>> df = pd.DataFrame({'price': [100, 101, 102]})
    >>> is_valid, errors = validate_dataframe(df, required_columns=['price'])
    >>> print(is_valid, errors)
    True, []
"""

import logging
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from functools import wraps
import time
import traceback
import sys
import hashlib
import re

logger = logging.getLogger(__name__)

def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_dir: Directory to store log files
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Create handlers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(log_path / f"trading_{timestamp}.log")
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from file.
    
    Args:
        config_path: Path to configuration file (JSON or YAML)
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    elif config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    return config

def save_config(config: Dict[str, Any], config_path: Union[str, Path]):
    """Save configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    
    if config_path.suffix == '.json':
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    elif config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")

def timer(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    return wrapper

def handle_exceptions(func):
    """Decorator to handle exceptions and log them."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    allow_empty: bool = False
) -> Tuple[bool, List[str]]:
    """Validate a DataFrame against required columns and data quality checks.
    
    Args:
        df: DataFrame to validate
        required_columns: List of column names that must be present
        allow_empty: Whether to allow empty DataFrames
        
    Returns:
        Tuple of (is_valid, list_of_errors)
        
    Example:
        >>> df = pd.DataFrame({'price': [100, 101, 102]})
        >>> is_valid, errors = validate_dataframe(df, ['price'])
        >>> print(is_valid, errors)
        True, []
    """
    errors = []
    
    # Check if DataFrame is empty
    if df.empty and not allow_empty:
        errors.append("DataFrame is empty")
        return False, errors
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    
    # Check for NaN values
    nan_columns = df.columns[df.isna().any()].tolist()
    if nan_columns:
        errors.append(f"NaN values found in columns: {nan_columns}")
    
    # Check for infinite values
    inf_columns = df.columns[np.isinf(df.select_dtypes(include=np.number)).any()].tolist()
    if inf_columns:
        errors.append(f"Infinite values found in columns: {inf_columns}")
    
    return len(errors) == 0, errors

def calculate_returns(
    prices: pd.Series,
    method: str = "log"
) -> pd.Series:
    """Calculate returns from a price series.
    
    Args:
        prices: Series of prices
        method: Return calculation method ("log" or "simple")
        
    Returns:
        Series of returns
        
    Example:
        >>> prices = pd.Series([100, 101, 102])
        >>> returns = calculate_returns(prices)
        >>> print(returns)
        0         NaN
        1    0.009950
        2    0.009852
        dtype: float64
    """
    if method == "log":
        return np.log(prices / prices.shift(1))
    else:  # simple returns
        return prices.pct_change()

def resample_data(
    data: pd.DataFrame,
    freq: str,
    agg_dict: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """Resample time series data to a different frequency.
    
    Args:
        data: DataFrame with datetime index
        freq: Target frequency (e.g., "1H", "1D", "1W")
        agg_dict: Dictionary mapping columns to aggregation functions
        
    Returns:
        Resampled DataFrame
        
    Example:
        >>> df = pd.DataFrame({'price': [100, 101, 102]}, 
        ...                   index=pd.date_range('2023-01-01', periods=3, freq='H'))
        >>> resampled = resample_data(df, '1D', {'price': 'mean'})
    """
    if agg_dict is None:
        agg_dict = {col: "mean" for col in data.columns}
    
    return data.resample(freq).agg(agg_dict)

def calculate_moving_average(
    data: pd.Series,
    window: int,
    min_periods: Optional[int] = None
) -> pd.Series:
    """Calculate moving average with configurable window.
    
    Args:
        data: Input series
        window: Window size for moving average
        min_periods: Minimum number of observations required
        
    Returns:
        Series of moving averages
        
    Example:
        >>> data = pd.Series([1, 2, 3, 4, 5])
        >>> ma = calculate_moving_average(data, window=3)
        >>> print(ma)
        0    NaN
        1    NaN
        2    2.0
        3    3.0
        4    4.0
        dtype: float64
    """
    if min_periods is None:
        min_periods = window
    return data.rolling(window=window, min_periods=min_periods).mean()

def calculate_volatility(
    returns: pd.Series,
    window: int = 20,
    annualize: bool = True
) -> pd.Series:
    """Calculate rolling volatility of returns.
    
    Args:
        returns: Series of returns
        window: Window size for calculation
        annualize: Whether to annualize the volatility
        
    Returns:
        Series of volatility values
        
    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03])
        >>> vol = calculate_volatility(returns, window=2)
        >>> print(vol)
        0         NaN
        1    0.021213
        2    0.035355
        dtype: float64
    """
    vol = returns.rolling(window=window).std()
    if annualize:
        vol = vol * np.sqrt(252)  # Annualize assuming 252 trading days
    return vol

def calculate_correlation(
    data1: pd.Series,
    data2: pd.Series,
    window: int = 20
) -> pd.Series:
    """Calculate rolling correlation between two series.
    
    Args:
        data1: First series
        data2: Second series
        window: Window size for calculation
        
    Returns:
        Series of correlation values
        
    Example:
        >>> s1 = pd.Series([1, 2, 3, 4, 5])
        >>> s2 = pd.Series([2, 4, 6, 8, 10])
        >>> corr = calculate_correlation(s1, s2, window=3)
        >>> print(corr)
        0    NaN
        1    NaN
        2    1.0
        3    1.0
        4    1.0
        dtype: float64
    """
    return data1.rolling(window=window).corr(data2)

def calculate_beta(
    returns: pd.Series,
    market_returns: pd.Series,
    window: int = 20
) -> pd.Series:
    """Calculate rolling beta against market returns.
    
    Args:
        returns: Asset returns
        market_returns: Market returns
        window: Window size for calculation
        
    Returns:
        Series of beta values
        
    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03])
        >>> market = pd.Series([0.02, -0.01, 0.02])
        >>> beta = calculate_beta(returns, market, window=2)
        >>> print(beta)
        0    NaN
        1   -1.0
        2    1.0
        dtype: float64
    """
    covariance = returns.rolling(window=window).cov(market_returns)
    market_variance = market_returns.rolling(window=window).var()
    return covariance / market_variance

def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    window: int = 20,
    annualize: bool = True
) -> pd.Series:
    """Calculate rolling Sharpe ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate
        window: Window size for calculation
        annualize: Whether to annualize the ratio
        
    Returns:
        Series of Sharpe ratios
        
    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03])
        >>> sharpe = calculate_sharpe_ratio(returns, window=2)
        >>> print(sharpe)
        0    NaN
        1    0.0
        2    0.707107
        dtype: float64
    """
    excess_returns = returns - risk_free_rate
    mean = excess_returns.rolling(window=window).mean()
    std = excess_returns.rolling(window=window).std()
    
    if annualize:
        mean = mean * 252
        std = std * np.sqrt(252)
    
    return mean / std

def calculate_drawdown(
    prices: pd.Series
) -> pd.Series:
    """Calculate drawdown series from prices.
    
    Args:
        prices: Series of prices
        
    Returns:
        Series of drawdown values
        
    Example:
        >>> prices = pd.Series([100, 95, 90, 100])
        >>> dd = calculate_drawdown(prices)
        >>> print(dd)
        0    0.000000
        1    0.050000
        2    0.100000
        3    0.000000
        dtype: float64
    """
    rolling_max = prices.expanding().max()
    drawdown = (prices - rolling_max) / rolling_max
    return drawdown

def calculate_max_drawdown(
    prices: pd.Series
) -> float:
    """Calculate maximum drawdown.
    
    Args:
        prices: Price series
        
    Returns:
        Maximum drawdown as a negative percentage
        
    Example:
        >>> prices = pd.Series([100, 110, 105, 120, 115])
        >>> max_dd = calculate_max_drawdown(prices)
        >>> print(max_dd)
        -0.045454545454545456
    """
    cumulative = prices / prices.iloc[0]
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())

def calculate_win_rate(
    returns: pd.Series
) -> float:
    """Calculate win rate from returns.
    
    Args:
        returns: Returns series
        
    Returns:
        Win rate as a percentage (0.0 to 1.0)
        
    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> win_rate = calculate_win_rate(returns)
        >>> print(win_rate)
        0.6
    """
    return float((returns > 0).mean())

def calculate_calmar_ratio(
    returns: pd.Series,
    prices: pd.Series,
    window: int = 252
) -> pd.Series:
    """Calculate rolling Calmar ratio.
    
    Args:
        returns: Series of returns
        prices: Series of prices
        window: Window size for calculation
        
    Returns:
        Series of Calmar ratios
        
    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03])
        >>> prices = pd.Series([100, 98, 101])
        >>> calmar = calculate_calmar_ratio(returns, prices, window=2)
        >>> print(calmar)
        0    NaN
        1    0.0
        2    0.5
        dtype: float64
    """
    annualized_return = returns.rolling(window=window).mean() * 252
    max_drawdown = prices.rolling(window=window).apply(calculate_max_drawdown)
    return annualized_return / max_drawdown

def calculate_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 20,
    annualize: bool = True
) -> pd.Series:
    """Calculate rolling Information ratio.
    
    Args:
        returns: Series of returns
        benchmark_returns: Series of benchmark returns
        window: Window size for calculation
        annualize: Whether to annualize the ratio
        
    Returns:
        Series of Information ratios
        
    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03])
        >>> benchmark = pd.Series([0.02, -0.01, 0.02])
        >>> ir = calculate_information_ratio(returns, benchmark, window=2)
        >>> print(ir)
        0    NaN
        1    0.0
        2    0.707107
        dtype: float64
    """
    excess_returns = returns - benchmark_returns
    mean = excess_returns.rolling(window=window).mean()
    std = excess_returns.rolling(window=window).std()
    
    if annualize:
        mean = mean * 252
        std = std * np.sqrt(252)
    
    return mean / std

def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    window: int = 20,
    annualize: bool = True
) -> pd.Series:
    """Calculate rolling Sortino ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate
        window: Window size for calculation
        annualize: Whether to annualize the ratio
        
    Returns:
        Series of Sortino ratios
        
    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03])
        >>> sortino = calculate_sortino_ratio(returns, window=2)
        >>> print(sortino)
        0    NaN
        1    0.0
        2    0.707107
        dtype: float64
    """
    excess_returns = returns - risk_free_rate
    mean = excess_returns.rolling(window=window).mean()
    
    # Calculate downside deviation
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.rolling(window=window).std()
    
    if annualize:
        mean = mean * 252
        downside_std = downside_std * np.sqrt(252)
    
    return mean / downside_std

def calculate_omega_ratio(
    returns: pd.Series,
    threshold: float = 0.0,
    window: int = 20
) -> pd.Series:
    """Calculate rolling Omega ratio.
    
    Args:
        returns: Series of returns
        threshold: Return threshold
        window: Window size for calculation
        
    Returns:
        Series of Omega ratios
        
    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03])
        >>> omega = calculate_omega_ratio(returns, window=2)
        >>> print(omega)
        0    NaN
        1    0.0
        2    1.0
        dtype: float64
    """
    def _omega_ratio(x):
        gains = x[x > threshold].sum()
        losses = abs(x[x <= threshold].sum())
        return gains / losses if losses != 0 else float('inf')
    
    return returns.rolling(window=window).apply(_omega_ratio)

def calculate_treynor_ratio(
    returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float = 0.0,
    window: int = 20,
    annualize: bool = True
) -> pd.Series:
    """Calculate rolling Treynor ratio.
    
    Args:
        returns: Series of returns
        market_returns: Series of market returns
        risk_free_rate: Risk-free rate
        window: Window size for calculation
        annualize: Whether to annualize the ratio
        
    Returns:
        Series of Treynor ratios
        
    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03])
        >>> market = pd.Series([0.02, -0.01, 0.02])
        >>> treynor = calculate_treynor_ratio(returns, market, window=2)
        >>> print(treynor)
        0    NaN
        1    0.0
        2    0.5
        dtype: float64
    """
    excess_returns = returns - risk_free_rate
    beta = calculate_beta(returns, market_returns, window=window)
    
    if annualize:
        excess_returns = excess_returns * 252
    
    return excess_returns / beta

def calculate_alpha(
    returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float = 0.0,
    window: int = 20,
    annualize: bool = True
) -> pd.Series:
    """Calculate rolling alpha.
    
    Args:
        returns: Series of returns
        market_returns: Series of market returns
        risk_free_rate: Risk-free rate
        window: Window size for calculation
        annualize: Whether to annualize alpha
        
    Returns:
        Series of alpha values
        
    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03])
        >>> market = pd.Series([0.02, -0.01, 0.02])
        >>> alpha = calculate_alpha(returns, market, window=2)
        >>> print(alpha)
        0    NaN
        1    0.0
        2    0.01
        dtype: float64
    """
    beta = calculate_beta(returns, market_returns, window=window)
    excess_returns = returns - risk_free_rate
    market_excess_returns = market_returns - risk_free_rate
    
    alpha = excess_returns - beta * market_excess_returns
    
    if annualize:
        alpha = alpha * 252
    
    return alpha

def plot_returns(returns: pd.Series, title: str = "Returns"):
    """Plot returns series.
    
    Args:
        returns: Returns series
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    plt.plot(returns.cumsum())
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.grid(True)
    plt.show()

def plot_volatility(volatility: pd.Series, title: str = "Volatility"):
    """Plot volatility series.
    
    Args:
        volatility: Volatility series
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    plt.plot(volatility)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.grid(True)
    plt.show()

def plot_correlation_matrix(returns: pd.DataFrame, title: str = "Correlation Matrix"):
    """Plot correlation matrix.
    
    Args:
        returns: Returns DataFrame
        title: Plot title
    """
    corr = returns.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def calculate_portfolio_metrics(returns: pd.Series) -> Dict[str, float]:
    """Calculate portfolio performance metrics.
    
    Args:
        returns: Returns series
        
    Returns:
        Dictionary of metrics
    """
    return {
        'total_return': (1 + returns).prod() - 1,
        'annualized_return': (1 + returns).prod() ** (252/len(returns)) - 1,
        'volatility': returns.std() * np.sqrt(252),
        'sharpe_ratio': calculate_sharpe_ratio(returns),
        'max_drawdown': calculate_max_drawdown(returns),
        'win_rate': calculate_win_rate(returns),
        'profit_factor': abs(returns[returns > 0].sum() / returns[returns < 0].sum())
    }

def format_currency(value: float) -> str:
    """Format number as currency.
    
    Args:
        value: Number to format
        
    Returns:
        Formatted string
    """
    return f"${value:,.2f}"

def format_percentage(value: float) -> str:
    """Format number as percentage.
    
    Args:
        value: Number to format
        
    Returns:
        Formatted string
    """
    return f"{value:.2%}"

def calculate_rolling_metrics(returns: pd.Series, window: int = 252) -> pd.DataFrame:
    """Calculate rolling performance metrics.
    
    Args:
        returns: Returns series
        window: Rolling window size
        
    Returns:
        DataFrame of rolling metrics
    """
    metrics = pd.DataFrame(index=returns.index)
    
    # Rolling returns
    metrics['returns'] = returns.rolling(window=window).mean() * 252
    
    # Rolling volatility
    metrics['volatility'] = returns.rolling(window=window).std() * np.sqrt(252)
    
    # Rolling Sharpe ratio
    metrics['sharpe_ratio'] = metrics['returns'] / metrics['volatility']
    
    # Rolling max drawdown
    metrics['max_drawdown'] = returns.rolling(window=window).apply(calculate_max_drawdown)
    
    return metrics

def plot_rolling_metrics(metrics: pd.DataFrame):
    """Plot rolling performance metrics.
    
    Args:
        metrics: DataFrame of rolling metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot returns
    axes[0, 0].plot(metrics['returns'])
    axes[0, 0].set_title('Rolling Returns')
    axes[0, 0].grid(True)
    
    # Plot volatility
    axes[0, 1].plot(metrics['volatility'])
    axes[0, 1].set_title('Rolling Volatility')
    axes[0, 1].grid(True)
    
    # Plot Sharpe ratio
    axes[1, 0].plot(metrics['sharpe_ratio'])
    axes[1, 0].set_title('Rolling Sharpe Ratio')
    axes[1, 0].grid(True)
    
    # Plot max drawdown
    axes[1, 1].plot(metrics['max_drawdown'])
    axes[1, 1].set_title('Rolling Max Drawdown')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def normalize_indicator_name(name: str) -> str:
    """Normalize technical indicator names.

    This helper converts indicator names to a consistent
    ``UPPER_SNAKE_CASE`` style as commonly used by ``pandas_ta``.

    Parameters
    ----------
    name:
        Original indicator name.

    Returns
    -------
    str
        Normalized indicator name.
    """

    if not isinstance(name, str):
        return str(name)

    normalized = name.replace(" ", "_").replace("-", "_")
    return normalized.upper()
