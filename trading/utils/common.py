import logging
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from functools import wraps
import time
import traceback
import sys

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

def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Save configuration to file with basic error handling.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary to persist.
    config_path : Union[str, Path]
        Path to the configuration file. Supported formats are ``.json`` and
        ``.yaml``/``.yml``.

    Raises
    ------
    IOError
        If the file cannot be written.
    ValueError
        If the file extension is unsupported.
    """

    config_path = Path(config_path)

    try:
        if config_path.suffix == ".json":
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)
        elif config_path.suffix in [".yaml", ".yml"]:
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError(
                f"Unsupported config file format: {config_path.suffix}"
            )
    except OSError as exc:
        raise IOError(f"Failed to save config to {config_path}: {exc}") from exc

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
            logger = logging.getLogger(func.__module__)
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

def calculate_returns(prices: pd.Series, method: str = 'log') -> pd.Series:
    """Calculate returns from price series.
    
    Args:
        prices: Price series
        method: Return calculation method ('log' or 'simple')
        
    Returns:
        Returns series
    """
    if method == 'log':
        returns = np.log(prices / prices.shift(1))
    else:  # simple
        returns = prices.pct_change()
    return returns

def calculate_volatility(returns: pd.Series, window: int = 252) -> pd.Series:
    """Calculate rolling volatility.
    
    Args:
        returns: Returns series
        window: Rolling window size
        
    Returns:
        Volatility series
    """
    return returns.rolling(window=window).std() * np.sqrt(252)

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio.
    
    Args:
        returns: Returns series
        risk_free_rate: Risk-free rate
        
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate/252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown.
    
    Args:
        returns: Returns series
        
    Returns:
        Maximum drawdown
    """
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = cumulative_returns / rolling_max - 1
    return drawdowns.min()

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
        'win_rate': (returns > 0).mean(),
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
