"""RSI strategy optimization module."""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from functools import partial

@dataclass
class RSIConfig:
    """Configuration for RSI strategy."""
    lookback_period: int
    overbought: float
    oversold: float
    smoothing: int
    stop_loss: float
    take_profit: float

def calculate_rsi(data: pd.DataFrame, lookback_period: int) -> pd.Series:
    """Calculate RSI indicator.
    
    Args:
        data: DataFrame containing price data
        lookback_period: Period for RSI calculation
        
    Returns:
        Series containing RSI values
    """
    # Calculate price changes
    delta = data["close"].diff()
    
    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).rolling(window=lookback_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=lookback_period).mean()
    
    # Calculate RS and RSI
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def generate_rsi_signals(data: pd.DataFrame, config: RSIConfig) -> pd.Series:
    """Generate trading signals based on RSI.
    
    Args:
        data: DataFrame containing price data
        config: RSI strategy configuration
        
    Returns:
        Series containing trading signals
    """
    # Calculate RSI
    rsi = calculate_rsi(data, config.lookback_period)
    
    # Generate signals
    signals = pd.Series(0, index=data.index)
    signals[rsi < config.oversold] = 1  # Oversold - Buy
    signals[rsi > config.overbought] = -1  # Overbought - Sell
    
    # Apply smoothing
    if config.smoothing > 1:
        signals = signals.rolling(window=config.smoothing).mean()
    
    return signals

def calculate_returns(data: pd.DataFrame, signals: pd.Series, config: RSIConfig) -> pd.Series:
    """Calculate strategy returns with stop-loss and take-profit.
    
    Args:
        data: DataFrame containing price data
        signals: Series containing trading signals
        config: RSI strategy configuration
        
    Returns:
        Series containing strategy returns
    """
    # Initialize returns
    returns = pd.Series(0.0, index=data.index)
    
    # Calculate price changes
    price_changes = data["close"].pct_change()
    
    # Apply signals
    returns = price_changes * signals.shift(1)
    
    # Apply stop-loss
    stop_loss_mask = (returns < -config.stop_loss) & (signals.shift(1) != 0)
    returns[stop_loss_mask] = -config.stop_loss
    
    # Apply take-profit
    take_profit_mask = (returns > config.take_profit) & (signals.shift(1) != 0)
    returns[take_profit_mask] = config.take_profit
    
    return returns

def calculate_metrics(returns: pd.Series) -> Dict[str, float]:
    """Calculate performance metrics for the strategy.
    
    Args:
        returns: Series containing strategy returns
        
    Returns:
        Dictionary containing performance metrics
    """
    # Calculate basic metrics
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
    
    # Calculate drawdown
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = cumulative_returns / rolling_max - 1
    max_drawdown = drawdowns.min()
    
    # Calculate win rate
    winning_trades = (returns > 0).sum()
    total_trades = (returns != 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate
    }

def optimize_rsi_parameters(
    data: pd.DataFrame,
    param_grid: Dict[str, List[Any]],
    metric: str = "sharpe_ratio"
) -> Tuple[RSIConfig, Dict[str, float]]:
    """Optimize RSI strategy parameters using grid search.
    
    Args:
        data: DataFrame containing price data
        param_grid: Dictionary of parameters to optimize
        metric: Performance metric to optimize
        
    Returns:
        Tuple of (optimal configuration, performance metrics)
    """
    # Generate parameter combinations
    param_combinations = []
    for lookback in param_grid["lookback_period"]:
        for overbought in param_grid["overbought"]:
            for oversold in param_grid["oversold"]:
                for smoothing in param_grid["smoothing"]:
                    for stop_loss in param_grid["stop_loss"]:
                        for take_profit in param_grid["take_profit"]:
                            param_combinations.append(RSIConfig(
                                lookback_period=lookback,
                                overbought=overbought,
                                oversold=oversold,
                                smoothing=smoothing,
                                stop_loss=stop_loss,
                                take_profit=take_profit
                            ))
    
    # Optimize parameters using parallel processing
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(
            partial(_evaluate_parameters, data),
            param_combinations
        ))
    
    # Find best configuration
    best_idx = np.argmax([r[1][metric] for r in results])
    best_config, best_metrics = results[best_idx]
    
    return best_config, best_metrics

def _evaluate_parameters(data: pd.DataFrame, config: RSIConfig) -> Tuple[RSIConfig, Dict[str, float]]:
    """Evaluate RSI strategy parameters.
    
    Args:
        data: DataFrame containing price data
        config: RSI strategy configuration
        
    Returns:
        Tuple of (configuration, performance metrics)
    """
    # Generate signals
    signals = generate_rsi_signals(data, config)
    
    # Calculate returns
    returns = calculate_returns(data, signals, config)
    
    # Calculate metrics
    metrics = calculate_metrics(returns)
    
    return config, metrics

def backtest_rsi_strategy(
    data: pd.DataFrame,
    config: RSIConfig,
    initial_capital: float = 100000.0
) -> Dict[str, Any]:
    """Backtest RSI strategy with detailed results.
    
    Args:
        data: DataFrame containing price data
        config: RSI strategy configuration
        initial_capital: Initial capital for backtest
        
    Returns:
        Dictionary containing backtest results
    """
    # Generate signals
    signals = generate_rsi_signals(data, config)
    
    # Calculate returns
    returns = calculate_returns(data, signals, config)
    
    # Calculate metrics
    metrics = calculate_metrics(returns)
    
    # Calculate equity curve
    equity_curve = initial_capital * (1 + returns).cumprod()
    
    # Calculate trade statistics
    trades = pd.DataFrame({
        "entry_date": data.index[signals != 0],
        "entry_price": data["close"][signals != 0],
        "signal": signals[signals != 0],
        "returns": returns[signals != 0]
    })
    
    # Calculate drawdown periods
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = cumulative_returns / rolling_max - 1
    
    return {
        "metrics": metrics,
        "equity_curve": equity_curve,
        "trades": trades,
        "drawdowns": drawdowns,
        "signals": signals,
        "returns": returns
    } 