"""
DEPRECATED: This file has been consolidated into trading\optimization\rsi_optimizer.py
Please use the consolidated version instead.
Last updated: 2025-06-18 13:06:19
"""

# -*- coding: utf-8 -*-
"""RSI strategy parameter optimizer."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Callable
import pandas as pd
import numpy as np
from itertools import product

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Default parameter grid
DEFAULT_PARAM_GRID = {
    "period": [7, 14, 21],
    "buy_threshold": [25, 30, 35],
    "sell_threshold": [65, 70, 75]
}

def calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio for a series of returns.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
        
    excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
    if excess_returns.std() == 0:
        return 0.0
        
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def optimize_rsi(
    df: pd.DataFrame,
    ticker: str,
    strategy_func: Callable,
    param_grid: Dict[str, List[Any]] = None
) -> Dict[str, Any]:
    """Optimize RSI strategy parameters using grid search.
    
    Args:
        df: Price data DataFrame
        ticker: Stock ticker symbol
        strategy_func: Function that generates RSI signals
        param_grid: Optional parameter grid to search
        
    Returns:
        Dictionary containing optimal parameters and performance
    """
    try:
        logger.info(f"Starting RSI optimization for {ticker}")
        
        # Use default grid if none provided
        param_grid = param_grid or DEFAULT_PARAM_GRID
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        best_sharpe = -np.inf
        best_params = None
        results = []
        
        # Test each parameter combination
        for params in param_combinations:
            param_dict = dict(zip(param_names, params))
            
            try:
                # Generate signals with current parameters
                signals = strategy_func(df, **param_dict)
                
                # Calculate returns
                returns = signals['returns']
                
                # Calculate Sharpe ratio
                sharpe = calculate_sharpe(returns)
                
                # Store results
                result = {
                    **param_dict,
                    'sharpe': sharpe
                }
                results.append(result)
                
                # Update best parameters
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = param_dict
                    
            except Exception as e:
                logger.warning(f"Error testing parameters {param_dict}: {str(e)}")
                continue
        
        if best_params is None:
            raise ValueError("No valid parameter combinations found")
            
        # Prepare optimal settings
        optimal_settings = {
            "optimal_period": best_params["period"],
            "buy_threshold": best_params["buy_threshold"],
            "sell_threshold": best_params["sell_threshold"],
            "sharpe": round(best_sharpe, 2)
        }
        
        # Save optimal settings
        settings_dir = Path("memory/strategy_settings/rsi")
        settings_dir.mkdir(parents=True, exist_ok=True)
        
        settings_file = settings_dir / f"{ticker}.json"
        with open(settings_file, "w") as f:
            json.dump(optimal_settings, f, indent=4)
            
        logger.info(f"RSI optimization complete for {ticker}. Best Sharpe: {best_sharpe:.2f}")
        
        # Save full results for analysis
        results_df = pd.DataFrame(results)
        results_file = settings_dir / f"{ticker}_results.csv"
        results_df.to_csv(results_file, index=False)
        
        return optimal_settings
        
    except Exception as e:
        error_msg = f"Error optimizing RSI for {ticker}: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

if __name__ == "__main__":
    # Test optimization
    import yfinance as yf
    from trading.strategies.rsi_signals import generate_rsi_signals
    
    # Get sample data
    ticker = "AAPL"
    df = yf.download(ticker, period="1y")
    
    # Run optimization
    optimal = optimize_rsi(df, ticker, generate_rsi_signals)
    print(json.dumps(optimal, indent=2))
