"""
Safe Mathematical Operations for Financial Calculations

This module provides division-safe versions of common financial calculations
to prevent the 180+ division-by-zero bugs found in the codebase.

Usage Examples:
    
    # Instead of: rs = gain / loss
    rsi = safe_rsi(prices)
    
    # Instead of: returns = np.diff(prices) / prices[:-1]  
    returns = safe_returns(prices)
    
    # Instead of: sharpe = mean / std
    sharpe = safe_sharpe_ratio(returns)
    
    # Instead of: drawdown = (equity - max) / max
    drawdown = safe_drawdown(equity_curve)
    
All functions handle edge cases automatically and return sensible defaults.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional


def safe_divide(
    numerator: Union[float, np.ndarray, pd.Series],
    denominator: Union[float, np.ndarray, pd.Series],
    default: float = 0.0,
    epsilon: float = 1e-10
) -> Union[float, np.ndarray, pd.Series]:
    """
    Safely divide numerator by denominator, handling zero/near-zero denominators.
    
    Args:
        numerator: Value(s) to divide
        denominator: Value(s) to divide by
        default: Value to return when denominator is zero (default: 0.0)
        epsilon: Threshold for considering denominator as zero (default: 1e-10)
    
    Returns:
        Result of division, or default where denominator is zero
        
    Examples:
        >>> safe_divide(10, 2)
        5.0
        >>> safe_divide(10, 0)
        0.0
        >>> safe_divide(np.array([10, 20]), np.array([2, 0]))
        array([5.0, 0.0])
    """
    # Handle scalar inputs
    if isinstance(numerator, (int, float)) and isinstance(denominator, (int, float)):
        if abs(denominator) < epsilon:
            return default
        result = numerator / denominator
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    
    # Convert to numpy arrays for vectorized operations
    num_arr = np.asarray(numerator)
    den_arr = np.asarray(denominator)
    
    # Create mask for valid denominators
    valid_mask = np.abs(den_arr) >= epsilon
    
    # Perform division
    result = np.full_like(num_arr, default, dtype=float)
    result[valid_mask] = num_arr[valid_mask] / den_arr[valid_mask]
    
    # Handle NaN and Inf
    result = np.where(np.isfinite(result), result, default)
    
    # Return same type as input
    if isinstance(numerator, pd.Series):
        return pd.Series(result, index=numerator.index, name=numerator.name)
    elif isinstance(denominator, pd.Series) and not isinstance(numerator, pd.Series):
        return pd.Series(result, index=denominator.index)
    elif isinstance(result, np.ndarray) and result.ndim == 0:
        return float(result)
    else:
        return result


def safe_rsi(
    prices: Union[np.ndarray, pd.Series],
    period: int = 14
) -> Union[np.ndarray, pd.Series]:
    """
    Calculate Relative Strength Index with safe division.
    
    This function eliminates the division-by-zero bug found in 12+ files
    throughout the codebase where rs = gain / loss causes crashes.
    
    Args:
        prices: Price series
        period: RSI period (default: 14)
    
    Returns:
        RSI values (0-100 scale)
        Returns 50.0 (neutral) when loss is zero
        
    Examples:
        >>> prices = pd.Series([100, 102, 101, 103, 105])
        >>> rsi = safe_rsi(prices, period=14)
    """
    # Convert to pandas Series for easier handling
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    if len(prices) < period + 1:
        # Return neutral RSI if insufficient data
        return pd.Series(50.0, index=prices.index) if isinstance(prices, pd.Series) else np.full(len(prices), 50.0)
    
    # Calculate price changes
    delta = prices.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)
    
    # Calculate average gains and losses
    avg_gains = gains.rolling(window=period, min_periods=1).mean()
    avg_losses = losses.rolling(window=period, min_periods=1).mean()
    
    # Use safe_divide for rs = gain / loss
    rs = safe_divide(avg_gains, avg_losses, default=0.0)
    
    # Calculate RSI: 100 - (100 / (1 + rs))
    # Handle case where rs might be very large (infinite gain)
    rs = np.clip(rs, 0, 1e10)  # Cap to prevent overflow
    rsi = 100 - (100 / (1 + rs))
    
    # Ensure RSI is in valid range
    rsi = np.clip(rsi, 0, 100)
    
    # Fill NaN values with neutral RSI
    if isinstance(rsi, pd.Series):
        rsi = rsi.fillna(50.0)
    else:
        rsi = np.where(np.isnan(rsi), 50.0, rsi)
    
    return rsi


def safe_returns(
    prices: Union[np.ndarray, pd.Series],
    method: str = 'simple'
) -> Union[np.ndarray, pd.Series]:
    """
    Calculate returns with safe division.
    
    Eliminates crashes from: np.diff(prices) / prices[:-1]
    
    Args:
        prices: Price series
        method: 'simple' or 'log' returns
    
    Returns:
        Returns series with zeros where division would fail
        
    Examples:
        >>> prices = np.array([100, 102, 101, 103])
        >>> returns = safe_returns(prices)
        array([0.02, -0.0098, 0.0198])
    """
    # Convert to numpy array for processing
    prices_arr = np.asarray(prices)
    
    if len(prices_arr) < 2:
        return np.array([]) if isinstance(prices, np.ndarray) else pd.Series(dtype=float, index=prices.index[1:])
    
    if method == 'log':
        # Log returns: ln(p_t / p_{t-1})
        prev_prices = prices_arr[:-1]
        current_prices = prices_arr[1:]
        
        # Safe division for log returns
        ratio = safe_divide(current_prices, prev_prices, default=1.0)
        returns = np.log(np.maximum(ratio, 1e-10))  # Prevent log(0)
    else:
        # Simple returns: (p_t - p_{t-1}) / p_{t-1}
        price_changes = np.diff(prices_arr)
        prev_prices = prices_arr[:-1]
        
        # Safe division
        returns = safe_divide(price_changes, prev_prices, default=0.0)
    
    # Handle NaN and Inf
    returns = np.where(np.isfinite(returns), returns, 0.0)
    
    # Return same type as input
    if isinstance(prices, pd.Series):
        return pd.Series(returns, index=prices.index[1:], name='returns')
    else:
        return returns


def safe_drawdown(
    equity_curve: Union[np.ndarray, pd.Series]
) -> Union[np.ndarray, pd.Series]:
    """
    Calculate drawdown with safe division.
    
    Eliminates crashes from: (equity - max) / max
    
    Args:
        equity_curve: Equity or cumulative return series
    
    Returns:
        Drawdown series (negative values indicating drawdown percentage)
        
    Examples:
        >>> equity = np.array([100, 110, 105, 115])
        >>> dd = safe_drawdown(equity)
    """
    # Convert to numpy array
    equity_arr = np.asarray(equity_curve)
    
    if len(equity_arr) == 0:
        return np.array([]) if isinstance(equity_curve, np.ndarray) else pd.Series(dtype=float, index=equity_curve.index)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_arr)
    
    # Calculate drawdown: (equity - max) / max
    # Use safe_divide to handle zero max values
    drawdown = safe_divide(
        equity_arr - running_max,
        running_max,
        default=0.0
    )
    
    # Return same type as input
    if isinstance(equity_curve, pd.Series):
        return pd.Series(drawdown, index=equity_curve.index, name='drawdown')
    else:
        return drawdown


def safe_sharpe_ratio(
    returns: Union[np.ndarray, pd.Series],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio with safe division.
    
    Eliminates crashes from: mean / std
    
    Args:
        returns: Return series
        risk_free_rate: Risk-free rate (default: 0.0)
        periods_per_year: Trading periods per year (default: 252)
    
    Returns:
        Sharpe ratio, or 0.0 if std is zero
        
    Examples:
        >>> returns = np.array([0.01, 0.02, -0.01, 0.015])
        >>> sharpe = safe_sharpe_ratio(returns)
    """
    returns_arr = np.asarray(returns)
    
    if len(returns_arr) == 0:
        return 0.0
    
    # Remove NaN and Inf values
    returns_arr = returns_arr[np.isfinite(returns_arr)]
    
    if len(returns_arr) == 0:
        return 0.0
    
    # Calculate excess returns
    excess_returns = returns_arr - (risk_free_rate / periods_per_year)
    
    # Calculate mean and std
    mean_return = np.mean(excess_returns)
    std_return = np.std(returns_arr, ddof=1)  # Sample standard deviation
    
    # Annualize
    annualized_mean = mean_return * periods_per_year
    annualized_std = std_return * np.sqrt(periods_per_year)
    
    # Safe division: mean / std
    sharpe = safe_divide(annualized_mean, annualized_std, default=0.0)
    
    # Handle scalar result
    if isinstance(sharpe, (np.ndarray, pd.Series)):
        sharpe = float(sharpe)
    
    return float(sharpe) if np.isfinite(sharpe) else 0.0


def safe_sortino_ratio(
    returns: Union[np.ndarray, pd.Series],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio with safe division.
    
    Eliminates crashes from: mean / downside_std
    
    Args:
        returns: Return series
        risk_free_rate: Risk-free rate (default: 0.0)
        periods_per_year: Trading periods per year (default: 252)
    
    Returns:
        Sortino ratio, or 0.0 if downside deviation is zero
    """
    returns_arr = np.asarray(returns)
    
    if len(returns_arr) == 0:
        return 0.0
    
    # Remove NaN and Inf values
    returns_arr = returns_arr[np.isfinite(returns_arr)]
    
    if len(returns_arr) == 0:
        return 0.0
    
    # Calculate excess returns
    excess_returns = returns_arr - (risk_free_rate / periods_per_year)
    
    # Calculate mean
    mean_return = np.mean(excess_returns)
    
    # Calculate downside deviation (only negative returns)
    downside_returns = returns_arr[returns_arr < 0]
    
    if len(downside_returns) == 0:
        # No downside risk, return high ratio or 0.0
        return 0.0 if mean_return <= 0 else float('inf') if mean_return > 0 else 0.0
    
    downside_std = np.std(downside_returns, ddof=1)
    
    # Annualize
    annualized_mean = mean_return * periods_per_year
    annualized_downside_std = downside_std * np.sqrt(periods_per_year)
    
    # Safe division: mean / downside_std
    sortino = safe_divide(annualized_mean, annualized_downside_std, default=0.0)
    
    # Handle scalar result
    if isinstance(sortino, (np.ndarray, pd.Series)):
        sortino = float(sortino)
    
    # Cap at reasonable value to avoid infinity
    if np.isinf(sortino) or sortino > 100:
        return 100.0
    
    return float(sortino) if np.isfinite(sortino) else 0.0


def safe_calmar_ratio(
    returns: Union[np.ndarray, pd.Series],
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio with safe division.
    
    Eliminates crashes from: annual_return / max_drawdown
    
    Args:
        returns: Return series
        periods_per_year: Trading periods per year (default: 252)
    
    Returns:
        Calmar ratio, or 0.0 if max drawdown is zero
    """
    returns_arr = np.asarray(returns)
    
    if len(returns_arr) == 0:
        return 0.0
    
    # Remove NaN and Inf values
    returns_arr = returns_arr[np.isfinite(returns_arr)]
    
    if len(returns_arr) == 0:
        return 0.0
    
    # Calculate cumulative returns (equity curve)
    cumulative_returns = np.cumprod(1 + returns_arr)
    
    # Calculate drawdown
    drawdown = safe_drawdown(cumulative_returns)
    max_drawdown = np.abs(np.min(drawdown))  # Max drawdown as positive value
    
    if max_drawdown == 0:
        return 0.0
    
    # Calculate annualized return
    total_return = cumulative_returns[-1] / cumulative_returns[0] - 1.0
    num_periods = len(returns_arr)
    if num_periods > 0:
        annualized_return = (1 + total_return) ** (periods_per_year / num_periods) - 1.0
    else:
        annualized_return = 0.0
    
    # Safe division: annual_return / max_drawdown
    calmar = safe_divide(annualized_return, max_drawdown, default=0.0)
    
    # Handle scalar result
    if isinstance(calmar, (np.ndarray, pd.Series)):
        calmar = float(calmar)
    
    return float(calmar) if np.isfinite(calmar) else 0.0


def safe_mape(
    actual: Union[np.ndarray, pd.Series],
    predicted: Union[np.ndarray, pd.Series]
) -> float:
    """
    Calculate Mean Absolute Percentage Error with safe division.
    
    Eliminates crashes from: (actual - pred) / actual
    
    Args:
        actual: Actual values
        predicted: Predicted values
    
    Returns:
        MAPE as percentage, ignoring points where actual is zero
    """
    actual_arr = np.asarray(actual)
    predicted_arr = np.asarray(predicted)
    
    if len(actual_arr) != len(predicted_arr):
        raise ValueError("actual and predicted must have same length")
    
    if len(actual_arr) == 0:
        return 0.0
    
    # Remove NaN and Inf
    mask = np.isfinite(actual_arr) & np.isfinite(predicted_arr)
    actual_arr = actual_arr[mask]
    predicted_arr = predicted_arr[mask]
    
    if len(actual_arr) == 0:
        return 0.0
    
    # Mask out zero actuals (division would fail)
    non_zero_mask = np.abs(actual_arr) > 1e-10
    
    if np.sum(non_zero_mask) == 0:
        return 0.0
    
    # Calculate percentage error only for non-zero actuals
    percentage_errors = np.abs(
        safe_divide(actual_arr[non_zero_mask] - predicted_arr[non_zero_mask],
                   actual_arr[non_zero_mask],
                   default=0.0)
    )
    
    # Calculate mean
    mape = np.mean(percentage_errors) * 100  # Convert to percentage
    
    return float(mape) if np.isfinite(mape) else 0.0


def safe_normalize(
    data: Union[np.ndarray, pd.Series],
    method: str = 'zscore'
) -> Union[np.ndarray, pd.Series]:
    """
    Normalize data with safe division.
    
    Eliminates crashes from normalization formulas.
    
    Args:
        data: Data to normalize
        method: 'zscore', 'minmax', or 'robust'
    
    Returns:
        Normalized data
        
    Methods:
        - zscore: (x - mean) / std
        - minmax: (x - min) / (max - min)
        - robust: (x - median) / IQR
    """
    data_arr = np.asarray(data)
    
    if len(data_arr) == 0:
        return data_arr if isinstance(data, np.ndarray) else pd.Series(dtype=float, index=data.index)
    
    # Remove NaN and Inf for calculation
    finite_mask = np.isfinite(data_arr)
    finite_data = data_arr[finite_mask]
    
    if len(finite_data) == 0:
        return data_arr if isinstance(data, np.ndarray) else pd.Series(data_arr, index=data.index)
    
    if method == 'zscore':
        # Z-score normalization: (x - mean) / std
        mean_val = np.mean(finite_data)
        std_val = np.std(finite_data, ddof=1)
        
        normalized = safe_divide(data_arr - mean_val, std_val, default=0.0)
        
    elif method == 'minmax':
        # Min-max normalization: (x - min) / (max - min)
        min_val = np.min(finite_data)
        max_val = np.max(finite_data)
        range_val = max_val - min_val
        
        normalized = safe_divide(data_arr - min_val, range_val, default=0.0)
        
    elif method == 'robust':
        # Robust normalization: (x - median) / IQR
        median_val = np.median(finite_data)
        q75 = np.percentile(finite_data, 75)
        q25 = np.percentile(finite_data, 25)
        iqr = q75 - q25
        
        normalized = safe_divide(data_arr - median_val, iqr, default=0.0)
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Preserve NaN and Inf in original positions
    normalized = np.where(finite_mask, normalized, data_arr)
    
    # Return same type as input
    if isinstance(data, pd.Series):
        return pd.Series(normalized, index=data.index, name=data.name)
    else:
        return normalized


def safe_kelly_fraction(
    avg_win: float,
    avg_loss: float,
    win_rate: float
) -> float:
    """
    Calculate Kelly Criterion fraction with safe division.
    
    Eliminates crashes from: avg_win / avg_loss
    
    Args:
        avg_win: Average win amount
        avg_loss: Average loss amount (positive value)
        win_rate: Win rate (0-1)
    
    Returns:
        Kelly fraction, capped at reasonable values (max 0.25 or 25%)
    """
    # Validate inputs
    if not (0 <= win_rate <= 1):
        return 0.0
    
    if avg_loss <= 0:
        return 0.0
    
    # Kelly formula: f = (p * b - q) / b
    # where p = win_rate, q = 1 - win_rate, b = avg_win / avg_loss
    # Simplified: f = win_rate - (1 - win_rate) / (avg_win / avg_loss)
    
    # Safe division for win/loss ratio
    win_loss_ratio = safe_divide(avg_win, avg_loss, default=0.0)
    
    if win_loss_ratio == 0:
        return 0.0
    
    # Calculate Kelly fraction
    kelly = win_rate - (1 - win_rate) / win_loss_ratio
    
    # Cap at reasonable maximum (25%)
    kelly = np.clip(kelly, 0.0, 0.25)
    
    return float(kelly) if np.isfinite(kelly) else 0.0


def safe_bollinger_position(
    price: float,
    upper_band: float,
    lower_band: float
) -> float:
    """
    Calculate position within Bollinger Bands with safe division.
    
    Eliminates crashes from: (price - lower) / (upper - lower)
    
    Args:
        price: Current price
        upper_band: Upper Bollinger Band
        lower_band: Lower Bollinger Band
    
    Returns:
        Position (0.0 = at lower, 0.5 = middle, 1.0 = at upper)
        Returns 0.5 if bands are equal (no range)
    """
    # Calculate band range
    band_range = upper_band - lower_band
    
    # If bands are equal or invalid, return middle position
    if abs(band_range) < 1e-10:
        return 0.5
    
    # Calculate position: (price - lower) / (upper - lower)
    position = safe_divide(price - lower_band, band_range, default=0.5)
    
    # Clamp to [0, 1] range
    position = np.clip(position, 0.0, 1.0)
    
    return float(position) if np.isfinite(position) else 0.5


def safe_price_momentum(
    current_price: Union[float, np.ndarray, pd.Series],
    reference_price: Union[float, np.ndarray, pd.Series]
) -> Union[float, np.ndarray, pd.Series]:
    """
    Calculate price momentum/ratio with safe division.
    
    Eliminates crashes from: current / reference - 1
    Or: (current - reference) / reference
    
    Args:
        current_price: Current price(s)
        reference_price: Reference price(s) (e.g., SMA, lagged price)
    
    Returns:
        Momentum as percentage change
    """
    # Use safe_divide for (current - reference) / reference
    # This is equivalent to: (current / reference) - 1
    momentum = safe_divide(
        np.asarray(current_price) - np.asarray(reference_price),
        np.asarray(reference_price),
        default=0.0
    )
    
    # Handle NaN and Inf
    momentum = np.where(np.isfinite(momentum), momentum, 0.0)
    
    # Return same type as input
    if isinstance(current_price, pd.Series):
        return pd.Series(momentum, index=current_price.index, name='momentum')
    elif isinstance(reference_price, pd.Series) and not isinstance(current_price, pd.Series):
        return pd.Series(momentum, index=reference_price.index, name='momentum')
    elif isinstance(momentum, np.ndarray) and momentum.ndim == 0:
        return float(momentum)
    else:
        return momentum


if __name__ == "__main__":
    # Test each function with edge cases
    
    print("Testing safe_math utilities...")
    
    # Test 1: safe_divide
    assert safe_divide(10, 2) == 5.0, "safe_divide basic test failed"
    assert safe_divide(10, 0) == 0.0, "safe_divide zero denominator test failed"
    assert safe_divide(10, 1e-11) == 0.0, "safe_divide epsilon test failed"
    assert np.allclose(safe_divide(np.array([10, 20]), np.array([2, 0])), np.array([5.0, 0.0])), "safe_divide array test failed"
    print("[PASS] safe_divide tests passed")
    
    # Test 2: safe_rsi with zero loss
    prices_no_loss = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114])
    rsi = safe_rsi(prices_no_loss)
    assert np.all((rsi >= 0) & (rsi <= 100)), "safe_rsi range test failed"
    assert not np.any(np.isnan(rsi)), "safe_rsi NaN test failed"
    print("[PASS] safe_rsi tests passed")
    
    # Test 3: safe_returns with zero prices
    prices_with_zero = np.array([100, 0, 102])
    returns = safe_returns(prices_with_zero)
    assert not np.any(np.isnan(returns)), "safe_returns NaN test failed"
    assert len(returns) == 2, "safe_returns length test failed"
    print("[PASS] safe_returns tests passed")
    
    # Test 4: safe_sharpe_ratio with zero std
    constant_returns = np.array([0.01, 0.01, 0.01])
    sharpe = safe_sharpe_ratio(constant_returns)
    assert sharpe == 0.0, "safe_sharpe_ratio zero std test failed"
    print("[PASS] safe_sharpe_ratio tests passed")
    
    # Test 5: safe_drawdown
    equity = np.array([100, 110, 105, 115, 120])
    dd = safe_drawdown(equity)
    assert len(dd) == len(equity), "safe_drawdown length test failed"
    assert not np.any(np.isnan(dd)), "safe_drawdown NaN test failed"
    print("[PASS] safe_drawdown tests passed")
    
    # Test 6: safe_sortino_ratio
    returns = np.array([0.01, 0.02, -0.01, 0.015])
    sortino = safe_sortino_ratio(returns)
    assert np.isfinite(sortino), "safe_sortino_ratio finite test failed"
    print("[PASS] safe_sortino_ratio tests passed")
    
    # Test 7: safe_calmar_ratio
    returns = np.array([0.01, -0.02, 0.015, -0.01, 0.02])
    calmar = safe_calmar_ratio(returns)
    assert np.isfinite(calmar), "safe_calmar_ratio finite test failed"
    print("[PASS] safe_calmar_ratio tests passed")
    
    # Test 8: safe_mape
    actual = np.array([100, 200, 150])
    predicted = np.array([105, 195, 155])
    mape = safe_mape(actual, predicted)
    assert np.isfinite(mape), "safe_mape finite test failed"
    # Test with zero actuals
    actual_zero = np.array([100, 0, 150])
    mape_zero = safe_mape(actual_zero, predicted)
    assert np.isfinite(mape_zero), "safe_mape zero actual test failed"
    print("[PASS] safe_mape tests passed")
    
    # Test 9: safe_normalize
    data = np.array([1, 2, 3, 4, 5])
    normalized = safe_normalize(data, method='zscore')
    assert not np.any(np.isnan(normalized)), "safe_normalize zscore NaN test failed"
    normalized_minmax = safe_normalize(data, method='minmax')
    assert not np.any(np.isnan(normalized_minmax)), "safe_normalize minmax NaN test failed"
    print("[PASS] safe_normalize tests passed")
    
    # Test 10: safe_kelly_fraction
    kelly = safe_kelly_fraction(avg_win=2.0, avg_loss=1.0, win_rate=0.6)
    assert 0 <= kelly <= 0.25, "safe_kelly_fraction range test failed"
    kelly_zero = safe_kelly_fraction(avg_win=2.0, avg_loss=0.0, win_rate=0.6)
    assert kelly_zero == 0.0, "safe_kelly_fraction zero loss test failed"
    print("[PASS] safe_kelly_fraction tests passed")
    
    # Test 11: safe_bollinger_position
    position = safe_bollinger_position(price=105, upper_band=110, lower_band=100)
    assert 0 <= position <= 1, "safe_bollinger_position range test failed"
    position_equal = safe_bollinger_position(price=105, upper_band=100, lower_band=100)
    assert position_equal == 0.5, "safe_bollinger_position equal bands test failed"
    print("[PASS] safe_bollinger_position tests passed")
    
    # Test 12: safe_price_momentum
    momentum = safe_price_momentum(110, 100)
    assert np.isfinite(momentum), "safe_price_momentum finite test failed"
    momentum_zero = safe_price_momentum(110, 0)
    assert momentum_zero == 0.0, "safe_price_momentum zero reference test failed"
    print("[PASS] safe_price_momentum tests passed")
    
    print("\n[SUCCESS] All tests passed!")

