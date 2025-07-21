"""
Math Helpers Module

Enhanced with Batch 11 features: Removed wrapper functions over NumPy and provides
only essential mathematical utilities that add value beyond NumPy's built-in functions.

This module provides mathematical utilities that extend NumPy functionality rather than
just wrapping it.
"""

import logging
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def calculate_rolling_statistics(
    data: Union[pd.Series, np.ndarray], window: int = 20, statistics: List[str] = None
) -> Dict[str, pd.Series]:
    """
    Calculate multiple rolling statistics efficiently.

    Args:
        data: Input data series
        window: Rolling window size
        statistics: List of statistics to calculate

    Returns:
        Dictionary of rolling statistics
    """
    if statistics is None:
        statistics = ["mean", "std", "min", "max", "median"]

    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    results = {}

    for stat in statistics:
        if stat == "mean":
            results[stat] = data.rolling(window=window).mean()
        elif stat == "std":
            results[stat] = data.rolling(window=window).std()
        elif stat == "min":
            results[stat] = data.rolling(window=window).min()
        elif stat == "max":
            results[stat] = data.rolling(window=window).max()
        elif stat == "median":
            results[stat] = data.rolling(window=window).median()
        elif stat == "skew":
            results[stat] = data.rolling(window=window).skew()
        elif stat == "kurt":
            results[stat] = data.rolling(window=window).kurt()
        else:
            logger.warning(f"Unknown statistic: {stat}")

    return results


def calculate_percentile_ranks(
    data: Union[pd.Series, np.ndarray], window: int = 252
) -> pd.Series:
    """
    Calculate percentile ranks within a rolling window.

    Args:
        data: Input data series
        window: Rolling window size

    Returns:
        Series of percentile ranks
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    def rolling_percentile_rank(x):
        if len(x) < 2:
            return 0.5
        return stats.percentileofscore(x[:-1], x[-1]) / 100

    return data.rolling(window=window).apply(rolling_percentile_rank)


def calculate_zscore(data: Union[pd.Series, np.ndarray], window: int = 20) -> pd.Series:
    """
    Calculate rolling z-score.

    Args:
        data: Input data series
        window: Rolling window size

    Returns:
        Series of z-scores
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()

    return (data - rolling_mean) / rolling_std


def calculate_momentum_score(
    data: Union[pd.Series, np.ndarray], short_window: int = 10, long_window: int = 50
) -> pd.Series:
    """
    Calculate momentum score based on short vs long term trends.

    Args:
        data: Input data series
        short_window: Short-term window
        long_window: Long-term window

    Returns:
        Series of momentum scores
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    short_ma = data.rolling(window=short_window).mean()
    long_ma = data.rolling(window=long_window).mean()

    # Normalize by long-term standard deviation
    long_std = data.rolling(window=long_window).std()

    momentum = (short_ma - long_ma) / long_std
    return momentum


def calculate_regime_probability(
    data: Union[pd.Series, np.ndarray], regimes: List[str] = None, window: int = 60
) -> Dict[str, pd.Series]:
    """
    Calculate probability of different market regimes.

    Args:
        data: Input data series (typically returns)
        regimes: List of regime names
        window: Rolling window size

    Returns:
        Dictionary of regime probabilities
    """
    if regimes is None:
        regimes = ["bull", "bear", "sideways"]

    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    # Calculate rolling statistics
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    rolling_skew = data.rolling(window=window).skew()

    probabilities = {}

    for regime in regimes:
        if regime == "bull":
            # High mean, low volatility, positive skew
            prob = (
                (rolling_mean > 0).astype(float) * 0.4
                + (rolling_std < rolling_std.median()).astype(float) * 0.3
                + (rolling_skew > 0).astype(float) * 0.3
            )
        elif regime == "bear":
            # Low mean, high volatility, negative skew
            prob = (
                (rolling_mean < 0).astype(float) * 0.4
                + (rolling_std > rolling_std.median()).astype(float) * 0.3
                + (rolling_skew < 0).astype(float) * 0.3
            )
        elif regime == "sideways":
            # Low mean, low volatility, near-zero skew
            prob = (
                (abs(rolling_mean) < rolling_std).astype(float) * 0.5
                + (rolling_std < rolling_std.median()).astype(float) * 0.3
                + (abs(rolling_skew) < 0.5).astype(float) * 0.2
            )

        probabilities[regime] = prob

    return probabilities


def calculate_tail_risk_metrics(
    data: Union[pd.Series, np.ndarray], confidence_level: float = 0.05
) -> Dict[str, float]:
    """
    Calculate comprehensive tail risk metrics.

    Args:
        data: Input data series (typically returns)
        confidence_level: Confidence level for tail risk

    Returns:
        Dictionary of tail risk metrics
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    # Remove NaN values
    data = data.dropna()

    if len(data) == 0:
        return {
            "var": 0.0,
            "cvar": 0.0,
            "tail_dependence": 0.0,
            "expected_shortfall": 0.0,
            "tail_risk_ratio": 0.0,
        }

    # Calculate VaR and CVaR
    var = np.percentile(data, confidence_level * 100)
    cvar = data[data <= var].mean()

    # Calculate tail dependence (probability of extreme losses)
    extreme_threshold = np.percentile(data, confidence_level * 100)
    tail_dependence = (data <= extreme_threshold).mean()

    # Calculate expected shortfall
    expected_shortfall = data[data <= var].mean()

    # Calculate tail risk ratio (ratio of tail risk to overall risk)
    overall_std = data.std()
    tail_std = data[data <= var].std()
    tail_risk_ratio = tail_std / overall_std if overall_std > 0 else 0.0

    return {
        "var": var,
        "cvar": cvar,
        "tail_dependence": tail_dependence,
        "expected_shortfall": expected_shortfall,
        "tail_risk_ratio": tail_risk_ratio,
    }


def calculate_correlation_regime(
    data1: Union[pd.Series, np.ndarray],
    data2: Union[pd.Series, np.ndarray],
    window: int = 60,
) -> pd.Series:
    """
    Calculate rolling correlation and classify into regimes.

    Args:
        data1: First data series
        data2: Second data series
        window: Rolling window size

    Returns:
        Series of correlation regimes
    """
    if isinstance(data1, np.ndarray):
        data1 = pd.Series(data1)
    if isinstance(data2, np.ndarray):
        data2 = pd.Series(data2)

    # Calculate rolling correlation
    rolling_corr = data1.rolling(window=window).corr(data2)

    # Classify into regimes
    def classify_correlation(corr):
        if pd.isna(corr):
            return "unknown"
        elif corr > 0.7:
            return "high_positive"
        elif corr > 0.3:
            return "moderate_positive"
        elif corr > -0.3:
            return "low"
        elif corr > -0.7:
            return "moderate_negative"
        else:
            return "high_negative"

    return rolling_corr.apply(classify_correlation)


def calculate_volatility_regime(
    data: Union[pd.Series, np.ndarray], window: int = 60, regimes: int = 3
) -> pd.Series:
    """
    Calculate volatility regimes using clustering.

    Args:
        data: Input data series (typically returns)
        window: Rolling window size
        regimes: Number of volatility regimes

    Returns:
        Series of volatility regimes
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    # Calculate rolling volatility
    rolling_vol = data.rolling(window=window).std() * np.sqrt(252)

    # Remove NaN values
    rolling_vol_clean = rolling_vol.dropna()

    if len(rolling_vol_clean) == 0:
        return pd.Series(index=data.index, dtype=str)

    # Use quantiles to classify regimes
    quantiles = np.linspace(0, 1, regimes + 1)
    thresholds = rolling_vol_clean.quantile(quantiles)

    def classify_volatility(vol):
        if pd.isna(vol):
            return "unknown"
        for i in range(regimes):
            if vol <= thresholds.iloc[i + 1]:
                return f"regime_{i + 1}"
        return f"regime_{regimes}"

    return rolling_vol.apply(classify_volatility)


def calculate_entropy(data: Union[pd.Series, np.ndarray], bins: int = 20) -> float:
    """
    Calculate Shannon entropy of a data series.

    Args:
        data: Input data series
        bins: Number of bins for histogram

    Returns:
        Shannon entropy value
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    # Remove NaN values
    data = data.dropna()

    if len(data) == 0:
        return 0.0

    # Calculate histogram
    hist, _ = np.histogram(data, bins=bins, density=True)

    # Remove zero probabilities
    hist = hist[hist > 0]

    # Calculate entropy
    entropy = -np.sum(hist * np.log2(hist))

    return entropy


def calculate_information_ratio(
    returns: Union[pd.Series, np.ndarray],
    benchmark_returns: Union[pd.Series, np.ndarray],
    window: int = 252,
) -> pd.Series:
    """
    Calculate rolling information ratio.

    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns
        window: Rolling window size

    Returns:
        Series of information ratios
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    if isinstance(benchmark_returns, np.ndarray):
        benchmark_returns = pd.Series(benchmark_returns)

    # Calculate excess returns
    excess_returns = returns - benchmark_returns

    # Calculate rolling information ratio
    rolling_mean = excess_returns.rolling(window=window).mean()
    rolling_std = excess_returns.rolling(window=window).std()

    information_ratio = rolling_mean / rolling_std

    return information_ratio


def calculate_treynor_ratio(
    returns: Union[pd.Series, np.ndarray],
    market_returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.02,
    window: int = 252,
) -> pd.Series:
    """
    Calculate rolling Treynor ratio.

    Args:
        returns: Portfolio returns
        market_returns: Market returns
        risk_free_rate: Risk-free rate
        window: Rolling window size

    Returns:
        Series of Treynor ratios
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    if isinstance(market_returns, np.ndarray):
        market_returns = pd.Series(market_returns)

    # Calculate excess returns
    excess_returns = returns - (risk_free_rate / 252)

    def rolling_beta(x):
        if len(x) < 2:
            return 1.0
        portfolio_ret = x.iloc[:, 0]
        market_ret = x.iloc[:, 1]
        covariance = portfolio_ret.cov(market_ret)
        market_variance = market_ret.var()
        return covariance / market_variance if market_variance > 0 else 1.0

    # Combine returns for rolling calculation
    combined = pd.concat([excess_returns, market_returns], axis=1)
    rolling_betas = combined.rolling(window=window).apply(rolling_beta)

    # Calculate Treynor ratio
    rolling_mean = excess_returns.rolling(window=window).mean()
    treynor_ratio = rolling_mean / rolling_betas

    return treynor_ratio


def calculate_jensen_alpha(
    returns: Union[pd.Series, np.ndarray],
    market_returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.02,
    window: int = 252,
) -> pd.Series:
    """
    Calculate rolling Jensen's alpha.

    Args:
        returns: Portfolio returns
        market_returns: Market returns
        risk_free_rate: Risk-free rate
        window: Rolling window size

    Returns:
        Series of Jensen's alpha values
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    if isinstance(market_returns, np.ndarray):
        market_returns = pd.Series(market_returns)

    # Calculate excess returns
    portfolio_excess = returns - (risk_free_rate / 252)
    market_excess = market_returns - (risk_free_rate / 252)

    def rolling_alpha(x):
        if len(x) < 2:
            return 0.0
        portfolio_ret = x.iloc[:, 0]
        market_ret = x.iloc[:, 1]

        # Calculate beta
        covariance = portfolio_ret.cov(market_ret)
        market_variance = market_ret.var()
        beta = covariance / market_variance if market_variance > 0 else 1.0

        # Calculate alpha
        alpha = portfolio_ret.mean() - beta * market_ret.mean()
        return alpha

    # Combine returns for rolling calculation
    combined = pd.concat([portfolio_excess, market_excess], axis=1)
    alpha = combined.rolling(window=window).apply(rolling_alpha)

    return alpha


def calculate_sortino_ratio(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.02,
    window: int = 252,
) -> pd.Series:
    """
    Calculate rolling Sortino ratio.

    Args:
        returns: Portfolio returns
        risk_free_rate: Risk-free rate
        window: Rolling window size

    Returns:
        Series of Sortino ratios
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)

    # Calculate excess returns
    excess_returns = returns - (risk_free_rate / 252)

    def rolling_sortino(x):
        if len(x) < 2:
            return 0.0

        # Calculate downside deviation
        downside_returns = x[x < 0]
        if len(downside_returns) == 0:
            return float("inf") if x.mean() > 0 else 0.0

        downside_deviation = downside_returns.std()
        if downside_deviation == 0:
            return float("inf") if x.mean() > 0 else 0.0

        return x.mean() / downside_deviation

    sortino_ratio = excess_returns.rolling(window=window).apply(rolling_sortino)

    return sortino_ratio


def calculate_calmar_ratio(
    equity_curve: Union[pd.Series, np.ndarray], window: int = 252
) -> pd.Series:
    """
    Calculate rolling Calmar ratio.

    Args:
        equity_curve: Portfolio equity curve
        window: Rolling window size

    Returns:
        Series of Calmar ratios
    """
    if isinstance(equity_curve, np.ndarray):
        equity_curve = pd.Series(equity_curve)

    def rolling_calmar(x):
        if len(x) < 2:
            return 0.0

        # Calculate returns
        returns = x.pct_change().dropna()
        if len(returns) == 0:
            return 0.0

        # Calculate annualized return
        annual_return = returns.mean() * 252

        # Calculate maximum drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = abs(drawdown.min())

        if max_drawdown == 0:
            return float("inf") if annual_return > 0 else 0.0

        return annual_return / max_drawdown

    calmar_ratio = equity_curve.rolling(window=window).apply(rolling_calmar)

    return calmar_ratio
