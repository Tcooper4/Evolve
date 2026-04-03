"""
Monte Carlo price path simulation (single-asset, GBM).

Used for forecasting fan charts and risk intervals. Standard tool at quant shops.
"""

import numpy as np


def simulate_price_paths(
    last_price: float,
    mu: float,
    sigma: float,
    horizon_days: int = 30,
    n_simulations: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """
    Geometric Brownian Motion Monte Carlo simulation.

    Returns array of shape (n_simulations, horizon_days) of price paths.
    Standard tool for risk and valuation.

    Args:
        last_price: Current price (starting point).
        mu: Annualized mean return (e.g. 0.10 for 10%).
        sigma: Annualized volatility (e.g. 0.20 for 20%).
        horizon_days: Number of days to simulate.
        n_simulations: Number of paths.
        seed: Random seed for reproducibility.

    Returns:
        np.ndarray of shape (n_simulations, horizon_days).
    """
    np.random.seed(seed)
    dt = 1.0 / 252.0
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)
    Z = np.random.standard_normal((n_simulations, horizon_days))
    log_returns = drift + diffusion * Z
    price_paths = last_price * np.exp(np.cumsum(log_returns, axis=1))
    return price_paths


def fan_chart_percentiles(
    price_paths: np.ndarray,
    percentiles: tuple = (5, 25, 50, 75, 95),
) -> dict:
    """
    Compute percentile bands across days for a fan chart.

    Args:
        price_paths: (n_simulations, horizon_days).
        percentiles: Which percentiles to compute (e.g. 5, 25, 50, 75, 95).

    Returns:
        Dict mapping percentile (e.g. 5) to 1d array of length horizon_days.
    """
    out = {}
    for p in percentiles:
        out[p] = np.percentile(price_paths, p, axis=0)
    return out


def probability_above_below(
    price_paths: np.ndarray,
    level: float,
    day_index: int = -1,
) -> tuple:
    """
    Probability that price is above or below level at day_index (default last day).

    Returns:
        (prob_above, prob_below).
    """
    if day_index < 0:
        day_index = price_paths.shape[1] + day_index
    prices_at_day = price_paths[:, day_index]
    prob_above = float(np.mean(prices_at_day >= level))
    prob_below = float(np.mean(prices_at_day < level))
    return prob_above, prob_below
