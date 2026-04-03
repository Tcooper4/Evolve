"""
Basic factor exposure model for attribution.

Explains what % of a stock's return came from momentum, reversal, volatility, volume trend.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd


def _close_col(df: pd.DataFrame):
    return df["Close"] if "Close" in df.columns else df["close"]

def _vol_col(df: pd.DataFrame):
    return df["Volume"] if "Volume" in df.columns else df["volume"]

STANDARD_FACTORS = {
    "momentum": lambda df: _close_col(df).pct_change(252),  # 12-month momentum
    "short_term_reversal": lambda df: _close_col(df).pct_change(5) * -1,  # 1-week reversal
    "volatility": lambda df: _close_col(df).pct_change().rolling(20).std() * np.sqrt(252),
    "volume_trend": lambda df: _vol_col(df).rolling(20).mean() / _vol_col(df).rolling(60).mean().replace(0, np.nan),
}


def compute_factor_exposures(
    df: pd.DataFrame,
    returns: pd.Series,
    window: int = 60,
) -> Dict[str, float]:
    """
    Regress returns on standard factor returns; return factor loadings (exposures).

    Args:
        df: OHLCV with Close, Volume.
        returns: Aligned daily returns (e.g. strategy or stock).
        window: Rolling window for regression.

    Returns:
        Dict factor_name -> coefficient (exposure).
    """
    if df is None or df.empty or ("Close" not in df.columns and "close" not in df.columns):
        return {k: 0.0 for k in STANDARD_FACTORS}
    if returns is None or returns.empty:
        return {k: 0.0 for k in STANDARD_FACTORS}

    out = {}
    try:
        factors_df = pd.DataFrame(index=df.index)
        for name, func in STANDARD_FACTORS.items():
            try:
                s = func(df)
                if s is not None and not s.empty:
                    factors_df[name] = s
            except Exception:
                pass
        factors_df = factors_df.dropna(how="all").replace([np.inf, -np.inf], np.nan).dropna()
        if factors_df.empty or len(factors_df) < window:
            return {k: 0.0 for k in STANDARD_FACTORS}
        common = returns.index.intersection(factors_df.index)
        if len(common) < 20:
            return {k: 0.0 for k in STANDARD_FACTORS}
        y = returns.reindex(common).dropna()
        X = factors_df.reindex(common).dropna()
        common = y.index.intersection(X.index)
        y = y.reindex(common).dropna()
        X = X.reindex(common).dropna()
        common = y.index.intersection(X.index)
        if len(common) < 20:
            return {k: 0.0 for k in STANDARD_FACTORS}
        y = y.loc[common]
        X = X.loc[common]
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression(fit_intercept=True)
        reg.fit(X, y)
        for i, name in enumerate(X.columns):
            out[name] = float(reg.coef_[i])
        for k in STANDARD_FACTORS:
            if k not in out:
                out[k] = 0.0
    except Exception:
        out = {k: 0.0 for k in STANDARD_FACTORS}
    return out


def factor_attribution_pct(
    df: pd.DataFrame,
    returns: pd.Series,
    window: int = 60,
) -> Dict[str, float]:
    """
    What % of total return (over window) is attributed to each factor (approximate).

    Returns dict factor -> contribution as fraction of total return (sum ≈ 1 or 0).
    """
    exposures = compute_factor_exposures(df, returns, window)
    total_ret = returns.tail(window).sum()
    if abs(total_ret) < 1e-12:
        return {k: 0.0 for k in exposures}
    try:
        factors_df = pd.DataFrame(index=df.index)
        for name, func in STANDARD_FACTORS.items():
            try:
                s = func(df)
                if s is not None and not s.empty:
                    factors_df[name] = s
            except Exception:
                pass
        factors_df = factors_df.dropna(how="all").replace([np.inf, -np.inf], np.nan).dropna()
        common = returns.index.intersection(factors_df.index).intersection(df.index)
        if len(common) < window:
            return {k: 0.0 for k in exposures}
        r = returns.reindex(common).dropna().tail(window)
        F = factors_df.reindex(common).dropna().tail(window)
        common = r.index.intersection(F.index)
        r = r.loc[common]
        F = F.loc[common]
        contributions = {}
        for name in F.columns:
            exp = exposures.get(name, 0.0)
            contributions[name] = float(exp * F[name].sum())
        total_contrib = sum(contributions.values())
        if abs(total_ret) < 1e-12:
            return {k: 0.0 for k in contributions}
        return {k: (v / total_ret) for k, v in contributions.items()}
    except Exception:
        return {k: 0.0 for k in STANDARD_FACTORS}
