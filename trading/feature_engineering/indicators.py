"""Common custom indicator functions for :mod:`feature_engineer`.

This module provides a collection of helper functions that can be
registered with :class:`FeatureEngineer` for additional feature
calculations. Custom indicators are kept here to keep
``feature_engineer.py`` focused on orchestration.
"""
from __future__ import annotations

import pandas as pd


def rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """Calculate a rolling z-score for a series."""
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return (series - mean) / std


# Example indicator that returns multiple columns
def price_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute high/low and close/open ratios."""
    out = pd.DataFrame(index=df.index)
    out["HL_RATIO"] = df["high"] / df["low"]
    out["CO_RATIO"] = df["close"] / df["open"]
    return out
