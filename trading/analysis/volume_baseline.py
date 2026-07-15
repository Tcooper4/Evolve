# -*- coding: utf-8 -*-
"""Robust volume baselines for spike detection.

The trailing 20-day *mean* is pulled up by a cluster of high-volume days,
silently weakening volume_ratio sensitivity exactly when the tape is already
turbulent (earnings clusters, macro shocks). Median and trimmed-mean
baselines are standard robust estimators of central tendency under outliers.

Default chosen via ``tests/test_volume_baseline_stress.py`` (trimmed_mean):
catches the moderate-news miss (ratio 1.91 → 2.09). On that suite median also
catches it but lifts FP by >5pp vs mean and fails the gate; trimmed_mean
lifts FP by ~3.3pp (0.033 → 0.066) and clears the +5pp gate. Absolute
|move|≥3% OR is available as an opt-in (``ABS_MOVE_OR_THRESHOLD``) but is
not the production default — alone it misses the 2.4% moderate event and
when stacked on trimmed_mean it adds further FP without helping that case.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd

BaselineMethod = Literal["mean", "median", "trimmed_mean"]

# Absolute |move| OR-branch (independent of volume-ratio distortion).
# 3% is above the ordinary 2% spike co-requirement — reserved for clearly
# sizeable sessions. On the stress suite it did not raise FP rate; it does
# not by itself catch the 2.4% moderate miss (needs baseline fix).
ABS_MOVE_OR_THRESHOLD = 0.03

# Drop this many highest-volume days from the window before averaging.
TRIMMED_MEAN_DROP_TOP = 2

# Stress-gated default (see module docstring / test_volume_baseline_stress).
DEFAULT_BASELINE_METHOD: BaselineMethod = "trimmed_mean"


def _window_stat(values: np.ndarray, method: BaselineMethod) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    if method == "mean":
        return float(np.mean(v))
    if method == "median":
        return float(np.median(v))
    if method == "trimmed_mean":
        if v.size <= TRIMMED_MEAN_DROP_TOP:
            return float(np.mean(v))
        trimmed = np.sort(v)[: v.size - TRIMMED_MEAN_DROP_TOP]
        return float(np.mean(trimmed))
    raise ValueError(f"unknown baseline method: {method}")


def rolling_volume_baseline(
    volume: pd.Series,
    *,
    window: int = 20,
    min_periods: int = 5,
    method: BaselineMethod = DEFAULT_BASELINE_METHOD,
) -> pd.Series:
    """Trailing robust baseline for each bar (includes the bar's own day
    in the window — same convention as the historical rolling mean)."""
    vol = volume.astype(float)
    if method == "mean":
        return vol.rolling(window, min_periods=min_periods).mean()
    if method == "median":
        return vol.rolling(window, min_periods=min_periods).median()

    def _apply(arr: np.ndarray) -> float:
        return _window_stat(arr, "trimmed_mean")

    return vol.rolling(window, min_periods=min_periods).apply(_apply, raw=True)


def prior_completed_baseline(
    volume: pd.Series,
    *,
    method: BaselineMethod = DEFAULT_BASELINE_METHOD,
    max_bars: int = 20,
    min_bars: int = 5,
) -> Optional[float]:
    """Baseline from a completed-bars series (today already excluded)."""
    if volume is None or len(volume) == 0:
        return None
    window = volume.astype(float).tail(max_bars)
    if len(window) < min_bars:
        return None
    val = _window_stat(window.to_numpy(), method)
    if not np.isfinite(val) or val <= 0:
        return None
    return float(val)


def volume_ratio_series(
    volume: pd.Series,
    *,
    window: int = 20,
    min_periods: int = 5,
    method: BaselineMethod = DEFAULT_BASELINE_METHOD,
) -> pd.Series:
    base = rolling_volume_baseline(
        volume, window=window, min_periods=min_periods, method=method
    )
    return volume.astype(float) / base.clip(lower=1.0)
