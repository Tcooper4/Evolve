# -*- coding: utf-8 -*-
"""Hand-verified series features for FFORMA-lite Phase 1.

get_series_features is descriptive only — no model selection. Series are
constructed so expected ranges are known a priori (same spirit as
tests/test_stub_sweep.py TestForecastRouterDetection).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.models.forecast_router import get_series_features


def _frame(close: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(close), freq="B")
    return pd.DataFrame({"Close": close.astype(float)}, index=idx)


class TestGetSeriesFeaturesShape:
    def test_keys_and_ranges_on_noise(self):
        rng = np.random.default_rng(0)
        df = _frame(100 + rng.normal(0, 1.0, 252))
        f = get_series_features(df)
        assert set(f) == {
            "trend_strength",
            "seasonality_strength",
            "noise_entropy",
            "volatility_regime",
            "data_length",
        }
        assert f["data_length"] == 252
        for k in ("trend_strength", "seasonality_strength", "noise_entropy"):
            assert 0.0 <= f[k] <= 1.0, k
        assert f["volatility_regime"] in {"low", "medium", "high", "insufficient"}

    def test_empty_and_short_series(self):
        empty = get_series_features(pd.DataFrame({"Close": []}))
        assert empty["data_length"] == 0
        assert empty["trend_strength"] == 0.0
        assert empty["volatility_regime"] == "insufficient"

        short = get_series_features(_frame(np.array([100.0, 101.0, 99.0])))
        assert short["data_length"] == 3
        assert short["volatility_regime"] == "insufficient"


class TestHandConstructedSeries:
    def test_pure_trend_high_strength_low_seasonality(self):
        # Exact geometric trend, zero noise → R² of log-linear fit = 1
        t = np.arange(252, dtype=float)
        close = 100.0 * np.exp(0.001 * t)
        f = get_series_features(_frame(close))
        assert f["trend_strength"] == pytest.approx(1.0, abs=1e-9)
        # Constant return path → weekly autocorr ≈ 0 (undefined/near-zero)
        assert f["seasonality_strength"] < 0.15
        # Degenerate / near-degenerate returns → very low entropy bins used
        assert f["noise_entropy"] < 0.35

    def test_pure_noise_low_trend(self):
        rng = np.random.default_rng(42)
        close = 100.0 + rng.normal(0.0, 2.0, 300)
        f = get_series_features(_frame(close))
        assert f["trend_strength"] < 0.15
        assert f["seasonality_strength"] < 0.35
        # Diffuse return histogram → high normalized entropy
        assert f["noise_entropy"] > 0.55

    def test_weekly_seasonal_high_seasonality_strength(self):
        # Same construction as stub-sweep seasonal detector test
        n = 252
        close = 100.0 + 3.0 * np.sin(2 * np.pi * np.arange(n) / 5.0)
        f = get_series_features(_frame(close))
        assert f["seasonality_strength"] > 0.30
        # No secular drift → weak trend R²
        assert f["trend_strength"] < 0.25

    def test_volatility_regime_high_at_end(self):
        # Long calm stretch, then a clearly louder tail — latest 21d RV in
        # the upper tercile of its own history.
        rng = np.random.default_rng(7)
        n_calm, n_loud = 320, 80
        rets = np.concatenate([
            rng.normal(0.0, 0.001, n_calm),
            rng.normal(0.0, 0.040, n_loud),
        ])
        close = 100.0 * np.exp(np.cumsum(rets))
        f = get_series_features(_frame(close))
        assert f["volatility_regime"] == "high"

    def test_volatility_regime_low_at_end(self):
        rng = np.random.default_rng(9)
        n_loud, n_calm = 320, 80
        rets = np.concatenate([
            rng.normal(0.0, 0.040, n_loud),
            rng.normal(0.0, 0.001, n_calm),
        ])
        close = 100.0 * np.exp(np.cumsum(rets))
        f = get_series_features(_frame(close))
        assert f["volatility_regime"] == "low"


class TestAnalyzeDataIncludesFeatures:
    def test_router_analyze_surfaces_continuous_keys(self):
        from trading.models.forecast_router import ForecastRouter

        rng = np.random.default_rng(1)
        df = _frame(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 120))))
        chars = ForecastRouter()._analyze_data(df)
        assert "trend_strength" in chars
        assert "seasonality_strength" in chars
        assert "noise_entropy" in chars
        assert "volatility_regime" in chars
        assert chars["data_length"] == chars["length"] == 120
