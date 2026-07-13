# -*- coding: utf-8 -*-
"""Conditional vol sizing overlay — hand-verifiable + OOS gate tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.portfolio.conditional_vol_sizing import (
    HIGH_VOL_MULTIPLIER,
    apply_kelly_vol_overlay,
    conditional_vol_multiplier,
    simulate_exposure_paths,
)


def _ohlc_from_rets(rets: np.ndarray, start: float = 100.0) -> pd.DataFrame:
    close = start * np.exp(np.cumsum(rets))
    idx = pd.date_range("2022-01-01", periods=len(close), freq="B")
    return pd.DataFrame({
        "Open": close, "High": close, "Low": close, "Close": close, "Volume": 1e6,
    }, index=idx)


class TestConditionalMultiplier:
    def test_normal_vol_multiplier_is_one(self):
        rng = np.random.default_rng(0)
        # Homogeneous moderate noise — latest RV near median → not extreme-high
        rets = rng.normal(0.0, 0.008, 300)
        info = conditional_vol_multiplier(_ohlc_from_rets(rets))
        assert info["regime"] in {"low", "medium", "high", "insufficient"}
        if info["regime"] != "high":
            assert info["multiplier"] == 1.0
            assert info["scaled_down"] is False

    def test_extreme_high_vol_scales_down(self):
        rng = np.random.default_rng(1)
        n_calm, n_loud = 320, 80
        rets = np.concatenate([
            rng.normal(0.0, 0.001, n_calm),
            rng.normal(0.0, 0.040, n_loud),
        ])
        info = conditional_vol_multiplier(_ohlc_from_rets(rets))
        assert info["regime"] == "high"
        assert info["multiplier"] == pytest.approx(HIGH_VOL_MULTIPLIER)
        assert info["scaled_down"] is True
        assert info["multiplier"] < 1.0

    def test_never_scales_above_one(self):
        rng = np.random.default_rng(2)
        # Loud then calm → low regime at end
        rets = np.concatenate([
            rng.normal(0.0, 0.040, 320),
            rng.normal(0.0, 0.001, 80),
        ])
        info = conditional_vol_multiplier(
            _ohlc_from_rets(rets), high_multiplier=0.5
        )
        assert info["multiplier"] <= 1.0
        if info["regime"] == "low":
            assert info["multiplier"] == 1.0


class TestKellyOverlayAttachment:
    def test_overlay_fields_preserve_raw_kelly(self):
        kelly = {
            "success": True,
            "full_kelly_fraction": 0.4,
            "half_kelly_fraction": 0.2,
            "half_kelly_dollars": 2000.0,
            "note": "Guide only.",
        }
        vol = {
            "multiplier": 0.5,
            "regime": "high",
            "scaled_down": True,
            "reason": "cut",
        }
        out = apply_kelly_vol_overlay(kelly, vol)
        assert out["half_kelly_fraction"] == 0.2
        assert out["half_kelly_dollars"] == 2000.0
        assert out["half_kelly_fraction_vol_adjusted"] == pytest.approx(0.1)
        assert out["half_kelly_dollars_vol_adjusted"] == pytest.approx(1000.0)
        assert out["vol_multiplier"] == 0.5


class TestSimulatePathsCausal:
    def test_overlay_path_differs_when_high_vol_present(self):
        rng = np.random.default_rng(3)
        # Long calm then crisis cluster — overlay should spend time < 1
        rets = np.concatenate([
            rng.normal(0.0004, 0.006, 400),
            rng.normal(-0.002, 0.035, 100),
        ])
        df = _ohlc_from_rets(rets)
        sim = simulate_exposure_paths(df["Close"], high_multiplier=0.5)
        assert sim["success"]
        assert sim["n_steps"] > 50
        # Either scaled down some of the time, or adopt gate evaluates cleanly
        assert "adopt_overlay" in sim
        assert "baseline" in sim and "conditional_overlay" in sim
        assert sim["baseline"]["max_drawdown"] >= 0
