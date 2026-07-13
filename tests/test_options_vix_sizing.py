# -*- coding: utf-8 -*-
"""Hand-verifiable tests for trading.portfolio.options_vix_sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.portfolio.options_vix_sizing import (
    ELEVATED_VIX_MULTIPLIER,
    LIVE_OPTIONS_VIX_SIZING_ENABLED,
    VIX_ABS_ELEVATED,
    VIX_PERCENTILE_ELEVATED,
    apply_kelly_options_vix_overlay,
    options_vix_multiplier,
    options_vix_multiplier_from_history,
    vix_trailing_percentile,
)


class TestMultiplierBands:
    def test_normal_vix_is_inert(self):
        # Low absolute VIX + low percentile → ×1.0
        info = options_vix_multiplier(vix=14.0, percentile=0.40)
        assert info["regime"] == "normal"
        assert info["multiplier"] == pytest.approx(1.0)
        assert info["scaled_down"] is False

    def test_elevated_percentile_scales_down(self):
        info = options_vix_multiplier(
            vix=22.0, percentile=VIX_PERCENTILE_ELEVATED + 0.01
        )
        assert info["regime"] == "elevated"
        assert info["multiplier"] == pytest.approx(ELEVATED_VIX_MULTIPLIER)
        assert info["scaled_down"] is True
        assert info["multiplier"] < 1.0

    def test_absolute_vix_band_scales_down(self):
        # Percentile "normal" but absolute VIX high
        info = options_vix_multiplier(
            vix=VIX_ABS_ELEVATED + 1.0, percentile=0.50
        )
        assert info["scaled_down"] is True
        assert info["multiplier"] == pytest.approx(ELEVATED_VIX_MULTIPLIER)

    def test_never_scales_above_one(self):
        info = options_vix_multiplier(
            vix=12.0, percentile=0.2, elevated_multiplier=1.5
        )
        # elevated_multiplier clamped to ≤1; and not elevated → 1.0
        assert info["multiplier"] <= 1.0
        info2 = options_vix_multiplier(
            vix=40.0, percentile=0.95, elevated_multiplier=1.5
        )
        assert info2["multiplier"] <= 1.0

    def test_boundary_percentile_inert_just_below(self):
        info = options_vix_multiplier(
            vix=20.0, percentile=VIX_PERCENTILE_ELEVATED - 1e-9
        )
        # Just below tercile and below abs band → inert
        if 20.0 < VIX_ABS_ELEVATED:
            assert info["multiplier"] == pytest.approx(1.0)


class TestFromHistory:
    def test_calm_then_spike_is_elevated(self):
        rng = np.random.default_rng(0)
        calm = rng.uniform(12, 16, 200)
        # Latest print clearly above the calm distribution
        series = np.concatenate([calm, [35.0]])
        info = options_vix_multiplier_from_history(series)
        assert info["vix"] == pytest.approx(35.0)
        assert info["scaled_down"] is True
        assert info["multiplier"] == pytest.approx(ELEVATED_VIX_MULTIPLIER)

    def test_percentile_hand_check(self):
        # 19 values of 10, then 20 → all 20 points ≤ 20 → percentile = 1.0
        s = pd.Series([10.0] * 19 + [20.0])
        stats = vix_trailing_percentile(s, lookback=20)
        assert stats["sufficient"] is True
        assert stats["percentile"] == pytest.approx(1.0)
        assert stats["vix"] == pytest.approx(20.0)


class TestKellyAttachment:
    def test_overlay_preserves_raw_kelly(self):
        kelly = {
            "success": True,
            "full_kelly_fraction": 0.4,
            "half_kelly_fraction": 0.2,
            "half_kelly_dollars": 2000.0,
            "note": "Guide only.",
        }
        vix = {
            "multiplier": 0.5,
            "regime": "elevated",
            "scaled_down": True,
            "vix": 30.0,
            "vix_percentile": 0.9,
            "reason": "cut",
        }
        out = apply_kelly_options_vix_overlay(kelly, vix)
        assert out["half_kelly_fraction"] == 0.2
        assert out["half_kelly_fraction_options_vix_adjusted"] == pytest.approx(0.1)
        assert out["half_kelly_dollars_options_vix_adjusted"] == pytest.approx(1000.0)
        assert out["options_vix_scaled_down"] is True

    def test_live_flag_off_in_phase3(self):
        assert LIVE_OPTIONS_VIX_SIZING_ENABLED is False
