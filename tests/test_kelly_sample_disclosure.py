# -*- coding: utf-8 -*-
"""Phase 4 — Kelly small-sample / premium-selling disclosure."""

from __future__ import annotations

from trading.portfolio.kelly_sample_disclosure import (
    PREMIUM_SOFT_N,
    SMALL_SAMPLE_N,
    assess_kelly_sample,
    attach_kelly_recommendation,
)
from trading.services.agent_tools import get_position_size


class TestAssessKellySample:
    def test_five_wins_triggers_provisional_caveat(self):
        a = assess_kelly_sample(5, 1.0)  # all wins, tiny n
        assert a["sample_size_flag"] in ("insufficient", "provisional")
        assert a["n_closed_trades"] == 5
        assert a["sample_size_caveat"]
        assert "5" in a["sample_size_caveat"]
        assert a["recommend_quarter_kelly"] is True

    def test_large_sample_no_strong_insufficient_flag(self):
        a = assess_kelly_sample(50, 0.55)
        assert a["sample_size_flag"] == "adequate"
        assert a["recommend_quarter_kelly"] is False
        # Light history note optional under soft band only
        assert a["n_closed_trades"] == 50

    def test_threshold_boundary(self):
        below = assess_kelly_sample(SMALL_SAMPLE_N - 1, 0.70)
        at = assess_kelly_sample(SMALL_SAMPLE_N, 0.70)
        assert below["sample_size_flag"] in ("insufficient", "provisional")
        assert at["sample_size_flag"] == "adequate"
        assert below["recommend_quarter_kelly"] is True
        assert at["recommend_quarter_kelly"] is False

    def test_premium_selling_recommends_quarter_even_when_adequate(self):
        a = assess_kelly_sample(50, 0.72, defined_risk_premium_selling=True)
        assert a["recommend_quarter_kelly"] is True
        assert a["sample_size_caveat"]
        assert "quarter-Kelly" in a["sample_size_caveat"]

    def test_premium_soft_band(self):
        a = assess_kelly_sample(
            PREMIUM_SOFT_N - 1, 0.70, defined_risk_premium_selling=True
        )
        assert a["sample_size_flag"] == "provisional"


class TestGetPositionSizeDisclosure:
    def test_small_sample_all_wins_shapes_output(self):
        # High WR, tiny n — the iron-condor mirage case
        r = get_position_size(
            0.8,
            1.2,
            10_000,
            apply_vol_overlay=False,
            n_closed_trades=5,
        )
        assert r["success"] is True
        assert r["n_closed_trades"] == 5
        assert r["sample_size_flag"] in ("insufficient", "provisional")
        assert r["sample_size_caveat"]
        assert "provisional" in r["sample_size_caveat"].lower() or "too small" in r[
            "sample_size_caveat"
        ].lower()
        # Guided size is quarter-Kelly; classic half fields still present
        assert r["recommended_basis"] == "quarter_kelly"
        assert r["recommended_fraction"] == r["quarter_kelly_fraction"]
        assert r["half_kelly_fraction"] > r["recommended_fraction"]
        assert abs(r["half_kelly_fraction"] - 2 * r["recommended_fraction"]) < 1e-3

    def test_larger_sample_keeps_half_kelly_recommendation(self):
        r = get_position_size(
            0.55,
            1.5,
            10_000,
            apply_vol_overlay=False,
            n_closed_trades=40,
        )
        assert r["sample_size_flag"] == "adequate"
        assert r["recommended_basis"] == "half_kelly"
        assert abs(r["recommended_fraction"] - r["half_kelly_fraction"]) < 1e-9

    def test_premium_flag_quarter_kelly(self):
        r = get_position_size(
            0.70,
            1.3,
            20_000,
            apply_vol_overlay=False,
            n_closed_trades=40,
            defined_risk_premium_selling=True,
        )
        assert r["defined_risk_premium_selling"] is True
        assert r["recommended_basis"] == "quarter_kelly"
        assert "quarter_kelly_fraction" in r

    def test_attach_recommendation_pure(self):
        base = {
            "success": True,
            "full_kelly_fraction": 0.4,
            "half_kelly_fraction": 0.2,
            "half_kelly_dollars": 2000.0,
            "note": "Guide only.",
        }
        a = assess_kelly_sample(5, 0.8)
        out = attach_kelly_recommendation(base, a, 10_000)
        assert out["recommended_basis"] == "quarter_kelly"
        assert out["quarter_kelly_dollars"] == 1000.0
