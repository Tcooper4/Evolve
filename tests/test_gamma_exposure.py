# -*- coding: utf-8 -*-
"""Hand-verifiable tests for trading.data.gamma_exposure."""

from __future__ import annotations

import pandas as pd
import pytest

from trading.data.gamma_exposure import (
    DATA_DISCLOSURE,
    CONTRACT_MULTIPLIER,
    GEX_PCT_MOVE,
    compute_gex_profile,
    gamma_flip_point,
    gex_contribution,
)


def _hand_mag(gamma: float, oi: float, spot: float) -> float:
    """Same magnitude formula as the module — hand-checkable."""
    return gamma * oi * CONTRACT_MULTIPLIER * (spot ** 2) * GEX_PCT_MOVE


class TestGexContributionSign:
    def test_call_positive_put_negative_hand_math(self):
        # S=100, γ=0.05, OI=10, mult=100
        # mag = 0.05 * 10 * 100 * 10000 * 0.01 = 5000
        spot, gamma, oi = 100.0, 0.05, 10.0
        assert _hand_mag(gamma, oi, spot) == pytest.approx(5000.0)
        assert gex_contribution(gamma, oi, spot, is_call=True) == pytest.approx(5000.0)
        assert gex_contribution(gamma, oi, spot, is_call=False) == pytest.approx(-5000.0)


class TestNetGexAndRegime:
    def test_net_short_gamma_from_heavier_puts(self):
        """
        Constructed chain (spot=100), explicit gammas:

        Call K=100: γ=0.02, OI=100 → +0.02*100*100*10000*0.01 = +20_000
        Put  K=100: γ=0.02, OI=200 → −40_000
        Net = −20_000 → dealers net short gamma
        """
        spot = 100.0
        calls = pd.DataFrame({"strike": [100.0], "openInterest": [100.0]})
        puts = pd.DataFrame({"strike": [100.0], "openInterest": [200.0]})
        out = compute_gex_profile(
            calls, puts, spot,
            expiry="2099-01-01",
            call_gammas=[0.02],
            put_gammas=[0.02],
        )
        assert out["success"] is True
        assert out["net_gex"] == pytest.approx(-20_000.0)
        assert out["regime_short"] == "short_gamma"
        assert "amplified" in out["regime"]
        assert out["delayed_data"] is True
        assert DATA_DISCLOSURE in out["disclosure"]

    def test_net_long_gamma_from_heavier_calls(self):
        """
        Call K=100: γ=0.02, OI=300 → +60_000
        Put  K=100: γ=0.02, OI=100 → −20_000
        Net = +40_000 → dealers net long gamma
        """
        spot = 100.0
        calls = pd.DataFrame({"strike": [100.0], "openInterest": [300.0]})
        puts = pd.DataFrame({"strike": [100.0], "openInterest": [100.0]})
        out = compute_gex_profile(
            calls, puts, spot,
            call_gammas=[0.02],
            put_gammas=[0.02],
        )
        assert out["net_gex"] == pytest.approx(40_000.0)
        assert out["regime_short"] == "long_gamma"
        assert "dampened" in out["regime"] or "pinning" in out["regime"]


class TestGammaFlipHand:
    def test_flip_interpolates_cumulative_zero(self):
        """
        Strike GEX: 90→−100, 100→+50, 110→+100
        Cumsum: −100, −50, +50
        Crosses zero between 100 and 110:
          w = 50/(50+50)=0.5 → flip = 100 + 0.5*10 = 105
        """
        by = {90.0: -100.0, 100.0: 50.0, 110.0: 100.0}
        assert gamma_flip_point(by) == pytest.approx(105.0)

    def test_flip_from_profile_matches_hand(self):
        """
        Spot=100. Build per-strike signed GEX via explicit γ/OI so profile
        matches {95: −1000, 100: −500, 105: +2000}:

        Put  K=95:  want −1000 → mag=1000 → γ*OI*100*100^2*0.01=1000
             0.01 * OI * 100 * 10000 * 0.01 = 1000 → 0.01*OI*100 = 1000
             wait: γ*OI*100*10000*0.01 = γ*OI*10000
             want 1000 → γ*OI = 0.1. Use γ=0.01, OI=10 → 0.01*10*10000=1000 ✓

        Put  K=100: −500 → γ*OI*10000=500 → γ*OI=0.05 → γ=0.01, OI=5

        Call K=105: +2000 → γ*OI*10000=2000 → γ*OI=0.2 → γ=0.02, OI=10

        Cumsum: −1000, −1500, +500 → cross between 100 and 105
          w = 1500/(1500+500)=0.75 → flip = 100 + 0.75*5 = 103.75
        """
        spot = 100.0
        calls = pd.DataFrame({"strike": [105.0], "openInterest": [10.0]})
        puts = pd.DataFrame({
            "strike": [95.0, 100.0],
            "openInterest": [10.0, 5.0],
        })
        out = compute_gex_profile(
            calls, puts, spot,
            call_gammas=[0.02],
            put_gammas=[0.01, 0.01],
        )
        by = {row["strike"]: row["gex"] for row in out["gex_by_strike"]}
        assert by[95.0] == pytest.approx(-1000.0)
        assert by[100.0] == pytest.approx(-500.0)
        assert by[105.0] == pytest.approx(2000.0)
        assert out["gamma_flip"] == pytest.approx(103.75)
        assert out["net_gex"] == pytest.approx(500.0)


class TestPinCandidates:
    def test_top_pins_by_abs_gex(self):
        spot = 100.0
        # Three strikes with known |GEX|
        calls = pd.DataFrame({
            "strike": [100.0, 110.0],
            "openInterest": [50.0, 10.0],
        })
        puts = pd.DataFrame({
            "strike": [90.0],
            "openInterest": [80.0],
        })
        # γ=0.01 all → mag = 0.01*OI*10000 = 100*OI
        out = compute_gex_profile(
            calls, puts, spot,
            call_gammas=[0.01, 0.01],
            put_gammas=[0.01],
            top_pins=2,
        )
        pins = out["pin_candidates"]
        assert len(pins) == 2
        # |put 90|=8000, |call 100|=5000, |call 110|=1000 → top 90, 100
        assert pins[0]["strike"] == pytest.approx(90.0)
        assert pins[1]["strike"] == pytest.approx(100.0)


class TestDisclosureAlwaysPresent:
    def test_missing_spot_still_discloses(self):
        out = compute_gex_profile(
            pd.DataFrame(), pd.DataFrame(), spot=0.0,
        )
        assert out["success"] is False
        assert out["delayed_data"] is True
        assert "OPRA" in out["disclosure"] or "delayed" in out["disclosure"].lower()
