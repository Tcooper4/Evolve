# -*- coding: utf-8 -*-
"""Hand-verifiable plain_language fields on technical outputs (Phase 1)."""

from __future__ import annotations

import pandas as pd

from tests.plain_language_test_utils import assert_plain_language_field
from trading.analysis.market_state import compose_market_state
from trading.analysis.options_structure_overlay import pick_options_structure
from trading.data.gamma_exposure import compute_gex_profile
from trading.data.options_skew import classify_skew_shape
from trading.optimization.deflated_sharpe import deflated_sharpe_ratio
from trading.portfolio.kelly_sample_disclosure import assess_kelly_sample


class TestGexPlainLanguage:
    def test_long_gamma_profile_has_plain_language(self):
        spot = 100.0
        calls = pd.DataFrame({"strike": [100.0], "openInterest": [300.0]})
        puts = pd.DataFrame({"strike": [100.0], "openInterest": [100.0]})
        out = compute_gex_profile(
            calls, puts, spot, expiry="2099-01-01", call_gammas=[0.02], put_gammas=[0.02]
        )
        assert "dampened" in out["regime"] or "pinning" in out["regime"]
        assert_plain_language_field(out["plain_language"])

    def test_short_gamma_profile_has_plain_language(self):
        spot = 100.0
        calls = pd.DataFrame({"strike": [100.0], "openInterest": [100.0]})
        puts = pd.DataFrame({"strike": [100.0], "openInterest": [200.0]})
        out = compute_gex_profile(
            calls, puts, spot, expiry="2099-01-01", call_gammas=[0.02], put_gammas=[0.02]
        )
        assert "amplified" in out["regime"]
        assert_plain_language_field(out["plain_language"])


class TestOptionsStructurePlainLanguage:
    def test_iron_condor_pick_plain_language(self):
        pick = pick_options_structure(regime_short="long_gamma")
        assert pick["rationale"]
        assert_plain_language_field(pick["plain_language"])

    def test_wait_pick_plain_language(self):
        pick = pick_options_structure(regime_short="short_gamma")
        assert_plain_language_field(pick["plain_language"])


class TestKellyPlainLanguage:
    def test_small_sample_plain_language(self):
        a = assess_kelly_sample(5, 0.9)
        assert a["sample_size_caveat"]
        assert_plain_language_field(a["plain_language"])

    def test_adequate_sample_plain_language(self):
        a = assess_kelly_sample(50, 0.55)
        assert_plain_language_field(a["plain_language"])


class TestDsrPlainLanguage:
    def test_high_dsr_plain_language(self):
        trials = [0.1, 0.2, 0.15, 0.05, 0.12, 0.18, 0.09, 0.11]
        out = deflated_sharpe_ratio(1.5, trials, n_obs=120)
        assert out is not None
        assert "survives the search" in out["interpretation"]
        assert_plain_language_field(out["plain_language"])

    def test_low_dsr_plain_language(self):
        trials = [0.1, 0.2, 0.15, 0.05, 0.12, 0.18, 0.09, 0.11]
        out = deflated_sharpe_ratio(0.05, trials, n_obs=120)
        assert out is not None
        assert_plain_language_field(out["plain_language"])


class TestSkewPlainLanguage:
    def test_put_smirk_shape_plain_language(self):
        out = classify_skew_shape(0.35, 0.25, 0.22)
        assert out["shape"] == "put_smirk"
        assert_plain_language_field(out["plain_language"])


class TestMarketStatePlainLanguage:
    def test_elevated_composite_plain_language(self):
        out = compose_market_state(
            gex_regime="short_gamma",
            event_severity=0.5,
            volatility_regime="high",
        )
        assert "short gamma" in out["label"]
        assert_plain_language_field(out["plain_language"])
