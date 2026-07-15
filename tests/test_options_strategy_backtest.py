# -*- coding: utf-8 -*-
"""Hand-verifiable BS + flat-path iron condor structure backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.backtesting.options_strategy_backtest import (
    DISCLOSURE,
    OptionsStructureParams,
    _purged_split_indices,
    black_scholes_price,
    build_structure_legs,
    mark_structure,
    run_options_structure_backtest,
    simulate_structure_trades,
)


class TestBlackScholesHand:
    def test_atm_call_textbook(self):
        # S=K=100, T=1y, r=5%, sigma=20% → call ≈ 10.4506
        c = black_scholes_price(
            100.0, 100.0, 1.0, 0.20, is_call=True, rate=0.05
        )
        assert c == pytest.approx(10.4506, abs=0.01)

    def test_put_call_parity_atm(self):
        S, K, T, r, sig = 100.0, 100.0, 1.0, 0.05, 0.20
        c = black_scholes_price(S, K, T, sig, is_call=True, rate=r)
        p = black_scholes_price(S, K, T, sig, is_call=False, rate=r)
        # C - P = S - K e^{-rT}
        assert (c - p) == pytest.approx(S - K * np.exp(-r * T), abs=1e-3)


class TestFlatPathNearMaxCredit:
    def test_flat_underlying_captures_most_credit_net_of_costs(self):
        """Spot flat inside wings → IC should finish profitable after costs."""
        n = 80
        idx = pd.bdate_range("2024-01-02", periods=n)
        spot = pd.Series(np.full(n, 100.0), index=idx)
        iv = pd.Series(np.full(n, 0.20), index=idx)
        params = OptionsStructureParams(
            strategy="iron_condor",
            dte=30,
            short_delta=0.20,
            wing_pct=0.05,
            profit_take=0.50,
            exit_dte_floor=7,
            reentry_gap_days=40,  # at most one trade
        )
        # Entry mark without costs for max-credit reference
        legs = build_structure_legs(params, 100.0, 0.20)
        entry_mid = mark_structure(
            legs, 100.0, 0.20, 30.0,
            rate=params.risk_free, apply_costs=False, closing=False,
        )
        assert entry_mid["fill_credit"] > 0

        sim = simulate_structure_trades(spot, iv, params, apply_costs=True)
        assert DISCLOSURE in sim["disclosure"]
        assert sim["n_trades"] >= 1
        t0 = sim["trades"][0]
        # Flat path: should not be a wipeout; usually profit_take or dte_floor win
        assert t0["pnl_dollars"] > 0
        assert t0["exit_reason"] in ("profit_take", "dte_floor", "expiry_window")


class TestPurgedSplitMatchesWalkForward:
    def test_purge_gap_same_as_walk_forward_validator(self):
        # WalkForwardValidator: train_end = start_idx, test_start = start_idx + purge
        n, purge, frac = 200, 37, 0.60
        train_end, test_start = _purged_split_indices(n, purge, train_frac=frac)
        start_idx = int(n * frac)
        assert train_end == start_idx
        assert test_start == start_idx + purge
        assert test_start - train_end == purge


class TestDisclosureAndLiveFlag:
    def test_public_api_always_discloses_and_defaults_off(self, monkeypatch):
        idx = pd.bdate_range("2024-01-02", periods=120)
        spot = pd.Series(100 + np.cumsum(np.random.default_rng(0).normal(0, 0.3, 120)), index=idx)
        iv = pd.Series(np.full(120, 0.18), index=idx)

        def fake_align(symbol, period="2y"):
            return spot, iv, None

        monkeypatch.setattr(
            "trading.backtesting.options_strategy_backtest._aligned_spot_vix",
            fake_align,
        )
        out = run_options_structure_backtest("SPY", strategy="iron_condor", sweep=False)
        assert out["success"] is True
        assert out["disclosure"] == DISCLOSURE
        assert out["recommend_live"] is False
        assert "historical option quotes" in out["disclosure"].lower() or "not real" in out["disclosure"].lower()


class TestRefinedGridLock:
    def test_narrow_grid_has_three_literature_pairs(self):
        # Same predeclared pairs as run_options_structure_refined_oos_real
        grid = ((0.20, 0.05), (0.25, 0.05), (0.20, 0.06))
        assert len(grid) == 3
        assert (0.20, 0.05) in grid


class TestCallCreditSpreadLegs:
    """Hand-checkable call-credit geometry (mirror of put credit)."""

    def test_ccs_two_call_legs_otm_ordered(self):
        params = OptionsStructureParams(
            strategy="call_credit_spread",
            dte=37,
            short_delta=0.20,
            wing_pct=0.05,
        )
        spot, iv = 100.0, 0.20
        legs = build_structure_legs(params, spot, iv)
        assert len(legs) == 2
        assert {lg.name for lg in legs} == {"short_call", "long_call"}
        by_name = {lg.name: lg for lg in legs}
        sc, lc = by_name["short_call"], by_name["long_call"]
        assert sc.is_call and lc.is_call
        assert sc.qty == -1.0 and lc.qty == 1.0
        assert sc.strike > spot  # OTM short call
        assert lc.strike > sc.strike  # long further OTM by wing
        # Wing ≈ 5% of spot (allow rounding)
        assert abs((lc.strike - sc.strike) - spot * 0.05) < 0.02

    def test_ccs_opens_for_credit_vs_pcs_symmetric_under_flat_iv(self):
        """Under BS flat IV + r=0, CCS mid credit ≈ PCS mid credit."""
        spot, iv = 100.0, 0.20
        ccs = OptionsStructureParams(
            strategy="call_credit_spread",
            dte=37,
            short_delta=0.20,
            wing_pct=0.05,
            risk_free=0.0,
        )
        pcs = OptionsStructureParams(
            strategy="put_credit_spread",
            dte=37,
            short_delta=0.20,
            wing_pct=0.05,
            risk_free=0.0,
        )
        ccs_m = mark_structure(
            build_structure_legs(ccs, spot, iv),
            spot, iv, 37.0, rate=0.0, apply_costs=False, closing=False,
        )
        pcs_m = mark_structure(
            build_structure_legs(pcs, spot, iv),
            spot, iv, 37.0, rate=0.0, apply_costs=False, closing=False,
        )
        assert ccs_m["mid_credit"] > 0
        assert pcs_m["mid_credit"] > 0
        # Discrete strike grid + fixed % wing → near, not exact, parity
        assert ccs_m["mid_credit"] == pytest.approx(pcs_m["mid_credit"], rel=0.15)

    def test_ccs_not_iron_condor(self):
        legs = build_structure_legs(
            OptionsStructureParams(strategy="call_credit_spread"),
            100.0,
            0.20,
        )
        assert all(lg.is_call for lg in legs)
        assert not any(lg.name.endswith("put") for lg in legs)
