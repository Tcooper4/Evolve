# -*- coding: utf-8 -*-
"""Depth-pass tests for trading/backtesting/position_sizing.py.

The audit had verified only equal-weighted, Kelly, and risk-based sizing;
this pass executed all 22 methods directly (bypassing the dispatcher's
silent fall-back-to-equal-weighted exception handler, which had masked
every failure below) and fixed six verified bugs:

1. risk_parity weighted assets PROPORTIONALLY to volatility (inverse of
   risk parity) - and even that never ran, because `asset in ndarray`
   compared a string against float values and always fell back.
2. black_litterman subtracted the ANNUAL risk-free rate from a DAILY mean
   return, so the excess was negative for every real asset and the method
   returned 0.0 unconditionally.
3. mean_variance had the same daily-vs-annual mismatch inside its Sharpe
   objective, silently inverting the optimization into a
   volatility-maximizer.
4. All four scipy-based sizers (mean_variance, minimum_variance,
   maximum_diversification, risk_efficient) crashed into the fallback for
   any asset ALREADY HELD: [asset] + positions duplicated the name, so the
   weight vector's length disagreed with the deduplicated covariance
   matrix.
5. momentum_weighted scaled by mean DAILY return (±0.1%), a functional
   no-op; now uses the 20-day window return with a [0.5x, 1.5x] clamp.
6. regime_based compared mean daily return against ±1% thresholds written
   for window returns - an unreachable branch; now compares window return.
"""

import numpy as np
import pandas as pd
import pytest

from trading.backtesting.position_sizing import (
    PositionSizing,
    PositionSizingEngine,
)


def _series(mu, sig, seed, n=300):
    r = np.random.default_rng(seed)
    ret = r.normal(0, sig, n)
    ret = ret - ret.mean() + mu  # exact sample mean = mu
    return 100 * np.exp(np.cumsum(ret))


@pytest.fixture()
def data():
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "UP_CALM": _series(0.0010, 0.008, 1, n),
            "UP_WILD": _series(0.0010, 0.030, 2, n),
            "FLATISH": _series(0.0001, 0.012, 3, n),
            "DOWN": _series(-0.0008, 0.015, 4, n),
        },
        index=idx,
    )


@pytest.fixture()
def engine():
    return PositionSizingEngine(cash=100000.0)


POSITIONS = {"UP_WILD": 0.1, "FLATISH": 0.1, "DOWN": 0.1}


def _size(engine, data, method, asset, signal=1.0):
    fn = getattr(engine, f"_calculate_{method}_size")
    return fn(asset, float(data[asset].iloc[-1]), "S", signal, data,
              dict(POSITIONS))


class TestRiskParity:
    def test_weights_inverse_to_volatility(self, engine, data):
        # Query each asset against the same 4-asset universe (positions =
        # the other three), so weights are comparable across queries.
        weights = {}
        for a in data.columns:
            others = {o: 0.1 for o in data.columns if o != a}
            fn = engine._calculate_risk_parity_size
            weights[a] = fn(a, float(data[a].iloc[-1]), "S", 1.0, data, others)
        vols = data.pct_change().std()
        vol_order = list(vols.sort_values().index)          # calmest first
        weight_order = sorted(weights, key=weights.get, reverse=True)
        assert vol_order == weight_order, (
            "risk parity must weight calmer assets more"
        )

    def test_not_stuck_on_equal_weighted_fallback(self, engine, data):
        w = _size(engine, data, "risk_parity", "UP_CALM")
        assert abs(w - 0.1) > 1e-9  # 0.1 was the perpetual fallback value


class TestBlackLitterman:
    def test_positive_in_uptrend_zero_in_downtrend(self, engine, data):
        up = _size(engine, data, "black_litterman", "UP_CALM", signal=0.8)
        dn = _size(engine, data, "black_litterman", "DOWN", signal=0.8)
        assert up > 0, "uptrending asset with positive view must size > 0"
        assert dn == 0


class TestScipyOptimizers:
    def test_mean_variance_prefers_best_sharpe(self, engine, data):
        weights = {
            a: _size(engine, data, "mean_variance", a) for a in data.columns
        }
        sharpes = (data.pct_change().mean() - 0.02 / 252) / data.pct_change().std()
        assert max(weights, key=weights.get) == sharpes.idxmax()

    @pytest.mark.parametrize(
        "method",
        ["mean_variance", "minimum_variance", "maximum_diversification",
         "risk_efficient"],
    )
    def test_held_asset_does_not_crash_into_fallback(self, engine, data, method):
        """UP_WILD is already in POSITIONS; before the dedupe fix, every
        scipy sizer raised on the duplicated asset and silently returned
        the 0.1 equal-weighted fallback."""
        held = _size(engine, data, method, "UP_WILD")
        assert np.isfinite(held)
        # min-variance should give the wild asset well UNDER equal weight;
        # the others give data-driven weights. All should differ from the
        # exact fallback for this configuration.
        assert abs(held - 0.1) > 1e-9


class TestMomentumScale:
    def _ramp_data(self):
        n = 300
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        # Strong deterministic drift plus small noise: constant-return
        # ramps have zero variance (skew undefined -> regime falls back),
        # so noise keeps every regime indicator well-defined while the
        # trend still dominates the 20-day window return.
        r = np.random.default_rng(7)
        up = 100 * np.exp(np.cumsum(0.004 + r.normal(0, 0.0005, n)))
        down = 100 * np.exp(np.cumsum(-0.004 + r.normal(0, 0.0005, n)))
        flat = 100 * np.exp(np.cumsum(r.normal(0, 0.0005, n)))
        return pd.DataFrame({"UP": up, "DOWN": down, "FLAT": flat}, index=idx)

    def test_momentum_weighted_actually_tilts(self, engine):
        data = self._ramp_data()
        base = engine._calculate_equal_weighted_size(
            "UP", float(data["UP"].iloc[-1]), "S", 1.0, data, {"FLAT": 0.1})
        up = engine._calculate_momentum_weighted_size(
            "UP", float(data["UP"].iloc[-1]), "S", 1.0, data, {"FLAT": 0.1})
        dn = engine._calculate_momentum_weighted_size(
            "DOWN", float(data["DOWN"].iloc[-1]), "S", 1.0, data, {"FLAT": 0.1})
        assert up > base > dn
        # And bounded by the clamp.
        assert base * 0.5 - 1e-9 <= dn and up <= base * 1.5 + 1e-9

    def test_regime_momentum_branch_reachable(self, engine):
        data = self._ramp_data()
        up = engine._calculate_regime_based_size(
            "UP", float(data["UP"].iloc[-1]), "S", 1.0, data, {"FLAT": 0.1})
        dn = engine._calculate_regime_based_size(
            "DOWN", float(data["DOWN"].iloc[-1]), "S", 1.0, data, {"FLAT": 0.1})
        assert up > dn, "momentum regime adjustment must fire on real trends"


class TestTradeHistoryMethods:
    def _seed(self, engine, win_rate=0.6, n=40):
        r = np.random.default_rng(9)
        for _ in range(n):
            pnl = (abs(r.normal(120, 40)) if r.random() < win_rate
                   else -abs(r.normal(90, 30)))
            engine.add_trade({"asset": "UP_CALM", "strategy": "S",
                              "pnl": float(pnl), "position_size": 0.1})

    def test_kelly_positive_with_edge_and_half_kelly_halves(self, engine, data):
        self._seed(engine)
        k = _size(engine, data, "kelly", "UP_CALM")
        assert k > 0
        assert _size(engine, data, "half_kelly", "UP_CALM") == pytest.approx(k / 2)

    def test_martingale_doubles_after_loss(self, engine, data):
        engine.add_trade({"asset": "UP_CALM", "strategy": "S",
                          "pnl": -50.0, "position_size": 0.08})
        m = _size(engine, data, "martingale", "UP_CALM")
        assert m == pytest.approx(0.16)

    def test_anti_martingale_scales_after_win(self, engine, data):
        engine.add_trade({"asset": "UP_CALM", "strategy": "S",
                          "pnl": 75.0, "position_size": 0.08})
        m = _size(engine, data, "anti_martingale", "UP_CALM")
        assert m == pytest.approx(0.12)


class TestDispatcher:
    def test_every_enum_method_dispatches_and_clamps(self, engine, data):
        for method in PositionSizing:
            out = engine.calculate_position_size(
                method=method, asset="UP_CALM",
                price=float(data["UP_CALM"].iloc[-1]),
                strategy="S", signal=0.8, data=data,
                positions=dict(POSITIONS),
            )
            assert np.isfinite(out)
            assert 0 <= out <= 1.0
