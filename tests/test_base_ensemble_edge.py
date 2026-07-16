# -*- coding: utf-8 -*-
"""Tests for BASE ensemble edge adapter (Phase 1) — mocked forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading.research.base_ensemble_edge import (
    PREDECLARED_TRIALS,
    TRIAL_JUSTIFICATION,
    direction_from_consensus,
    make_bh_excess_weight_fn,
    make_direction_signal_fn,
)
from trading.research.signal_edge_harness import TargetSpec, TrialSpec, run_signal_edge_oos


class TestDirectionMapping:
    def test_bull_bear_neutral(self):
        assert direction_from_consensus({"direction": "BULLISH"}) == 1.0
        assert direction_from_consensus({"direction": "BEARISH"}) == -1.0
        assert direction_from_consensus({"direction": "NEUTRAL"}) == 0.0
        assert direction_from_consensus({"error": "x"}) == 0.0


class TestTrialLock:
    def test_four_predeclared_cells(self):
        assert len(PREDECLARED_TRIALS) == 4
        labels = {t.label for t in PREDECLARED_TRIALS}
        assert labels == {"h5_core3", "h5_base5", "h7_core3", "h7_base5"}
        assert "locked before" in TRIAL_JUSTIFICATION.lower()


class TestMockedEnsembleOOS:
    def test_planted_directional_edge_can_clear(self):
        """Mock consensus that knows the next-day return sign."""
        n = 500
        idx = pd.bdate_range("2018-01-02", periods=n)
        rng = np.random.default_rng(0)
        signal = rng.choice([-1.0, 1.0], size=n)
        fwd = 0.01 * signal + rng.normal(0, 0.002, size=n)
        close = np.empty(n)
        close[0] = 100.0
        for t in range(n - 1):
            close[t + 1] = close[t] * (1.0 + float(fwd[t]))
        hist = pd.DataFrame(
            {"Close": close, "Open": close, "High": close, "Low": close},
            index=idx,
        )
        truth = pd.Series(signal, index=idx)

        def forecast_fn(ctx, horizon, models):
            last = ctx.index[-1]
            d = float(truth.loc[last])
            return {"direction": "BULLISH" if d > 0 else "BEARISH", "error": None}

        from trading.research import base_ensemble_edge as be

        be._DIRECTION_CACHE.clear()
        trials = (
            TrialSpec(params={"horizon": 1, "models": "core3", "step_size": 5}, label="a"),
            TrialSpec(params={"horizon": 1, "models": "base5", "step_size": 5}, label="b"),
        )
        sfn = make_direction_signal_fn(forecast_fn=forecast_fn)
        result = run_signal_edge_oos(
            sfn,
            {"SYN": hist},
            signal_name="mock_base",
            trials=trials,
            trial_justification=TRIAL_JUSTIFICATION,
            target=TargetSpec(kind="hit_miss", horizon=1, benchmark="zero"),
            purge_bars=1,
            min_train_obs=10,
        )
        assert result["success"] is True
        assert result["recommend_live"] is True

    def test_bh_excess_weight_geometry(self):
        idx = pd.bdate_range("2020-01-02", periods=200)
        close = pd.Series(np.linspace(100, 120, 200), index=idx)
        hist = pd.DataFrame({"Close": close})

        def forecast_fn(ctx, horizon, models):
            return {
                "direction": "BULLISH" if len(ctx) % 2 == 0 else "BEARISH",
                "error": None,
            }

        from trading.research import base_ensemble_edge as be

        be._DIRECTION_CACHE.clear()
        fn = make_bh_excess_weight_fn(forecast_fn=forecast_fn)
        w = fn("SPY", hist, {"horizon": 5, "models": "core3", "step_size": 20})
        vals = w.dropna()
        assert set(np.unique(vals.to_numpy())) <= {0.0, -2.0}
