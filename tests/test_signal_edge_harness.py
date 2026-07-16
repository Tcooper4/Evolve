# -*- coding: utf-8 -*-
"""Hand-verifiable tests for the shared signal-edge harness (Phase 0)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.research.signal_edge_harness import (
    DSR_LIVE_THRESHOLD,
    MIN_OOS_OBS,
    TargetSpec,
    TrialSpec,
    forward_returns,
    observation_stats,
    purged_split_indices,
    recommend_live_gate,
    run_signal_edge_oos,
)


def _planted_history(
    n: int = 700,
    beta: float = 1.0,
    noise: float = 0.25,
    seed: int = 7,
) -> pd.DataFrame:
    """Close path where forward 1d return ≈ beta * planted_signal + noise."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    signal = rng.normal(0.0, 1.0, size=n)
    fwd = beta * signal + rng.normal(0.0, noise, size=n)
    close = np.empty(n, dtype=float)
    close[0] = 100.0
    for t in range(n - 1):
        close[t + 1] = close[t] * (1.0 + float(fwd[t]))
    return pd.DataFrame({"Close": close, "planted_signal": signal}, index=idx)


def _noise_history(n: int = 700, seed: int = 99) -> pd.DataFrame:
    """Signal independent of returns — pure noise."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    rets = rng.normal(0.0, 0.01, size=n)
    close = 100.0 * np.cumprod(1.0 + rets)
    signal = rng.normal(0.0, 1.0, size=n)
    return pd.DataFrame({"Close": close, "planted_signal": signal}, index=idx)


def _scale_signal(symbol: str, history: pd.DataFrame, params):
    scale = float(params.get("scale", 1.0))
    return history["planted_signal"].astype(float) * scale


PREDECLARED = (
    TrialSpec(params={"scale": 0.5}, label="half"),
    TrialSpec(params={"scale": 1.0}, label="unit"),
    TrialSpec(params={"scale": 1.5}, label="one_half"),
)
JUSTIFICATION = (
    "Harness self-test only: three scale multipliers on a synthetic score. "
    "Locked before inspecting DSR; not a real trading signal."
)


class TestPurgedSplitMatchesOptions:
    def test_matches_options_helper(self):
        from trading.backtesting.options_strategy_backtest import (
            _purged_split_indices as opt_split,
        )

        for n, purge, frac in ((200, 5, 0.60), (500, 37, 0.60), (80, 1, 0.5)):
            assert purged_split_indices(n, purge, frac) == opt_split(n, purge, frac)


class TestForwardReturnHand:
    def test_one_day_forward_exact(self):
        idx = pd.bdate_range("2024-01-02", periods=4)
        close = pd.Series([100.0, 110.0, 99.0, 108.9], index=idx)
        fwd = forward_returns(close, 1)
        assert fwd.iloc[0] == pytest.approx(0.10)
        assert fwd.iloc[1] == pytest.approx(-0.10)
        assert fwd.iloc[2] == pytest.approx(0.10)
        assert np.isnan(fwd.iloc[3])


class TestObservationStatsHand:
    def test_sharpe_mu_over_sd(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        st = observation_stats(vals)
        mu = vals.mean()
        sd = vals.std(ddof=1)
        assert st["n"] == 4
        assert st["sharpe"] == pytest.approx(mu / sd, rel=1e-4)
        assert st["win_rate"] == 1.0


class TestRecommendLiveGate:
    def test_requires_all_three(self):
        dsr_ok = {"deflated_sharpe": 0.96}
        assert recommend_live_gate(dsr_ok, 0.2, 20) is True
        assert recommend_live_gate(dsr_ok, 0.2, MIN_OOS_OBS - 1) is False
        assert recommend_live_gate(dsr_ok, -0.01, 20) is False
        assert recommend_live_gate({"deflated_sharpe": 0.94}, 0.2, 20) is False
        assert recommend_live_gate(None, 0.2, 20) is False
        assert DSR_LIVE_THRESHOLD == 0.95


class TestPlantedEdgeClearsGate:
    def test_known_edge_recommend_live(self):
        hist = _planted_history(n=800, beta=1.2, noise=0.2, seed=3)
        prices = {"SYN": hist}
        out = run_signal_edge_oos(
            _scale_signal,
            prices,
            signal_name="synthetic_planted_edge",
            trials=PREDECLARED,
            trial_justification=JUSTIFICATION,
            target=TargetSpec(kind="forward_return", horizon=1),
            universe=["SYN"],
            purge_bars=1,
        )
        assert out["success"] is True
        assert out["error"] is None
        assert out["n_trials"] == 3
        assert out["trial_justification"] == JUSTIFICATION
        assert "ordering_note" in out
        test = out["test"] or {}
        stats = test.get("stats") or {}
        assert int(stats.get("n") or 0) >= MIN_OOS_OBS
        assert float(stats.get("sharpe") or 0) > 0
        dsr = out.get("deflated_sharpe") or {}
        assert float(dsr.get("deflated_sharpe") or 0) >= DSR_LIVE_THRESHOLD
        assert out["recommend_live"] is True


class TestNoiseDoesNotClearGate:
    def test_pure_noise_not_live(self):
        hist = _noise_history(n=800, seed=11)
        prices = {"NOISE": hist}
        out = run_signal_edge_oos(
            _scale_signal,
            prices,
            signal_name="synthetic_pure_noise",
            trials=PREDECLARED,
            trial_justification=JUSTIFICATION,
            target=TargetSpec(kind="forward_return", horizon=1),
            universe=["NOISE"],
            purge_bars=1,
        )
        assert out["success"] is True
        assert out["recommend_live"] is False
        # Noise may still have a lucky champion sharpe; gate must stay closed.
        dsr = out.get("deflated_sharpe")
        if dsr is not None:
            assert float(dsr.get("deflated_sharpe") or 0) < DSR_LIVE_THRESHOLD or (
                (out.get("test") or {}).get("stats") or {}
            ).get("sharpe") in (None, 0) or float(
                ((out.get("test") or {}).get("stats") or {}).get("sharpe") or 0
            ) <= 0


class TestHitMissAndVolTargetsSmoke:
    def test_hit_miss_and_vol_run(self):
        hist = _planted_history(n=400, beta=0.9, noise=0.3, seed=5)
        prices = {"SYN": hist}

        hit = run_signal_edge_oos(
            _scale_signal,
            prices,
            signal_name="synthetic_hit_miss",
            trials=PREDECLARED,
            trial_justification=JUSTIFICATION,
            target=TargetSpec(kind="hit_miss", horizon=1, benchmark="zero"),
            universe=["SYN"],
        )
        assert hit["success"] is True
        assert hit["target"]["kind"] == "hit_miss"

        vol = run_signal_edge_oos(
            _scale_signal,
            prices,
            signal_name="synthetic_vol",
            trials=PREDECLARED,
            trial_justification=JUSTIFICATION,
            target=TargetSpec(kind="forward_realized_vol", horizon=5),
            universe=["SYN"],
        )
        assert vol["success"] is True
        assert vol["target"]["kind"] == "forward_realized_vol"


class TestJustificationRequired:
    def test_empty_justification_errors(self):
        hist = _noise_history(n=200, seed=1)
        out = run_signal_edge_oos(
            _scale_signal,
            {"X": hist},
            signal_name="bad",
            trials=PREDECLARED,
            trial_justification="   ",
            target=TargetSpec(kind="forward_return", horizon=1),
        )
        assert out["success"] is False
        assert "trial_justification" in (out.get("error") or "")
