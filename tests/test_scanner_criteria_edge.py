# -*- coding: utf-8 -*-
"""Tests for scanner-criteria edge adapter (Phase 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading.research.scanner_criteria_edge import (
    CONTROL_SEED,
    PREDECLARED_TRIALS,
    TRIAL_JUSTIFICATION,
    compute_pass_series,
    control_panel_from_real,
    passes_quick_technical_at_end,
)


class TestTrialLock:
    def test_six_predeclared_cells(self):
        assert len(PREDECLARED_TRIALS) == 6
        labels = {t.label for t in PREDECLARED_TRIALS}
        assert "qs6.0_h5" in labels
        assert "qs6.5_h20" in labels
        assert "locked before" in TRIAL_JUSTIFICATION.lower()
        assert CONTROL_SEED == 42


class TestQuickTechnicalGate:
    def test_short_series_fails(self):
        assert passes_quick_technical_at_end(np.array([1.0, 2.0]), None) is False

    def test_compute_pass_series_smoke(self):
        n = 100
        idx = pd.bdate_range("2022-01-03", periods=n)
        # Mild uptrend with noise — may or may not pass; just ensure 0/1 shape
        rng = np.random.default_rng(0)
        close = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, n))
        hist = pd.DataFrame(
            {"Close": close, "Volume": np.full(n, 1e6)},
            index=idx,
        )
        s = compute_pass_series(hist, min_quick_score=6.0, min_bars=40)
        fired = s.dropna()
        assert set(fired.unique()).issubset({0.0, 1.0})
        assert fired.index.min() >= idx[40]


class TestRandomControl:
    def test_matches_daily_pass_count(self):
        idx = pd.bdate_range("2023-01-02", periods=5)
        cols = ["A", "B", "C", "D"]
        real = pd.DataFrame(
            [
                [1, 0, 1, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 0],
                [np.nan, np.nan, np.nan, np.nan],
                [0, 1, 0, 0],
            ],
            index=idx,
            columns=cols,
            dtype=float,
        )
        ctrl = control_panel_from_real(real, seed=42)
        for dt in idx:
            r = real.loc[dt]
            c = ctrl.loc[dt]
            if not np.isfinite(r.to_numpy()).any():
                continue
            assert int(np.nansum(c == 1.0)) == int(np.nansum(r == 1.0))
