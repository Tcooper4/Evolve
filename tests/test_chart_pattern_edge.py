# -*- coding: utf-8 -*-
"""Tests for chart-pattern edge adapter (Phase 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading.research.chart_pattern_edge import (
    PATTERN_SETS,
    PREDECLARED_TRIALS,
    SHUFFLE_SEED,
    TRIAL_JUSTIFICATION,
    compute_pattern_signal_series,
    shuffle_signal_dates,
)


class TestTrialLock:
    def test_six_predeclared_cells(self):
        assert len(PREDECLARED_TRIALS) == 6
        labels = {t.label for t in PREDECLARED_TRIALS}
        assert "classic_reversal_h5" in labels
        assert "triangles_h20" in labels
        assert "locked" in TRIAL_JUSTIFICATION.lower()
        assert SHUFFLE_SEED == 42

    def test_classic_set_is_literature_core(self):
        rev = PATTERN_SETS["classic_reversal"]
        assert "Head and Shoulders" in rev
        assert "Double Bottom" in rev
        assert "Golden Cross" not in rev  # MA crosses excluded from Phase 2


class TestShuffleControl:
    def test_preserves_event_count_and_signs(self):
        idx = pd.bdate_range("2020-01-02", periods=100)
        sig = pd.Series(np.nan, index=idx)
        sig.iloc[10] = 1.0
        sig.iloc[20] = -1.0
        sig.iloc[40] = 1.0
        sh = shuffle_signal_dates(sig, horizon=5, seed=42)
        assert sh.dropna().shape[0] == 3
        assert sorted(sh.dropna().tolist()) == sorted([1.0, -1.0, 1.0])
        # Dates moved (very likely with seed 42)
        assert list(sh.dropna().index) != list(sig.dropna().index)

    def test_seed_is_deterministic(self):
        idx = pd.bdate_range("2020-01-02", periods=80)
        sig = pd.Series(np.nan, index=idx)
        sig.iloc[5] = -1.0
        sig.iloc[15] = 1.0
        a = shuffle_signal_dates(sig, horizon=5, seed=42)
        b = shuffle_signal_dates(sig, horizon=5, seed=42)
        pd.testing.assert_series_equal(a, b)


class TestCausalEmitUsesLag:
    def test_synthetic_detector_path_smoke(self, monkeypatch):
        """Inject a fake Pattern completing at slice end; emit after lag."""
        from trading.analysis import chart_pattern_detector as cpd
        from trading.analysis.chart_pattern_detector import Pattern

        n = 120
        idx = pd.bdate_range("2021-01-04", periods=n)
        close = np.linspace(100, 130, n) + np.sin(np.linspace(0, 8, n))
        hist = pd.DataFrame(
            {
                "Open": close,
                "High": close + 1,
                "Low": close - 1,
                "Close": close,
                "Volume": np.full(n, 1e6),
            },
            index=idx,
        )

        original_detect = cpd.ChartPatternDetector.detect_all

        def fake_detect(self):
            # Only fire once when slice is long enough
            if len(self.closes) < 90:
                self._patterns = []
                return {"patterns": [], "support_resistance": [], "trend": {}, "signals": []}
            # Pattern "completes" on last bar of this slice
            end = len(self.closes) - 1
            self._patterns = [
                Pattern(
                    name="Double Bottom",
                    pattern_type="bullish",
                    confidence=0.8,
                    start_idx=end - 20,
                    end_idx=end,
                    start_date=str(self.dates[end - 20].date()),
                    end_date=str(self.dates[end].date()),
                    description="test",
                )
            ]
            return {"patterns": [], "support_resistance": [], "trend": {}, "signals": []}

        monkeypatch.setattr(cpd.ChartPatternDetector, "detect_all", fake_detect)
        series = compute_pattern_signal_series(
            "SPY",
            hist,
            pattern_names=PATTERN_SETS["classic_reversal"],
            min_confidence=0.6,
            peak_lag=7,
            scan_step=5,
            min_bars=80,
        )
        monkeypatch.setattr(cpd.ChartPatternDetector, "detect_all", original_detect)
        fired = series.dropna()
        assert len(fired) >= 1
        assert set(fired.unique()).issubset({1.0, -1.0})
