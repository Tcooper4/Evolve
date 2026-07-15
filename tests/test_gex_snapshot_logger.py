# -*- coding: utf-8 -*-
"""GEX snapshot logger + near_flip honesty (Phase 2 — no fake backtest)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from trading.data.gex_snapshot_logger import (
    NEAR_FLIP_BOUNDARY_NOTE,
    PURPOSE_DISCLOSURE,
    append_snapshot_row,
    backfill_pending_outcomes,
    count_snapshots,
    gex_snapshot_logging_enabled,
    log_daily_gex_snapshots,
    near_flip_distance_pct,
    snapshot_symbols,
)
from trading.data.gamma_exposure import (
    DATA_DISCLOSURE,
    NEAR_FLIP_PCT,
    compute_gex_profile,
)
from trading.analysis.options_structure_overlay import DISCLOSURE


class TestNearFlipIsDesignChoice:
    def test_boundary_constant_is_half_pct(self):
        assert NEAR_FLIP_PCT == pytest.approx(0.005)

    def test_hand_geometry_within_and_outside(self):
        # spot=100, flip=100.4 → dist=0.4% → near_flip
        # spot=100, flip=100.6 → dist=0.6% → not near by boundary alone
        assert near_flip_distance_pct(100.0, 100.4) == pytest.approx(0.004)
        assert near_flip_distance_pct(100.0, 100.6) == pytest.approx(0.006)
        assert near_flip_distance_pct(100.0, 100.4) <= NEAR_FLIP_PCT
        assert near_flip_distance_pct(100.0, 100.6) > NEAR_FLIP_PCT

    def test_regime_uses_named_boundary(self):
        # Net long via heavier calls; flip far from spot → long_gamma
        calls = pd.DataFrame({"strike": [100.0], "openInterest": [300.0]})
        puts = pd.DataFrame({"strike": [100.0], "openInterest": [100.0]})
        far = compute_gex_profile(
            calls, puts, 100.0, call_gammas=[0.02], put_gammas=[0.02]
        )
        assert far["regime_short"] == "long_gamma"
        assert far["near_flip_validated"] is False
        assert far["near_flip_pct"] == pytest.approx(NEAR_FLIP_PCT)

        # Force near_flip: spot within 0.5% of flip from a cross profile
        # Use flip ≈ 103.75 from existing hand case; spot=103.75 → near
        calls2 = pd.DataFrame({"strike": [105.0], "openInterest": [10.0]})
        puts2 = pd.DataFrame({
            "strike": [95.0, 100.0],
            "openInterest": [10.0, 5.0],
        })
        near = compute_gex_profile(
            calls2,
            puts2,
            103.75,
            call_gammas=[0.02],
            put_gammas=[0.01, 0.01],
        )
        assert near["regime_short"] == "near_flip"
        assert "design" in (near.get("near_flip_note") or "").lower() or (
            near["near_flip_validated"] is False
        )

    def test_disclosures_mention_unvalidated_boundary(self):
        assert "design" in DATA_DISCLOSURE.lower() or "0.5%" in DATA_DISCLOSURE
        assert "EVOLVE_GEX_SNAPSHOT_LOG" in DATA_DISCLOSURE
        assert "near_flip" in DISCLOSURE.lower()
        assert "EVOLVE_GEX_SNAPSHOT_LOG" in DISCLOSURE
        assert "future" in PURPOSE_DISCLOSURE.lower()
        assert "0.5%" in NEAR_FLIP_BOUNDARY_NOTE or "design" in NEAR_FLIP_BOUNDARY_NOTE


class TestGexSnapshotLogger:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("EVOLVE_GEX_SNAPSHOT_LOG", raising=False)
        assert gex_snapshot_logging_enabled() is False
        out = log_daily_gex_snapshots(fetch_gex=False)
        assert out["enabled"] is False
        assert out["logged"] == 0

    def test_append_and_backfill_hand_returns(self, monkeypatch, tmp_path):
        monkeypatch.setenv("EVOLVE_GEX_SNAPSHOT_LOG", "1")
        monkeypatch.setattr(
            "trading.data.gex_snapshot_logger._db_path",
            lambda: Path(tmp_path) / "gex_regime_snapshots.db",
        )
        assert gex_snapshot_logging_enabled() is True
        assert snapshot_symbols() == ["SPY"]

        d0 = date(2026, 7, 10)
        d1 = date(2026, 7, 13)  # next business day in synthetic index
        wrote = append_snapshot_row(
            as_of_date=d0,
            symbol="SPY",
            spot=100.0,
            net_gex=1.0,
            gamma_flip=100.3,
            regime_short="near_flip",
            structure_pick="wait_mixed",
        )
        assert wrote is True
        # Duplicate day ignored
        assert append_snapshot_row(
            as_of_date=d0,
            symbol="SPY",
            spot=100.0,
            net_gex=1.0,
            gamma_flip=100.3,
            regime_short="near_flip",
            structure_pick="wait_mixed",
        ) is False
        assert count_snapshots("SPY") == 1

        # Hand: close 100 → 103 next day → |ret|=0.03; H-L/close = 2/103
        idx = pd.to_datetime([d0, d1])
        hist = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [101.0, 104.0],
                "Low": [99.0, 102.0],
                "Close": [100.0, 103.0],
                "Volume": [1e6, 1e6],
            },
            index=idx,
        )
        n = backfill_pending_outcomes("SPY", hist)
        assert n == 1
        # Re-read via sqlite through a second append path check
        from trading.data import gex_snapshot_logger as mod

        with mod._connect() as conn:
            row = conn.execute(
                "SELECT next_day_abs_return, next_day_range_pct, structure_pick "
                "FROM gex_regime_snapshots WHERE symbol='SPY'"
            ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(0.03)
        assert row[1] == pytest.approx(2.0 / 103.0)
        assert row[2] == "wait_mixed"
