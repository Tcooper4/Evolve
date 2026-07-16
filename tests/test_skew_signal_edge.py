# -*- coding: utf-8 -*-
"""Tests for skew availability gate + snapshot logger (Phase 5)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from trading.data.skew_snapshot_logger import (
    append_snapshot_row,
    count_snapshots,
)
from trading.research.skew_signal_edge import (
    MIN_ROWS_PER_SYMBOL,
    TRIAL_JUSTIFICATION_WHEN_READY,
    audit_skew_snapshot_availability,
    run_skew_signal_oos_real,
)


class TestTrialLockWhenReady:
    def test_horizons_and_floor_locked(self):
        assert MIN_ROWS_PER_SYMBOL == 60
        assert "skew_diff" in TRIAL_JUSTIFICATION_WHEN_READY
        assert "butterfly_proxy" in TRIAL_JUSTIFICATION_WHEN_READY
        assert "horizon=5" in TRIAL_JUSTIFICATION_WHEN_READY
        assert "horizon=21" in TRIAL_JUSTIFICATION_WHEN_READY
        assert "moneyness" in TRIAL_JUSTIFICATION_WHEN_READY.lower()
        assert "delta" in TRIAL_JUSTIFICATION_WHEN_READY.lower()


class TestEmptyStoreDefers:
    def test_missing_db_defers(self, tmp_path, monkeypatch):
        missing = tmp_path / "skew_snapshots.db"
        monkeypatch.setattr(
            "trading.research.skew_signal_edge._db_path",
            lambda: missing,
        )
        monkeypatch.setattr(
            "trading.research.skew_signal_edge.count_snapshots",
            lambda symbol=None: 0,
        )
        audit = audit_skew_snapshot_availability()
        assert audit["enough_for_oos"] is False

        out = run_skew_signal_oos_real(
            out_path=str(tmp_path / "skew_signal_oos_real.json")
        )
        assert out["status"] == "insufficient_data"
        assert out["deferred"] is True
        assert out["recommend_live"] is False
        assert Path(out["wrote"]).is_file()


class TestSnapshotLoggerSmoke:
    def test_append_and_count(self, tmp_path, monkeypatch):
        db = tmp_path / "skew_snapshots.db"
        monkeypatch.setattr(
            "trading.data.skew_snapshot_logger._db_path",
            lambda: db,
        )
        assert count_snapshots() == 0
        ok = append_snapshot_row(
            as_of_date=date(2026, 7, 15),
            symbol="SPY",
            skew_diff=0.04,
            shape="put_smirk",
            put_otm_iv=0.22,
            call_otm_iv=0.18,
            atm_iv=0.17,
            butterfly_proxy=0.03,
        )
        assert ok is True
        assert count_snapshots("SPY") == 1
        assert (
            append_snapshot_row(
                as_of_date=date(2026, 7, 15),
                symbol="SPY",
                skew_diff=0.01,
                shape="flat",
                put_otm_iv=0.20,
                call_otm_iv=0.19,
                atm_iv=0.18,
                butterfly_proxy=0.015,
            )
            is False
        )
        assert count_snapshots("SPY") == 1
