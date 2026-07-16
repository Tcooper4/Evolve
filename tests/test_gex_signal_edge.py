# -*- coding: utf-8 -*-
"""Tests for GEX signal availability gate (Phase 3)."""

from __future__ import annotations

from pathlib import Path

from trading.research.gex_signal_edge import (
    MIN_ROWS_PER_SYMBOL,
    PREFERRED_ROWS_PER_SYMBOL,
    audit_gex_snapshot_availability,
    run_gex_signal_oos_real,
)


class TestAvailabilityFloor:
    def test_floor_matches_logger_doc(self):
        assert MIN_ROWS_PER_SYMBOL == 60
        assert PREFERRED_ROWS_PER_SYMBOL == 120

    def test_empty_store_defers(self, tmp_path, monkeypatch):
        missing = tmp_path / "gex_regime_snapshots.db"
        monkeypatch.setattr(
            "trading.research.gex_signal_edge._db_path",
            lambda: missing,
        )
        monkeypatch.setattr(
            "trading.research.gex_signal_edge.count_snapshots",
            lambda symbol=None: 0,
        )
        audit = audit_gex_snapshot_availability()
        assert audit["db_exists"] is False
        assert audit["total_rows"] == 0
        assert audit["enough_for_oos"] is False

        out = run_gex_signal_oos_real(out_path=str(tmp_path / "gex_signal_oos_real.json"))
        assert out["status"] == "insufficient_data"
        assert out["deferred"] is True
        assert out["recommend_live"] is False
        assert Path(out["wrote"]).is_file()
