# -*- coding: utf-8 -*-
"""Tests for AI Score availability gate (Phase 4)."""

from __future__ import annotations

from pathlib import Path

from trading.research.ai_score_edge import (
    GLOBAL_MIN_ROWS,
    MIN_ROWS_PER_SYMBOL,
    audit_ai_score_availability,
    run_ai_score_oos_real,
)


class TestAvailabilityFloor:
    def test_floors_match_ic_gates(self):
        assert MIN_ROWS_PER_SYMBOL == 30
        assert GLOBAL_MIN_ROWS == 100

    def test_missing_db_defers(self, tmp_path, monkeypatch):
        missing = tmp_path / "signal_scores.db"
        monkeypatch.setattr(
            "trading.research.ai_score_edge._db_path",
            lambda: missing,
        )
        audit = audit_ai_score_availability(run_backfill=False)
        assert audit["db_exists"] is False
        assert audit["enough_for_oos"] is False

        out = run_ai_score_oos_real(
            out_path=str(tmp_path / "ai_score_oos_real.json"),
            run_backfill=False,
        )
        assert out["status"] == "insufficient_data"
        assert out["deferred"] is True
        assert out["recommend_live"] is False
        assert Path(out["wrote"]).is_file()
