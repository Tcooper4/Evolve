# -*- coding: utf-8 -*-
"""Tests for sentiment availability gate + snapshot logger (Phase 3)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from trading.data.news_sentiment_snapshot_logger import (
    append_snapshot_row,
    count_snapshots,
)
from trading.research.sentiment_edge import (
    MIN_ROWS_PER_SYMBOL,
    TRIAL_JUSTIFICATION_WHEN_READY,
    audit_sentiment_snapshot_availability,
    run_sentiment_oos_real,
)


class TestTrialLockWhenReady:
    def test_horizons_are_short_lived_only(self):
        assert MIN_ROWS_PER_SYMBOL == 30
        assert "1" in TRIAL_JUSTIFICATION_WHEN_READY
        assert "3" in TRIAL_JUSTIFICATION_WHEN_READY
        assert "long" not in TRIAL_JUSTIFICATION_WHEN_READY.lower() or "not a long" in TRIAL_JUSTIFICATION_WHEN_READY.lower()


class TestEmptyStoreDefers:
    def test_missing_db_defers(self, tmp_path, monkeypatch):
        missing = tmp_path / "news_sentiment_snapshots.db"
        monkeypatch.setattr(
            "trading.research.sentiment_edge._db_path",
            lambda: missing,
        )
        monkeypatch.setattr(
            "trading.research.sentiment_edge.count_snapshots",
            lambda symbol=None: 0,
        )
        audit = audit_sentiment_snapshot_availability()
        assert audit["enough_for_oos"] is False

        out = run_sentiment_oos_real(
            out_path=str(tmp_path / "sentiment_oos_real.json")
        )
        assert out["status"] == "insufficient_data"
        assert out["deferred"] is True
        assert out["recommend_live"] is False
        assert Path(out["wrote"]).is_file()


class TestSnapshotLoggerSmoke:
    def test_append_and_count(self, tmp_path, monkeypatch):
        db = tmp_path / "news_sentiment_snapshots.db"
        monkeypatch.setattr(
            "trading.data.news_sentiment_snapshot_logger._db_path",
            lambda: db,
        )
        assert count_snapshots() == 0
        ok = append_snapshot_row(
            as_of_date=date(2026, 7, 15),
            symbol="SPY",
            sentiment_score=0.25,
            sentiment_label="BULLISH",
            n_headlines=10,
            engine="finbert+vader",
        )
        assert ok is True
        assert count_snapshots("SPY") == 1
        # Duplicate day ignored
        assert (
            append_snapshot_row(
                as_of_date=date(2026, 7, 15),
                symbol="SPY",
                sentiment_score=0.1,
                sentiment_label="NEUTRAL",
                n_headlines=5,
                engine="vader",
            )
            is False
        )
        assert count_snapshots("SPY") == 1
