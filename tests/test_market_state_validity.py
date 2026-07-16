# -*- coding: utf-8 -*-
"""Tests for market-state validity gate (Phase 3) — availability first."""

from __future__ import annotations

from pathlib import Path

from trading.research.market_state_validity import (
    MIN_ROWS_GEX,
    TRIAL_JUSTIFICATION_WHEN_READY,
    audit_market_state_data_availability,
    run_market_state_validity_real,
)


class TestTrialLockWhenReady:
    def test_validity_not_directional(self):
        assert MIN_ROWS_GEX == 60
        assert "forward_realized_vol" in TRIAL_JUSTIFICATION_WHEN_READY
        assert "forward_abs_drawdown" in TRIAL_JUSTIFICATION_WHEN_READY
        assert "directional" in TRIAL_JUSTIFICATION_WHEN_READY.lower()
        assert "No directional return target" in TRIAL_JUSTIFICATION_WHEN_READY
        assert "elevated" in TRIAL_JUSTIFICATION_WHEN_READY
        assert "calm" in TRIAL_JUSTIFICATION_WHEN_READY


class TestEmptyStoresDefer:
    def test_zero_rows_defers(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "trading.research.market_state_validity.count_gex_snapshots",
            lambda: 0,
        )
        monkeypatch.setattr(
            "trading.research.market_state_validity.count_skew_snapshots",
            lambda: 0,
        )
        monkeypatch.setattr(
            "trading.research.market_state_validity.gex_db_path",
            lambda: tmp_path / "gex.db",
        )
        monkeypatch.setattr(
            "trading.research.market_state_validity.skew_db_path",
            lambda: tmp_path / "skew.db",
        )

        audit = audit_market_state_data_availability()
        assert audit["enough_for_validity_oos"] is False
        assert audit["gex"]["total_rows"] == 0
        assert audit["skew"]["total_rows"] == 0

        out = run_market_state_validity_real(
            out_path=str(tmp_path / "market_state_validity_real.json")
        )
        assert out["status"] == "insufficient_data"
        assert out["deferred"] is True
        assert out["recommend_live"] is False
        assert out["real_counts"]["gex_snapshot_rows"] == 0
        assert out["real_counts"]["skew_snapshot_rows"] == 0
        assert Path(out["wrote"]).is_file()
