# -*- coding: utf-8 -*-
"""Phase 1 — price alerts fire once until explicit re-arm."""

from __future__ import annotations

from trading.services.alert_checker import (
    STATUS_ACTIVE,
    STATUS_TRIGGERED,
    check_alerts_for_user,
    is_alert_armed,
    mark_alert_triggered,
    rearm_alert_for_user,
    rearm_alert_record,
)


class TestAlertRecordHelpers:
    def test_legacy_missing_status_is_armed(self):
        assert is_alert_armed({"id": "a", "symbol": "SPY"}) is True

    def test_triggered_not_armed(self):
        assert is_alert_armed({"status": STATUS_TRIGGERED}) is False

    def test_mark_and_rearm_roundtrip(self):
        base = {
            "id": "x",
            "symbol": "QQQ",
            "condition": "price_above",
            "threshold": 100,
        }
        fired = mark_alert_triggered(base, price=101.5, at="2026-07-14T12:00:00Z")
        assert fired["status"] == STATUS_TRIGGERED
        assert fired["triggered_price"] == 101.5
        assert fired["triggered_at"] == "2026-07-14T12:00:00Z"
        rearms = rearm_alert_record(fired)
        assert rearms["status"] == STATUS_ACTIVE
        assert "triggered_at" not in rearms
        assert "triggered_price" not in rearms


def _patch_alert_store(monkeypatch, store):
    import config.user_store as us
    import trading.data.price_cache as pc
    from trading.utils.time_utils import MarketHours

    def load(_sid):
        return dict(store)

    def save(_sid, prefs):
        store.clear()
        store.update(prefs)

    monkeypatch.setattr(us, "load_user_preferences", load)
    monkeypatch.setattr(us, "save_user_preferences", save)
    # Price alerts skip when the session is closed — force open for unit tests.
    monkeypatch.setattr(MarketHours, "is_market_open", lambda self, dt=None: True)
    return pc


class TestCheckAlertsOneShot:
    def test_fires_once_while_price_stays_past_threshold(self, monkeypatch):
        store = {
            "evolve_alerts": [
                {
                    "id": "a1",
                    "symbol": "SPY",
                    "condition": "price_above",
                    "threshold": 100.0,
                    "status": "active",
                }
            ]
        }
        pc = _patch_alert_store(monkeypatch, store)
        monkeypatch.setattr(pc, "get_quote", lambda _s: {"price": 105.0})

        first = check_alerts_for_user("user:test")
        assert len(first) == 1
        assert first[0]["alert_id"] == "a1"
        assert first[0]["status"] == STATUS_TRIGGERED
        assert store["evolve_alerts"][0]["status"] == STATUS_TRIGGERED
        assert store["evolve_alerts"][0]["triggered_price"] == 105.0

        second = check_alerts_for_user("user:test")
        assert second == []
        assert store["evolve_alerts"][0]["status"] == STATUS_TRIGGERED

        ok, alerts, err = rearm_alert_for_user("user:test", "a1")
        assert ok is True and err is None
        assert alerts[0]["status"] == STATUS_ACTIVE

        third = check_alerts_for_user("user:test")
        assert len(third) == 1
        assert third[0]["alert_id"] == "a1"

    def test_background_wrapper_shares_oneshot_store(self, monkeypatch):
        """background_jobs.run_alert_checks_for_user wraps the same checker."""
        store = {
            "evolve_alerts": [
                {
                    "id": "b1",
                    "symbol": "AAPL",
                    "condition": "price_below",
                    "threshold": 200.0,
                }
            ]
        }
        pc = _patch_alert_store(monkeypatch, store)
        monkeypatch.setattr(pc, "get_quote", lambda _s: {"price": 150.0})

        from trading.services.background_jobs import run_alert_checks_for_user

        via_bg = run_alert_checks_for_user("user:test")
        assert len(via_bg) == 1
        via_pull = check_alerts_for_user("user:test")
        assert via_pull == []


class TestRearmPersistence:
    def test_rearm_missing_alert_returns_error(self, monkeypatch):
        store = {"evolve_alerts": [{"id": "keep", "symbol": "SPY", "status": "triggered"}]}
        _patch_alert_store(monkeypatch, store)
        ok, alerts, err = rearm_alert_for_user("user:test", "nope")
        assert ok is False
        assert err == "alert not found"
        assert alerts[0]["id"] == "keep"