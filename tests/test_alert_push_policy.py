# -*- coding: utf-8 -*-
"""Phase 3 — watch vs action mode + notification rate limit."""

from __future__ import annotations

import asyncio

import pytest

from trading.services.alert_push_policy import (
    MODE_ACTION,
    MODE_WATCH,
    AlertPushRateLimiter,
    alert_allows_push,
    normalize_alert_mode,
    should_push_alert_notification,
)


class TestAlertMode:
    def test_legacy_missing_mode_allows_push(self):
        assert alert_allows_push({"symbol": "SPY"}) is True
        assert normalize_alert_mode(None, default=MODE_ACTION) == MODE_ACTION

    def test_watch_blocks_push(self):
        assert alert_allows_push({"mode": MODE_WATCH}) is False
        assert should_push_alert_notification(
            "alice", {"symbol": "SPY", "mode": MODE_WATCH}
        ) == (False, "watch")

    def test_action_allows_push(self):
        lim = AlertPushRateLimiter(max_per_symbol_hour=5)
        ok, reason = should_push_alert_notification(
            "alice",
            {"symbol": "SPY", "mode": MODE_ACTION},
            limiter=lim,
            now=1_000.0,
        )
        assert ok is True and reason == "ok"


class TestPushRateLimit:
    def test_nth_push_suppressed_same_symbol(self):
        lim = AlertPushRateLimiter(max_per_symbol_hour=2, window_seconds=3600)
        row = {"symbol": "SPY", "mode": MODE_ACTION}
        assert should_push_alert_notification("u", row, limiter=lim, now=100)[0]
        assert should_push_alert_notification("u", row, limiter=lim, now=200)[0]
        ok, reason = should_push_alert_notification(
            "u", row, limiter=lim, now=300
        )
        assert ok is False and reason == "rate_limited"

    def test_other_symbol_independent(self):
        lim = AlertPushRateLimiter(max_per_symbol_hour=1, window_seconds=3600)
        assert should_push_alert_notification(
            "u", {"symbol": "SPY", "mode": "action"}, limiter=lim, now=1
        )[0]
        assert should_push_alert_notification(
            "u", {"symbol": "QQQ", "mode": "action"}, limiter=lim, now=2
        )[0]

    def test_window_expiry_allows_again(self):
        lim = AlertPushRateLimiter(max_per_symbol_hour=1, window_seconds=100)
        row = {"symbol": "IWM", "mode": MODE_ACTION}
        assert should_push_alert_notification("u", row, limiter=lim, now=0)[0]
        assert should_push_alert_notification("u", row, limiter=lim, now=50)[0] is False
        assert should_push_alert_notification("u", row, limiter=lim, now=101)[0]


class TestBackgroundSeparatesExecutionFromPush:
    def test_watch_alert_still_triggers_but_no_publish(self, monkeypatch):
        """Execution (one-shot) runs; WebSocket publish does not for watch."""
        store = {
            "evolve_alerts": [
                {
                    "id": "w1",
                    "symbol": "SPY",
                    "condition": "price_above",
                    "threshold": 100.0,
                    "mode": MODE_WATCH,
                    "status": "active",
                }
            ]
        }
        import config.user_store as us
        import trading.data.price_cache as pc
        from trading.utils.time_utils import MarketHours
        from trading.services.alert_checker import check_alerts_for_user

        monkeypatch.setattr(us, "load_user_preferences", lambda _s: dict(store))
        monkeypatch.setattr(
            us,
            "save_user_preferences",
            lambda _s, p: (store.clear(), store.update(p)),
        )
        monkeypatch.setattr(pc, "get_quote", lambda _s: {"price": 105.0})
        monkeypatch.setattr(MarketHours, "is_market_open", lambda self, dt=None: True)

        fired = check_alerts_for_user("user:alice")
        assert len(fired) == 1
        assert fired[0]["mode"] == MODE_WATCH
        assert store["evolve_alerts"][0]["status"] == "triggered"

        published = []

        async def _fake_publish(username, payload):
            published.append((username, payload))

        monkeypatch.setattr(
            "trading.services.background_jobs._publish", _fake_publish
        )
        monkeypatch.setattr(
            "trading.services.background_jobs.collect_target_session_ids",
            lambda: ["user:alice"],
        )
        monkeypatch.setattr(
            "trading.services.background_jobs.run_limit_checks_for_user",
            lambda _u: [],
        )
        monkeypatch.setattr(
            "trading.services.background_jobs.run_alert_checks_for_user",
            lambda _s: fired,
        )

        from trading.services.background_jobs import background_tick

        stats = asyncio.run(background_tick())
        assert stats["alerts"] == 1  # execution counted
        assert published == []  # no push for watch

    def test_rate_limit_suppresses_push_not_execution_count(self, monkeypatch):
        lim = AlertPushRateLimiter(max_per_symbol_hour=1, window_seconds=3600)
        monkeypatch.setattr(
            "trading.services.alert_push_policy.alert_push_rate_limiter", lim
        )

        rows = [
            {"symbol": "SPY", "mode": MODE_ACTION, "condition": "price_above",
             "threshold": 1, "alert_id": "a1"},
            {"symbol": "SPY", "mode": MODE_ACTION, "condition": "price_below",
             "threshold": 2, "alert_id": "a2"},
        ]
        published = []

        async def _fake_publish(username, payload):
            published.append(payload)

        monkeypatch.setattr(
            "trading.services.background_jobs._publish", _fake_publish
        )
        monkeypatch.setattr(
            "trading.services.background_jobs.collect_target_session_ids",
            lambda: ["user:bob"],
        )
        monkeypatch.setattr(
            "trading.services.background_jobs.run_limit_checks_for_user",
            lambda _u: [],
        )
        monkeypatch.setattr(
            "trading.services.background_jobs.run_alert_checks_for_user",
            lambda _s: rows,
        )

        from trading.services.background_jobs import background_tick

        stats = asyncio.run(background_tick())
        assert stats["alerts"] == 2
        assert len(published) == 1  # 2nd same-symbol push rate-limited
