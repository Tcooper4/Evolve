# -*- coding: utf-8 -*-
"""Real multi-user concurrency checks for background_jobs + notification hub.

Findings this test guards (fixed 2026-07):
* Users are processed concurrently — a slow/failing peer must not
  head-of-line-block others in the same tick.
* A 5+ user tick with simulated network delay finishes well under 45s.
* Hub fan-out is username-scoped (all tabs for A, never B).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Tuple

import pytest

from trading.services.background_jobs import (
    DEFAULT_INTERVAL_SEC,
    background_tick,
)
from trading.services.notification_hub import NotificationHub


class FakeWS:
    """Minimal WebSocket stand-in for hub fan-out tests."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.messages: List[Dict[str, Any]] = []
        self.alive = True

    async def send_json(self, payload: Dict[str, Any]) -> None:
        if not self.alive:
            raise RuntimeError("closed")
        self.messages.append(dict(payload))


class TestMultiUserBackgroundConcurrency:
    def test_slow_and_failing_user_do_not_block_peers(self, monkeypatch):
        """Execution evidence: fast users finish while slow user is still running."""
        sessions = [
            "user:u0", "user:u1", "user:u2", "user:u3", "user:u4", "user:slow",
        ]
        monkeypatch.setattr(
            "trading.services.background_jobs.collect_target_session_ids",
            lambda: list(sessions),
        )

        timings: Dict[str, Tuple[float, float]] = {}

        def limits(uid: str):
            t0 = time.perf_counter()
            if uid.endswith("slow"):
                time.sleep(0.55)  # simulated network / quote lag
            elif uid.endswith("u2"):
                raise RuntimeError("deliberate limit check failure")
            else:
                time.sleep(0.08)  # realistic simulated per-user IO
            t1 = time.perf_counter()
            timings[uid] = (t0, t1)
            if uid.endswith("u1"):
                return [{
                    "id": "ord-u1",
                    "symbol": "SPY",
                    "side": "buy",
                    "quantity": 1,
                    "filled_price": 100.0,
                }]
            return []

        def alerts(sid: str):
            if sid.endswith("u3"):
                raise RuntimeError("deliberate alert check failure")
            time.sleep(0.05)
            if sid.endswith("u0"):
                return [{
                    "symbol": "QQQ",
                    "condition": "price_above",
                    "threshold": 1.0,
                    "alert_id": "a0",
                    "current_price": 2.0,
                    "mode": "action",
                }]
            return []

        published: List[Tuple[str, Dict[str, Any]]] = []

        async def capture(username: str, payload: Dict[str, Any]) -> None:
            published.append((username, payload))

        monkeypatch.setattr(
            "trading.services.background_jobs.run_limit_checks_for_user",
            limits,
        )
        monkeypatch.setattr(
            "trading.services.background_jobs.run_alert_checks_for_user",
            alerts,
        )
        monkeypatch.setattr(
            "trading.services.background_jobs._publish",
            capture,
        )
        # Avoid optional snapshot-logger side effects
        monkeypatch.setenv("EVOLVE_GEX_SNAPSHOT_LOG", "0")
        monkeypatch.setenv("EVOLVE_NEWS_SENTIMENT_SNAPSHOT_LOG", "0")
        monkeypatch.setenv("EVOLVE_SKEW_SNAPSHOT_LOG", "0")

        t_tick0 = time.perf_counter()
        stats = asyncio.run(background_tick())
        tick_elapsed = time.perf_counter() - t_tick0

        assert stats["users"] == 6
        assert stats["fills"] == 1
        assert stats["alerts"] == 1
        # Comfortably under 45s interval under simulated network delays
        assert tick_elapsed < DEFAULT_INTERVAL_SEC
        # Parallel: wall time ≪ sum of per-user times (~0.55+5*0.08+…)
        # Sequential would be ≥ ~0.55 + 5*0.08 + alerts ≈ 1.0s+ and
        # specifically ≥ slow alone + a peer. Require clear overlap.
        assert tick_elapsed < 1.2, (
            f"tick took {tick_elapsed:.3f}s — looks sequential/hogged"
        )

        # Fast users must finish before slow finishes (overlap evidence)
        slow = timings.get("user:slow")
        assert slow is not None
        fast_ends = [
            timings[k][1]
            for k in timings
            if k != "user:slow" and k in timings
        ]
        assert fast_ends, "expected peer timing samples"
        assert min(fast_ends) < slow[1], (
            "a peer did not finish before the slow user — possible HOL blocking"
        )
        # Slow user's window must overlap a fast user's window
        overlapped = False
        for k, (a0, a1) in timings.items():
            if k == "user:slow":
                continue
            if a0 < slow[1] and slow[0] < a1:
                overlapped = True
                break
        assert overlapped, "expected overlapping execution windows across users"

        # Notifications only to the owning username
        users = {u for u, _ in published}
        assert users <= {"u0", "u1"}
        assert all(
            (u == "u1" and p.get("type") == "limit_fill")
            or (u == "u0" and p.get("type") == "alert_trigger")
            for u, p in published
        )

    def test_hub_fans_out_to_same_user_tabs_never_cross_user(self):
        hub = NotificationHub()

        async def _run():
            a1, a2 = FakeWS("alice-tab1"), FakeWS("alice-tab2")
            b1 = FakeWS("bob-tab1")
            await hub.connect("alice", a1)
            await hub.connect("alice", a2)
            await hub.connect("bob", b1)

            n = await hub.publish("alice", {
                "type": "limit_fill",
                "symbol": "AAPL",
                "message": "Alice fill",
            })
            assert n == 2
            assert len(a1.messages) == 1 and len(a2.messages) == 1
            assert a1.messages[0]["symbol"] == "AAPL"
            assert a2.messages[0]["message"] == "Alice fill"
            assert b1.messages == [], "Bob must not receive Alice's fill"

            n2 = await hub.publish("bob", {
                "type": "alert_trigger",
                "symbol": "SPY",
                "message": "Bob alert",
            })
            assert n2 == 1
            assert len(b1.messages) == 1
            assert b1.messages[0]["symbol"] == "SPY"
            # Alice tabs unchanged
            assert len(a1.messages) == 1 and len(a2.messages) == 1

            await hub.disconnect("alice", a1)
            n3 = await hub.publish("alice", {"type": "ping"})
            assert n3 == 1
            assert len(a2.messages) == 2
            assert len(a1.messages) == 1  # disconnected tab gets nothing new

        asyncio.run(_run())

    def test_five_user_tick_under_interval_with_network_delay(self, monkeypatch):
        """Five users × ~0.35s simulated network work — parallel wall ≪ 45s."""
        users = [f"user:p{i}" for i in range(5)]
        monkeypatch.setattr(
            "trading.services.background_jobs.collect_target_session_ids",
            lambda: users,
        )
        monkeypatch.setenv("EVOLVE_GEX_SNAPSHOT_LOG", "0")
        monkeypatch.setenv("EVOLVE_NEWS_SENTIMENT_SNAPSHOT_LOG", "0")
        monkeypatch.setenv("EVOLVE_SKEW_SNAPSHOT_LOG", "0")

        def limits(uid: str):
            time.sleep(0.35)  # network-ish delay per user
            return []

        def alerts(sid: str):
            time.sleep(0.10)
            return []

        async def noop_publish(username: str, payload: Dict[str, Any]) -> None:
            return None

        monkeypatch.setattr(
            "trading.services.background_jobs.run_limit_checks_for_user",
            limits,
        )
        monkeypatch.setattr(
            "trading.services.background_jobs.run_alert_checks_for_user",
            alerts,
        )
        monkeypatch.setattr(
            "trading.services.background_jobs._publish",
            noop_publish,
        )

        t0 = time.perf_counter()
        stats = asyncio.run(background_tick())
        elapsed = time.perf_counter() - t0

        assert stats["users"] == 5
        # Sequential would be ≥ 5*(0.35+0.10) = 2.25s; parallel ~0.45s
        assert elapsed < 1.5, f"elapsed {elapsed:.3f}s suggests serialization"
        assert elapsed < DEFAULT_INTERVAL_SEC
        # Record measured time in assertion message for the Phase 2 report
        print(f"MEASURED_TICK_ELAPSED_SEC={elapsed:.4f}")
