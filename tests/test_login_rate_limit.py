# -*- coding: utf-8 -*-
"""Hand-verifiable login rate-limit behavior."""

from __future__ import annotations

import pytest

from trading.auth.login_rate_limit import (
    BASE_COOLDOWN_SEC,
    MAX_FREE_FAILURES,
    LoginRateLimiter,
)


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += float(seconds)


@pytest.fixture()
def lim_clock():
    clock = FakeClock()
    lim = LoginRateLimiter(
        max_free_failures=5,
        base_cooldown_sec=15.0,
        cooldown_growth=2.0,
        max_cooldown_sec=900.0,
        clock=clock,
    )
    return lim, clock


class TestLoginRateLimiter:
    def test_free_failures_then_cooldown(self, lim_clock):
        lim, clock = lim_clock
        for i in range(MAX_FREE_FAILURES):
            d = lim.record_failure("alice", "1.2.3.4")
            assert d.allowed is True, f"failure {i+1} should still be free"
            assert lim.check("alice", "1.2.3.4").allowed is True

        # 6th consecutive failure arms cooldown
        d = lim.record_failure("alice", "1.2.3.4")
        assert d.allowed is False
        assert d.retry_after_sec == pytest.approx(BASE_COOLDOWN_SEC)
        assert "locked" in d.detail.lower() or "too many" in d.detail.lower()

        blocked = lim.check("alice", "1.2.3.4")
        assert blocked.allowed is False
        assert "second" in blocked.detail.lower()

    def test_success_resets_counter(self, lim_clock):
        lim, clock = lim_clock
        for _ in range(MAX_FREE_FAILURES):
            lim.record_failure("bob", "10.0.0.1")
        lim.record_success("bob", "10.0.0.1")
        # Fresh typo budget after success
        for _ in range(MAX_FREE_FAILURES):
            assert lim.record_failure("bob", "10.0.0.1").allowed is True
        assert lim.check("bob", "10.0.0.1").allowed is True

    def test_cooldown_expires_and_grows_on_repeat(self, lim_clock):
        lim, clock = lim_clock
        for _ in range(MAX_FREE_FAILURES + 1):
            lim.record_failure("cara", "9.9.9.9")
        assert lim.check("cara", "9.9.9.9").allowed is False

        clock.advance(BASE_COOLDOWN_SEC + 0.1)
        assert lim.check("cara", "9.9.9.9").allowed is True

        # Fail again (failures still > free threshold) → 2nd lockout doubles
        d = lim.record_failure("cara", "9.9.9.9")
        assert d.allowed is False
        assert d.retry_after_sec == pytest.approx(BASE_COOLDOWN_SEC * 2)

    def test_ip_isolation_and_user_isolation(self, lim_clock):
        lim, clock = lim_clock
        for _ in range(MAX_FREE_FAILURES + 1):
            lim.record_failure("dave", "5.5.5.5")
        assert lim.check("dave", "5.5.5.5").allowed is False
        # Different user, different IP — unaffected
        assert lim.check("erin", "6.6.6.6").allowed is True
        # Same IP hits IP lock even for another username
        blocked_ip = lim.check("erin", "5.5.5.5")
        assert blocked_ip.allowed is False
        assert "ip" in blocked_ip.detail.lower()

    def test_endpoint_returns_429_when_locked(self, monkeypatch, tmp_path):
        monkeypatch.setenv("EVOLVE_AUTH_SECRET", "t")
        import trading.auth.accounts as A
        import trading.auth.login_rate_limit as L

        monkeypatch.setattr(A, "DB_PATH", tmp_path / "a.db")
        clock = FakeClock()
        lim = LoginRateLimiter(
            max_free_failures=5,
            base_cooldown_sec=15.0,
            cooldown_growth=2.0,
            clock=clock,
        )
        monkeypatch.setattr(L, "login_rate_limiter", lim)

        A.create_user("thomas", "hunter2secure")
        from fastapi.testclient import TestClient

        from web.backend.main import app

        c = TestClient(app)
        for _ in range(MAX_FREE_FAILURES):
            r = c.post(
                "/api/auth/token",
                data={"username": "thomas", "password": "wrong-password"},
            )
            assert r.status_code == 401

        r = c.post(
            "/api/auth/token",
            data={"username": "thomas", "password": "wrong-password"},
        )
        assert r.status_code == 429
        body = r.json()
        detail = body.get("detail") or ""
        assert "fail" in detail.lower() or "locked" in detail.lower()
        assert "Retry-After" in r.headers

        # Successful login after cooldown clears + resets
        clock.advance(20)
        ok = c.post(
            "/api/auth/token",
            data={"username": "thomas", "password": "hunter2secure"},
        )
        assert ok.status_code == 200
        assert "access_token" in ok.json()

        # Counter reset: can fail again without immediate 429
        r2 = c.post(
            "/api/auth/token",
            data={"username": "thomas", "password": "still-wrong"},
        )
        assert r2.status_code == 401
