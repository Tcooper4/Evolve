# -*- coding: utf-8 -*-
"""Invite-gated signup: codes, password policy, rate limit, admin APIs."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from trading.auth import accounts as A
from trading.auth import invites as I
from trading.auth.login_rate_limit import LoginRateLimiter, MAX_FREE_FAILURES
from trading.auth.password_policy import (
    MIN_SIGNUP_PASSWORD_LEN,
    validate_signup_password,
)


STRONG_PW = "CorrectHorse9"
ADMIN_PW = "AdminPassw0rd"


@pytest.fixture()
def acc_db(tmp_path, monkeypatch):
    db = tmp_path / "accounts.db"
    monkeypatch.setattr(A, "DB_PATH", db)
    monkeypatch.setenv("EVOLVE_AUTH_SECRET", "test-secret-invite-signup")
    return db


@pytest.fixture()
def admin_client(acc_db, monkeypatch):
    A.create_user("admin", ADMIN_PW, "Admin", role="admin", db_path=acc_db)
    A.create_user("bob", STRONG_PW, "Bob", role="user", db_path=acc_db)
    from trading.auth import login_rate_limit as L

    lim = LoginRateLimiter(
        max_free_failures=5,
        base_cooldown_sec=15.0,
        cooldown_growth=2.0,
        max_cooldown_sec=900.0,
    )
    monkeypatch.setattr(L, "login_rate_limiter", lim)
    from web.backend.main import app

    return TestClient(app), lim


def _auth(client: TestClient, user: str, pw: str) -> dict:
    r = client.post("/api/auth/token", data={"username": user, "password": pw})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


# ---------------------------------------------------------------------------
# Password policy (hand-verifiable messages)
# ---------------------------------------------------------------------------


class TestPasswordPolicy:
    def test_accepts_strong(self):
        validate_signup_password(STRONG_PW)

    def test_rejects_short_with_reason(self):
        with pytest.raises(ValueError) as ei:
            validate_signup_password("Ab1")
        assert str(MIN_SIGNUP_PASSWORD_LEN) in str(ei.value)

    def test_rejects_no_upper(self):
        with pytest.raises(ValueError) as ei:
            validate_signup_password("correcthorse9")
        assert "uppercase" in str(ei.value).lower()

    def test_rejects_no_lower(self):
        with pytest.raises(ValueError) as ei:
            validate_signup_password("CORRECTHORSE9")
        assert "lowercase" in str(ei.value).lower()

    def test_rejects_no_digit(self):
        with pytest.raises(ValueError) as ei:
            validate_signup_password("CorrectHorse")
        assert "digit" in str(ei.value).lower()

    def test_rejects_edge_whitespace(self):
        with pytest.raises(ValueError) as ei:
            validate_signup_password(" CorrectHorse9")
        assert "whitespace" in str(ei.value).lower()


# ---------------------------------------------------------------------------
# Invite store
# ---------------------------------------------------------------------------


class TestInviteStore:
    def test_generate_peek_consume(self, acc_db):
        inv = I.generate_invite_code("admin", db_path=acc_db)
        assert inv["status"] == "outstanding"
        assert "-" in inv["code"]
        peeked = I.peek_invite(inv["code"], db_path=acc_db)
        assert peeked["code"] == inv["code_raw"]
        I.consume_invite(inv["code"], "alice", db_path=acc_db)
        with pytest.raises(ValueError) as ei:
            I.peek_invite(inv["code"], db_path=acc_db)
        assert "already been used" in str(ei.value)

    def test_expired_rejected(self, acc_db):
        inv = I.generate_invite_code("admin", expires_in_days=1, db_path=acc_db)
        # Force expiry in DB
        import sqlite3

        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        con = sqlite3.connect(acc_db)
        con.execute(
            "UPDATE invite_codes SET expires_at=? WHERE code=?",
            (past, inv["code_raw"]),
        )
        con.commit()
        con.close()
        with pytest.raises(ValueError) as ei:
            I.peek_invite(inv["code"], db_path=acc_db)
        assert "expired" in str(ei.value).lower()

    def test_unknown_code_rejected(self, acc_db):
        with pytest.raises(ValueError) as ei:
            I.peek_invite("ZZZZ-ZZZZ-ZZZZ", db_path=acc_db)
        assert "not found" in str(ei.value).lower()

    def test_consume_is_single_use(self, acc_db):
        inv = I.generate_invite_code("admin", db_path=acc_db)
        I.consume_invite(inv["code_raw"], "alice", db_path=acc_db)
        with pytest.raises(ValueError) as ei:
            I.consume_invite(inv["code"], "bob", db_path=acc_db)
        assert "already been used" in str(ei.value)


# ---------------------------------------------------------------------------
# Admin API
# ---------------------------------------------------------------------------


class TestAdminInvitesAPI:
    def test_non_admin_forbidden(self, admin_client):
        client, _ = admin_client
        H = _auth(client, "bob", STRONG_PW)
        r = client.post("/api/admin/invites", json={}, headers=H)
        assert r.status_code == 403
        r2 = client.get("/api/admin/invites", headers=H)
        assert r2.status_code == 403

    def test_admin_can_create_and_list(self, admin_client):
        client, _ = admin_client
        H = _auth(client, "admin", ADMIN_PW)
        r = client.post(
            "/api/admin/invites",
            json={"expires_in_days": 7},
            headers=H,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["success"] is True
        assert body["status"] == "outstanding"
        assert len(body["code"].replace("-", "")) == 12

        listed = client.get("/api/admin/invites", headers=H).json()
        assert listed["success"] is True
        assert any(i["code"] == body["code"] for i in listed["invites"])


# ---------------------------------------------------------------------------
# Signup API
# ---------------------------------------------------------------------------


class TestSignupAPI:
    def test_missing_invite_rejected(self, admin_client):
        client, _ = admin_client
        r = client.post(
            "/api/auth/signup",
            json={"username": "carol", "password": STRONG_PW, "invite_code": ""},
        )
        assert r.status_code == 400
        assert "invite" in r.json()["detail"].lower()

    def test_invalid_invite_rejected(self, admin_client):
        client, _ = admin_client
        r = client.post(
            "/api/auth/signup",
            json={
                "username": "carol",
                "password": STRONG_PW,
                "invite_code": "AAAA-AAAA-AAAA",
            },
        )
        assert r.status_code == 400
        assert "not found" in r.json()["detail"].lower()

    def test_weak_password_clear_reason(self, admin_client):
        client, _ = admin_client
        H = _auth(client, "admin", ADMIN_PW)
        code = client.post("/api/admin/invites", json={}, headers=H).json()["code"]
        r = client.post(
            "/api/auth/signup",
            json={"username": "carol", "password": "short1A", "invite_code": code},
        )
        assert r.status_code == 400
        detail = r.json()["detail"].lower()
        assert "password" in detail
        assert str(MIN_SIGNUP_PASSWORD_LEN) in detail or "at least" in detail

    def test_valid_signup_then_code_unusable(self, admin_client, acc_db):
        client, _ = admin_client
        H = _auth(client, "admin", ADMIN_PW)
        code = client.post("/api/admin/invites", json={}, headers=H).json()["code"]

        r = client.post(
            "/api/auth/signup",
            json={
                "username": "carol",
                "password": STRONG_PW,
                "invite_code": code,
                "display_name": "Carol",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["username"] == "carol"
        assert body["display_name"] == "Carol"
        assert body["access_token"]

        # Account works
        assert A.authenticate("carol", STRONG_PW, db_path=acc_db)
        row = A.get_user("carol", db_path=acc_db)
        assert row is not None
        assert row["role"] == "user"

        # Same code cannot be reused
        r2 = client.post(
            "/api/auth/signup",
            json={
                "username": "dave",
                "password": STRONG_PW,
                "invite_code": code,
            },
        )
        assert r2.status_code == 400
        assert "already been used" in r2.json()["detail"].lower()

    def test_expired_invite_rejected(self, admin_client, acc_db):
        client, _ = admin_client
        H = _auth(client, "admin", ADMIN_PW)
        code = client.post("/api/admin/invites", json={}, headers=H).json()["code"]
        raw = I.normalize_invite_code(code)
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        import sqlite3

        con = sqlite3.connect(acc_db)
        con.execute(
            "UPDATE invite_codes SET expires_at=? WHERE code=?",
            (past, raw),
        )
        con.commit()
        con.close()

        r = client.post(
            "/api/auth/signup",
            json={"username": "eve", "password": STRONG_PW, "invite_code": code},
        )
        assert r.status_code == 400
        assert "expired" in r.json()["detail"].lower()

    def test_signup_rate_limited_like_login(self, admin_client, monkeypatch):
        client, lim = admin_client
        H = _auth(client, "admin", ADMIN_PW)
        # Burn free failures with bad invite codes
        for i in range(MAX_FREE_FAILURES):
            r = client.post(
                "/api/auth/signup",
                json={
                    "username": "ratelim",
                    "password": STRONG_PW,
                    "invite_code": f"BAD{i}BADBADBA",  # wrong length / not found
                },
            )
            assert r.status_code == 400, r.text

        # One more triggers lockout (same pattern as login)
        r = client.post(
            "/api/auth/signup",
            json={
                "username": "ratelim",
                "password": STRONG_PW,
                "invite_code": "BADXBADXBADX",
            },
        )
        assert r.status_code == 429, r.text
        assert "Retry-After" in r.headers
        detail = (r.json().get("detail") or "").lower()
        assert "fail" in detail or "locked" in detail or "try again" in detail

    def test_duplicate_username_clear_error(self, admin_client):
        client, _ = admin_client
        H = _auth(client, "admin", ADMIN_PW)
        code = client.post("/api/admin/invites", json={}, headers=H).json()["code"]
        r = client.post(
            "/api/auth/signup",
            json={
                "username": "bob",  # already exists
                "password": STRONG_PW,
                "invite_code": code,
            },
        )
        assert r.status_code == 400
        assert "already exists" in r.json()["detail"].lower()
        # Invite must remain usable (create_user failed before consume)
        listed = client.get("/api/admin/invites", headers=H).json()["invites"]
        match = [i for i in listed if i["code"] == code]
        assert match and match[0]["status"] == "outstanding"
