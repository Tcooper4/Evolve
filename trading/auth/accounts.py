# -*- coding: utf-8 -*-
"""Account store for Evolve's multi-user mode.

SQLite-backed user accounts with bcrypt password hashes. This is the
identity source for the login gate (trading/auth/gate.py); once a user
authenticates, their username becomes the platform-wide user id
(``user:<username>``) that the memory store, settings, and watchlist all
key on.

Passwords are never stored or logged in plaintext; only bcrypt hashes are
persisted. Account management is via scripts/manage_users.py or the
first-run bootstrap in the login gate.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import bcrypt

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "accounts.db"


def _conn(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS accounts (
            username      TEXT PRIMARY KEY,
            display_name  TEXT NOT NULL,
            email         TEXT,
            password_hash TEXT NOT NULL,
            role          TEXT NOT NULL DEFAULT 'user',
            active        INTEGER NOT NULL DEFAULT 1,
            created_at    TEXT NOT NULL,
            last_login    TEXT
        )
        """
    )
    con.commit()
    return con


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_user(
    username: str,
    password: str,
    display_name: Optional[str] = None,
    email: Optional[str] = None,
    role: str = "user",
    db_path: Optional[Path] = None,
) -> None:
    """Create an account. Raises ValueError on bad input or duplicate."""
    username = (username or "").strip().lower()
    if not username.isidentifier():
        raise ValueError(
            "Username must be letters/digits/underscore, starting with a letter"
        )
    if len(password or "") < 8:
        raise ValueError("Password must be at least 8 characters")
    con = _conn(db_path)
    try:
        con.execute(
            "INSERT INTO accounts (username, display_name, email, password_hash,"
            " role, active, created_at) VALUES (?,?,?,?,?,1,?)",
            (username, display_name or username, email,
             hash_password(password), role, _now()),
        )
        con.commit()
        logger.info("Created account %r (role=%s)", username, role)
    except sqlite3.IntegrityError as e:
        raise ValueError(f"Username '{username}' already exists") from e
    finally:
        con.close()


def authenticate(username: str, password: str, db_path: Optional[Path] = None) -> bool:
    """True only for an ACTIVE account with a matching password."""
    username = (username or "").strip().lower()
    con = _conn(db_path)
    try:
        row = con.execute(
            "SELECT password_hash, active FROM accounts WHERE username=?",
            (username,),
        ).fetchone()
        if not row or not row[1]:
            # Burn comparable time so absent users aren't distinguishable
            # from wrong passwords by response timing.
            bcrypt.checkpw(b"x", bcrypt.hashpw(b"y", bcrypt.gensalt()))
            return False
        ok = verify_password(password, row[0])
        if ok:
            con.execute(
                "UPDATE accounts SET last_login=? WHERE username=?",
                (_now(), username),
            )
            con.commit()
        return ok
    finally:
        con.close()


def set_password(username: str, new_password: str, db_path: Optional[Path] = None) -> None:
    if len(new_password or "") < 8:
        raise ValueError("Password must be at least 8 characters")
    con = _conn(db_path)
    try:
        cur = con.execute(
            "UPDATE accounts SET password_hash=? WHERE username=?",
            (hash_password(new_password), (username or "").strip().lower()),
        )
        if cur.rowcount == 0:
            raise ValueError(f"No such user: {username}")
        con.commit()
    finally:
        con.close()


def set_active(username: str, active: bool, db_path: Optional[Path] = None) -> None:
    con = _conn(db_path)
    try:
        cur = con.execute(
            "UPDATE accounts SET active=? WHERE username=?",
            (1 if active else 0, (username or "").strip().lower()),
        )
        if cur.rowcount == 0:
            raise ValueError(f"No such user: {username}")
        con.commit()
    finally:
        con.close()


def get_user(username: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Return one account row (no password hash), or None."""
    username = (username or "").strip().lower()
    if not username:
        return None
    con = _conn(db_path)
    try:
        row = con.execute(
            "SELECT username, display_name, email, role, active, created_at,"
            " last_login FROM accounts WHERE username=?",
            (username,),
        ).fetchone()
        if not row:
            return None
        keys = ["username", "display_name", "email", "role", "active",
                "created_at", "last_login"]
        return dict(zip(keys, row))
    finally:
        con.close()


def list_users(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    con = _conn(db_path)
    try:
        rows = con.execute(
            "SELECT username, display_name, email, role, active, created_at,"
            " last_login FROM accounts ORDER BY username"
        ).fetchall()
        keys = ["username", "display_name", "email", "role", "active",
                "created_at", "last_login"]
        return [dict(zip(keys, r)) for r in rows]
    finally:
        con.close()


def user_count(db_path: Optional[Path] = None) -> int:
    con = _conn(db_path)
    try:
        return con.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]
    finally:
        con.close()


def credentials_dict(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Active accounts in the shape streamlit-authenticator expects."""
    users = {}
    for u in list_users(db_path):
        if not u["active"]:
            continue
        con = _conn(db_path)
        try:
            pw_hash = con.execute(
                "SELECT password_hash FROM accounts WHERE username=?",
                (u["username"],),
            ).fetchone()[0]
        finally:
            con.close()
        users[u["username"]] = {
            "name": u["display_name"],
            "email": u["email"] or f"{u['username']}@local",
            "password": pw_hash,  # already bcrypt-hashed
        }
    return {"usernames": users}
