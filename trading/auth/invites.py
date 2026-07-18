# -*- coding: utf-8 -*-
"""Invite codes for gated multi-user signup.

Only an admin-role account may create codes. Signup consumes a code
exactly once (atomic UPDATE). Codes live in the same SQLite file as
accounts so backups stay coherent.
"""

from __future__ import annotations

import logging
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from trading.auth import accounts as _accounts

logger = logging.getLogger(__name__)

# Hand-typeable alphabet — no 0/O, 1/I/L ambiguity.
_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_CODE_LEN = 12
DEFAULT_EXPIRES_DAYS = 14


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def _conn(db_path: Optional[Path] = None) -> sqlite3.Connection:
    # Resolve accounts.DB_PATH at call time so test monkeypatches apply.
    path = Path(db_path) if db_path else Path(_accounts.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS invite_codes (
            code        TEXT PRIMARY KEY,
            created_by  TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            used_by     TEXT,
            used_at     TEXT,
            expires_at  TEXT
        )
        """
    )
    con.commit()
    return con


def _format_code(raw: str) -> str:
    """XXXX-XXXX-XXXX for readability when handing to a friend."""
    body = "".join(c for c in (raw or "").upper() if c.isalnum())
    if len(body) != _CODE_LEN:
        return body
    return f"{body[0:4]}-{body[4:8]}-{body[8:12]}"


def normalize_invite_code(code: str) -> str:
    """Strip separators/whitespace; store and match without hyphens."""
    return "".join(c for c in (code or "").upper() if c.isalnum())


def generate_invite_code(
    created_by: str,
    *,
    expires_in_days: Optional[int] = DEFAULT_EXPIRES_DAYS,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Create a fresh invite. ``created_by`` must be an admin (caller enforces)."""
    admin = (created_by or "").strip().lower()
    if not admin:
        raise ValueError("created_by is required")
    days = DEFAULT_EXPIRES_DAYS if expires_in_days is None else int(expires_in_days)
    if days < 1 or days > 365:
        raise ValueError("expires_in_days must be between 1 and 365")

    raw = "".join(secrets.choice(_CODE_ALPHABET) for _ in range(_CODE_LEN))
    expires_at = (_now() + timedelta(days=days)).isoformat()
    created_at = _now_iso()

    con = _conn(db_path)
    try:
        # Extremely unlikely collision; retry a few times.
        for _ in range(5):
            try:
                con.execute(
                    "INSERT INTO invite_codes "
                    "(code, created_by, created_at, used_by, used_at, expires_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (raw, admin, created_at, None, None, expires_at),
                )
                con.commit()
                break
            except sqlite3.IntegrityError:
                raw = "".join(
                    secrets.choice(_CODE_ALPHABET) for _ in range(_CODE_LEN)
                )
        else:
            raise RuntimeError("Could not allocate a unique invite code")
    finally:
        con.close()

    logger.info("Invite code created by %s (expires %s)", admin, expires_at)
    return {
        "code": _format_code(raw),
        "code_raw": raw,
        "created_by": admin,
        "created_at": created_at,
        "used_by": None,
        "used_at": None,
        "expires_at": expires_at,
        "status": "outstanding",
    }


def _status_for(row: Dict[str, Any], now: Optional[datetime] = None) -> str:
    now = now or _now()
    if row.get("used_by"):
        return "used"
    exp = row.get("expires_at")
    if exp:
        try:
            exp_dt = datetime.fromisoformat(exp)
            if exp_dt.tzinfo is None:
                exp_dt = exp_dt.replace(tzinfo=timezone.utc)
            if exp_dt <= now:
                return "expired"
        except ValueError:
            pass
    return "outstanding"


def list_invite_codes(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    con = _conn(db_path)
    try:
        rows = con.execute(
            "SELECT code, created_by, created_at, used_by, used_at, expires_at "
            "FROM invite_codes ORDER BY created_at DESC"
        ).fetchall()
        keys = [
            "code", "created_by", "created_at", "used_by", "used_at", "expires_at",
        ]
        out: List[Dict[str, Any]] = []
        now = _now()
        for r in rows:
            item = dict(zip(keys, r))
            item["code_display"] = _format_code(item["code"])
            item["status"] = _status_for(item, now)
            out.append(item)
        return out
    finally:
        con.close()


def peek_invite(code: str, db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Validate an invite without consuming it. Raises ValueError with reason."""
    raw = normalize_invite_code(code)
    if len(raw) != _CODE_LEN:
        raise ValueError("Invite code is invalid or incomplete")
    con = _conn(db_path)
    try:
        row = con.execute(
            "SELECT code, created_by, created_at, used_by, used_at, expires_at "
            "FROM invite_codes WHERE code=?",
            (raw,),
        ).fetchone()
        if not row:
            raise ValueError("Invite code not found")
        keys = [
            "code", "created_by", "created_at", "used_by", "used_at", "expires_at",
        ]
        item = dict(zip(keys, row))
        status = _status_for(item)
        if status == "used":
            raise ValueError("Invite code has already been used")
        if status == "expired":
            raise ValueError("Invite code has expired")
        return item
    finally:
        con.close()


def consume_invite(
    code: str,
    used_by: str,
    db_path: Optional[Path] = None,
) -> None:
    """Mark invite used. Raises ValueError if missing / used / expired.

    Uses a single conditional UPDATE so two concurrent signups cannot
    both succeed on the same code.
    """
    raw = normalize_invite_code(code)
    user = (used_by or "").strip().lower()
    if not user:
        raise ValueError("used_by is required")
    if len(raw) != _CODE_LEN:
        raise ValueError("Invite code is invalid or incomplete")

    now = _now_iso()
    con = _conn(db_path)
    try:
        # Reject already-used or expired in one shot.
        cur = con.execute(
            """
            UPDATE invite_codes
               SET used_by=?, used_at=?
             WHERE code=?
               AND used_by IS NULL
               AND (expires_at IS NULL OR expires_at > ?)
            """,
            (user, now, raw, now),
        )
        con.commit()
        if cur.rowcount != 1:
            # Distinguish reasons for honest API errors.
            row = con.execute(
                "SELECT used_by, expires_at FROM invite_codes WHERE code=?",
                (raw,),
            ).fetchone()
            if not row:
                raise ValueError("Invite code not found")
            if row[0]:
                raise ValueError("Invite code has already been used")
            raise ValueError("Invite code has expired")
    finally:
        con.close()


__all__ = [
    "DEFAULT_EXPIRES_DAYS",
    "normalize_invite_code",
    "generate_invite_code",
    "list_invite_codes",
    "peek_invite",
    "consume_invite",
]
