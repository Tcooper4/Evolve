# -*- coding: utf-8 -*-
"""Admin-global settings (non-per-user), stored in accounts.db.

Lives beside invite codes — same SQLite file, same admin audience.
Currently: the live shared-keys override for ``resolve_api_key``.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from trading.auth import accounts as _accounts

logger = logging.getLogger(__name__)

SHARED_KEYS_SETTING = "shared_keys"

# Process-local cache: None = miss; value is Optional[bool] override.
# Invalidated on every write so admin toggles take effect immediately.
_cache_lock = threading.Lock()
_shared_keys_cache: Optional[Optional[bool]] = None  # outer Optional = miss
_cache_loaded = False


def _conn(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else Path(_accounts.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS admin_settings (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            updated_by TEXT
        )
        """
    )
    con.commit()
    return con


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def invalidate_shared_keys_cache() -> None:
    """Drop the hot-path cache (call after admin writes)."""
    global _shared_keys_cache, _cache_loaded
    with _cache_lock:
        _shared_keys_cache = None
        _cache_loaded = False


def get_shared_keys_override(
    db_path: Optional[Path] = None,
) -> Optional[bool]:
    """Return True/False if an admin has set the toggle, else None
    (caller should fall back to EVOLVE_SHARED_KEYS env).
    """
    global _shared_keys_cache, _cache_loaded
    # Hot path: serve from memory when DB path is the default (production).
    use_cache = db_path is None
    if use_cache:
        with _cache_lock:
            if _cache_loaded:
                return _shared_keys_cache

    try:
        con = _conn(db_path)
        try:
            row = con.execute(
                "SELECT value FROM admin_settings WHERE key=?",
                (SHARED_KEYS_SETTING,),
            ).fetchone()
        finally:
            con.close()
    except Exception as e:
        logger.warning("admin_settings: read shared_keys failed: %s", e)
        return None

    override: Optional[bool] = None
    if row and row[0] is not None:
        raw = str(row[0]).strip().lower()
        if raw in ("0", "false", "no", "off"):
            override = False
        elif raw in ("1", "true", "yes", "on"):
            override = True

    if use_cache:
        with _cache_lock:
            _shared_keys_cache = override
            _cache_loaded = True
    return override


def set_shared_keys_override(
    allowed: bool,
    *,
    updated_by: str = "",
    db_path: Optional[Path] = None,
) -> bool:
    """Persist the admin shared-keys toggle. Returns the stored value."""
    admin = (updated_by or "").strip().lower() or None
    value = "1" if allowed else "0"
    con = _conn(db_path)
    try:
        con.execute(
            """
            INSERT INTO admin_settings(key, value, updated_at, updated_by)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,
                updated_at=excluded.updated_at,
                updated_by=excluded.updated_by
            """,
            (SHARED_KEYS_SETTING, value, _now_iso(), admin),
        )
        con.commit()
    finally:
        con.close()
    invalidate_shared_keys_cache()
    return allowed


def shared_keys_status(db_path: Optional[Path] = None) -> dict:
    """Snapshot for the admin UI: effective policy + provenance."""
    override = get_shared_keys_override(db_path=db_path)
    import os

    env_raw = os.getenv("EVOLVE_SHARED_KEYS", "1").strip()
    env_allowed = env_raw.lower() not in ("0", "false", "no")
    if override is None:
        return {
            "allowed": env_allowed,
            "source": "env",
            "env_default": env_allowed,
            "persisted": False,
        }
    return {
        "allowed": bool(override),
        "source": "persisted",
        "env_default": env_allowed,
        "persisted": True,
    }


__all__ = [
    "SHARED_KEYS_SETTING",
    "get_shared_keys_override",
    "set_shared_keys_override",
    "shared_keys_status",
    "invalidate_shared_keys_cache",
]
