"""
Per-user persistent storage for beta: encrypted API keys and preferences in SQLite.
Do not commit data/users.db; keep EVOLVE_ENCRYPTION_KEY in .env and out of version control.

Preferences JSON is **user-stated** only (e.g. Settings risk_tolerance). Do not
treat this store as an engagement / click-learning model, and do not write
inferred preference fields from watchlist or UI activity. See
docs/PERSONALIZATION.md.
"""

import json
import os
import sqlite3
import logging
from pathlib import Path

from cryptography.fernet import Fernet

os.makedirs("data", exist_ok=True)
os.makedirs(".cache", exist_ok=True)
USER_DB_PATH = Path("data/users.db")
USER_DB_PATH.parent.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(USER_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            session_id TEXT PRIMARY KEY,
            created_at TEXT,
            last_seen TEXT,
            encrypted_keys TEXT,
            preferences TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_api_keys (
            session_id TEXT PRIMARY KEY,
            keys TEXT NOT NULL
        )
    """)
    return conn


def _get_cipher():
    key = os.getenv("EVOLVE_ENCRYPTION_KEY", "")
    if not key:
        # BUG FIX: this previously generated a new random key and wrote
        # it only to the .env FILE, never updating this process's
        # os.environ. Since os.getenv() re-reads the environment (not
        # the file) on every call, EVERY call to _get_cipher() within
        # the same running process generated a genuinely different
        # random key whenever EVOLVE_ENCRYPTION_KEY wasn't pre-set.
        # save_user_keys() would encrypt with one key, and a later
        # load_user_keys() call in the same session would try to decrypt
        # with a different key - failing with InvalidToken, silently
        # caught and returning an empty dict as if no keys were ever
        # saved. Verified concretely. Setting the key into os.environ
        # immediately fixes consistency for the current process, on top
        # of the existing .env write for future process restarts.
        key = Fernet.generate_key().decode()
        os.environ["EVOLVE_ENCRYPTION_KEY"] = key
        # write to .env
        env_path = Path(".env")
        with open(env_path, "a") as f:
            f.write(f"\nEVOLVE_ENCRYPTION_KEY={key}")
    return Fernet(key.encode() if isinstance(key, str) else key)


def init_user_db():
    with _get_conn() as conn:
        pass


def save_user_keys(session_id: str, keys: dict):
    cipher = _get_cipher()
    encrypted = cipher.encrypt(json.dumps(keys).encode()).decode()
    with _get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO users (session_id, created_at, last_seen, encrypted_keys)
               VALUES (?, datetime('now'), datetime('now'), ?)""",
            (session_id, encrypted),
        )
        conn.commit()


def load_user_keys(session_id: str) -> dict:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT encrypted_keys FROM users WHERE session_id=?", (session_id,)
        ).fetchone()
    if not row or not row[0]:
        return {}
    try:
        cipher = _get_cipher()
        return json.loads(cipher.decrypt(row[0].encode()).decode())
    except Exception:
        # Corrupted data or wrong key — return empty rather than crash
        return {}


def save_user_preferences(session_id: str, prefs: dict):
    """
    Persist preferences for a session. Upserts the users row so saves work
    even when no row existed yet (e.g. before onboarding wrote encrypted_keys).
    """
    if not session_id:
        return
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO users (session_id, created_at, last_seen, preferences)
            VALUES (?, datetime('now'), datetime('now'), ?)
            ON CONFLICT(session_id) DO UPDATE SET
                preferences = excluded.preferences,
                last_seen = datetime('now')
            """,
            (session_id, json.dumps(prefs)),
        )
        conn.commit()


def load_user_preferences(session_id: str) -> dict:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT preferences FROM users WHERE session_id=?", (session_id,)
        ).fetchone()
    if not row or not row[0]:
        return {}
    return json.loads(row[0])


def list_session_ids_with_alerts() -> list:
    """Session IDs whose preferences contain a non-empty evolve_alerts list."""
    out: list = []
    try:
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT session_id, preferences FROM users"
            ).fetchall()
        for session_id, prefs_raw in rows:
            if not session_id or not prefs_raw:
                continue
            try:
                prefs = json.loads(prefs_raw)
            except Exception:
                continue
            raw = prefs.get("evolve_alerts") if isinstance(prefs, dict) else None
            if isinstance(raw, list) and len(raw) > 0:
                out.append(str(session_id))
    except Exception as e:
        logger.debug("list_session_ids_with_alerts failed: %s", e)
    return out


def inject_user_keys_to_env(session_id: str) -> None:
    """
    Load stored API keys for the given session and set them in os.environ.
    Does not overwrite keys already set in the environment (env takes precedence).
    Idempotent; safe to call multiple times.

    MULTI-USER GUARD: os.environ is process-global and Streamlit serves
    every logged-in user from one process, so injecting one user's keys
    into the environment would make everyone else's requests spend that
    user's quota. In live-site mode (EVOLVE_REQUIRE_LOGIN=1) this is
    therefore a hard no-op; per-request resolution happens in
    config/api_keys.resolve_api_key instead. Personal mode (single user)
    keeps the original behavior.
    """
    if not session_id:
        return
    if os.getenv("EVOLVE_REQUIRE_LOGIN", "0").strip() in ("1", "true", "yes"):
        logger.debug(
            "inject_user_keys_to_env skipped in multi-user mode (%s); "
            "per-request resolver handles keys", session_id,
        )
        return
    try:
        keys = {}
        try:
            keys.update(load_user_api_keys(session_id) or {})
        except Exception:
            pass
        try:
            keys.update(load_user_keys(session_id) or {})
        except Exception:
            pass
        if not keys:
            return
        for key, value in keys.items():
            if not value or not isinstance(value, str):
                continue
            if key in os.environ and os.environ[key]:
                continue  # Do not overwrite existing env
            os.environ[key] = value
    except Exception:
        pass


def inject_user_keys_to_session(session_id: str) -> None:
    """
    Load stored API keys for the given session and store them in st.session_state.
    Never touches os.environ — safe for multi-user cloud deployments.
    """
    if not session_id:
        return
    try:
        import streamlit as st
        keys: dict = {}
        try:
            keys.update(load_user_api_keys(session_id) or {})
        except Exception:
            pass
        try:
            keys.update(load_user_keys(session_id) or {})
        except Exception:
            pass
        if not keys:
            return
        for key, value in keys.items():
            if not value or not isinstance(value, str):
                continue
            st.session_state[f"user_key_{key}"] = value
    except Exception as e:
        logger.warning("user_store: inject_user_keys_to_session failed: %s", e)


def save_user_api_keys(session_id: str, keys: dict) -> None:
    """
    Save API keys for a user session.
    Keys dict: {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "OPENAI_API_KEY": "sk-...",
    }
    Keys are encrypted before storage.
    """
    if not session_id:
        return
    try:
        cipher = _get_cipher()
        # Encrypt each key value
        encrypted: dict = {}
        for k, v in (keys or {}).items():
            if v and isinstance(v, str):
                encrypted[k] = cipher.encrypt(v.encode()).decode()
            elif isinstance(v, str) and v == "":
                # Explicit clear
                encrypted[k] = ""
        existing = load_user_api_keys(session_id) or {}
        existing.update(encrypted)
        with _get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO
                   user_api_keys(session_id, keys)
                   VALUES (?, ?)""",
                (session_id, json.dumps(existing)),
            )
            conn.commit()
    except Exception as e:
        logger.warning(
            "user_store: save_user_api_keys failed: %s", e
        )


def load_user_api_keys(session_id: str) -> dict:
    """
    Load API keys for a user session and decrypt them.
    Returns plaintext keys dict. Never raises.
    """
    if not session_id:
        return {}
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT keys FROM user_api_keys WHERE session_id=?",
                (session_id,),
            ).fetchone()
        if not row or not row[0]:
            return {}
        payload = json.loads(row[0])
        if not isinstance(payload, dict):
            return {}
        cipher = _get_cipher()
        out: dict = {}
        for k, enc in payload.items():
            if enc is None:
                continue
            if isinstance(enc, str) and enc == "":
                # Cleared key
                out[k] = ""
                continue
            if not isinstance(enc, str):
                continue
            try:
                out[k] = cipher.decrypt(enc.encode()).decode()
            except Exception:
                continue
        return out
    except Exception as e:
        logger.warning("user_store: load_user_api_keys failed: %s", e)
        return {}