"""
Per-user persistent storage for beta: encrypted API keys and preferences in SQLite.
Do not commit data/users.db; keep EVOLVE_ENCRYPTION_KEY in .env and out of version control.
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
        # auto-generate and save on first run
        key = Fernet.generate_key().decode()
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
    with _get_conn() as conn:
        conn.execute(
            "UPDATE users SET preferences=?, last_seen=datetime('now') WHERE session_id=?",
            (json.dumps(prefs), session_id),
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


def inject_user_keys_to_env(session_id: str) -> None:
    """
    Load stored API keys for the given session and set them in os.environ.
    Does not overwrite keys already set in the environment (env takes precedence).
    Idempotent; safe to call multiple times.
    """
    if not session_id:
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
        conn = _get_conn()
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
        conn.execute(
            """INSERT OR REPLACE INTO
               user_api_keys(session_id, keys)
               VALUES (?, ?)""",
            (session_id, json.dumps(existing)),
        )
        conn.commit()
        conn.close()
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