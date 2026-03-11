"""
Per-user persistent storage for beta: encrypted API keys and preferences in SQLite.
Do not commit data/users.db; keep EVOLVE_ENCRYPTION_KEY in .env and out of version control.
"""

import json
import os
import sqlite3
from pathlib import Path

from cryptography.fernet import Fernet

os.makedirs("data", exist_ok=True)
os.makedirs(".cache", exist_ok=True)
USER_DB_PATH = Path("data/users.db")
USER_DB_PATH.parent.mkdir(exist_ok=True)


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
    with sqlite3.connect(USER_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                session_id TEXT PRIMARY KEY,
                created_at TEXT,
                last_seen TEXT,
                encrypted_keys TEXT,
                preferences TEXT
            )
        """)


def save_user_keys(session_id: str, keys: dict):
    cipher = _get_cipher()
    encrypted = cipher.encrypt(json.dumps(keys).encode()).decode()
    with sqlite3.connect(USER_DB_PATH) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO users (session_id, created_at, last_seen, encrypted_keys)
               VALUES (?, datetime('now'), datetime('now'), ?)""",
            (session_id, encrypted),
        )
        conn.commit()


def load_user_keys(session_id: str) -> dict:
    with sqlite3.connect(USER_DB_PATH) as conn:
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
    with sqlite3.connect(USER_DB_PATH) as conn:
        conn.execute(
            "UPDATE users SET preferences=?, last_seen=datetime('now') WHERE session_id=?",
            (json.dumps(prefs), session_id),
        )
        conn.commit()


def load_user_preferences(session_id: str) -> dict:
    with sqlite3.connect(USER_DB_PATH) as conn:
        row = conn.execute(
            "SELECT preferences FROM users WHERE session_id=?", (session_id,)
        ).fetchone()
    if not row or not row[0]:
        return {}
    return json.loads(row[0])
