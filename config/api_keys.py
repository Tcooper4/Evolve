# -*- coding: utf-8 -*-
"""Per-user API key resolution — the single source of key truth.

Each account stores its own API keys (encrypted at rest via
config/user_store.py; the Settings page is the UI). This module answers
"which key should THIS request use?" without ever writing user keys into
``os.environ`` — the environment is process-global, and Streamlit serves
every logged-in user from one process, so env injection makes one user's
key silently pay for everyone else's requests. That leak is exactly what
this resolver replaces.

Resolution priority for ``resolve_api_key(name)``:

1. The current user's stored key (encrypted store, keyed by the login
   identity ``user:<name>``, or ``local`` in personal mode).
2. The same key mirrored into ``st.session_state`` by
   ``inject_user_keys_to_session`` (fast path; also covers Streamlit
   Cloud where the store may lag a save within the same rerun).
3. The server environment — the operator's own keys — ONLY when the
   shared-keys policy allows it: always in personal mode; in live mode
   (``EVOLVE_REQUIRE_LOGIN=1``) unless ``EVOLVE_SHARED_KEYS=0``. Setting
   ``EVOLVE_SHARED_KEYS=0`` on a hosted site means users MUST supply
   their own keys and can never spend the operator's quota.

Alias handling: NEWS_API_KEY / NEWSAPI_KEY and GOOGLE_API_KEY /
GEMINI_API_KEY are treated as the same credential.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_ALIASES: Dict[str, List[str]] = {
    "NEWS_API_KEY": ["NEWS_API_KEY", "NEWSAPI_KEY"],
    "NEWSAPI_KEY": ["NEWSAPI_KEY", "NEWS_API_KEY"],
    "GOOGLE_API_KEY": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "GEMINI_API_KEY": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "HUGGINGFACE_API_KEY": ["HUGGINGFACE_API_KEY", "HF_TOKEN"],
}


def current_user_id() -> str:
    """The identity of the current request: the logged-in user in live
    mode, 'local' in personal mode."""
    try:
        import streamlit as st

        sid = st.session_state.get("evolve_session_id")
        if sid:
            return str(sid)
    except Exception:
        pass
    return os.getenv("EVOLVE_SESSION_ID") or "local"


def shared_keys_allowed() -> bool:
    """May requests fall back to the server's own env keys?"""
    require_login = os.getenv("EVOLVE_REQUIRE_LOGIN", "0").strip() in (
        "1", "true", "yes",
    )
    if not require_login:
        return True  # personal mode: env keys are the user's own keys
    return os.getenv("EVOLVE_SHARED_KEYS", "1").strip() not in ("0", "false", "no")


def _user_stored_keys(session_id: str) -> Dict[str, str]:
    try:
        from config.user_store import load_user_api_keys, load_user_keys

        keys: Dict[str, str] = {}
        keys.update(load_user_api_keys(session_id) or {})
        keys.update(load_user_keys(session_id) or {})
        return {k: v for k, v in keys.items() if isinstance(v, str) and v}
    except Exception as e:  # noqa: BLE001 - key lookup must never crash a request
        logger.warning("api_keys: user store lookup failed: %s", e)
        return {}


def resolve_api_key(name: str, session_id: Optional[str] = None) -> Optional[str]:
    """The key the CURRENT request should use for `name`, or None."""
    names = _ALIASES.get(name, [name])
    sid = session_id or current_user_id()

    # 1. Encrypted per-user store
    stored = _user_stored_keys(sid)
    for n in names:
        if stored.get(n):
            return stored[n]

    # 2. Session-state mirror (set on login / on Settings save)
    try:
        import streamlit as st

        for n in names:
            v = st.session_state.get(f"user_key_{n}")
            if isinstance(v, str) and v:
                return v
    except Exception:
        pass

    # 3. Server env, policy permitting
    if shared_keys_allowed():
        for n in names:
            v = os.getenv(n)
            if v:
                return v
    return None


def resolve_many(names: List[str], session_id: Optional[str] = None) -> Dict[str, Optional[str]]:
    sid = session_id or current_user_id()
    return {n: resolve_api_key(n, session_id=sid) for n in names}
