# -*- coding: utf-8 -*-
"""Login gate for Evolve's multi-user live mode.

Call :func:`require_login` at the top of app.py and every page. Behavior
is controlled by the ``EVOLVE_REQUIRE_LOGIN`` environment variable:

* unset / "0" (default) — **personal mode**: no login UI, identity falls
  back to the existing anonymous stable id. Local single-user usage is
  completely unchanged.
* "1" — **live-site mode**: unauthenticated visitors see only a login
  form; nothing else on any page renders. On success the username becomes
  the platform-wide user id (``user:<username>`` in
  ``st.session_state["evolve_session_id"]``), which the memory store,
  settings, chat learning, and watchlist already key on — so every
  account automatically gets its own persistent, adapting workspace.

Sessions persist across refreshes via streamlit-authenticator's signed
cookie. The cookie-signing secret comes from ``EVOLVE_AUTH_SECRET``; if
unset, a random secret is generated once and persisted to
``data/.auth_secret`` (chmod 600) so restarts don't log everyone out.

First run: if no accounts exist yet and login is required, the gate shows
a one-time bootstrap form to create the initial admin account. The
bootstrap is only reachable while the accounts table is empty.
"""

from __future__ import annotations

import logging
import os
import secrets
from pathlib import Path
from typing import Optional

import streamlit as st

logger = logging.getLogger(__name__)

_SECRET_FILE = Path(__file__).resolve().parents[2] / "data" / ".auth_secret"
_COOKIE_NAME = "evolve_auth"
_COOKIE_DAYS = 14


def login_required() -> bool:
    return os.getenv("EVOLVE_REQUIRE_LOGIN", "0").strip() in ("1", "true", "yes")


def _cookie_secret() -> str:
    env = os.getenv("EVOLVE_AUTH_SECRET")
    if env:
        return env
    try:
        if _SECRET_FILE.exists():
            return _SECRET_FILE.read_text().strip()
        _SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
        secret = secrets.token_hex(32)
        _SECRET_FILE.write_text(secret)
        try:
            os.chmod(_SECRET_FILE, 0o600)
        except OSError:
            pass
        return secret
    except OSError as e:
        logger.warning("Auth secret persistence failed (%s); using per-process secret", e)
        return secrets.token_hex(32)


def _bootstrap_first_admin() -> None:
    """One-time form to create the initial admin when no accounts exist."""
    from trading.auth import accounts

    st.markdown("### Welcome to Evolve — create the admin account")
    st.caption(
        "No accounts exist yet. This form appears only once; afterwards, "
        "accounts are managed with `python scripts/manage_users.py`."
    )
    with st.form("evolve_bootstrap_admin"):
        username = st.text_input("Username")
        display = st.text_input("Display name")
        password = st.text_input("Password (min 8 chars)", type="password")
        confirm = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Create admin account", type="primary")
    if submitted:
        if password != confirm:
            st.error("Passwords do not match.")
        else:
            try:
                accounts.create_user(
                    username, password, display_name=display or username,
                    role="admin",
                )
                st.success("Admin account created — log in below.")
                st.rerun()
            except ValueError as e:
                st.error(str(e))
    st.stop()


def require_login() -> Optional[str]:
    """Gate the current page. Returns the username in live-site mode, or
    None in personal mode. Unauthenticated visitors never get past this
    call — it renders the login form and stops the script."""
    if not login_required():
        return None

    from trading.auth import accounts

    if accounts.user_count() == 0:
        _bootstrap_first_admin()

    try:
        import streamlit_authenticator as stauth
    except ImportError:
        st.error(
            "Login is required (EVOLVE_REQUIRE_LOGIN=1) but "
            "streamlit-authenticator is not installed: "
            "pip install streamlit-authenticator"
        )
        st.stop()

    if "evolve_authenticator" not in st.session_state:
        st.session_state["evolve_authenticator"] = stauth.Authenticate(
            accounts.credentials_dict(),
            cookie_name=_COOKIE_NAME,
            cookie_key=_cookie_secret(),
            cookie_expiry_days=_COOKIE_DAYS,
        )
    authenticator = st.session_state["evolve_authenticator"]

    try:
        authenticator.login(location="main")
    except Exception as e:  # noqa: BLE001 - a component hiccup must not blank the app
        logger.warning("Login widget error: %s", e)

    status = st.session_state.get("authentication_status")
    if status is True:
        username = str(st.session_state.get("username", "")).strip().lower()
        st.session_state["evolve_session_id"] = f"user:{username}"
        st.session_state["evolve_stable_user_id"] = f"user:{username}"
        os.environ["EVOLVE_SESSION_ID"] = f"user:{username}"
        with st.sidebar:
            st.caption(f"Signed in as **{st.session_state.get('name', username)}**")
            authenticator.logout("Log out", location="sidebar")
        return username
    if status is False:
        st.error("Username or password is incorrect.")
    else:
        st.caption("Sign in to your Evolve workspace.")
    st.stop()
    return None  # unreachable; st.stop() raises
