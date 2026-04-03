# -*- coding: utf-8 -*-
"""
Settings page — Watchlist, Alerts, System (from Alerts, Admin, Watchlist).
"""
import sys
from pathlib import Path
import runpy
import os

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st

from components.theme import inject_theme, render_top_bar, keyboard_shortcut_js

try:
    from trading.utils.notification_system import NotificationSystem
except Exception:
    NotificationSystem = None
try:
    import psutil
except Exception:
    psutil = None
try:
    from config.user_store import load_user_preferences, save_user_preferences
except Exception:
    load_user_preferences = save_user_preferences = None

try:
    st.markdown(keyboard_shortcut_js(), unsafe_allow_html=True)
except Exception:
    pass
inject_theme()
render_top_bar()

st.title("⚙️ Settings")
st.caption("Watchlist, alerts, and system configuration")

tab_wl, tab_keys, tab_alerts, tab_admin = st.tabs([
    "Watchlist",
    "🔑 API Keys",
    "Alerts",
    "System"
])

with tab_wl:
    st.subheader("Watchlist")
    try:
        if load_user_preferences:
            prefs = load_user_preferences(st.session_state.get("evolve_session_id", "") or "default")
            if prefs:
                st.caption("User preferences loaded from user_store.")
        from components.watchlist_widget import render_watchlist
        render_watchlist()
    except Exception as e:
        st.caption(f"Feature unavailable: {e}")

with tab_keys:
    st.subheader("API Keys")
    st.caption(
        "Your keys are encrypted and stored "
        "locally. They persist across sessions "
        "and are never shared."
    )
    try:
        from utils.session_utils import (
            get_stable_user_id
        )
        from config.user_store import (
            save_user_api_keys,
            load_user_api_keys,
            inject_user_keys_to_env,
        )
        _uid = get_stable_user_id()
        _saved = load_user_api_keys(_uid) or {}

        # Show masked existing keys
        _has_anthropic = bool(
            _saved.get("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        _has_openai = bool(
            _saved.get("OPENAI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )

        st.markdown("#### Anthropic (Claude)")
        if _has_anthropic:
            st.success("✅ Anthropic key saved")
        anthropic_key = st.text_input(
            "Anthropic API Key",
            type="password",
            placeholder="sk-ant-... (leave blank to keep existing)",
            key="settings_anthropic_key"
        )

        st.markdown("#### OpenAI")
        if _has_openai:
            st.success("✅ OpenAI key saved")
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-... (leave blank to keep existing)",
            key="settings_openai_key"
        )

        if st.button(
            "Save API Keys",
            key="settings_save_keys"
        ):
            keys_to_save = {}
            if anthropic_key.strip():
                keys_to_save[
                    "ANTHROPIC_API_KEY"
                ] = anthropic_key.strip()
            if openai_key.strip():
                keys_to_save[
                    "OPENAI_API_KEY"
                ] = openai_key.strip()
            if keys_to_save:
                save_user_api_keys(
                    _uid, keys_to_save
                )
                for k, v in keys_to_save.items():
                    st.session_state[f"user_key_{k}"] = v
                inject_user_keys_to_env(_uid)
                st.success(
                    "Keys saved and activated. "
                    "They will load automatically "
                    "next time you open the app."
                )
            else:
                st.info(
                    "No new keys entered. "
                    "Existing keys unchanged."
                )

        # Clear keys option
        if st.button(
            "Clear Saved Keys",
            key="settings_clear_keys"
        ):
            save_user_api_keys(_uid, {
                "ANTHROPIC_API_KEY": "",
                "OPENAI_API_KEY": "",
            })
            st.warning("Keys cleared.")

    except Exception as e:
        st.caption(
            f"API key storage unavailable: {e}"
        )

with tab_alerts:
    st.subheader("Alerts")
    try:
        # TODO S44: inline this (1995 lines)
        old_path = project_root / "scripts" / "old_10_Alerts.py"
        runpy.run_path(str(old_path), run_name="__main__")
    except Exception as e:
        st.caption(f"Feature unavailable: {e}")

with tab_admin:
    st.subheader("System")
    try:
        if psutil:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            st.metric("CPU Usage", f"{cpu_percent}%")
            st.metric("RAM Usage", f"{mem.percent}%")
        # TODO S44: inline this (4448 lines)
        old_path = project_root / "scripts" / "old_11_Admin.py"
        runpy.run_path(str(old_path), run_name="__main__")
    except Exception as e:
        st.caption(f"Feature unavailable: {e}")


# Page Assistant
try:
    from ui.page_assistant import render_page_assistant
    render_page_assistant("Settings")
except Exception:
    pass
