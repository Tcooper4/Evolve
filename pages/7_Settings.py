# -*- coding: utf-8 -*-
"""
Settings page — Watchlist, Alerts, System (from Alerts, Admin, Watchlist).
"""
import sys
from pathlib import Path
import runpy

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st

from components.theme import market_status_html, render_top_bar, keyboard_shortcut_js

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
render_top_bar()
st.markdown(market_status_html(), unsafe_allow_html=True)

st.title("⚙️ Settings")
st.caption("Watchlist, alerts, and system configuration")

tab_wl, tab_alerts, tab_admin = st.tabs(["Watchlist", "Alerts", "System"])

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

with tab_alerts:
    st.subheader("Alerts")
    try:
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
        old_path = project_root / "scripts" / "old_11_Admin.py"
        runpy.run_path(str(old_path), run_name="__main__")
    except Exception as e:
        st.caption(f"Feature unavailable: {e}")
