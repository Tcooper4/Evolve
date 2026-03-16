# -*- coding: utf-8 -*-
"""
Trade page — Execute, Positions, History, Risk (reorganized from Trade Execution, Performance, Portfolio).
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
    st.markdown(keyboard_shortcut_js(), unsafe_allow_html=True)
except Exception:
    pass
render_top_bar()
st.markdown(market_status_html(), unsafe_allow_html=True)

st.title("💰 Trade")
st.caption("Execution, positions, history, and risk")

# Summary metrics row (placeholder when no portfolio)
try:
    if "portfolio_manager" not in st.session_state:
        from trading.portfolio.portfolio_manager import PortfolioManager
        st.session_state.portfolio_manager = PortfolioManager()
    pm = st.session_state.portfolio_manager
    positions = pm.get_all_positions() if hasattr(pm, "get_all_positions") else []
    total_val = sum(float(p.get("market_value", 0) or 0) for p in positions) if isinstance(positions, list) else 0
    if not total_val and hasattr(pm, "get_portfolio_value"):
        total_val = float(pm.get_portfolio_value() or 0)
except Exception:
    total_val = 0
    positions = []

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Portfolio Value", f"${total_val:,.2f}")
with c2:
    st.metric("Day P&L", "—", delta=None)
with c3:
    st.metric("Total Return", "—", delta=None)

tab_exec, tab_pos, tab_hist, tab_risk = st.tabs(["Execute", "Positions", "History", "Risk"])

with tab_exec:
    st.subheader("Execute")
    try:
        old_path = project_root / "scripts" / "old_4_Trade_Execution.py"
        runpy.run_path(str(old_path), run_name="__main__")
        st.caption("Est. slippage: 5bps applied to all orders")
    except Exception as e:
        st.caption(f"Feature unavailable: {e}")

with tab_pos:
    st.subheader("Positions")
    try:
        old_path = project_root / "scripts" / "old_5_Portfolio.py"
        runpy.run_path(str(old_path), run_name="__main__")
    except Exception as e:
        st.caption(f"Feature unavailable: {e}")

with tab_hist:
    st.subheader("History")
    try:
        old_path = project_root / "scripts" / "old_7_Performance.py"
        runpy.run_path(str(old_path), run_name="__main__")
    except Exception as e:
        st.caption(f"Feature unavailable: {e}")

with tab_risk:
    st.subheader("Risk")
    try:
        old_path = project_root / "scripts" / "old_6_Risk_Management.py"
        runpy.run_path(str(old_path), run_name="__main__")
    except Exception as e:
        st.caption(f"Feature unavailable: {e}")
