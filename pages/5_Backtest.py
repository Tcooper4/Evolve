# -*- coding: utf-8 -*-
"""
Backtest page -- Strategy testing, walk-forward, RL trainer, reports.
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
from components.theme import inject_theme, market_status_html, render_top_bar, keyboard_shortcut_js

try:
    st.markdown(keyboard_shortcut_js(), unsafe_allow_html=True)
except Exception:
    pass
inject_theme()
render_top_bar()
st.markdown(market_status_html(), unsafe_allow_html=True)

import runpy
try:
    runpy.run_path(
        str(project_root / "scripts" / "old_3_Strategy_Testing.py"),
        run_name="__main__"
    )
except Exception as e:
    st.error(f"Backtest page error: {e}")
    import traceback
    with st.expander("Details"):
        st.code(traceback.format_exc())
