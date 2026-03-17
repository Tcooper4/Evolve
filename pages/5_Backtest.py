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
from components.theme import inject_theme, render_top_bar, keyboard_shortcut_js

try:
    st.markdown(keyboard_shortcut_js(), unsafe_allow_html=True)
except Exception:
    pass
inject_theme()
render_top_bar()

import runpy

# TODO S44: inline old_3_Strategy_Testing.py
# Keeping runpy — file is too large to
# safely inline in one session
try:
    runpy.run_path(
        str(project_root / "scripts" / "old_3_Strategy_Testing.py"),
        run_name="__main__",
    )
except Exception as e:
    import traceback

    st.warning(
        "⚠️ Strategy testing module failed "
        "to load."
    )
    with st.expander("Error details", expanded=False):
        st.code(traceback.format_exc())
    if st.button("🔄 Retry", key="backtest_retry"):
        st.rerun()


# Page Assistant
try:
    from ui.page_assistant import render_page_assistant
    render_page_assistant("Backtest")
except Exception:
    pass
