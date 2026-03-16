# -*- coding: utf-8 -*-
"""
Global theme and CSS for Evolve Trading Platform.
Injects dark theme, market status, top bar, and keyboard shortcut.
"""

import streamlit as st
from datetime import datetime, time as dtime, timezone, timedelta
from typing import Dict, Any


def inject_theme() -> None:
    """Inject global CSS via st.markdown(unsafe_allow_html=True)."""
    css = """
    <style>
    :root {
      --ev-bg: #0a0e1a;
      --ev-surface: #0f1525;
      --ev-surface2: #131d30;
      --ev-border: #1e2d45;
      --ev-accent: #00d4ff;
      --ev-text: #e0e6f0;
      --ev-muted: #4a6080;
      --ev-up: #26a69a;
      --ev-dn: #ef5350;
      --ev-warn: #ff9800;
    }

    /* Page background */
    .stApp { background-color: var(--ev-bg) !important; }

    /* Main content text */
    .stApp, .stMarkdown, p, div {
      color: var(--ev-text);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
      background-color: #070b14 !important;
      border-right: 1px solid var(--ev-border);
    }

    /* Hide Streamlit branding and top bar */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* Metric deltas */
    [data-testid="stMetricDelta"] svg { display: none; }

    /* Dataframes */
    .stDataFrame { background: var(--ev-surface) !important; }

    /* Buttons */
    .stButton > button {
      background: var(--ev-surface) !important;
      border: 1px solid var(--ev-border) !important;
      color: var(--ev-text) !important;
      font-family: 'Courier New', monospace;
    }
    .stButton > button:hover {
      border-color: var(--ev-accent) !important;
      color: var(--ev-accent) !important;
    }

    /* Text inputs and selectboxes */
    .stTextInput input, .stSelectbox select {
      background: var(--ev-surface) !important;
      border: 1px solid var(--ev-border) !important;
      color: var(--ev-text) !important;
      font-family: 'Courier New', monospace;
    }
    .stTextInput input:focus {
      border-color: var(--ev-accent) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
      background: var(--ev-surface);
      color: var(--ev-muted);
      border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
      color: var(--ev-accent) !important;
      border-bottom: 2px solid var(--ev-accent) !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
      background: var(--ev-surface);
      border: 1px solid var(--ev-border);
      border-radius: 4px;
      padding: 8px 12px;
    }
    [data-testid="stMetricValue"] {
      font-family: 'Courier New', monospace;
      color: var(--ev-text);
    }

    /* Price flash animations */
    @keyframes flashUp {
      0%  { background-color: rgba(38,166,154,0.35); }
      100%{ background-color: transparent; }
    }
    @keyframes flashDn {
      0%  { background-color: rgba(239,83,80,0.35); }
      100%{ background-color: transparent; }
    }
    .flash-up { animation: flashUp 0.8s ease-out; }
    .flash-dn { animation: flashDn 0.8s ease-out; }

    /* Market status badge */
    .market-badge {
      font-size: 11px;
      padding: 2px 8px;
      border-radius: 3px;
      font-family: 'Courier New', monospace;
      letter-spacing: 0.5px;
    }
    .market-open   { color:#26a69a; border:1px solid #26a69a;
                   background:#0d2a1a; }
    .market-closed { color:#4a6080; border:1px solid #4a6080;
                   background:#0f1525; }
    .market-pre    { color:#ff9800; border:1px solid #ff9800;
                   background:#2a1a0a; }
    .market-post   { color:#ff9800; border:1px solid #ff9800;
                   background:#2a1a0a; }

    /* Global ticker bar */
    .ticker-bar {
      display: flex;
      gap: 16px;
      padding: 4px 0;
      font-family: 'Courier New', monospace;
      font-size: 12px;
      border-bottom: 1px solid var(--ev-border);
      margin-bottom: 12px;
      flex-wrap: wrap;
    }
    .ticker-item { display: flex; gap: 6px; align-items: center; }
    .ticker-sym  { color: var(--ev-accent); font-weight: bold; }
    .ticker-px   { color: var(--ev-text); }
    .ticker-up   { color: var(--ev-up); }
    .ticker-dn   { color: var(--ev-dn); }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def market_status_html() -> str:
    """Return an HTML string with the current market status badge."""
    try:
        try:
            import pytz
            et = pytz.timezone("America/New_York")
            now = datetime.now(et)
        except ImportError:
            now = datetime.now(timezone(timedelta(hours=-4)))
        t = now.time()
        wd = now.weekday()
        if wd >= 5:
            return (
                '<span class="market-badge market-closed">'
                "MARKET CLOSED</span>"
            )
        if dtime(4, 0) <= t < dtime(9, 30):
            return (
                '<span class="market-badge market-pre">'
                "PRE-MARKET</span>"
            )
        if dtime(9, 30) <= t < dtime(16, 0):
            close_mins = (16 * 60) - (t.hour * 60 + t.minute)
            return (
                f'<span class="market-badge market-open">'
                f"MARKET OPEN &nbsp;·&nbsp; {close_mins}m to close</span>"
            )
        if dtime(16, 0) <= t < dtime(20, 0):
            return (
                '<span class="market-badge market-post">'
                "AFTER-HOURS</span>"
            )
        return (
            '<span class="market-badge market-closed">'
            "MARKET CLOSED</span>"
        )
    except Exception:
        return ""


def render_top_bar() -> None:
    """Render the persistent index ticker bar at the top of each page."""
    try:
        from trading.data.price_cache import get_quote
        tickers_to_fetch = ["SPY", "QQQ", "IWM", "^VIX"]
        items = []
        for sym in tickers_to_fetch:
            try:
                q = get_quote(sym)
                price = q.get("price")
                prev = q.get("prev_close")
                if price is not None and prev is not None and prev != 0:
                    chg = (float(price) - float(prev)) / float(prev) * 100
                    cls = "ticker-up" if chg >= 0 else "ticker-dn"
                    items.append(
                        f'<span class="ticker-item">'
                        f'<span class="ticker-sym">{sym}</span>'
                        f'<span class="ticker-px">${float(price):.2f}</span>'
                        f'<span class="{cls}">{chg:+.2f}%</span>'
                        f"</span>"
                    )
            except Exception:
                continue
        status = market_status_html()
        bar = '<div class="ticker-bar">' + "".join(items)
        if status:
            bar += f'<span style="margin-left:auto">{status}</span>'
        bar += "</div>"
        st.markdown(bar, unsafe_allow_html=True)
    except Exception:
        pass


def keyboard_shortcut_js() -> str:
    """Return a <script> tag that focuses global search on '/' key."""
    return """
<script>
document.addEventListener('keydown', function(e) {
    if (e.key === '/' && e.target.tagName !== 'INPUT'
        && e.target.tagName !== 'TEXTAREA') {
        e.preventDefault();
        var inputs = window.parent.document
            .querySelectorAll('input[placeholder]');
        for (var i = 0; i < inputs.length; i++) {
            if (inputs[i].placeholder.toLowerCase()
                .includes('search') ||
                inputs[i].placeholder.toLowerCase()
                .includes('ticker')) {
                inputs[i].focus();
                break;
            }
        }
    }
});
</script>
"""
