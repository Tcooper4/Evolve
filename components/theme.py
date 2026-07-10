# -*- coding: utf-8 -*-
"""
Global theme and CSS for Evolve Trading Platform.

Design system (Fable pass, July 2026)
-------------------------------------
Evolve's established identity - deep navy canvas with a cyan signal accent -
is kept (Plotly traces across every page already use it) but the execution
is rebuilt around an explicit token system:

* Color: layered surfaces (--bg0 canvas, --bg1 surface, --bg2 raised) with
  1px --line borders instead of flat same-color blocks; semantic --up /
  --down / --warn for anything that encodes direction or state.
* Type: Inter for UI, JetBrains Mono with tabular numerals for every
  number - tickers, metrics, inputs, tables. Numbers that line up in
  columns and never shift width as they tick are the terminal signature.
* Motion: 140ms ease transitions on interactive surfaces, a slow pulse on
  the market-open dot, and nothing else. prefers-reduced-motion disables
  all of it.

Public API unchanged: inject_theme, market_status_html, render_top_bar,
keyboard_shortcut_js.
"""

import streamlit as st
from datetime import datetime, time as dtime, timezone, timedelta
from typing import Dict, Any


def inject_theme() -> None:
    """Inject global CSS via st.markdown(unsafe_allow_html=True)."""
    css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg0: #070b14;
    --bg1: #0c1322;
    --bg2: #121b30;
    --line: #1c2a44;
    --line-hi: #2c4066;
    --text: #e6ecf7;
    --dim: #8b9ab8;
    --faint: #5a6a8c;
    --accent: #00d4ff;
    --accent-soft: rgba(0, 212, 255, 0.12);
    --up: #00e08a;
    --down: #ff4d6d;
    --warn: #ffb454;
    --font-ui: 'Inter', -apple-system, 'Segoe UI', sans-serif;
    --font-data: 'JetBrains Mono', 'SF Mono', Consolas, monospace;
    --radius: 10px;
    --speed: 140ms;
}

/* ============================ Base ============================ */
.stApp {
    background:
        radial-gradient(1200px 500px at 75% -10%, rgba(0, 212, 255, 0.045), transparent 60%),
        var(--bg0) !important;
    color: var(--text) !important;
    font-family: var(--font-ui) !important;
}
.main .block-container {
    padding-top: 0.9rem !important;
    max-width: 1440px;
}
.stApp p, .stApp div, .stApp span, .stApp label, .stApp li {
    color: var(--text);
    font-family: var(--font-ui);
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    color: var(--text) !important;
    font-family: var(--font-ui) !important;
    font-weight: 600 !important;
    letter-spacing: -0.015em !important;
}
.stApp h3 {
    padding-bottom: 0.35rem;
    border-bottom: 1px solid var(--line);
}
.stCaption, [data-testid="stCaptionContainer"] p, .stApp small {
    color: var(--dim) !important;
    font-size: 0.8rem !important;
}
a, a:visited { color: var(--accent) !important; }
code, pre, kbd {
    font-family: var(--font-data) !important;
    background: var(--bg2) !important;
    border: 1px solid var(--line);
    border-radius: 5px;
    color: var(--accent) !important;
    font-size: 0.82em !important;
    padding: 0.08em 0.35em;
}

/* ========================= Scrollbars ========================= */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--bg0); }
::-webkit-scrollbar-thumb {
    background: var(--line);
    border-radius: 6px;
    border: 2px solid var(--bg0);
}
::-webkit-scrollbar-thumb:hover { background: var(--line-hi); }

/* ========================== Sidebar =========================== */
[data-testid="stSidebar"] {
    background: #060a12 !important;
    border-right: 1px solid var(--line) !important;
}
[data-testid="stSidebar"] * { color: var(--text); }
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    justify-content: flex-start;
    background: transparent !important;
    border: 1px solid transparent !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--accent-soft) !important;
    border-color: var(--line) !important;
}

/* =========================== Inputs =========================== */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
.stTextArea textarea,
input[type="number"], input[type="text"] {
    background: var(--bg1) !important;
    color: var(--text) !important;
    border: 1px solid var(--line) !important;
    border-radius: 8px !important;
    font-family: var(--font-data) !important;
    font-size: 0.88rem !important;
    font-variant-numeric: tabular-nums;
    transition: border-color var(--speed) ease, box-shadow var(--speed) ease;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-soft) !important;
}
[data-testid="stSelectbox"] > div > div,
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--bg1) !important;
    color: var(--text) !important;
    border-color: var(--line) !important;
    border-radius: 8px !important;
    transition: border-color var(--speed) ease;
}
[data-testid="stSelectbox"] > div > div:hover,
.stMultiSelect > div > div:hover {
    border-color: var(--line-hi) !important;
}
[data-baseweb="popover"], [data-baseweb="menu"] {
    background: var(--bg2) !important;
    border: 1px solid var(--line) !important;
    border-radius: 8px !important;
}
[data-baseweb="option"], [role="option"] {
    background: transparent !important;
    color: var(--text) !important;
}
[data-baseweb="option"]:hover, [role="option"]:hover {
    background: var(--accent-soft) !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 4px var(--accent-soft) !important;
}
.stRadio label, .stCheckbox label { color: var(--text) !important; }

/* =========================== Buttons ========================== */
.stButton > button, .stFormSubmitButton > button,
[data-testid="baseButton-secondary"], [data-testid="stBaseButton-secondary"] {
    background: var(--bg1) !important;
    color: var(--text) !important;
    border: 1px solid var(--line) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em;
    transition: border-color var(--speed) ease, color var(--speed) ease,
                background var(--speed) ease, transform var(--speed) ease;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: var(--accent-soft) !important;
}
.stButton > button:active { transform: translateY(1px); }
.stButton > button[kind="primary"],
[data-testid="baseButton-primary"], [data-testid="stBaseButton-primary"] {
    background: linear-gradient(180deg, #06e2ff 0%, #00b8de 100%) !important;
    color: #04222c !important;
    border: 1px solid rgba(0, 212, 255, 0.55) !important;
    font-weight: 600 !important;
    text-shadow: none;
}
.stButton > button[kind="primary"]:hover,
[data-testid="baseButton-primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
    filter: brightness(1.08);
    color: #04222c !important;
    box-shadow: 0 2px 14px rgba(0, 212, 255, 0.28) !important;
}

/* ============================ Tabs ============================ */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--line) !important;
    gap: 0.25rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--faint) !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em;
    padding: 0.45rem 0.9rem !important;
    border-radius: 8px 8px 0 0 !important;
    transition: color var(--speed) ease, background var(--speed) ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--dim) !important;
    background: rgba(255, 255, 255, 0.025) !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: var(--accent) !important;
    height: 2px !important;
}
[data-testid="stTabContent"] { background: transparent !important; }

/* =========================== Metrics ========================== */
[data-testid="stMetric"] {
    background: linear-gradient(180deg, var(--bg2) 0%, var(--bg1) 100%) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--radius) !important;
    padding: 12px 14px !important;
    transition: border-color var(--speed) ease, transform var(--speed) ease;
}
[data-testid="stMetric"]:hover {
    border-color: var(--line-hi) !important;
    transform: translateY(-1px);
}
[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-family: var(--font-data) !important;
    font-variant-numeric: tabular-nums;
    font-weight: 500 !important;
    font-size: 1.5rem !important;
}
[data-testid="stMetricLabel"] {
    color: var(--faint) !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricDelta"] {
    font-family: var(--font-data) !important;
    font-variant-numeric: tabular-nums;
    font-size: 0.82rem !important;
}
[data-testid="stMetricDelta"] svg { transform: scale(0.85); }

/* ===================== Tables & dataframes ==================== */
[data-testid="stDataFrame"], .stDataFrame, .dataframe, [data-testid="stTable"] {
    background: var(--bg1) !important;
    color: var(--text) !important;
    border-radius: var(--radius) !important;
    font-family: var(--font-data) !important;
    font-variant-numeric: tabular-nums;
}
[data-testid="stDataFrame"] * {
    font-family: var(--font-data) !important;
    font-size: 0.82rem !important;
}

/* ================== Expanders, forms, alerts ================== */
[data-testid="stExpander"] {
    background: var(--bg1) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--radius) !important;
    transition: border-color var(--speed) ease;
}
[data-testid="stExpander"]:hover { border-color: var(--line-hi) !important; }
[data-testid="stExpander"] summary {
    color: var(--dim) !important;
    font-weight: 500;
}
[data-testid="stForm"] {
    background: var(--bg1) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
}
[data-testid="stAlert"] {
    background: var(--bg1) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stPlotlyChart"] {
    background: transparent !important;
}
[data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"],
[data-testid="column"] { background: transparent !important; }
hr { border-color: var(--line) !important; }

/* ===================== Top bar / market badge ================= */
.evolve-topbar {
    display: flex;
    align-items: center;
    gap: 18px;
    flex-wrap: wrap;
    padding: 7px 2px 9px 2px;
    margin-bottom: 10px;
    border-bottom: 1px solid var(--line);
    background: linear-gradient(180deg, rgba(12, 19, 34, 0.6), transparent);
}
.evolve-tick {
    font-family: var(--font-data);
    font-size: 12px;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}
.evolve-tick .sym { color: var(--accent); font-weight: 600; letter-spacing: 0.04em; }
.evolve-tick .px { color: var(--text); margin-left: 6px; }
.evolve-tick .chg { margin-left: 6px; }
.evolve-tick .chg.up { color: var(--up); }
.evolve-tick .chg.down { color: var(--down); }

.market-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 10px;
    border-radius: 999px;
    font-family: var(--font-data);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.market-badge .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: currentColor;
}
.market-badge.market-open {
    background: rgba(0, 224, 138, 0.1);
    color: var(--up);
    border: 1px solid rgba(0, 224, 138, 0.45);
}
.market-badge.market-open .dot { animation: evolve-pulse 2.2s ease-in-out infinite; }
.market-badge.market-closed {
    background: rgba(139, 154, 184, 0.1);
    color: var(--dim);
    border: 1px solid rgba(139, 154, 184, 0.35);
}
.market-badge.market-pre, .market-badge.market-post {
    background: rgba(255, 180, 84, 0.1);
    color: var(--warn);
    border: 1px solid rgba(255, 180, 84, 0.5);
}
@keyframes evolve-pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(0, 224, 138, 0.5); }
    50% { opacity: 0.55; box-shadow: 0 0 0 5px rgba(0, 224, 138, 0); }
}

/* ================= Chrome removal & a11y floor ================ */
#MainMenu { visibility: hidden !important; }
footer { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stHeader"] { background: transparent !important; }
.stDeployButton { display: none !important; }

:focus-visible {
    outline: 2px solid var(--accent) !important;
    outline-offset: 2px;
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation: none !important;
        transition: none !important;
    }
}
"""
    st.markdown("<style>" + css + "</style>", unsafe_allow_html=True)


def market_status_html() -> str:
    """Return an HTML string with the current market status badge."""
    try:
        try:
            import pytz

            et = pytz.timezone("America/New_York")
            now = datetime.now(et)
        except ImportError:
            try:
                # Python 3.9+ stdlib fallback with full DST support
                from zoneinfo import ZoneInfo

                now = datetime.now(ZoneInfo("America/New_York"))
            except Exception:
                # Last-resort: no tz library. UTC is 4–5h ahead of ET; compare
                # against ET boundaries by converting UTC to approximate ET.
                now = datetime.utcnow() + timedelta(hours=-4)
        t = now.time()
        wd = now.weekday()
        _dot = '<span class="dot"></span>'
        if wd >= 5:
            return (
                f'<span class="market-badge market-closed">{_dot}'
                "MARKET CLOSED</span>"
            )
        if dtime(4, 0) <= t < dtime(9, 30):
            return (
                f'<span class="market-badge market-pre">{_dot}'
                "PRE-MARKET</span>"
            )
        if dtime(9, 30) <= t < dtime(16, 0):
            close_mins = (16 * 60) - (t.hour * 60 + t.minute)
            return (
                f'<span class="market-badge market-open">{_dot}'
                f"MARKET OPEN &nbsp;·&nbsp; {close_mins}m to close</span>"
            )
        if dtime(16, 0) <= t < dtime(20, 0):
            return (
                f'<span class="market-badge market-post">{_dot}'
                "AFTER-HOURS</span>"
            )
        return (
            f'<span class="market-badge market-closed">{_dot}'
            "MARKET CLOSED</span>"
        )
    except Exception:
        return ""


def render_top_bar() -> None:
    """Render the persistent index ticker bar at the top of each page."""
    try:
        from trading.data.price_cache import get_quote

        _col_toggle, _col_bar = st.columns([1, 16])
        with _col_toggle:
            # Custom sidebar toggle removed: Streamlit Cloud provides native collapse.
            # Keep column for layout alignment with ticker bar.
            pass

        tickers_to_fetch = ["SPY", "QQQ", "IWM", "^VIX"]
        ticker_data = {}
        for sym in tickers_to_fetch:
            try:
                q = get_quote(sym)
                price = q.get("price")
                prev = q.get("prev_close")
                if price is not None and prev is not None and prev != 0:
                    chg = (float(price) - float(prev)) / float(prev) * 100
                    ticker_data[sym] = {"price": float(price), "change_pct": chg}
            except Exception:
                continue

        items_html = ""
        for sym, info in ticker_data.items():
            price = info.get("price", "")
            chg = info.get("change_pct", 0)
            sign = "+" if chg >= 0 else ""
            klass = "up" if chg >= 0 else "down"
            items_html += (
                f'<span class="evolve-tick">'
                f'<span class="sym">{sym}</span>'
                f'<span class="px">${price:.2f}</span>'
                f'<span class="chg {klass}">{sign}{chg:.2f}%</span>'
                f"</span>"
            )

        status_html = market_status_html()
        bar = (
            f'<div class="evolve-topbar">'
            f"{items_html}"
            f'<span style="margin-left:auto">{status_html}</span>'
            f"</div>"
        )
        with _col_bar:
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
