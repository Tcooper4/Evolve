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
/* Force dark background on everything */
.stApp {
    background-color: #0a0e1a !important;
    color: #e0e6f0 !important;
}

/* Main content area */
.main .block-container {
    background-color: #0a0e1a !important;
    padding-top: 1rem !important;
}

/* All text */
.stApp p, .stApp div, .stApp span, .stApp label {
    color: #e0e6f0 !important;
}

/* Headings */
.stApp h1, .stApp h2, .stApp h3 {
    color: #e0e6f0 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #070b14 !important;
    border-right: 1px solid #1e2d45 !important;
}
[data-testid="stSidebar"] * {
    color: #e0e6f0 !important;
}

/* Text inputs */
[data-testid="stTextInput"] input {
    background-color: #0f1525 !important;
    color: #e0e6f0 !important;
    border: 1px solid #1e2d45 !important;
}

/* Selectboxes */
[data-testid="stSelectbox"] > div > div {
    background-color: #0f1525 !important;
    color: #e0e6f0 !important;
    border: 1px solid #1e2d45 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: #0a0e1a !important;
    border-bottom: 1px solid #1e2d45 !important;
}
.stTabs [data-baseweb="tab"] {
    background-color: #0a0e1a !important;
    color: #4a6080 !important;
}
.stTabs [aria-selected="true"] {
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #0f1525 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 4px !important;
    padding: 8px 12px !important;
}
[data-testid="stMetricValue"] {
    color: #e0e6f0 !important;
}
[data-testid="stMetricLabel"] {
    color: #4a6080 !important;
}

/* Buttons */
.stButton > button {
    background-color: #0f1525 !important;
    color: #e0e6f0 !important;
    border: 1px solid #1e2d45 !important;
}
.stButton > button:hover {
    border-color: #00d4ff !important;
    color: #00d4ff !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    background-color: #0f1525 !important;
}

/* Radio buttons */
[data-testid="stRadio"] label {
    color: #e0e6f0 !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: #0f1525 !important;
    border: 1px solid #1e2d45 !important;
}
[data-testid="stExpander"] summary {
    color: #e0e6f0 !important;
}

/* Plotly chart containers */
[data-testid="stPlotlyChart"] {
    background-color: #0a0e1a !important;
}

/* Info/warning/error boxes */
[data-testid="stAlert"] {
    background-color: #0f1525 !important;
    border: 1px solid #1e2d45 !important;
}

/* Hide Streamlit branding */
#MainMenu { visibility: hidden !important; }
footer { visibility: hidden !important; }
header { visibility: hidden !important; }
.stDeployButton { display: none !important; }
"""
    # Force dark theme config + CSS injection
    st.markdown("<style>" + css + "</style>", unsafe_allow_html=True)


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
