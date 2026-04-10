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

/* Force dark on all form elements */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background-color: #0f1525 !important;
    color: #e0e6f0 !important;
    border-color: #1e2d45 !important;
}

/* Force dark on all containers and columns */
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
[data-testid="column"] {
    background-color: transparent !important;
}

/* Force dark on form submit buttons */
.stFormSubmitButton > button,
[data-testid="baseButton-secondary"],
[data-testid="baseButton-primary"] {
    background-color: #0f1525 !important;
    color: #e0e6f0 !important;
    border: 1px solid #1e2d45 !important;
}

/* Force dark on all markdown text */
.stMarkdown, .stMarkdown p,
.stMarkdown h1, .stMarkdown h2,
.stMarkdown h3, .stMarkdown li {
    color: #e0e6f0 !important;
}

/* Force dark on all tables */
.stDataFrame, .dataframe,
[data-testid="stTable"] {
    background-color: #0f1525 !important;
    color: #e0e6f0 !important;
}

/* Force dark on tab panels */
[data-testid="stTabContent"] {
    background-color: #0a0e1a !important;
}

/* Force dark on all form containers */
[data-testid="stForm"] {
    background-color: #0f1525 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 4px !important;
    padding: 1rem !important;
}

/* Force dark on number and text inputs */
input[type="number"], input[type="text"] {
    background-color: #0f1525 !important;
    color: #e0e6f0 !important;
}

/* Radio and checkbox labels */
.stRadio label, .stCheckbox label {
    color: #e0e6f0 !important;
}

/* Selectbox dropdown options */
[data-baseweb="popover"] {
    background-color: #0f1525 !important;
}
[data-baseweb="option"] {
    background-color: #0f1525 !important;
    color: #e0e6f0 !important;
}

/* Market status badge */
.market-badge {
    display: inline-flex;
    align-items: center;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}
.market-badge.market-open {
    background: rgba(0, 212, 255, 0.12);
    color: #00d4ff;
    border: 1px solid rgba(0, 212, 255, 0.6);
}
.market-badge.market-closed {
    background: rgba(128, 139, 166, 0.18);
    color: #808ba6;
    border: 1px solid rgba(128, 139, 166, 0.5);
}
.market-badge.market-pre,
.market-badge.market-post {
    background: rgba(255, 171, 64, 0.14);
    color: #ffab40;
    border: 1px solid rgba(255, 171, 64, 0.7);
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

        _sidebar_toggle_html = """
<div style="display:flex;align-items:center;height:100%;">
<button type="button" title="Toggle sidebar"
style="background:#0f1525;color:#00d4ff;border:1px solid #1e2d45;border-radius:4px;
padding:6px 12px;cursor:pointer;font-size:16px;line-height:1;"
onclick="(function(){var d=window.parent.document;var x=d.querySelector(
'[data-testid=&quot;collapsedControl&quot;]');if(x){x.click();return;}var s=d.querySelector(
'section[data-testid=&quot;stSidebar&quot;]');if(s){var b=s.querySelector(
'button[kind=&quot;headerNoPadding&quot;]')||s.querySelector(
'button[kind=&quot;header&quot;]')||s.querySelector('button');if(b){b.click();return;}}
var h=d.querySelector('[data-testid=&quot;stHeader&quot;] button');if(h){h.click();}})();">
☰</button>
</div>
"""
        _col_toggle, _col_bar = st.columns([1, 16])
        with _col_toggle:
            st.html(
                f'<div style="height:44px">{_sidebar_toggle_html}</div>',
                unsafe_allow_javascript=True,
            )

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
            color = "#26a69a" if chg >= 0 else "#ef5350"
            items_html += (
                f'<span style="margin-right:20px;'
                f'font-family:monospace;font-size:12px">'
                f'<span style="color:#00d4ff;'
                f'font-weight:bold">{sym}</span>'
                f'<span style="color:#e0e6f0;'
                f'margin-left:4px">${price:.2f}</span>'
                f'<span style="color:{color};'
                f'margin-left:4px">{sign}{chg:.2f}%</span>'
                f'</span>'
            )

        status_html = market_status_html()
        bar = (
            f'<div style="display:flex;align-items:center;'
            f'padding:6px 0;border-bottom:1px solid #1e2d45;'
            f'margin-bottom:8px;flex-wrap:wrap;">'
            f'{items_html}'
            f'<span style="margin-left:auto;font-size:11px">'
            f'{status_html}</span>'
            f'</div>'
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
