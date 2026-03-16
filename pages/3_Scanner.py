# -*- coding: utf-8 -*-
"""
Market Scanner page — screen stocks by technical and AI-driven filters.
Reorganized from 13_Scanner with theme, price_cache, News Score, and fragment refresh.
"""
import json
import os
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from components.theme import market_status_html, render_top_bar, keyboard_shortcut_js
from trading.data.price_cache import get_history, get_news, batch_quotes

try:
    st.markdown(keyboard_shortcut_js(), unsafe_allow_html=True)
except Exception:
    pass
render_top_bar()
st.markdown(market_status_html(), unsafe_allow_html=True)

st.title("🔍 Scanner")
st.caption("Screen stocks by technical conditions and AI Score ranking")

try:
    from trading.analysis.market_scanner import scan_market, get_available_filters, DEFAULT_UNIVERSE
    scanner_available = True
except Exception as e:
    st.error(f"Scanner unavailable: {e}")
    scanner_available = False

UNIVERSE_FALLBACK = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
    "JPM", "UNH", "V", "XOM", "LLY", "MA", "JNJ", "PG", "HD", "MRK",
    "COST", "ABBV", "BAC", "WMT", "KO", "PEP", "CVX", "CRM", "AMD",
]

RUSSELL_1000_FALLBACK = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","LLY","AVGO",
    "TSLA","WMT","JPM","V","XOM","UNH","ORCL","MA","COST","HD",
    "PG","JNJ","ABBV","NFLX","BAC","CRM","CVX","MRK","AMD","PEP",
    "TMO","ADBE","ACN","LIN","MCD","CSCO","ABT","GE","TXN","DHR",
    "PM","CAT","ISRG","INTU","AMGN","VZ","NOW","MS","GS","RTX",
]

if not scanner_available:
    st.stop()


def _load_universe(name: str, fallback: list) -> list:
    _dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(_dir, "..", "data", "universes", f"{name}.json")
    try:
        with open(json_path, encoding="utf-8", errors="replace") as f:
            tickers = json.load(f)
        if tickers:
            return tickers
    except Exception:
        pass
    try:
        if name == "sp500":
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            if tables:
                table = tables[0]
                tickers = (
                    table[[c for c in table.columns if str(c).lower() in ("symbol", "ticker")][0]]
                    .astype(str)
                    .str.replace(".", "-", regex=False)
                    .str.upper()
                    .tolist()
                )
                if tickers:
                    return tickers
        elif name in ("russell1000", "russell3000"):
            url = "https://en.wikipedia.org/wiki/Russell_3000_Index"
            tables = pd.read_html(url)
            tickers_list = []
            for t in tables:
                cols = [c.lower() for c in t.columns.astype(str)]
                if any("ticker" in c or "symbol" in c for c in cols):
                    for col in t.columns:
                        if "ticker" in str(col).lower() or "symbol" in str(col).lower():
                            series = t[col].astype(str).str.replace(".", "-", regex=False).str.upper()
                            tickers_list.extend(series.tolist())
                    break
            tickers_list = [t for t in tickers_list if t and t != "nan"]
            if tickers_list:
                return tickers_list[:1000] if name == "russell1000" else tickers_list
        elif name == "sp500_nasdaq100":
            sp500 = _load_universe("sp500", fallback)
            nasdaq100 = _load_universe("nasdaq100", fallback)
            if sp500 or nasdaq100:
                return sorted(set(sp500 or []).union(nasdaq100 or []))
    except Exception:
        pass
    return fallback


@st.cache_data(ttl=86400)
def _load_scanner_universe(universe_label: str) -> list:
    universe_label = (universe_label or "").strip()
    if universe_label.startswith("S&P 100"):
        name = "sp100"
    elif universe_label.startswith("S&P 500 (~500"):
        name = "sp500"
    elif universe_label.startswith("S&P 500 + Nasdaq 100"):
        name = "sp500_nasdaq100"
    elif "Russell 3000" in universe_label:
        name = "russell3000"
    elif "Russell 1000" in universe_label:
        name = "russell1000"
    else:
        return DEFAULT_UNIVERSE or []
    return _load_universe(name, UNIVERSE_FALLBACK)


def _news_score(ticker: str) -> tuple:
    """Return (label, color) for News Score: HOT/POS/NEU/NEG."""
    pos_kw = ["surge", "beat", "rise", "gain", "growth", "up", "bull", "upgrade", "strong"]
    neg_kw = ["fall", "drop", "miss", "cut", "down", "bear", "downgrade", "weak", "loss"]
    try:
        items = get_news(ticker)
        now = datetime.utcnow()
        score = 0
        for item in items:
            title = (item.get("title") or item.get("headline") or "").lower()
            pub = item.get("published") or item.get("providerPublishTime")
            if pub:
                try:
                    if isinstance(pub, (int, float)):
                        dt = datetime.utcfromtimestamp(pub / 1000 if pub > 1e12 else pub)
                    else:
                        dt = datetime.fromisoformat(str(pub).replace("Z", "+00:00"))
                    if (now - dt.replace(tzinfo=None) if getattr(dt, "tzinfo", None) else now - dt).days > 1:
                        continue
                except Exception:
                    pass
            score += sum(1 for w in pos_kw if w in title) - sum(1 for w in neg_kw if w in title)
        if score >= 3:
            return "HOT", "#ff9800"
        if score >= 1:
            return "POS", "#26a69a"
        if score <= -1:
            return "NEG", "#ef5350"
        return "NEU", "#4a6080"
    except Exception:
        return "NEU", "#4a6080"


col_left, col_right = st.columns([2, 1])
with col_left:
    available_filters = get_available_filters()
    universe_choice = st.selectbox(
        "Stock Universe",
        [
            "S&P 100 (~100, fastest)",
            "S&P 500 (~500, fast)",
            "S&P 500 + Nasdaq 100 (~600, moderate)",
            "Russell 1000 (~1000, slow)",
            "Russell 3000 (~3000, very slow)",
        ],
        key="scanner_universe_choice",
    )
    selected_filters = st.multiselect(
        "Scan Filters",
        options=list(available_filters.keys()),
        default=["top_ai_score"],
        format_func=lambda k: f"{k}: {available_filters[k]}",
        help="Select one or more filters. Stocks must pass ALL selected filters.",
        key="scanner_filters",
    )
with col_right:
    max_results = st.slider("Max results", 5, 50, 20, key="scanner_max_results")
    custom_universe = st.text_input(
        "Custom universe (optional)",
        placeholder="AAPL,MSFT,NVDA,TSLA",
        help="Comma-separated tickers. Leave blank to use selected stock universe.",
        key="scanner_custom_universe",
    )

universe = None
if custom_universe.strip():
    universe = [t.strip().upper() for t in custom_universe.split(",") if t.strip()]
else:
    label_map = {
        "S&P 100 (~100, fastest)": "S&P 100",
        "S&P 500 (~500, fast)": "S&P 500 (~500, fast)",
        "S&P 500 + Nasdaq 100 (~600, moderate)": "S&P 500 + Nasdaq 100 (~600, moderate)",
        "Russell 1000 (~1000, slow)": "Russell 1000 (~1000, slow)",
        "Russell 3000 (~3000, very slow)": "Russell 3000 (~3000, very slow)",
    }
    loader_label = label_map.get(universe_choice, "S&P 100")
    universe = _load_scanner_universe(loader_label)
    if "Russell 1000" in universe_choice or "Russell 3000" in universe_choice:
        st.warning("⚠️ Scanning 1000+ stocks may take 2-3 minutes.")

if not selected_filters:
    st.warning("Select at least one filter to run a scan.")
    st.stop()

if st.button("🚀 Run Scan", type="primary", key="scanner_run_btn"):
    progress_bar = st.progress(0.0, text="Scanning...")
    def _progress(done, total):
        pct = done / total if total > 0 else 0
        progress_bar.progress(pct, text=f"Scanning {done}/{total}...")
    with st.spinner("Running scan..."):
        scan_result = scan_market(
            filters=selected_filters,
            universe=universe,
            max_results=max_results,
            progress_callback=_progress,
        )
    progress_bar.empty()
    if scan_result.get("error"):
        st.error(f"Scan error: {scan_result['error']}")
    else:
        st.session_state.scanner_results = scan_result
    st.rerun()

if "scanner_results" not in st.session_state:
    st.session_state.scanner_results = None

_st_version = tuple(int(x) for x in st.__version__.split(".")[:2])
_FRAGMENT_OK = _st_version >= (1, 37)


def _scanner_table():
    scan_result = st.session_state.get("scanner_results")
    if not scan_result or scan_result.get("error"):
        st.info("Configure filters above and click **Run Scan** to screen the market.")
        st.markdown("#### Available Filters")
        for k, v in get_available_filters().items():
            st.markdown(f"- **{k}**: {v}")
        return

    results = scan_result.get("results") or []
    st.success(
        f"✅ Scanned {scan_result.get('scanned', 0)} stocks — "
        f"{scan_result.get('passed', 0)} passed filters",
    )
    if not results:
        st.info("No stocks passed the selected filters. Try relaxing criteria.")
        return

    # News Score column
    for r in results:
        label, color = _news_score(r.get("symbol", ""))
        r["news_score"] = label
        r["_news_color"] = color

    df = pd.DataFrame(results)

    # Signal filter buttons (client-side filter)
    if "scanner_signal_filter" not in st.session_state:
        st.session_state.scanner_signal_filter = "All"
    filter_opts = ["All", "Score A+", "Breakout", "Oversold", "News Surge"]
    cols = st.columns(len(filter_opts))
    for i, opt in enumerate(filter_opts):
        with cols[i]:
            if st.button(opt, key=f"scanner_filter_{opt}", use_container_width=True):
                st.session_state.scanner_signal_filter = opt
                st.rerun()
    current_filter = st.session_state.scanner_signal_filter

    if current_filter == "Score A+":
        df = df[df.get("ai_grade", "") == "A"] if "ai_grade" in df.columns else df
    elif current_filter == "Breakout":
        df = df[df.get("vs_sma20", 0) > 2] if "vs_sma20" in df.columns else df
    elif current_filter == "Oversold":
        df = df[df.get("rsi", 50) < 30] if "rsi" in df.columns else df
    elif current_filter == "News Surge":
        df = df[df.get("news_score", "") == "HOT"] if "news_score" in df.columns else df

    df_display = df.rename(columns={
        "symbol": "Symbol", "price": "Price", "change_20d": "20d Chg%",
        "rsi": "RSI", "vs_sma20": "vs SMA20%", "pct_from_52w_high": "vs 52w High%",
        "volume_ratio": "Vol Ratio", "ai_score": "AI Score", "ai_grade": "Grade",
        "news_score": "News",
    })
    if "News" not in df_display.columns and "news_score" in df.columns:
        df_display["News"] = df["news_score"]

    def _color_score(val):
        try:
            v = float(val)
            if v >= 8:
                return "background-color: #d4edda"
            if v >= 6.5:
                return "background-color: #cce5ff"
            if v >= 5:
                return "background-color: #fff3cd"
            return "background-color: #f8d7da"
        except Exception:
            return ""

    def _color_news_cell(val):
        colors = {"HOT": "#ff9800", "POS": "#26a69a", "NEG": "#ef5350", "NEU": "#4a6080"}
        c = colors.get(str(val).strip(), "#4a6080")
        return f"background-color: {c}; color: #fff"

    styler = df_display.style
    if "AI Score" in df_display.columns:
        try:
            styler = styler.map(_color_score, subset=["AI Score"])
        except Exception:
            styler = styler.applymap(_color_score, subset=["AI Score"])
    if "News" in df_display.columns:
        try:
            styler = styler.map(_color_news_cell, subset=["News"])
        except Exception:
            styler = styler.applymap(_color_news_cell, subset=["News"])
    st.dataframe(styler, use_container_width=True, height=400, key="scanner_results_df")

    if len(df_display) >= 3 and "AI Score" in df_display.columns:
        st.markdown("#### AI Score Distribution")
        fig = px.bar(
            df_display.sort_values("AI Score", ascending=False),
            x="Symbol", y="AI Score", color="AI Score",
            color_continuous_scale="RdYlGn", range_color=[1, 10],
            text="Grade", height=300,
        )
        fig.update_layout(template="plotly_dark", showlegend=False)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True, key="scanner_dist_chart")

    st.markdown("#### Drill Down")
    selected_sym = st.selectbox(
        "Select stock to analyze",
        options=[r.get("symbol") for r in results if r.get("symbol")],
        key="scanner_drilldown",
    )
    if selected_sym:
        try:
            from trading.analysis.ai_score import compute_ai_score
            _hist = get_history(selected_sym, period="6mo")
            _ai = compute_ai_score(selected_sym, _hist)
            if _ai.get("error") is None:
                st.markdown(f"**{selected_sym}** — {_ai['summary']}")
                _sigs = pd.DataFrame(_ai.get("signals", []))
                if not _sigs.empty and "name" in _sigs.columns:
                    st.dataframe(_sigs[["name", "value", "impact", "description"]], use_container_width=True, key="scanner_drill_sigs")
        except Exception as _e:
            st.caption(f"Feature unavailable: {_e}")


if _FRAGMENT_OK:
    @st.fragment(run_every=30)
    def _scanner_results():
        _scanner_table()
    _scanner_results()
else:
    _scanner_table()
