# -*- coding: utf-8 -*-
"""
Market Scanner page — screen stocks by technical and AI-driven filters.
Reorganized from 13_Scanner with theme, price_cache, News Score, and fragment refresh.
"""
import json
import logging
import os
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from components.theme import inject_theme, render_top_bar, keyboard_shortcut_js
from trading.data.price_cache import get_history, get_news, batch_quotes

logger = logging.getLogger(__name__)

try:
    from utils.dataframe_utils import normalize_for_display
except ImportError:
    normalize_for_display = lambda df: df

try:
    st.markdown(keyboard_shortcut_js(), unsafe_allow_html=True)
except Exception:
    pass
inject_theme()
render_top_bar()

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
    except Exception as _e:
        logger.warning("Scanner: universe load failed (json) for %s: %s", name, _e)
        st.caption(f"⚠️ Universe load failed: {_e}")
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
    except Exception as _e:
        logger.warning(
            "Scanner: universe load failed (web) for %s: %s",
            name,
            _e,
        )
        st.caption(f"⚠️ Universe load failed: {_e}")
    return fallback


def _get_short_float(ticker: str) -> str:
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info
        pct = info.get("shortPercentOfFloat", None)
        if pct is None:
            return "N/A"
        return f"{pct * 100:.1f}%"
    except Exception:
        return "N/A"


@st.cache_data(ttl=300)
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

    try:
        from trading.data.price_cache import get_history
        from trading.strategies.pairs_trading_engine import PairsTradingEngine

        _top = [
            str(r.get("symbol"))
            for r in results[:10]
            if r.get("symbol")
        ]
        if len(_top) >= 2:
            _engine = PairsTradingEngine()
            _pd: dict = {}
            for _s in _top:
                _h = get_history(_s, period="1y")
                if _h is not None and not _h.empty:
                    _cm = {c.lower(): c for c in _h.columns}
                    _cc = _cm.get("close", _h.columns[0])
                    _pd[_s] = pd.DataFrame(
                        {"close": _h[_cc].astype(float)}
                    )
            _pairs = _engine.find_cointegrated_pairs(_pd, _top)
            if _pairs:
                _a, _b, _r0 = _pairs[0]
                _pv = getattr(_r0, "p_value", float("nan"))
                st.info(
                    f"Pairs signal: {_a}/{_b} cointegrated "
                    f"(p={_pv:.3f})"
                )
    except Exception:
        pass

    for r in results:
        label, color = _news_score(r.get("symbol", ""))
        r["news_score"] = label
        r["_news_color"] = color
        sym = r.get("symbol")
        if sym and "short_float" not in r:
            r["short_float"] = _get_short_float(sym)

    df = pd.DataFrame(results)

    if "scanner_signal_filter" not in st.session_state:
        st.session_state.scanner_signal_filter = "All"
    filter_opts = ["All", "Score A+", "Breakout", "Oversold", "News Surge"]
    st.markdown(
        '<div style="background: var(--secondary-bg, var(--secondary-background-color, '
        "#1e1e1e)); padding: 8px 12px; border-radius: 8px; margin-bottom: 12px;\">",
        unsafe_allow_html=True,
    )
    cols = st.columns(len(filter_opts))
    for i, opt in enumerate(filter_opts):
        with cols[i]:
            if st.button(opt, key=f"scanner_filter_{opt}", width="stretch"):
                st.session_state.scanner_signal_filter = opt
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    current_filter = st.session_state.scanner_signal_filter

    if current_filter == "Score A+":
        df = df[df.get("ai_grade", "") == "A"] if "ai_grade" in df.columns else df
    elif current_filter == "Breakout":
        df = df[df.get("vs_sma20", 0) > 2] if "vs_sma20" in df.columns else df
    elif current_filter == "Oversold":
        df = df[df.get("rsi", 50) < 30] if "rsi" in df.columns else df
    elif current_filter == "News Surge":
        df = df[df.get("news_score", "") == "HOT"] if "news_score" in df.columns else df

    df_display = df.rename(
        columns={
            "symbol": "Symbol",
            "price": "Price",
            "change_20d": "20d Chg%",
            "rsi": "RSI",
            "vs_sma20": "vs SMA20%",
            "pct_from_52w_high": "vs 52w High%",
            "volume_ratio": "Vol Ratio",
            "ai_score": "AI Score",
            "ai_grade": "Grade",
            "news_score": "News",
            "short_float": "Short Float",
        }
    )
    if "News" not in df_display.columns and "news_score" in df.columns:
        df_display["News"] = df["news_score"]

    def _color_score(val):
        try:
            v = float(val)
            if v >= 8:
                return "background-color: #1a4a2a; color: #26a69a"
            if v >= 6.5:
                return "background-color: #1a2a3a; color: #64b5f6"
            if v >= 5:
                return "background-color: #3a2a0a; color: #ff9800"
            return "background-color: #3a1a1a; color: #ef5350"
        except Exception:
            return ""

    def _color_news_cell(val):
        styles = {
            "HOT": "background-color: #2a1a0a; color: #ff9800",
            "POS": "background-color: #1a4a2a; color: #26a69a",
            "NEG": "background-color: #3a1a1a; color: #ef5350",
            "NEU": "background-color: #1a1a2a; color: #4a6080",
        }
        return styles.get(str(val).strip(), styles["NEU"])

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
    st.dataframe(styler, width="stretch", height=400, key="scanner_results_df")

    if len(df_display) >= 3 and "AI Score" in df_display.columns:
        st.markdown("#### AI Score Distribution")
        fig = px.bar(
            df_display.sort_values("AI Score", ascending=False),
            x="Symbol",
            y="AI Score",
            color="AI Score",
            color_continuous_scale="RdYlGn",
            range_color=[1, 10],
            text="Grade",
            height=300,
        )
        fig.update_layout(template="plotly_dark", showlegend=False)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, width="stretch", key="scanner_dist_chart")

    st.markdown("#### Signal breakdown")
    st.caption(
        "Select any symbol from the results above to see what's driving its score."
    )
    sym_opts = [r.get("symbol") for r in results if r.get("symbol")]
    if sym_opts:
        selected = st.selectbox(
            "Symbol",
            options=sym_opts,
            key="scanner_signal_sym",
            label_visibility="collapsed",
        )
        if selected:
            try:
                from trading.analysis.ai_score import compute_ai_score
                from trading.data.price_cache import get_history as _gh

                _hist = _gh(selected, period="6mo")
                _ai = compute_ai_score(selected, _hist)
                if _ai.get("error") is None:
                    st.markdown(f"**{selected}** — {_ai['summary']}")
                    _sigs = pd.DataFrame(_ai.get("signals", []))
                    if not _sigs.empty and "name" in _sigs.columns:
                        st.dataframe(
                            normalize_for_display(
                                _sigs[["name", "value", "impact", "description"]]
                            ),
                            width="stretch",
                            key="scanner_drill_sigs",
                        )
            except Exception as _e:
                st.caption(f"Feature unavailable: {_e}")


col_title, col_universe, col_score, col_stream = st.columns([3, 2, 2, 2])
with col_title:
    st.markdown("### Scanner")
with col_universe:
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
        label_visibility="collapsed",
    )
with col_score:
    min_score = st.slider(
        "Min AI Score",
        min_value=0.0,
        max_value=10.0,
        value=6.5,
        step=0.5,
        key="scanner_min_ai_score",
        label_visibility="collapsed",
    )
with col_stream:
    st.toggle(
        "Live mode",
        key="scanner_streaming_mode",
        help="Auto-refresh every 60 seconds",
    )

st.caption("Screen stocks by technical conditions and AI Score ranking")

tab_scan, tab_pairs = st.tabs([
    "Scanner",
    "Pairs trading",
])

available_filters = get_available_filters()

with tab_scan:
    col_left, col_right = st.columns([2, 1])
    with col_left:
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

    if st.session_state.get("scanner_streaming_mode"):
        try:
            from data.streaming_pipeline import create_streaming_pipeline

            _u = universe if universe else ["SPY"]
            create_streaming_pipeline(_u[:20], timeframes=["1d"], providers=["yfinance"])
            st.caption(
                "Streaming pipeline ready — quotes refresh with the scanner fragment."
            )
        except Exception as _se:
            st.caption(
                f"Streaming unavailable: {_se}. Using standard mode."
            )

    if not selected_filters:
        st.warning("Select at least one filter to run a scan.")
        st.stop()

    if st.button("Run Scan", type="primary", key="scanner_run_btn"):
        progress_bar = st.progress(0.0, text="Scanning...")

        def _progress(done, total):
            pct = done / total if total > 0 else 0
            progress_bar.progress(pct, text=f"Scanning {done}/{total}...")

        with st.spinner("Running scan..."):
            _filters_for_scan = [f for f in selected_filters if f != "top_ai_score"]
            scan_result = scan_market(
                filters=_filters_for_scan,
                universe=universe,
                max_results=max_results,
                progress_callback=_progress,
            )
        progress_bar.empty()
        if scan_result.get("error"):
            st.error(f"Scan error: {scan_result['error']}")
        else:
            if "top_ai_score" in selected_filters:
                _results = scan_result.get("results") or []
                _results = [
                    r for r in _results
                    if float(r.get("ai_score", 0) or 0) >= float(min_score)
                ]
                scan_result["results"] = _results
                scan_result["passed"] = len(_results)
            st.session_state.scanner_results = scan_result
            st.rerun()

    if "scanner_results" not in st.session_state:
        st.session_state.scanner_results = None

    _st_version = tuple(int(x) for x in st.__version__.split(".")[:2])
    _FRAGMENT_OK = _st_version >= (1, 37)
    if _FRAGMENT_OK:
        @st.fragment(run_every=60)
        def _scanner_results():
            _scanner_table()

        _scanner_results()
    else:
        _scanner_table()

with tab_pairs:
    st.markdown("#### Pairs trading")
    st.caption("Run a cointegration scan on symbols from your last scan results.")
    scan_result = st.session_state.get("scanner_results") or {}
    results = scan_result.get("results") or []
    if not results:
        st.info("Run a scan in the Scanner tab first to populate candidate symbols.")
    else:
        try:
            from trading.strategies.pairs_trading_engine import PairsTradingEngine
            from trading.data.price_cache import get_history

            _pe = PairsTradingEngine()
            if st.button(
                "Find cointegrated pairs",
                key="pairs_scan_btn",
                type="primary",
            ):
                with st.spinner("Running cointegration tests…"):
                    _syms = [
                        str(r.get("symbol"))
                        for r in results
                        if r.get("symbol")
                    ][:12]
                    _pd: dict = {}
                    for _s in _syms:
                        _h = get_history(_s, period="1y")
                        if _h is not None and not _h.empty:
                            _cm = {c.lower(): c for c in _h.columns}
                            _cc = _cm.get("close", _h.columns[0])
                            _pd[_s] = pd.DataFrame({"close": _h[_cc].astype(float)})
                    _pairs = _pe.find_cointegrated_pairs(_pd, _syms)
                    if _pairs:
                        st.dataframe(
                            normalize_for_display(
                                pd.DataFrame(
                                    [
                                        {
                                            "A": a,
                                            "B": b,
                                            "p_value": getattr(r, "p_value", None),
                                            "hedge_ratio": getattr(
                                                r, "hedge_ratio", None
                                            ),
                                        }
                                        for a, b, r in _pairs[:20]
                                    ]
                                )
                            ),
                            width="stretch",
                        )
                    else:
                        st.caption("No cointegrated pairs found in this sample.")
        except Exception as _pe:
            st.caption(f"Pairs trading unavailable: {_pe}")


# Page Assistant
try:
    from ui.page_assistant import render_page_assistant
    render_page_assistant("Scanner")
except Exception:
    pass
