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
from trading.data.ticker_resolver import resolve_ticker

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

_scanner_uid = ""
_scanner_prefs = {}
try:
    from utils.session_utils import get_stable_user_id
    from config.user_store import load_user_preferences

    _scanner_uid = st.session_state.get("evolve_session_id") or get_stable_user_id()
    _scanner_prefs = load_user_preferences(_scanner_uid) or {}
except Exception:
    pass

if "scanner_prefs_hydrated" not in st.session_state:
    st.session_state["scanner_min_ai_score"] = float(
        _scanner_prefs.get("min_ai_score", 6.5)
    )
    _pu = str(_scanner_prefs.get("briefing_universe", ""))
    _def_scan_uni = "S&P 100 (~100, fastest)"
    if "NASDAQ100" in _pu or "NASDAQ" in _pu:
        _def_scan_uni = "S&P 500 + Nasdaq 100 (~600, moderate)"
    elif "SP500" in _pu:
        _def_scan_uni = "S&P 500 (~500, fast)"
    elif "Top 25" in _pu or "SP100" in _pu:
        _def_scan_uni = "S&P 100 (~100, fastest)"
    st.session_state["scanner_universe_choice"] = _def_scan_uni
    st.session_state["scanner_prefs_hydrated"] = True

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


def _batch_short_floats(symbols: list) -> dict:
    """Fetch short float % for multiple tickers in one yfinance call."""
    out: dict = {}
    if not symbols:
        return out
    try:
        import yfinance as yf

        _t = yf.Tickers(" ".join(symbols))
        for sym in symbols:
            try:
                _tk = getattr(_t, "tickers", {}).get(sym)
                if _tk is None and hasattr(_t, "tickers"):
                    _tk = _t.tickers.get(sym.upper())
                if _tk is None:
                    out[sym] = "N/A"
                    continue
                pct = _tk.info.get("shortPercentOfFloat")
                out[sym] = (
                    f"{pct * 100:.1f}%"
                    if pct is not None
                    else "N/A"
                )
            except Exception:
                out[sym] = "N/A"
    except Exception:
        for sym in symbols:
            out[sym] = "N/A"
    return out


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
    _uni_lbl = st.session_state.get("scanner_universe_choice", "selected universe")
    _fa = scan_result.get("filters_applied") or []
    _fa_s = ", ".join(_fa) if _fa else "none"
    st.caption(
        f"{scan_result.get('passed', 0)} of {scan_result.get('scanned', 0)} stocks "
        f"passed filters ({_uni_lbl}). Filters applied: {_fa_s}."
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

    _syms_no_sf = [
        r.get("symbol")
        for r in results
        if r.get("symbol") and "short_float" not in r
    ]
    if _syms_no_sf:
        _sf_map = _batch_short_floats(_syms_no_sf)
        for r in results:
            _s = r.get("symbol")
            if _s in _sf_map and "short_float" not in r:
                r["short_float"] = _sf_map[_s]

    for r in results:
        label, color = _news_score(r.get("symbol", ""))
        r["news_score"] = label
        r["_news_color"] = color

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
        if "quick_score" in df.columns:
            df = df[df["quick_score"] >= 7.5]
        elif "ai_grade" in df.columns:
            df = df[df["ai_grade"] == "A"]
    elif current_filter == "Breakout":
        if "vs_sma20" in df.columns:
            df = df[
                pd.to_numeric(
                    df["vs_sma20"],
                    errors="coerce"
                ).fillna(0) > 2
            ]
    elif current_filter == "Oversold":
        if "rsi" in df.columns:
            df = df[
                pd.to_numeric(
                    df["rsi"],
                    errors="coerce"
                ).fillna(50) < 30
            ]
    elif current_filter == "News Surge":
        if "news_score" in df.columns:
            df = df[
                df["news_score"] == "HOT"
            ]

    df_display = df.rename(
        columns={
            "symbol": "Symbol",
            "price": "Price",
            "change_20d": "20d Chg%",
            "rsi": "RSI",
            "vs_sma20": "vs SMA20%",
            "pct_from_52w_high": "vs 52w High%",
            "volume_ratio": "Vol Ratio",
            "quick_score": "Quick Score ⚡",
            "short_quick_score": "Short Score ⬇️",
            "ai_score": "AI Score",
            "ai_grade": "Grade",
            "news_score": "News",
            "short_float": "Short Float",
        }
    )
    if "News" not in df_display.columns and "news_score" in df.columns:
        df_display["News"] = df["news_score"]

    # Drop internal display columns
    _drop = ["_news_color", "news_color"]
    df_display = df_display.drop(
        columns=[
            c for c in _drop
            if c in df_display.columns
        ]
    )

    # Normalize signals column for Arrow
    # compatibility — convert lists to
    # comma-separated strings
    if "signals" in df_display.columns:
        def _fmt_signals(val):
            if isinstance(val, list):
                names = [
                    s.get("name", str(s))
                    if isinstance(s, dict)
                    else str(s)
                    for s in val
                ]
                return ", ".join(names) if names else "—"
            if val is None or (
                isinstance(val, float)
                and val != val
            ):
                return "—"
            return str(val) if val else "—"

        df_display["signals"] = (
            df_display["signals"].apply(
                _fmt_signals)
        )

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
    if "Quick Score ⚡" in df_display.columns:
        try:
            styler = styler.map(
                _color_score,
                subset=["Quick Score ⚡"],
            )
        except Exception:
            styler = styler.applymap(
                _color_score,
                subset=["Quick Score ⚡"],
            )
    if "News" in df_display.columns:
        try:
            styler = styler.map(_color_news_cell, subset=["News"])
        except Exception:
            styler = styler.applymap(_color_news_cell, subset=["News"])

    def _color_short(val):
        try:
            v = float(val)
            if v >= 7:
                return (
                    "background-color: #3a1a1a; color: #ef5350"
                )
            if v >= 6:
                return (
                    "background-color: #3a2a0a; color: #ff9800"
                )
            return ""
        except Exception:
            return ""

    if "Short Score ⬇️" in df_display.columns:
        try:
            styler = styler.map(
                _color_short,
                subset=["Short Score ⬇️"],
            )
        except Exception:
            styler = styler.applymap(
                _color_short,
                subset=["Short Score ⬇️"],
            )

    st.dataframe(styler, width="stretch", height=400, key="scanner_results_df")

    st.caption(
        "⚡ Quick Score is a fast technical estimate (RSI, momentum, volume, trend). "
        "For the full 16-signal AI Score, select a ticker in Signal Breakdown below."
    )

    _score_col = (
        "AI Score"
        if "AI Score" in df_display.columns
        else (
            "Quick Score ⚡"
            if "Quick Score ⚡" in df_display.columns
            else None
        )
    )
    if len(df_display) >= 3 and _score_col:
        st.markdown(f"#### {_score_col} Distribution")
        _sort_df = df_display.sort_values(_score_col, ascending=False)
        _bar_kw = dict(
            x="Symbol",
            y=_score_col,
            color=_score_col,
            color_continuous_scale="RdYlGn",
            range_color=[1, 10],
            height=300,
        )
        if "Grade" in _sort_df.columns:
            _bar_kw["text"] = "Grade"
        fig = px.bar(_sort_df, **_bar_kw)
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
        "Min Quick Score",
        min_value=0.0,
        max_value=10.0,
        value=6.0,
        step=0.5,
        key="scanner_min_ai_score",
        label_visibility="collapsed",
        help=(
            "Minimum Quick Score ⚡ for quick_technical filter. "
            "Quick Score is a fast technical estimate — use Signal Breakdown below "
            "for full AI Score."
        ),
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
            default=["quick_technical"],
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
        universe = [
            resolve_ticker(t.strip().upper(), validate=False)
            for t in custom_universe.split(",")
            if t.strip()
        ]
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

    if st.button("Run Scan", type="primary", key="scanner_run_btn"):
        progress_bar = st.progress(0.0, text="Scanning...")

        def _progress(done, total, phase="filter"):
            pct = (done / total
                   if total > 0 else 0)
            if phase == "filter":
                # Scale filter phase to first 80% of bar
                scaled = pct * 0.8
                progress_bar.progress(
                    scaled,
                    text=(
                        f"Filtering universe... "
                        f"{done}/{total} stocks"
                    ),
                )
            else:
                # Finalize phase: 80-100%
                scaled = 0.8 + pct * 0.2
                progress_bar.progress(
                    min(scaled, 1.0),
                    text=(
                        f"Ranking... "
                        f"{done}/{total} "
                        f"passed filters"
                    ),
                )

        with st.spinner("Running scan..."):
            scan_result = scan_market(
                filters=selected_filters,
                universe=universe,
                max_results=max_results,
                min_quick_score=float(min_score),
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
    if _FRAGMENT_OK:
        if st.session_state.get("scanner_streaming_mode"):
            @st.fragment(run_every=60)
            def _scanner_results():
                _scanner_table()
        else:
            @st.fragment
            def _scanner_results():
                _scanner_table()
        _scanner_results()
    else:
        _scanner_table()

    _sr = st.session_state.get("scanner_results") or {}
    if _sr and not _sr.get("error") and (_sr.get("results") or []):
        _brief_map = {
            "S&P 100 (~100, fastest)": "SP100 (balanced, ~60s)",
            "S&P 500 (~500, fast)": "SP500 (broadest, ~3min)",
            "S&P 500 + Nasdaq 100 (~600, moderate)": (
                "NASDAQ100 (tech-heavy)"
            ),
            "Russell 1000 (~1000, slow)": "SP500 (broadest, ~3min)",
            "Russell 3000 (~3000, very slow)": "SP500 (broadest, ~3min)",
        }
        if st.button(
            "Save current filters as briefing defaults",
            key="save_scanner_as_defaults",
        ):
            try:
                from config.user_store import (
                    load_user_preferences as _lp,
                    save_user_preferences as _sv,
                )
                from utils.session_utils import get_stable_user_id as _gsid

                _uid_sv = st.session_state.get("evolve_session_id") or _gsid()
                _sp = _lp(_uid_sv) or {}
                _sv(
                    _uid_sv,
                    {
                        **_sp,
                        "min_ai_score": float(min_score),
                        "briefing_universe": _brief_map.get(
                            universe_choice,
                            "SP100 (balanced, ~60s)",
                        ),
                    },
                )
                st.success(
                    "Saved as briefing defaults. "
                    "Home briefing will use these settings on the next run."
                )
            except Exception as e:
                st.caption(f"Could not save: {e}")

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
