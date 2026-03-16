# -*- coding: utf-8 -*-
"""
Evolve Dashboard — Personalized morning briefing + live market events monitor.
Reorganized from 0_Home.py with theme, price_cache, and fragment-based refresh.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import streamlit as st

from components.theme import market_status_html, render_top_bar, keyboard_shortcut_js
from trading.data.price_cache import get_quote, get_history, get_info, get_news

try:
    st.markdown(keyboard_shortcut_js(), unsafe_allow_html=True)
except Exception:
    pass
render_top_bar()
st.markdown(market_status_html(), unsafe_allow_html=True)

from config.user_store import load_user_preferences, save_user_preferences
from trading.analysis.market_monitor import scan_watchlist, DEFAULT_WATCHLIST
from trading.analysis.event_news_fetcher import fetch_news_around_event
from trading.analysis.news_ranker import rank_news_by_relevance
from trading.data.earnings_calendar import get_upcoming_earnings

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

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 60
COMPANY_NAMES = {
    "AAPL": "Apple", "NVDA": "Nvidia", "MSFT": "Microsoft",
    "TSLA": "Tesla", "AMZN": "Amazon", "META": "Meta",
    "GOOGL": "Google", "SPY": "S&P 500", "QQQ": "Nasdaq",
    "JPM": "JPMorgan", "BAC": "Bank of America", "GS": "Goldman Sachs",
    "AMD": "AMD", "NFLX": "Netflix", "UBER": "Uber",
    "PLTR": "Palantir", "ARM": "ARM Holdings",
}

PULSE_TICKERS = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq",
    "IWM": "Russell 2000",
    "^VIX": "VIX",
    "GLD": "Gold",
    "BTC-USD": "Bitcoin",
}


@st.cache_data(ttl=900)
def get_market_pulse() -> dict:
    """Fetch live data for major market indicators using price_cache.get_quote."""
    results = {}
    for ticker, name in PULSE_TICKERS.items():
        try:
            q = get_quote(ticker)
            price = q.get("price")
            if price is None:
                hist = get_history(ticker, period="1d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])
            if price is None:
                continue
            prev_close = q.get("prev_close")
            if prev_close is None or prev_close == 0:
                hist = get_history(ticker, period="5d")
                if len(hist) >= 2:
                    prev_close = float(hist["Close"].iloc[-2])
            if prev_close is None or prev_close == 0:
                continue
            chg = (float(price) - float(prev_close)) / float(prev_close) * 100
            results[ticker] = {"name": name, "price": float(price), "change": chg}
        except Exception:
            continue
    return results


@st.cache_data(ttl=900)
def get_prepost_price(symbol: str) -> dict:
    """Get pre/post market price using price_cache get_quote and get_info."""
    try:
        q = get_quote(symbol)
        regular = q.get("price")
        if regular is None:
            hist = get_history(symbol, period="1d")
            if not hist.empty:
                regular = float(hist["Close"].iloc[-1])
        prev_close = q.get("prev_close")
        info = get_info(symbol)
        pre = info.get("preMarketPrice")
        post = info.get("postMarketPrice")
        pre_chg_pct = None
        post_chg_pct = None
        if prev_close is not None and prev_close != 0:
            if pre is not None:
                pre_chg_pct = (float(pre) - float(prev_close)) / float(prev_close) * 100
            if post is not None:
                post_chg_pct = (float(post) - float(prev_close)) / float(prev_close) * 100
        return {
            "regular": regular,
            "pre": pre,
            "post": post,
            "pre_chg_pct": pre_chg_pct,
            "post_chg_pct": post_chg_pct,
        }
    except Exception:
        return {"regular": None, "pre": None, "post": None, "pre_chg_pct": None, "post_chg_pct": None}


TOP_MOVERS_UNIVERSE_OPTIONS = [
    "S&P 100",
    "S&P 500",
    "S&P 500 + Nasdaq 100",
    "Russell 1000",
]


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
def load_universe_tickers(universe: str) -> list:
    universe = (universe or "S&P 100").strip()
    label_to_name = {
        "S&P 100": "sp100", "S&P 500": "sp500", "Nasdaq 100": "nasdaq100",
        "S&P 500 + Nasdaq 100": "sp500_nasdaq100", "Russell 1000": "russell1000", "Russell 3000": "russell3000",
    }
    name = label_to_name.get(universe, "sp100")
    return _load_universe(name, UNIVERSE_FALLBACK)


@st.cache_data(ttl=900)
def scan_top_movers(universe: str) -> dict:
    try:
        import yfinance as yf
        import numpy as np

        tickers = load_universe_tickers(universe)
        if not tickers:
            return {"as_of": None, "gainers": [], "losers": []}
        data = yf.download(tickers, period="1d", interval="1d", auto_adjust=False, progress=False, threads=True)
        if data.empty:
            return {"as_of": None, "gainers": [], "losers": []}
        as_of_ts = None
        movers = []
        if isinstance(data.columns, pd.MultiIndex):
            open_row = data["Open"].iloc[0]
            close_row = data["Close"].iloc[-1]
            as_of_ts = data.index[-1].to_pydatetime() if hasattr(data.index[-1], "to_pydatetime") else data.index[-1]
            for sym in close_row.index:
                try:
                    o, c = float(open_row.get(sym, np.nan)), float(close_row.get(sym, np.nan))
                    if not np.isfinite(o) or not np.isfinite(c) or o == 0:
                        continue
                    movers.append({"symbol": sym, "price": c, "change": (c - o) / o * 100.0})
                except Exception:
                    continue
        else:
            try:
                o, c = float(data["Open"].iloc[0]), float(data["Close"].iloc[-1])
                as_of_ts = data.index[-1].to_pydatetime() if hasattr(data.index[-1], "to_pydatetime") else data.index[-1]
                if o != 0:
                    movers.append({"symbol": tickers[0], "price": c, "change": (c - o) / o * 100.0})
            except Exception:
                pass
        for m in movers:
            m["volume_ratio"] = 1.0
        if not movers:
            return {"as_of": as_of_ts, "gainers": [], "losers": []}
        gainers = sorted([m for m in movers if m["change"] > 0], key=lambda x: x["change"], reverse=True)[:5]
        losers = sorted([m for m in movers if m["change"] < 0], key=lambda x: x["change"])[:5]
        return {
            "as_of": as_of_ts.isoformat() if hasattr(as_of_ts, "isoformat") else str(as_of_ts),
            "gainers": gainers,
            "losers": losers,
        }
    except Exception:
        return {"as_of": None, "gainers": [], "losers": []}


def _fallback_briefing(market_pulse: dict, top_movers: list) -> str:
    spy = (market_pulse or {}).get("SPY", {})
    vix = (market_pulse or {}).get("^VIX", {}).get("price", 20)
    direction = "higher" if spy.get("change", 0) > 0 else "lower"
    movers = ", ".join([f"{m['symbol']} ({m['change']:+.1f}%)" for m in (top_movers or [])[:3]]) or "None"
    return f"Markets opened {direction} with SPY {spy.get('change', 0):+.2f}%. VIX is {vix:.1f}. Notable movers: {movers}."


@st.cache_data(ttl=600)
def generate_morning_briefing(market_pulse: dict, top_movers: list) -> str:
    try:
        from agents.llm.agent import get_prompt_agent
        agent = get_prompt_agent()
        if not agent:
            return _fallback_briefing(market_pulse, top_movers)
        spy_chg = (market_pulse or {}).get("SPY", {}).get("change", 0)
        vix = (market_pulse or {}).get("^VIX", {}).get("price", 20)
        movers_str = ", ".join([f"{m['symbol']} {m['change']:+.1f}%" for m in (top_movers or [])[:5]])
        prompt = f"""Write a concise 3-paragraph morning market briefing (like opening a newspaper).
Today's data: SPY {spy_chg:+.2f}%, VIX {vix:.1f}, Top movers: {movers_str}.
Paragraph 1: Overall market tone. Paragraph 2: The 2-3 biggest stock stories. Paragraph 3: What to watch. Be specific. Do NOT mention Apple unless it is a top mover."""
        response = agent.process_prompt(prompt)
        if isinstance(response, dict):
            return response.get("message", str(response))
        return response.message if hasattr(response, "message") else str(response)
    except Exception:
        return _fallback_briefing(market_pulse, top_movers)


st.title("🏠 Good morning")
st.caption("Your personalized briefing. Simple, jargon-free.")

pulse = get_market_pulse()
if pulse:
    cols = st.columns(6)
    for i, (ticker, data) in enumerate(pulse.items()):
        with cols[i]:
            fmt = f"{data['price']:.1f}" if ticker == "^VIX" else f"${data['price']:.2f}"
            st.metric(data["name"], fmt, f"{data['change']:+.2f}%")
            _pp = get_prepost_price(ticker)
            if _pp.get("pre") is not None:
                st.caption(f"Pre-mkt: ${_pp['pre']:.2f}" + (f" ({_pp['pre_chg_pct']:+.2f}%)" if _pp.get("pre_chg_pct") is not None else ""))
            elif _pp.get("post") is not None:
                st.caption(f"After-hrs: ${_pp['post']:.2f}" + (f" ({_pp['post_chg_pct']:+.2f}%)" if _pp.get("post_chg_pct") is not None else ""))

vix = pulse.get("^VIX", {}).get("price", 20)
fg_label = "Extreme Greed" if vix < 15 else "Greed" if vix < 20 else "Neutral" if vix < 25 else "Fear" if vix < 30 else "Extreme Fear"
fg_color = "green" if vix < 15 else "lightgreen" if vix < 20 else "gray" if vix < 25 else "orange" if vix < 30 else "red"
st.markdown(f"**Market Sentiment:** VIX {vix:.1f} — :{fg_color}[{fg_label}]")

session_id = st.session_state.get("evolve_session_id") or st.session_state.get("session_id", "")
saved_prefs = load_user_preferences(session_id) if session_id else {}
top_movers_universe = saved_prefs.get("home_top_movers_universe", TOP_MOVERS_UNIVERSE_OPTIONS[0])
if top_movers_universe not in TOP_MOVERS_UNIVERSE_OPTIONS:
    top_movers_universe = TOP_MOVERS_UNIVERSE_OPTIONS[0]

with st.sidebar:
    st.markdown("### ⚙️ Monitor Settings")
    vol_threshold = st.slider(
        "Volume spike threshold", 1.5, 5.0, 3.0, 0.5,
        help="Minimum volume multiple over 20-period average to trigger alert",
        key="dash_vol_threshold",
    )
    price_threshold = st.slider("Min price move %", 0.5, 5.0, 2.0, 0.5, key="dash_price_threshold")

@st.cache_data(ttl=3600)
def _universe_stock_count(universe: str) -> int:
    return len(load_universe_tickers(universe))

_universe_opts = TOP_MOVERS_UNIVERSE_OPTIONS
_universe_idx = _universe_opts.index(top_movers_universe) if top_movers_universe in _universe_opts else 0
_selected_universe = st.selectbox(
    "Top Movers universe",
    options=_universe_opts,
    index=_universe_idx,
    format_func=lambda u: f"{u} ({_universe_stock_count(u)} stocks)",
    key="dash_top_movers_universe",
)
if _selected_universe != top_movers_universe:
    save_user_preferences(session_id, {**saved_prefs, "home_top_movers_universe": _selected_universe})
    st.rerun()
top_movers_universe = _selected_universe

current_movers = scan_top_movers(top_movers_universe)
if current_movers.get("gainers") or current_movers.get("losers"):
    st.session_state["home_last_top_movers"] = current_movers

movers_state = st.session_state.get("home_last_top_movers", current_movers)
gainers = movers_state.get("gainers") or []
losers = movers_state.get("losers") or []
movers = gainers + losers

if gainers or losers:
    st.subheader("Today's Top Movers")
    as_of = movers_state.get("as_of")
    caption_parts = [f"Universe: {top_movers_universe}"]
    if as_of:
        try:
            ts = as_of
            if isinstance(ts, str) and "T" in ts:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if getattr(dt, "tzinfo", None):
                    dt = dt.replace(tzinfo=None)
                as_of_fmt = dt.strftime("%b %d, %Y") if dt.date() != datetime.now().date() else "today"
            else:
                as_of_fmt = str(ts)[:10] if ts else "today"
        except Exception:
            as_of_fmt = "today"
        caption_parts.append(f"as of {as_of_fmt}")
    st.caption(" • ".join(caption_parts))
    col_gainers, col_losers = st.columns(2)

    def _render_mover_column(col, title, items, positive: bool):
        with col:
            st.markdown(f"**{title}**")
            if not items:
                st.caption("No data available.")
                return
            for j, mover in enumerate(items):
                icon = "🟢" if positive else "🔴"
                try:
                    st.metric(f"{icon} {mover['symbol']}", f"${mover['price']:.2f}", f"{mover['change']:+.2f}%")
                    from trading.analysis.ai_score import compute_ai_score
                    _score = compute_ai_score(mover["symbol"])
                    if _score.get("error") is None:
                        st.caption(f"AI Score: {_score['overall_score']}/10 ({_score['grade']})")
                except Exception as e:
                    st.caption(f"Feature unavailable: {e}")

    _render_mover_column(col_gainers, "Top 5 Gainers", gainers[:5], True)
    _render_mover_column(col_losers, "Top 5 Losers", losers[:5], False)
    st.markdown("---")
    st.markdown("#### 🔍 Quick Scan")
    if st.button("Run Top AI Score Scan", key="dash_quick_scan"):
        st.switch_page("pages/3_Scanner.py")

    try:
        _watchlist = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM", "BAC", "SPY"]
        _upcoming = [e for sym in _watchlist for e in [get_upcoming_earnings(sym, 14)] if e.get("is_within_window")]
        _upcoming.sort(key=lambda x: x.get("days_until", 99))
        if _upcoming:
            with st.expander(f"Upcoming Earnings — next 14 days ({len(_upcoming)} stocks)", expanded=True):
                for _e in _upcoming[:6]:
                    c1, c2, c3, c4 = st.columns([1, 2, 1, 2])
                    c1.write(f"**{_e['symbol']}**")
                    c2.write(_e["next_earnings_date"])
                    c3.write(f"{_e['days_until']}d")
                    c4.write(f"EPS est ${_e['eps_estimate']:.2f}" if _e.get("eps_estimate") else "—")
    except Exception as e:
        st.caption(f"Feature unavailable: {e}")

st.markdown("---")
st.subheader("📊 Volume & News Events")
_vol_th = vol_threshold
_price_th = price_threshold
_all_movers = (movers_state.get("gainers") or []) + (movers_state.get("losers") or [])
_qualified = [m for m in _all_movers if m.get("volume_ratio", 1.0) >= _vol_th and abs(m.get("change", 0)) >= _price_th]
_qualified.sort(key=lambda x: x.get("volume_ratio", 0), reverse=True)
_top4 = _qualified[:4]
try:
    from components.news_candle_chart import render_news_candle_chart
except Exception:
    render_news_candle_chart = None

if render_news_candle_chart:
    _row1_cols = st.columns(2)
    _row2_cols = st.columns(2)
    for _idx in range(4):
        _col = _row1_cols[_idx % 2] if _idx < 2 else _row2_cols[_idx % 2]
        with _col:
            if _idx < len(_top4):
                _sym = _top4[_idx].get("symbol", "")
                try:
                    render_news_candle_chart(_sym, period="3mo", interval="1d", show_annotations=True)
                except Exception as _e:
                    st.caption(f"Chart unavailable for {_sym}")
            else:
                st.info("No additional volume events detected")

st.markdown("---")
st.subheader("Watchlist")
with st.expander("Watchlist", expanded=True):
    try:
        from components.watchlist_widget import render_watchlist
        render_watchlist()
    except Exception as e:
        st.caption(f"Watchlist unavailable: {e}")
    st.markdown("#### 📈 Watchlist Chart")
    try:
        from trading.data.watchlist import WatchlistManager
        from components.news_candle_chart import render_news_candle_chart as _rcc
        _wm = WatchlistManager()
        _tickers = [e.get("symbol", "").upper() for e in (_wm.get_all() or []) if e.get("symbol")]
        if not _tickers:
            st.info("Add tickers to your watchlist above to see their chart here.")
        else:
            _sel = st.selectbox("Ticker", _tickers, key="dash_watchlist_chart_ticker")
            if _sel:
                try:
                    _rcc(_sel, period="6mo", show_annotations=True)
                except Exception as _e:
                    st.caption(f"Chart unavailable for {_sel}")
    except Exception as _e:
        st.caption(f"Watchlist chart unavailable: {_e}")

st.markdown("---")
st.markdown("### 🎯 Top Opportunities")
st.caption("Stocks from your watchlist ranked by AI Score")
try:
    from trading.analysis.market_scanner import scan_market
    import time as _time
    _quick_universe = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM", "BAC", "SPY", "QQQ", "GLD"]
    _scan_key, _scan_ts_key, _scan_ttl = "home_scan_result", "home_scan_ts", 900
    _now = _time.time()
    _cached = st.session_state.get(_scan_key)
    _cached_ts = st.session_state.get(_scan_ts_key, 0)
    if _cached is None or (_now - _cached_ts) > _scan_ttl:
        with st.spinner("Refreshing opportunities..."):
            _cached = scan_market(filters=["top_ai_score"], universe=_quick_universe, max_results=5)
        st.session_state[_scan_key] = _cached
        st.session_state[_scan_ts_key] = _now
    _scan = _cached
    _cache_age_min = int((_now - _cached_ts) / 60) if _cached_ts else 0
    _refresh_col, _age_col = st.columns([1, 4])
    with _refresh_col:
        if st.button("🔄 Refresh", key="dash_scan_refresh"):
            st.session_state.pop(_scan_key, None)
            st.rerun()
    with _age_col:
        if _cached_ts > 0:
            st.caption(f"Last scanned {_cache_age_min}m ago")
    if _scan.get("error") is None and _scan.get("results"):
        _cols = st.columns(min(5, len(_scan["results"])))
        for _idx, _res in enumerate(_scan["results"]):
            with _cols[_idx]:
                _grade_color = {"A": "🟢", "B": "🔵", "C": "🟡", "D": "🟠", "F": "🔴"}
                st.metric(
                    label=f"{_grade_color.get(_res.get('ai_grade', ''), '⚪')} {_res.get('symbol', '')}",
                    value=f"${_res.get('price', 0):,.2f}",
                    delta=f"AI {_res.get('ai_score', 0)}/10 ({_res.get('ai_grade', '')})",
                )
    elif _scan.get("passed", 0) == 0:
        st.caption("No stocks above AI Score 7.0 in watchlist right now.")
    else:
        st.caption(f"Scanner: {_scan.get('error', 'no results')}")
except Exception as _e:
    st.caption(f"Quick scan unavailable: {_e}")

if "last_scan_time" not in st.session_state:
    st.session_state.last_scan_time = 0
if "event_feed" not in st.session_state:
    st.session_state.event_feed = []
if "featured_event" not in st.session_state:
    st.session_state.featured_event = None
if "selected_event" not in st.session_state:
    st.session_state.selected_event = None
if "home_briefing_text" not in st.session_state:
    st.session_state.home_briefing_text = None
if "home_briefing_cards" not in st.session_state:
    st.session_state.home_briefing_cards = []
if "home_briefing_market_data" not in st.session_state:
    st.session_state.home_briefing_market_data = {}

if not st.session_state.get("monitoring_ran_this_session"):
    def _run_monitoring_once():
        try:
            from trading.services.monitoring_tools import check_model_degradation, check_strategy_degradation
            check_model_degradation()
            check_strategy_degradation()
        except Exception as e:
            logger.debug("Background monitoring: %s", e)
    import threading
    t = threading.Thread(target=_run_monitoring_once, daemon=True)
    t.start()
    st.session_state["monitoring_ran_this_session"] = True

if st.sidebar.button("🔄 Refresh briefing", key="dash_refresh_briefing"):
    st.session_state.home_briefing_text = None
    st.session_state.home_briefing_cards = []
    st.session_state.home_briefing_market_data = {}
    st.rerun()

session_id = st.session_state.get("evolve_session_id") or st.session_state.get("session_id", "")
saved_prefs = load_user_preferences(session_id) if session_id else {}
default_watchlist = saved_prefs.get("watchlist", "AAPL,NVDA,MSFT,TSLA,AMZN,META,GOOGL,SPY,QQQ,JPM,AMD,NFLX")
watchlist_pref = saved_prefs.get("watchlist", default_watchlist)
watchlist = [s.strip().upper() for s in watchlist_pref.split(",") if s.strip()]
if not watchlist:
    watchlist = DEFAULT_WATCHLIST


# Fragment version check (st.fragment available in Streamlit >= 1.37)
_st_version = tuple(int(x) for x in st.__version__.split(".")[:2])
_FRAGMENT_OK = _st_version >= (1, 37)

def _run_market_scan():
    vol_th = st.session_state.get("dash_vol_threshold", 3.0)
    price_th = st.session_state.get("dash_price_threshold", 2.0)
    now = time.time()
    time_since_scan = now - st.session_state.last_scan_time
    if time_since_scan >= POLL_INTERVAL_SECONDS:
        with st.spinner("🔍 Scanning markets..."):
            new_spikes = scan_watchlist(
                watchlist=watchlist,
                volume_multiplier=vol_th,
                min_price_move_pct=price_th,
            )
            st.session_state.last_scan_time = time.time()
            if new_spikes:
                top = new_spikes[0]
                company = COMPANY_NAMES.get(top["symbol"], top["symbol"])
                try:
                    articles = fetch_news_around_event(
                        top["symbol"], company,
                        top["timestamp"].to_pydatetime() if hasattr(top["timestamp"], "to_pydatetime") else top["timestamp"],
                    )
                except Exception as e:
                    logger.warning("Fetch news failed: %s", e)
                    articles = []
                ranked = rank_news_by_relevance(articles, top["symbol"], company, top["direction"])
                top["news"] = ranked[:5]
                top["top_headline"] = ranked[0]["title"] if ranked else "No news found"
                top["top_url"] = ranked[0]["url"] if ranked else ""
                existing_keys = {(e["symbol"], str(e["timestamp"])) for e in st.session_state.event_feed}
                key = (top["symbol"], str(top["timestamp"]))
                if key not in existing_keys:
                    st.session_state.event_feed.insert(0, top)
                    st.session_state.event_feed = st.session_state.event_feed[:20]
                    st.session_state.featured_event = top

if _FRAGMENT_OK:
    @st.fragment(run_every=60)
    def _market_monitor_fragment():
        _run_market_scan()
    _market_monitor_fragment()
else:
    _run_market_scan()

st.markdown("---")

if st.session_state.home_briefing_text is None:
    movers_for_briefing = gainers + losers
    with st.spinner("Preparing your briefing..."):
        try:
            from trading.memory import get_memory_store
            from trading.services.home_briefing_service import generate_briefing
            store = get_memory_store()
            out = generate_briefing(store)
            st.session_state.home_briefing_text = generate_morning_briefing(pulse, movers_for_briefing)
            st.session_state.home_briefing_cards = out.get("cards", [])
            st.session_state.home_briefing_market_data = out.get("market_data", {})
        except Exception as e:
            logger.exception("Briefing generation failed: %s", e)
            st.session_state.home_briefing_text = _fallback_briefing(pulse, movers_for_briefing)
            st.session_state.home_briefing_cards = []
            st.session_state.home_briefing_market_data = {}
    st.rerun()

market_data = st.session_state.home_briefing_market_data
with st.container(border=True):
    st.markdown(st.session_state.home_briefing_text)

cards = st.session_state.home_briefing_cards
if cards:
    st.markdown("---")
    st.subheader("What to know")
    cols = st.columns(min(len(cards), 4))
    for i, card in enumerate(cards[:4]):
        headline = card.get("headline", "Update")
        detail = (card.get("detail", "") or "").replace("downfrom", "down from ")
        card_type = card.get("card_type", "news")
        with cols[i % len(cols)]:
            with st.container(border=True):
                st.markdown(f"**{headline}**")
                if detail:
                    st.caption(detail)
                if card_type == "price_chart" and market_data:
                    symbol = card.get("symbol") or (list(market_data.keys())[0] if market_data else None)
                    if symbol and symbol in market_data:
                        entry = market_data[symbol]
                        series, dates = entry.get("series"), entry.get("dates") or []
                        if series and len(series) >= 2:
                            try:
                                df = pd.DataFrame({"price": series}, index=pd.Index(dates)) if len(dates) == len(series) else pd.DataFrame({"price": series})
                                st.line_chart(df)
                            except Exception:
                                pass

st.markdown("---")
st.markdown("**Ask the AI anything about your portfolio or the markets:**")
follow_up = st.text_input(
    "Ask a follow-up question",
    placeholder="e.g. Why did my NVDA position go up? What should I do next?",
    key="dash_follow_up",
    label_visibility="collapsed",
)
if follow_up and follow_up.strip():
    st.session_state["chat_prefill"] = follow_up.strip()
    try:
        st.switch_page("pages/6_Chat.py")
    except Exception as e:
        logger.warning("switch_page failed: %s", e)
        st.info("Go to **Chat** in the sidebar and paste your question there.")

try:
    from ui.page_assistant import render_page_assistant
    render_page_assistant("Home")
except Exception:
    pass
