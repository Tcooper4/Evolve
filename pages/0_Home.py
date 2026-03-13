"""
Evolve Home — Personalized morning briefing + live market events monitor.

Uses MemoryStore, market data (FallbackDataProvider), and risk snapshot to generate
a plain-English briefing and 2–4 dynamic cards. Includes a background polling loop
that scans a watchlist for volume/price spikes and shows a featured event with chart and news.
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
from streamlit_autorefresh import st_autorefresh

from config.user_store import load_user_preferences, save_user_preferences
from trading.analysis.market_monitor import scan_watchlist, DEFAULT_WATCHLIST
from trading.analysis.event_news_fetcher import fetch_news_around_event
from trading.analysis.news_ranker import rank_news_by_relevance
from trading.analysis.chart_builder import build_event_chart
from trading.data.earnings_calendar import get_upcoming_earnings

# Emergency fallback when JSON and Wikipedia both fail (e.g. Streamlit Cloud 403)
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

st.title("🏠 Good morning")
st.caption("Your personalized briefing. Simple, jargon-free.")


@st.cache_data(ttl=900)
def get_market_pulse() -> dict:
    """Fetch live data for major market indicators using yfinance.fast_info."""
    try:
        import yfinance as yf
        tickers = {
            "SPY": "S&P 500",
            "QQQ": "Nasdaq",
            "IWM": "Russell 2000",
            "^VIX": "VIX",
            "GLD": "Gold",
            "BTC-USD": "Bitcoin",
        }
        results: dict[str, dict] = {}
        for ticker, name in tickers.items():
            try:
                t = yf.Ticker(ticker)
                fast = getattr(t, "fast_info", {}) or {}
                price = fast.get("last_price")
                if price is None:
                    hist = t.history(period="1d")
                    if not hist.empty:
                        price = float(hist["Close"].iloc[-1])
                if price is None:
                    continue

                prev_close = fast.get("previous_close")
                if prev_close is None or prev_close == 0:
                    hist = t.history(period="2d")
                    if len(hist) >= 2:
                        prev_close = float(hist["Close"].iloc[-2])

                if prev_close is None or prev_close == 0:
                    continue

                chg = (float(price) - float(prev_close)) / float(prev_close) * 100
                results[ticker] = {"name": name, "price": float(price), "change": chg}
            except Exception:
                # Gracefully skip tickers that fail to load
                continue
        return results
    except Exception:
        return {}


@st.cache_data(ttl=900)
def get_prepost_price(symbol: str) -> dict:
    """Get pre/post market price if available, using yfinance.fast_info."""
    try:
        import yfinance as yf

        t = yf.Ticker(symbol)
        fast = getattr(t, "fast_info", {}) or {}

        regular = fast.get("last_price")
        if regular is None:
            hist = t.history(period="1d")
            if not hist.empty:
                regular = float(hist["Close"].iloc[-1])

        pre = fast.get("pre_market_price")
        post = fast.get("post_market_price")

        pre_chg_pct = None
        post_chg_pct = None

        prev_close = fast.get("previous_close")
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
        return {
            "regular": None,
            "pre": None,
            "post": None,
            "pre_chg_pct": None,
            "post_chg_pct": None,
        }


TOP_MOVERS_UNIVERSE_OPTIONS = [
    "S&P 100",
    "S&P 500",
    "S&P 500 + Nasdaq 100",
    "Russell 1000",
]


def _load_universe(name: str, fallback: list) -> list:
    """Load universe: JSON file → Wikipedia scrape → fallback list."""
    # Tier 1: pre-built JSON (fast, works on Cloud)
    _dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(_dir, "..", "data", "universes", f"{name}.json")
    try:
        with open(json_path, encoding="utf-8") as f:
            tickers = json.load(f)
        if tickers:
            return tickers
    except Exception:
        pass

    # Tier 2: Wikipedia scrape (works locally, may fail on Cloud)
    try:
        import pandas as pd  # type: ignore

        if name == "sp500":
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            if tables:
                table = tables[0]
                tickers = (
                    table[
                        [c for c in table.columns if str(c).lower() in ("symbol", "ticker")][0]
                    ]
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
            tickers_list: list[str] = []
            for t in tables:
                cols = [c.lower() for c in t.columns.astype(str)]
                if any("ticker" in c or "symbol" in c for c in cols):
                    for col in t.columns:
                        col_lower = str(col).lower()
                        if "ticker" in col_lower or "symbol" in col_lower:
                            series = (
                                t[col]
                                .astype(str)
                                .str.replace(".", "-", regex=False)
                                .str.upper()
                            )
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

    # Tier 3: emergency fallback
    return fallback


@st.cache_data(ttl=86400)
def load_universe_tickers(universe: str) -> list[str]:
    """Load ticker universe for the Home page Top Movers section."""
    universe = (universe or "S&P 100").strip()
    # Map UI label → JSON filename
    label_to_name = {
        "S&P 100": "sp100",
        "S&P 500": "sp500",
        "Nasdaq 100": "nasdaq100",
        "S&P 500 + Nasdaq 100": "sp500_nasdaq100",
        "Russell 1000": "russell1000",
        "Russell 3000": "russell3000",
    }
    name = label_to_name.get(universe, "sp100")
    return _load_universe(name, UNIVERSE_FALLBACK)


@st.cache_data(ttl=900)
def scan_top_movers(universe: str) -> dict:
    """Scan for top movers (gainers/losers) in the selected universe."""
    try:
        import yfinance as yf  # type: ignore
        import numpy as np  # type: ignore

        tickers = load_universe_tickers(universe)
        if not tickers:
            return {"as_of": None, "gainers": [], "losers": []}

        # Use batch download for efficiency (1d for same-day change)
        data = yf.download(
            tickers,
            period="1d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
        )

        if data.empty:
            return {"as_of": None, "gainers": [], "losers": []}

        as_of_ts = None

        movers = []
        if isinstance(data.columns, pd.MultiIndex):
            # Columns like ('Open', 'AAPL'), ('Close', 'AAPL'), ...
            open_row = data["Open"].iloc[0]
            close_row = data["Close"].iloc[-1]
            as_of_ts = data.index[-1].to_pydatetime() if hasattr(data.index[-1], "to_pydatetime") else data.index[-1]
            for sym in close_row.index:
                try:
                    o = float(open_row.get(sym, np.nan))
                    c = float(close_row.get(sym, np.nan))
                    if not np.isfinite(o) or not np.isfinite(c) or o == 0:
                        continue
                    chg = (c - o) / o * 100.0
                    movers.append({"symbol": sym, "price": c, "change": chg})
                except Exception:
                    continue
        else:
            # Single-ticker case
            try:
                o = float(data["Open"].iloc[0])
                c = float(data["Close"].iloc[-1])
                as_of_ts = data.index[-1].to_pydatetime() if hasattr(data.index[-1], "to_pydatetime") else data.index[-1]
                if o != 0:
                    chg = (c - o) / o * 100.0
                    movers.append({"symbol": tickers[0], "price": c, "change": chg})
            except Exception:
                pass

        # Add volume_ratio for filtering: 5d download to get volume history
        if movers and len(tickers) > 1:
            try:
                vol_data = yf.download(
                    tickers,
                    period="5d",
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    threads=True,
                    group_by="ticker",
                )
                # With group_by="ticker", columns are (Ticker, Field): level 0 = tickers, level 1 = Open/Close/Volume
                has_vol = not vol_data.empty and (
                    ("Volume" in vol_data.columns) if not isinstance(vol_data.columns, pd.MultiIndex) else ("Volume" in vol_data.columns.get_level_values(1))
                )
                if has_vol:
                    for m in movers:
                        sym = m["symbol"]
                        try:
                            if isinstance(vol_data.columns, pd.MultiIndex) and sym in vol_data.columns.get_level_values(0):
                                # MultiIndex: first level = ticker, second = field (e.g. Volume)
                                vol_series = vol_data[sym]["Volume"].dropna()
                            elif not isinstance(vol_data.columns, pd.MultiIndex):
                                vol_series = vol_data["Volume"].dropna()
                            else:
                                m["volume_ratio"] = 1.0
                                continue
                            if len(vol_series) >= 2:
                                last_vol = float(vol_series.iloc[-1])
                                avg_vol = float(vol_series.iloc[:-1].mean())
                                m["volume_ratio"] = round(last_vol / avg_vol, 2) if avg_vol and avg_vol > 0 else 1.0
                            else:
                                m["volume_ratio"] = 1.0
                        except Exception:
                            m["volume_ratio"] = 1.0
                else:
                    for m in movers:
                        m["volume_ratio"] = 1.0
            except Exception:
                for m in movers:
                    m["volume_ratio"] = 1.0
        else:
            for m in movers:
                m["volume_ratio"] = 1.0

        if not movers:
            return {"as_of": as_of_ts, "gainers": [], "losers": []}

        # Sort into gainers and losers
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
    return (
        f"Markets opened {direction} with SPY {spy.get('change', 0):+.2f}%. "
        f"VIX is {vix:.1f}. Notable movers: {movers}."
    )


@st.cache_data(ttl=600)
def generate_morning_briefing(market_pulse: dict, top_movers: list) -> str:
    """Generate a dynamic briefing based on actual market data."""
    try:
        from agents.llm.agent import get_prompt_agent

        agent = get_prompt_agent()
        if not agent:
            return _fallback_briefing(market_pulse, top_movers)

        spy_chg = (market_pulse or {}).get("SPY", {}).get("change", 0)
        vix = (market_pulse or {}).get("^VIX", {}).get("price", 20)
        movers_str = ", ".join(
            [f"{m['symbol']} {m['change']:+.1f}%" for m in (top_movers or [])[:5]]
        )

        prompt = f"""Write a concise 3-paragraph morning market briefing (like opening a newspaper).

Today's data: SPY {spy_chg:+.2f}%, VIX {vix:.1f}, Top movers: {movers_str}.
Paragraph 1: Overall market tone and what's driving it.
Paragraph 2: The 2-3 biggest individual stock stories today.
Paragraph 3: What to watch for the rest of the day.
Be specific; cite the actual % moves. Do NOT mention Apple unless it is actually one of the top movers."""

        response = agent.process_prompt(prompt)
        if isinstance(response, dict):
            return response.get("message", str(response))
        return response.message if hasattr(response, "message") else str(response)
    except Exception:
        return _fallback_briefing(market_pulse, top_movers)


# Market Pulse row (6 tiles)
pulse = get_market_pulse()
if pulse:
    cols = st.columns(6)
    for i, (ticker, data) in enumerate(pulse.items()):
        with cols[i]:
            fmt = f"{data['price']:.1f}" if ticker == "^VIX" else f"${data['price']:.2f}"
            st.metric(data["name"], fmt, f"{data['change']:+.2f}%")
            _pp = get_prepost_price(ticker)
            if _pp.get("pre") is not None:
                if _pp.get("pre_chg_pct") is not None:
                    st.caption(f"Pre-mkt: ${_pp['pre']:.2f} ({_pp['pre_chg_pct']:+.2f}%)")
                else:
                    st.caption(f"Pre-mkt: ${_pp['pre']:.2f}")
            elif _pp.get("post") is not None:
                if _pp.get("post_chg_pct") is not None:
                    st.caption(f"After-hrs: ${_pp['post']:.2f} ({_pp['post_chg_pct']:+.2f}%)")
                else:
                    st.caption(f"After-hrs: ${_pp['post']:.2f}")

# Fear & Greed proxy + Top Movers
vix = pulse.get("^VIX", {}).get("price", 20)
if vix < 15:
    fg_label, fg_color = "Extreme Greed", "green"
elif vix < 20:
    fg_label, fg_color = "Greed", "lightgreen"
elif vix < 25:
    fg_label, fg_color = "Neutral", "gray"
elif vix < 30:
    fg_label, fg_color = "Fear", "orange"
else:
    fg_label, fg_color = "Extreme Fear", "red"
st.markdown(f"**Market Sentiment:** VIX {vix:.1f} — :{fg_color}[{fg_label}]")

# Determine Top Movers universe from preferences (default to S&P 100)
session_id = st.session_state.get("evolve_session_id") or st.session_state.get("session_id", "")
saved_prefs = load_user_preferences(session_id) if session_id else {}
top_movers_universe = saved_prefs.get("home_top_movers_universe", TOP_MOVERS_UNIVERSE_OPTIONS[0])
if top_movers_universe not in TOP_MOVERS_UNIVERSE_OPTIONS:
    top_movers_universe = TOP_MOVERS_UNIVERSE_OPTIONS[0]

# Monitor settings in sidebar (early so vol/price thresholds available for Volume & News section)
with st.sidebar:
    st.markdown("### ⚙️ Monitor Settings")
    vol_threshold = st.slider(
        "Volume spike threshold",
        1.5, 5.0, 3.0, 0.5,
        help="Minimum volume multiple over 20-period average to trigger alert",
    )
    price_threshold = st.slider("Min price move %", 0.5, 5.0, 2.0, 0.5)

# Top Movers universe selectbox: show count, persist to user_store
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
    key="home_top_movers_universe_select",
)
if _selected_universe != top_movers_universe:
    save_user_preferences(session_id, {**saved_prefs, "home_top_movers_universe": _selected_universe})
    st.rerun()
top_movers_universe = _selected_universe

current_movers = scan_top_movers(top_movers_universe)
if (current_movers.get("gainers") or current_movers.get("losers")):
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
            for mover in items:
                icon = "🟢" if positive else "🔴"
                st.metric(
                    f"{icon} {mover['symbol']}",
                    f"${mover['price']:.2f}",
                    f"{mover['change']:+.2f}%",
                )
                try:
                    from trading.analysis.ai_score import compute_ai_score

                    _score = compute_ai_score(mover["symbol"])
                    if _score.get("error") is None:
                        st.caption(f"AI Score: {_score['overall_score']}/10 ({_score['grade']})")
                except Exception:
                    # If AI Score is unavailable, silently skip
                    pass

    _render_mover_column(col_gainers, "Top 5 Gainers", gainers[:5], positive=True)
    _render_mover_column(col_losers, "Top 5 Losers", losers[:5], positive=False)

    st.markdown("---")
    st.markdown("#### 🔍 Quick Scan")
    if st.button("Run Top AI Score Scan", key="home_quick_scan"):
        st.switch_page("pages/13_Scanner.py")

    # Upcoming earnings for a core watchlist
    try:
        _watchlist = [
            "AAPL",
            "MSFT",
            "NVDA",
            "GOOGL",
            "AMZN",
            "META",
            "TSLA",
            "JPM",
            "BAC",
            "SPY",
        ]
        _upcoming = [
            e
            for sym in _watchlist
            for e in [get_upcoming_earnings(sym, 14)]
            if e.get("is_within_window")
        ]
        _upcoming.sort(key=lambda x: x.get("days_until", 99))
        if _upcoming:
            with st.expander(
                f"Upcoming Earnings — next 14 days ({len(_upcoming)} stocks)",
                expanded=True,
            ):
                for _e in _upcoming[:6]:
                    c1, c2, c3, c4 = st.columns([1, 2, 1, 2])
                    c1.write(f"**{_e['symbol']}**")
                    c2.write(_e["next_earnings_date"])
                    c3.write(f"{_e['days_until']}d")
                    if _e.get("eps_estimate"):
                        try:
                            c4.write(f"EPS est ${_e['eps_estimate']:.2f}")
                        except Exception:
                            c4.write("EPS est —")
                    else:
                        c4.write("—")
    except Exception:
        pass

# 📊 Volume & News Events — 2×2 grid from top movers meeting volume/price thresholds
st.markdown("---")
st.subheader("📊 Volume & News Events")
_vol_th = vol_threshold
_price_th = price_threshold
_all_movers = (movers_state.get("gainers") or []) + (movers_state.get("losers") or [])
_qualified = [
    m for m in _all_movers
    if m.get("volume_ratio", 1.0) >= _vol_th and abs(m.get("change", 0)) >= _price_th
]
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

# Watchlist section
st.markdown("---")
st.subheader("Watchlist")
with st.expander("Watchlist", expanded=True):
    try:
        from components.watchlist_widget import render_watchlist

        render_watchlist()
    except Exception as e:
        st.caption(f"Watchlist unavailable: {e}")

    # 📈 Watchlist Chart — select one ticker from watchlist
    st.markdown("#### 📈 Watchlist Chart")
    try:
        from trading.data.watchlist import WatchlistManager
        _wm = WatchlistManager()
        _tickers = [e.get("symbol", "").upper() for e in (_wm.get_all() or []) if e.get("symbol")]
        if not _tickers:
            st.info("Add tickers to your watchlist above to see their chart here.")
        else:
            _sel = st.selectbox("Ticker", _tickers, key="home_watchlist_chart_ticker")
            if _sel:
                try:
                    render_news_candle_chart(_sel, period="6mo", show_annotations=True)
                except Exception as _e:
                    st.caption(f"Chart unavailable for {_sel}")
    except Exception as _e:
        st.caption(f"Watchlist chart unavailable: {_e}")

# Top Opportunities — quick AI Score scan on watchlist
st.markdown("---")
st.markdown("### 🎯 Top Opportunities")
st.caption("Stocks from your watchlist ranked by AI Score")

try:
    from trading.analysis.market_scanner import scan_market

    _quick_universe = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
                       "TSLA", "JPM", "BAC", "SPY", "QQQ", "GLD"]
    import time as _time
    _scan_key = "home_scan_result"
    _scan_ts_key = "home_scan_ts"
    _scan_ttl = 900  # 15 minutes

    _now = _time.time()
    _cached = st.session_state.get(_scan_key)
    _cached_ts = st.session_state.get(_scan_ts_key, 0)

    if _cached is None or (_now - _cached_ts) > _scan_ttl:
        with st.spinner("Refreshing opportunities..."):
            _cached = scan_market(
                filters=["top_ai_score"],
                universe=_quick_universe,
                max_results=5,
            )
        st.session_state[_scan_key] = _cached
        st.session_state[_scan_ts_key] = _now

    _scan = _cached
    _cache_age_min = int((_now - _cached_ts) / 60) if _cached_ts else 0

    _refresh_col, _age_col = st.columns([1, 4])
    with _refresh_col:
        if st.button("🔄 Refresh", key="home_scan_refresh"):
            st.session_state.pop(_scan_key, None)
            st.rerun()
    with _age_col:
        if _cached_ts > 0:
            st.caption(f"Last scanned {_cache_age_min}m ago")

    if _scan.get('error') is None and _scan.get('results'):
        _cols = st.columns(min(5, len(_scan['results'])))
        for _idx, _res in enumerate(_scan['results']):
            with _cols[_idx]:
                _grade_color = {'A': '🟢', 'B': '🔵', 'C': '🟡', 'D': '🟠', 'F': '🔴'}
                st.metric(
                    label=f"{_grade_color.get(_res.get('ai_grade', ''), '⚪')} {_res.get('symbol', '')}",
                    value=f"${_res.get('price', 0):,.2f}",
                    delta=f"AI {_res.get('ai_score', 0)}/10 ({_res.get('ai_grade', '')})",
                )
    elif _scan.get('passed', 0) == 0:
        st.caption("No stocks above AI Score 7.0 in watchlist right now.")
    else:
        st.caption(f"Scanner: {_scan.get('error', 'no results')}")

except Exception as _e:
    st.caption(f"Quick scan unavailable: {_e}")

# Session state for live market monitor
if "last_scan_time" not in st.session_state:
    st.session_state.last_scan_time = 0
if "event_feed" not in st.session_state:
    st.session_state.event_feed = []
if "featured_event" not in st.session_state:
    st.session_state.featured_event = None
if "selected_event" not in st.session_state:
    st.session_state.selected_event = None

# Session state for cached briefing
if "home_briefing_text" not in st.session_state:
    st.session_state.home_briefing_text = None
if "home_briefing_cards" not in st.session_state:
    st.session_state.home_briefing_cards = []
if "home_briefing_market_data" not in st.session_state:
    st.session_state.home_briefing_market_data = {}

# Once per session: run monitoring checks in background (do not block UI)
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

# Refresh button (clear cache and rerun)
if st.sidebar.button("🔄 Refresh briefing"):
    st.session_state.home_briefing_text = None
    st.session_state.home_briefing_cards = []
    st.session_state.home_briefing_market_data = {}
    st.rerun()

# Load persisted watchlist for this user
session_id = st.session_state.get("evolve_session_id") or st.session_state.get("session_id", "")
saved_prefs = load_user_preferences(session_id) if session_id else {}
default_watchlist = saved_prefs.get(
    "watchlist",
    "AAPL,NVDA,MSFT,TSLA,AMZN,META,GOOGL,SPY,QQQ,JPM,AMD,NFLX",
)

watchlist_pref = saved_prefs.get("watchlist", default_watchlist)
watchlist = [s.strip().upper() for s in watchlist_pref.split(",") if s.strip()]
if not watchlist:
    watchlist = DEFAULT_WATCHLIST

# Auto-refresh: check if we should run a scan
now = time.time()
time_since_scan = now - st.session_state.last_scan_time
should_scan = time_since_scan >= POLL_INTERVAL_SECONDS

if should_scan:
    with st.spinner("🔍 Scanning markets..."):
        new_spikes = scan_watchlist(
            watchlist=watchlist,
            volume_multiplier=vol_threshold,
            min_price_move_pct=price_threshold,
        )
        st.session_state.last_scan_time = time.time()
        if new_spikes:
            top = new_spikes[0]
            company = COMPANY_NAMES.get(top["symbol"], top["symbol"])
            try:
                articles = fetch_news_around_event(
                    top["symbol"],
                    company,
                    top["timestamp"].to_pydatetime() if hasattr(top["timestamp"], "to_pydatetime") else top["timestamp"],
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("Fetch news failed: %s", e)
                articles = []
            ranked = rank_news_by_relevance(
                articles, top["symbol"], company, top["direction"],
            )
            top["news"] = ranked[:5]
            top["top_headline"] = ranked[0]["title"] if ranked else "No news found"
            top["top_url"] = ranked[0]["url"] if ranked else ""
            existing_keys = {(e["symbol"], str(e["timestamp"])) for e in st.session_state.event_feed}
            key = (top["symbol"], str(top["timestamp"]))
            if key not in existing_keys:
                st.session_state.event_feed.insert(0, top)
                st.session_state.event_feed = st.session_state.event_feed[:20]
                st.session_state.featured_event = top

seconds_left = max(0, int(POLL_INTERVAL_SECONDS - time_since_scan))

    # Old live feed UI has been replaced by the news-linked chart above.

st.markdown("---")

# Generate briefing if not cached
movers = []
if st.session_state.home_briefing_text is None:
    movers = gainers + losers
    with st.spinner("Preparing your briefing..."):
        try:
            from trading.memory import get_memory_store
            from trading.services.home_briefing_service import generate_briefing
            store = get_memory_store()
            out = generate_briefing(store)
            # Prefer a dynamic briefing driven by live market pulse and top movers.
            st.session_state.home_briefing_text = generate_morning_briefing(pulse, movers)
            st.session_state.home_briefing_cards = out.get("cards", [])
            st.session_state.home_briefing_market_data = out.get("market_data", {})
        except Exception as e:
            logger.exception(f"Briefing generation failed: {e}")
            st.session_state.home_briefing_text = _fallback_briefing(pulse, movers)
            st.session_state.home_briefing_cards = []
            st.session_state.home_briefing_market_data = {}
    st.rerun()

market_data = st.session_state.home_briefing_market_data

# Remove duplicate SPY/AAPL weekly metric row; briefing and top index metrics already cover this section.
with st.container(border=True):
    st.markdown(st.session_state.home_briefing_text)

# Dynamic cards (2–4): container with border, content visible
cards = st.session_state.home_briefing_cards

if cards:
    st.markdown("---")
    st.subheader("What to know")
    n = len(cards)
    cols = st.columns(min(n, 4))
    for i, card in enumerate(cards[:4]):
        headline = card.get("headline", "Update")
        detail = card.get("detail", "")
        if isinstance(detail, str):
            # Minor cleanup for cramped phrases like "downfrom"
            detail = detail.replace("downfrom", "down from ")
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
                        series = entry.get("series")
                        dates = entry.get("dates") or []
                        if series and len(series) >= 2:
                            try:
                                if len(dates) == len(series):
                                    df = pd.DataFrame({"price": series}, index=pd.Index(dates))
                                    st.line_chart(df)
                                else:
                                    df = pd.DataFrame({"price": series})
                                    st.line_chart(df)
                            except Exception:
                                pass

# Follow-up: prominent label
st.markdown("---")
st.markdown("**Ask the AI anything about your portfolio or the markets:**")
follow_up = st.text_input(
    "Ask a follow-up question",
    placeholder="e.g. Why did my NVDA position go up? What should I do next?",
    key="home_follow_up",
    label_visibility="collapsed",
)
if follow_up and follow_up.strip():
    st.session_state["chat_prefill"] = follow_up.strip()
    try:
        st.switch_page("pages/1_Chat.py")
    except Exception as e:
        logger.warning(f"switch_page failed: {e}")
        st.info("Go to **Chat** in the sidebar and paste your question there.")

try:
    from ui.page_assistant import render_page_assistant
    render_page_assistant("Home")
except Exception:
    pass

# Auto-refresh every 60 seconds (no thread blocking)
st_autorefresh(interval=60_000, limit=None, key="market_monitor_refresh")
