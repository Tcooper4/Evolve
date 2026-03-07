"""
Evolve Home — Personalized morning briefing + live market events monitor.

Uses MemoryStore, market data (FallbackDataProvider), and risk snapshot to generate
a plain-English briefing and 2–4 dynamic cards. Includes a background polling loop
that scans a watchlist for volume/price spikes and shows a featured event with chart and news.
"""

import logging
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

st.set_page_config(
    page_title="Evolve Home",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="auto",
)

st.title("🏠 Good morning")
st.caption("Your personalized briefing. Simple, jargon-free.")

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

# Monitor settings in sidebar
with st.sidebar:
    st.markdown("### ⚙️ Monitor Settings")
    vol_threshold = st.slider(
        "Volume spike threshold",
        1.5, 5.0, 3.0, 0.5,
        help="Minimum volume multiple over 20-period average to trigger alert",
    )
    price_threshold = st.slider("Min price move %", 0.5, 5.0, 2.0, 0.5)
    custom_watchlist = st.text_input(
        "Watchlist (comma separated)",
        default_watchlist,
        key="watchlist_input",
    )
if custom_watchlist.strip() != default_watchlist.strip() and session_id:
    save_user_preferences(session_id, {**saved_prefs, "watchlist": custom_watchlist.strip()})

watchlist = [s.strip().upper() for s in custom_watchlist.split(",") if s.strip()]
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
            articles = fetch_news_around_event(
                top["symbol"],
                company,
                top["timestamp"].to_pydatetime() if hasattr(top["timestamp"], "to_pydatetime") else top["timestamp"],
            )
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

# Two columns: featured event (left 70%), live feed (right 30%)
col_main, col_feed = st.columns([7, 3])
with col_main:
    if st.session_state.featured_event or st.session_state.selected_event:
        event = st.session_state.selected_event or st.session_state.featured_event
        symbol = event["symbol"]
        direction_emoji = "🟢" if event["direction"] == "up" else "🔴"
        st.markdown(
            f"### {direction_emoji} {symbol} — {event['price_move_pct']:.1f}% move on {event['volume_ratio']:.1f}x volume"
        )
        if event.get("df") is not None:
            fig = build_event_chart(event["df"], symbol, event["timestamp"])
            st.plotly_chart(fig, use_container_width=True)
        if event.get("news"):
            st.markdown("#### 📰 Related News")
            for i, article in enumerate(event["news"][:3]):
                score = article.get("relevance_score", 0)
                badge = "🟢" if score >= 4 else "🟡" if score >= 2 else "⚪"
                st.markdown(f"{badge} **{article['title']}**")
                st.caption(
                    f"{article.get('source', {}).get('name', '')} · "
                    f"{article.get('publishedAt', '')[:10]} · {article.get('relevance_reason', '')}"
                )
                if article.get("url"):
                    st.markdown(f"[Read article →]({article['url']})")
                if i < 2:
                    st.divider()
        if st.button("🤖 Explain this move", key=f"explain_{symbol}"):
            headlines = "\n".join([f"- {a['title']}" for a in event.get("news", [])[:5]])
            prompt = (
                f"{symbol} moved {event['price_move_pct']:.1f}% "
                f"{'up' if event['direction'] == 'up' else 'down'} on {event['volume_ratio']:.1f}x average volume "
                f"at {event['timestamp']}. These news headlines were published around that time:\n{headlines}\n\n"
                "Which headline most likely caused this move and why? Answer in 3 sentences."
            )
            with st.spinner("Analyzing..."):
                try:
                    from agents.llm.active_llm_calls import call_active_llm_simple
                    explanation = call_active_llm_simple(prompt)
                    st.info(explanation)
                except Exception as e:
                    st.error(f"Could not generate explanation: {e}")
    else:
        st.info(
            "🔍 Monitoring markets... First scan runs on page load. "
            "Significant volume spikes will appear here automatically."
        )
        st.caption(f"Watching: {', '.join(watchlist)}")

with col_feed:
    st.markdown("### 📡 Live Feed")
    st.caption(
        f"Next scan in {seconds_left}s · Last scan: "
        f"{datetime.fromtimestamp(st.session_state.last_scan_time).strftime('%H:%M:%S') if st.session_state.last_scan_time else 'Never'}"
    )
    st.progress(1 - (seconds_left / POLL_INTERVAL_SECONDS))
    if not st.session_state.event_feed:
        st.caption("No significant events detected yet. Threshold: 3x volume + 2% move.")
    for event in st.session_state.event_feed:
        direction_emoji = "🟢" if event["direction"] == "up" else "🔴"
        time_str = (
            event["timestamp"].strftime("%H:%M")
            if hasattr(event["timestamp"], "strftime") else str(event["timestamp"])
        )
        headline_short = (
            (event.get("top_headline", "")[:60] + "...")
            if len(event.get("top_headline", "")) > 60
            else event.get("top_headline", "No news")
        )
        if st.button(
            f"{direction_emoji} {event['symbol']} {event['price_move_pct']:.1f}% · {time_str}",
            key=f"feed_{event['symbol']}_{time_str}",
            use_container_width=True,
        ):
            st.session_state.selected_event = event
            st.rerun()
        st.caption(headline_short)
        st.divider()

st.markdown("---")

# Generate briefing if not cached
if st.session_state.home_briefing_text is None:
    with st.spinner("Preparing your briefing..."):
        try:
            from trading.memory import get_memory_store
            from trading.services.home_briefing_service import generate_briefing
            store = get_memory_store()
            out = generate_briefing(store)
            st.session_state.home_briefing_text = out.get("briefing_text", "")
            st.session_state.home_briefing_cards = out.get("cards", [])
            st.session_state.home_briefing_market_data = out.get("market_data", {})
        except Exception as e:
            logger.exception(f"Briefing generation failed: {e}")
            st.session_state.home_briefing_text = (
                "We couldn't generate your briefing right now. Please try **Refresh briefing** in the sidebar."
            )
            st.session_state.home_briefing_cards = []
            st.session_state.home_briefing_market_data = {}
    st.rerun()

market_data = st.session_state.home_briefing_market_data

# Metric row: SPY and AAPL price + weekly change %
if market_data:
    m1, m2 = st.columns(2)
    for col, symbol in zip([m1, m2], ["SPY", "AAPL"]):
        if symbol in market_data:
            data = market_data[symbol]
            cur = data.get("current")
            week_ago = data.get("week_ago")
            with col:
                if cur is not None:
                    delta = None
                    if week_ago is not None and week_ago != 0:
                        pct = ((cur - week_ago) / week_ago) * 100
                        delta = f"{pct:+.1f}% vs 1w"
                    st.metric(label=symbol, value=f"${cur:,.2f}", delta=delta)

st.markdown("---")
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
