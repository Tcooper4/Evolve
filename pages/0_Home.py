"""
Evolve Home — Personalized morning briefing.

Uses MemoryStore, market data (FallbackDataProvider), and risk snapshot to generate
a plain-English briefing and 2–4 dynamic cards. Designed for any user, including
those with no finance background. Cached per session; refresh button for a new take.
"""

import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Evolve Home",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="auto",
)

st.title("🏠 Good morning")
st.caption("Your personalized briefing. Simple, jargon-free.")

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

# Show briefing text
st.markdown(st.session_state.home_briefing_text)

# Dynamic cards (2–4)
cards = st.session_state.home_briefing_cards
market_data = st.session_state.home_briefing_market_data

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
            with st.container():
                st.markdown(f"**{headline}**")
                if detail:
                    with st.expander("See details"):
                        st.caption(detail)
                # Optional: small chart for price_chart if we have series
                if card_type == "price_chart" and market_data:
                    symbol = card.get("symbol") or list(market_data.keys())[0] if market_data else None
                    if symbol and symbol in market_data:
                        series = market_data[symbol].get("series")
                        if series and len(series) >= 2:
                            try:
                                import pandas as pd
                                df = pd.DataFrame({"price": series})
                                st.line_chart(df)
                            except Exception:
                                pass

# Follow-up question: routes to Chat with pre-filled question
st.markdown("---")
st.caption("Ask a follow-up about anything in your briefing.")
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
