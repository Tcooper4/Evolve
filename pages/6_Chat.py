# -*- coding: utf-8 -*-
"""
Chat page — LLM chat (left) + news/research panel (right). Reorganized from 1_Chat + Research.
"""
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st

from components.theme import inject_theme, render_top_bar, keyboard_shortcut_js

try:
    from agents.llm.active_llm_calls import call_active_llm_chat
except Exception:
    call_active_llm_chat = None

try:
    st.markdown(keyboard_shortcut_js(), unsafe_allow_html=True)
except Exception:
    pass
inject_theme()
render_top_bar()

logger = logging.getLogger(__name__)

st.title("💬 Chat")
st.caption("Ask about portfolio, strategies, risk. News and research on the right.")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "chat_last_action_data" not in st.session_state:
    st.session_state.chat_last_action_data = None


def get_chat_router():
    if "chat_router" not in st.session_state:
        try:
            from trading.agents.enhanced_prompt_router import EnhancedPromptRouterAgent
            st.session_state.chat_router = EnhancedPromptRouterAgent()
        except Exception as e:
            logger.warning(f"Router init failed: {e}")
            st.session_state.chat_router = None
    return st.session_state.get("chat_router")


def _render_action_data(data: dict) -> None:
    if not data:
        return
    try:
        import pandas as pd
        metrics = data.get("metrics") or {}
        if metrics and isinstance(metrics, dict):
            with st.expander("📊 Metrics", expanded=True):
                cols = st.columns(min(4, len(metrics)))
                for i, (k, v) in enumerate(list(metrics.items())[:8]):
                    if isinstance(v, (int, float)):
                        cols[i % len(cols)].metric(k.replace("_", " ").title(), f"{v:.2%}" if 0 < abs(v) < 2 and "ratio" not in k.lower() and "rate" not in k.lower() else f"{v:.2f}")
        equity = data.get("equity_curve")
        if equity is not None and hasattr(equity, "__len__"):
            try:
                df = pd.DataFrame(equity) if not isinstance(equity, pd.DataFrame) else equity
                if not df.empty and hasattr(df, "columns"):
                    col = "equity_curve" if "equity_curve" in df.columns else df.columns[0]
                    st.line_chart(df[col] if col in df.columns else df.iloc[:, 0])
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"Could not render action data: {e}")

col_chat, col_news = st.columns([2, 1])

with col_chat:
    for msg in st.session_state.chat_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant" and msg.get("action_data"):
                _render_action_data(msg["action_data"])

    prompt = st.chat_input("Ask about your portfolio, strategies, risk, or request a backtest...")
    if not prompt and st.session_state.get("chat_prefill"):
        prompt = st.session_state.pop("chat_prefill", "").strip()

    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    from trading.memory import get_memory_store
                    from trading.services import chat_nl_service
                    store = get_memory_store()
                    try:
                        store.ingest_preference_text(prompt, source="chat")
                    except Exception:
                        pass
                    router = get_chat_router()
                    route_result = chat_nl_service.parse_intent(router, prompt) if router else {"intent": "unknown", "args": {}}
                    intent = route_result.get("intent", "unknown") if isinstance(route_result, dict) else getattr(route_result, "intent", "unknown")
                    memory_context = chat_nl_service.get_memory_context(store)
                    agent_response = chat_nl_service.run_agent_action(prompt)
                    context_block = chat_nl_service.build_context_block(
                        memory_context, agent_response, intent=intent, store=store,
                    )
                    conv = [{"role": m["role"], "content": m.get("content", "")} for m in st.session_state.chat_messages[:-1]]
                    if call_active_llm_chat:
                        reply = call_active_llm_chat(
                            chat_nl_service.EVOLVE_CHAT_SYSTEM_PROMPT, context_block, conv, prompt,
                            max_tokens=2048,
                        )
                        reply = reply.strip() or "I didn't get a response. Please try again."
                    else:
                        reply = chat_nl_service.call_claude(
                            chat_nl_service.EVOLVE_CHAT_SYSTEM_PROMPT, context_block, conv, prompt,
                        )
                    action_data = None
                    _ar = agent_response if isinstance(agent_response, dict) else {"data": getattr(agent_response, "data", None)}
                    if isinstance(_ar.get("data"), dict):
                        action_data = _ar["data"]
                    elif _ar.get("data") is not None:
                        action_data = {"raw": str(_ar["data"])[:500]}
                    st.markdown(reply)
                    st.session_state.chat_messages.append({"role": "assistant", "content": reply, "action_data": action_data})
                    if action_data:
                        _render_action_data(action_data)
                except Exception as e:
                    logger.exception(f"Chat turn failed: {e}")
                    err_msg = f"Something went wrong: {e}. Please try again."
                    st.markdown(err_msg)
                    st.session_state.chat_messages.append({"role": "assistant", "content": err_msg, "action_data": None})

with col_news:
    if "chat_news_results" not in st.session_state:
        try:
            from trading.data.news_aggregator import get_news
            st.session_state.chat_news_results = get_news("SPY", max_items=10)
            st.session_state.chat_news_ticker = "SPY"
        except Exception:
            st.session_state.chat_news_results = []
            st.session_state.chat_news_ticker = "SPY"

    def _fetch_and_store_news(ticker: str):
        ticker = (ticker or "SPY").strip().upper() or "SPY"
        try:
            from trading.data.news_aggregator import get_news
            st.session_state.chat_news_results = get_news(ticker, max_items=10)
            st.session_state.chat_news_ticker = ticker
        except Exception as e:
            st.session_state.chat_news_results = []
            st.caption(f"News unavailable: {e}")

    st.subheader("📰 News")
    news_ticker = st.text_input("Ticker", value=st.session_state.get("chat_news_ticker", "SPY"), key="chat_news_ticker_input_6", placeholder="SPY, AAPL").strip().upper() or "SPY"
    if st.button("Get News", key="chat_get_news_6"):
        _fetch_and_store_news(news_ticker)
        st.rerun()
    items = st.session_state.get("chat_news_results") or []
    if not items:
        st.caption("News temporarily unavailable")
    else:
        for i, item in enumerate(items[:8]):
            title = item.get("title") or item.get("headline", "")
            if not title:
                continue
            publisher = item.get("source") or item.get("publisher", "")
            link = item.get("url") or item.get("link", "")
            st.markdown(f"**{title[:80]}** — {publisher}")
            if link:
                st.caption(f"[Read more]({link})")
            st.divider()

    st.subheader("Research")
    st.caption("arXiv and research panels are on the Model Lab page.")

with st.sidebar:
    agent_mode = st.toggle("🤖 Multi-Agent Mode", value=False, help="Coordinate with Forecasting, Market Analysis, Strategy agents.", key="chat_agent_mode_6")
    st.session_state["agent_orchestration_mode"] = agent_mode
    if st.button("Save conversation to memory", key="chat_save_6"):
        if not st.session_state.chat_messages:
            st.warning("No messages to save.")
        else:
            try:
                from trading.memory import get_memory_store
                from trading.memory.memory_store import MemoryType
                from trading.services import chat_nl_service
                summary = chat_nl_service.summarize_conversation(st.session_state.chat_messages)
                store = get_memory_store()
                store.upsert(MemoryType.SHORT_TERM, namespace="Chat", key="conversation_summary", value={"summary": summary, "turns": len(st.session_state.chat_messages)}, category="conversation")
                st.success("Saved to short-term memory.")
            except Exception as e:
                st.error(f"Could not save: {e}")
    if st.button("Clear chat", key="chat_clear_6"):
        st.session_state.chat_messages = []
        st.session_state.chat_last_action_data = None
        st.rerun()

try:
    from ui.page_assistant import render_page_assistant
    render_page_assistant("Chat")
except Exception:
    pass
