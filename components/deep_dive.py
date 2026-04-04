# -*- coding: utf-8 -*-
"""Single-ticker deep analysis surface (scroll + 4 tabs)."""
import asyncio
import logging
from datetime import datetime

import streamlit as st

from components.analyze_ai_score import get_ai_recommendation_dict, top_signals_summary
from components.analyze_chart import render_price_chart
from components.analyze_diagnostics import render_diagnostics
from components.analyze_forecast import render_forecast
from components.analyze_news import render_news
from components.analyze_options import render_options
from trading.data.price_cache import get_history, get_info, get_quote

logger = logging.getLogger(__name__)


def _deep_dive_platform_suffix(
    sym: str,
    price,
    chg,
    score,
    rec,
) -> str:
    """Pre-loaded ticker context for tool router + synthesis (no tool execution here)."""
    parts = [f"Deep dive symbol: {sym}"]
    if price is not None:
        chg_s = f"{float(chg):+.2f}%" if chg is not None else "—"
        parts.append(f"Last price: ${float(price):.2f} · session change {chg_s}")
    if score and not score.get("error"):
        parts.append(
            f"AI score: overall {score.get('overall_score')} grade {score.get('grade')} "
            f"(T/M/S/F: {score.get('technical_score')}/"
            f"{score.get('momentum_score')}/{score.get('sentiment_score')}/"
            f"{score.get('fundamental_score')})"
        )
        summ = score.get("summary") or ""
        if summ:
            parts.append(f"Score summary: {summ[:600]}")
    sigs = (score or {}).get("signals") or []
    if sigs:
        parts.append("Top signals:")
        for s in sigs[:5]:
            parts.append(
                f"  - [{s.get('impact')}] {s.get('name')}: {s.get('value')}"
            )
    if rec:
        parts.append(
            f"Recommendation snapshot: {rec.get('action')} · "
            f"conviction {rec.get('conviction')} · entry {rec.get('entry')} · "
            f"target {rec.get('target')} · stop {rec.get('stop')}"
        )
    try:
        from trading.data.earnings_calendar import get_upcoming_earnings

        er = get_upcoming_earnings(sym)
        if er and er.get("days_until") is not None:
            parts.append(f"Upcoming earnings: in {er.get('days_until')} days")
    except Exception:
        pass
    try:
        from trading.data.news_aggregator import get_news

        items = get_news(sym, max_items=5)
        if items:
            parts.append("Recent headlines (snapshot):")
            for it in items[:5]:
                t = (it.get("title") or it.get("headline") or str(it))[:200]
                parts.append(f"  - {t}")
    except Exception:
        pass
    return "\n".join(parts)


def _render_deep_dive_chat(
    sym: str,
    price,
    chg,
    score,
    rec,
) -> None:
    """Bottom chat bar: same tool loop as Home, with ticker-heavy context."""
    st.markdown("---")
    st.subheader(f"Ask about {sym}")
    _dk = f"deep_dive_chat_{sym}"
    if _dk not in st.session_state:
        st.session_state[_dk] = []

    platform_suffix = _deep_dive_platform_suffix(sym, price, chg, score, rec)
    _inp = st.chat_input(f"Ask about {sym}...", key=f"dd_chat_in_{sym}")
    _raw = (_inp or "").strip()

    if _raw:
        st.session_state[_dk].append({"role": "user", "content": _raw})

    for msg in st.session_state[_dk]:
        with st.chat_message(msg.get("role", "user")):
            if msg.get("role") == "assistant":
                for _c in msg.get("tool_captions") or []:
                    st.caption(_c)
            st.markdown(msg.get("content", ""))

    if not _raw:
        return

    res = None
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                from agents.llm.tool_executor import execute_with_tools
                from trading.memory import get_memory_store
                from trading.services import chat_nl_service

                store = get_memory_store()
                um = f"[Deep dive: {sym}]\n{_raw}"
                try:
                    store.ingest_preference_text(um, source="deep_dive_chat")
                except Exception as e:
                    logger.warning("Deep dive chat: preference ingest failed: %s", e)
                if "chat_router" not in st.session_state:
                    try:
                        from trading.agents.enhanced_prompt_router import (
                            EnhancedPromptRouterAgent,
                        )

                        st.session_state.chat_router = EnhancedPromptRouterAgent()
                    except Exception as e:
                        logger.warning("Deep dive chat: router failed: %s", e)
                        st.session_state.chat_router = None
                router = st.session_state.get("chat_router")
                route_result = (
                    chat_nl_service.parse_intent(router, um)
                    if router
                    else {"intent": "unknown", "args": {}}
                )
                intent = (
                    route_result.get("intent", "unknown")
                    if isinstance(route_result, dict)
                    else getattr(route_result, "intent", "unknown")
                )
                memory_context = chat_nl_service.get_memory_context(store)
                agent_response = chat_nl_service.run_agent_action(um)
                context_block = chat_nl_service.build_context_block(
                    memory_context,
                    agent_response,
                    intent=intent,
                    store=store,
                )
                conv = [
                    {"role": m["role"], "content": m.get("content", "")}
                    for m in st.session_state[_dk][:-1]
                ]
                res = execute_with_tools(
                    user_message=um,
                    context_block=context_block,
                    conversation_messages=conv,
                    system_prompt=chat_nl_service.EVOLVE_CHAT_SYSTEM_PROMPT,
                    platform_context_suffix=platform_suffix,
                    focus_symbol=sym,
                    available_tools=[
                        "scan_universe",
                        "get_ai_score",
                        "get_forecast",
                        "get_news",
                        "get_risk_metrics",
                    ],
                    max_tokens=2048,
                )
                for _c in res.tool_captions or []:
                    st.caption(_c)
                st.markdown(res.text)
            except Exception as e:
                logger.exception("Deep dive chat failed: %s", e)
                from agents.llm.tool_executor import ToolChatResult

                res = ToolChatResult(
                    text=f"Chat unavailable: {e}",
                    tool_captions=[],
                )
                st.caption(res.text)
    st.session_state[_dk].append(
        {
            "role": "assistant",
            "content": res.text if res else "",
            "tool_captions": (res.tool_captions or []) if res else [],
        }
    )


def render_deep_dive(ticker: str) -> None:
    sym = (ticker or "").strip().upper()
    if not sym:
        return
    try:
        q = get_quote(sym)
        price = q.get("price")
        prev = q.get("prev_close")
        hist = get_history(sym, period="3mo", interval="1d")
        if price is None and hist is not None and not hist.empty:
            price = float(hist["Close"].iloc[-1])
        chg = None
        if price is not None and prev not in (None, 0):
            chg = (float(price) - float(prev)) / float(prev) * 100
        info = get_info(sym) or {}
        name = info.get("longName") or info.get("shortName") or sym
        c1, c2, c3 = st.columns([1, 4, 1])
        with c1:
            if st.button("← Back", key="deep_dive_back"):
                st.session_state.pop("deep_dive_ticker", None)
                st.session_state.pop("home_chat_pending", None)
                st.rerun()
        with c2:
            chg_s = f"{chg:+.2f}%" if chg is not None else "—"
            st.markdown(f"## {sym} · {name}")
            st.caption(f"Last **${float(price):.2f}** · Day **{chg_s}**" if price else sym)
        with c3:
            st.session_state["analyze_ticker"] = sym

        if hist is not None and not hist.empty:
            render_price_chart(
                sym,
                hist,
                period="3mo",
                period_label="3M",
                _interval="1d",
                _tf_label="1d",
                trader_mode="Short-term",
                st_ver=tuple(int(x) for x in st.__version__.split(".")[:2]),
            )
            try:
                from trading.analysis.chart_pattern_detector import ChartPatternDetector

                with st.expander("Chart patterns", expanded=False):
                    ChartPatternDetector(sym, hist).render_streamlit()
            except Exception as e:
                st.caption(f"unavailable: {e}")

        score = None
        try:
            from trading.analysis.ai_score import compute_ai_score

            score = compute_ai_score(sym, hist)
        except Exception:
            score = None
        rec = get_ai_recommendation_dict(sym, hist, trader_mode="Short-term")
        st.markdown("### Recommendation")
        if rec:
            act = rec.get("action", "HOLD")
            st.markdown(f"**{act}** · conviction **{rec.get('conviction', '—')}**")
            st.caption(
                f"Entry {rec.get('entry')} · Target {rec.get('target')} · "
                f"Stop {rec.get('stop')} · Expected move **{rec.get('pct_move', '—')}%**"
            )
            _tc, _pc = st.columns(2)
            with _tc:
                if st.button(
                    "Track recommendation",
                    key=f"track_{sym}",
                ):
                    try:
                        from trading.services.recommendation_tracker import (
                            RecommendationTracker,
                        )
                        from utils.session_utils import get_stable_user_id

                        _uid = get_stable_user_id()
                        if not _uid:
                            st.warning(
                                "Sign in or complete onboarding to save tracking."
                            )
                        else:
                            _ai = float(
                                (score or {}).get("overall_score")
                                or rec.get("signal_score")
                                or 0
                            )
                            tracker = RecommendationTracker()
                            tracker.save_recommendation(
                                session_id=_uid,
                                symbol=sym,
                                action=str(rec.get("action", "HOLD")),
                                entry_price=float(rec.get("entry") or 0),
                                target_price=float(rec.get("target") or 0),
                                stop_price=float(rec.get("stop") or 0),
                                ai_score=_ai,
                                timestamp=datetime.now().isoformat(),
                            )
                            st.success(
                                "Recommendation saved for tracking"
                            )
                    except Exception as e:
                        st.caption(f"Tracking unavailable: {e}")
            with _pc:
                if st.button("+ Paper Trade", key=f"paper_{sym}"):
                    try:
                        from trading.execution.models import OrderType
                        from trading.execution.trade_execution_simulator import (
                            TradeExecutionSimulator,
                        )

                        _act = str(rec.get("action", "HOLD")).upper()
                        _side = (
                            "buy"
                            if _act == "BUY"
                            else "sell"
                            if _act == "SELL"
                            else "buy"
                        )
                        _sim = TradeExecutionSimulator()
                        _entry = float(rec.get("entry") or 0)
                        _sim.place_order(
                            symbol=sym,
                            order_type=OrderType.MARKET,
                            side=_side,
                            quantity=1.0,
                            price=_entry if _entry > 0 else None,
                            stop_price=float(rec.get("stop") or 0)
                            or None,
                        )
                        st.success(
                            f"Paper trade opened: {_act} {sym} @ {_entry}"
                        )
                    except Exception as e:
                        st.caption(f"Paper trade unavailable: {e}")
        elif score and not score.get("error"):
            st.caption(score.get("summary", ""))
        else:
            st.caption("Recommendation data unavailable.")

        st.markdown("### What's driving this")
        signals = (score or {}).get("signals") or []
        for s in signals[:5]:
            imp = s.get("impact", "neutral")
            icon = "🟢" if imp == "positive" else "🔴" if imp == "negative" else "🟡"
            nm = s.get("name", "")
            val = s.get("value", "")
            st.markdown(f"{icon} **{nm}** — {val}")
        st.caption(top_signals_summary(signals))

        try:
            from trading.commentary.commentary_engine import (
                CommentaryRequest,
                CommentaryType,
                create_commentary_engine,
            )

            _eng = create_commentary_engine()
            _req = CommentaryRequest(
                commentary_type=CommentaryType.MARKET_REGIME,
                symbol=sym,
                timestamp=datetime.now(),
                market_data=hist,
            )
            _resp = asyncio.run(_eng.generate_commentary(_req))
            _txt = (
                getattr(_resp, "detailed_analysis", None)
                or getattr(_resp, "summary", None)
                or ""
            )
            if _txt:
                st.markdown("**AI Commentary**")
                st.markdown(_txt[:4000])
        except Exception:
            pass

        tf, tn, tr, tp = st.tabs(["Forecast", "News", "Risk", "Patterns"])
        with tf:
            render_forecast(sym, hist, horizon=7)
        with tn:
            render_news(sym)
        with tr:
            render_diagnostics(sym, hist)
            st.markdown("---")
            render_options(sym)
        with tp:
            try:
                from trading.analysis.chart_pattern_detector import ChartPatternDetector

                if hist is not None and not hist.empty:
                    ChartPatternDetector(sym, hist).render_streamlit()
                else:
                    st.caption("No history for pattern detection.")
            except Exception as e:
                st.caption(f"unavailable: {e}")

        ctx = sym
        if rec:
            ctx = (
                f"{sym} (action {rec.get('action')}, "
                f"target {rec.get('target')})"
            )
        st.session_state["home_chat_context_ticker"] = ctx
        _render_deep_dive_chat(sym, price, chg, score, rec)
    except Exception as e:
        st.caption(f"unavailable: {e}")
