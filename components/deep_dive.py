# -*- coding: utf-8 -*-
"""Single-ticker deep analysis surface (scroll + 5 tabs)."""
import logging
import time
from datetime import datetime

import streamlit as st

from components.analyze_ai_score import top_signals_summary
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
            _tf_options = {
                "1W": ("5d", "15m", "15m"),
                "1M": ("1mo", "1h", "1h"),
                "3M": ("3mo", "1d", "1d"),
                "6M": ("6mo", "1d", "1d"),
                "1Y": ("1y", "1d", "1d"),
                "2Y": ("2y", "1wk", "1W"),
            }
            _tf_key = f"dd_tf_{sym}"
            if _tf_key not in st.session_state:
                st.session_state[_tf_key] = "3M"

            _tf_cols = st.columns(len(_tf_options))
            for _i, (_lbl, _) in enumerate(_tf_options.items()):
                with _tf_cols[_i]:
                    if st.button(
                        _lbl,
                        key=f"dd_tf_{sym}_{_lbl}",
                        type=(
                            "primary"
                            if st.session_state[_tf_key] == _lbl
                            else "secondary"
                        ),
                    ):
                        st.session_state[_tf_key] = _lbl
                        st.rerun()

            _sel_tf = st.session_state[_tf_key]
            _period, _interval, _tf_label = _tf_options[_sel_tf]

            _chart_hist = get_history(sym, period=_period, interval=_interval)
            if _chart_hist is None or _chart_hist.empty:
                _chart_hist = hist
                # Align chart params with fallback series (default 3mo / 1d)
                _period = "3mo"
                _interval = "1d"
                _tf_label = "1d"
                _sel_tf = "3M"
                st.session_state[_tf_key] = "3M"

            render_price_chart(
                sym,
                _chart_hist,
                period=_period,
                period_label=_sel_tf,
                _interval=_interval,
                _tf_label=_tf_label,
                trader_mode="Short-term",
                st_ver=tuple(int(x) for x in st.__version__.split(".")[:2]),
            )

        _prefs_dd: dict = {}
        try:
            from config.user_store import load_user_preferences
            from utils.session_utils import get_stable_user_id

            _uid_dd = (
                st.session_state.get("evolve_session_id")
                or get_stable_user_id()
            )
            _prefs_dd = load_user_preferences(_uid_dd) or {}
        except Exception:
            _prefs_dd = {}

        _scoring_style = str(
            _prefs_dd.get("scoring_style", "Balanced (default)"),
        )

        score = None
        try:
            from trading.analysis.ai_score import compute_ai_score

            score = compute_ai_score(
                sym, hist, scoring_style=_scoring_style,
            )
        except Exception:
            score = None

        _show_short = False
        try:
            _od_dd = _prefs_dd.get(
                "opportunity_direction",
                "Bullish only (BUY signals)",
            )
            _show_short = (
                "Both" in _od_dd
                or "Bearish" in _od_dd
            )
        except Exception:
            pass

        short_score_result = None
        if _show_short and score and not score.get("error"):
            try:
                from trading.analysis.ai_score import compute_short_score

                short_score_result = compute_short_score(
                    sym,
                    hist,
                    ai_result=score,
                    scoring_style=_scoring_style,
                )
            except Exception:
                short_score_result = None

        _rec_key = f"deep_dive_rec_{sym}"
        rec = st.session_state.get(_rec_key)
        if rec is None:
            _fc_key_for_rec = f"deep_dive_forecast_{sym}"
            if st.session_state.get(_fc_key_for_rec):
                try:
                    from components.analyze_ai_score import (
                        get_ai_recommendation_dict,
                    )

                    rec = get_ai_recommendation_dict(
                        sym,
                        hist,
                        trader_mode="Short-term",
                        scoring_style=_scoring_style,
                    )
                    if rec:
                        st.session_state[_rec_key] = rec
                except Exception as _rec_e:
                    logger.debug("Deep dive recommendation: %s", _rec_e)
        st.markdown("### Recommendation")
        if rec:
            act = str(rec.get("action", "HOLD"))
            conv = str(rec.get("conviction", "MEDIUM"))
            _act_color = (
                "green"
                if act == "BUY"
                else "red"
                if act in ("SELL", "SHORT")
                else "orange"
            )
            st.markdown(f":{_act_color}[## {act}]")
            _sc = float((score or {}).get("overall_score") or 0)
            _grade = str((score or {}).get("grade", "C"))
            col_sc, col_cv, col_en = st.columns(3)
            with col_sc:
                st.metric(
                    "AI Score",
                    f"{_sc:.1f}/10",
                    delta=f"Grade {_grade}",
                    delta_color="off",
                )
            with col_cv:
                st.metric("Conviction", conv)
            with col_en:
                _entry = rec.get("entry")
                if _entry not in (None, "", 0, "0"):
                    try:
                        st.metric(
                            "Entry",
                            f"${float(_entry):.2f}",
                        )
                    except (TypeError, ValueError):
                        st.metric("Entry", str(_entry))

            _target = rec.get("target")
            _stop = rec.get("stop")
            _pct = rec.get("pct_move")
            if _target or _stop or _pct not in (None, "", "—"):
                col_t, col_s, col_p = st.columns(3)
                with col_t:
                    if _target not in (None, "", 0):
                        try:
                            st.metric(
                                "Target",
                                f"${float(_target):.2f}",
                            )
                        except (TypeError, ValueError):
                            st.metric("Target", str(_target))
                with col_s:
                    if _stop not in (None, "", 0):
                        try:
                            st.metric(
                                "Stop Loss",
                                f"${float(_stop):.2f}",
                            )
                        except (TypeError, ValueError):
                            st.metric("Stop Loss", str(_stop))
                with col_p:
                    if _pct not in (None, "", "—"):
                        st.metric(
                            "Expected Move",
                            f"{_pct}%",
                        )

            _summ = (score or {}).get("summary", "")
            if _summ:
                st.caption(str(_summ))

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
            _sc2 = float(score.get("overall_score") or 0)
            _grade2 = str(score.get("grade", "C"))
            st.metric(
                "AI Score",
                f"{_sc2:.1f}/10",
                delta=f"Grade {_grade2}",
                delta_color="off",
            )
            st.caption(score.get("summary", "") or "")
        else:
            st.caption("Recommendation unavailable.")

        st.markdown("### What's driving this")
        signals = (score or {}).get("signals") or []
        for s in signals[:5]:
            imp = s.get("impact", "neutral")
            icon = "🟢" if imp == "positive" else "🔴" if imp == "negative" else "🟡"
            nm = s.get("name", "")
            val = s.get("value", "")
            st.markdown(f"{icon} **{nm}** — {val}")
        st.caption(top_signals_summary(signals))

        if _show_short and short_score_result and not short_score_result.get(
            "error"
        ):
            _ss = short_score_result
            _sc = float(_ss.get("short_score", 5.0))
            st.markdown("### Short Score")
            _sc_color = (
                "red"
                if _sc >= 7.0
                else "orange"
                if _sc >= 6.0
                else "green"
            )
            st.markdown(
                f":{_sc_color}[**{_ss.get('label', '?')} — {_sc:.1f}/10**]"
            )
            for s in (_ss.get("signals") or [])[:4]:
                imp = str(s.get("impact", "")).lower()
                icon = (
                    "🔴"
                    if imp in ("bearish", "negative")
                    else "⚠️"
                    if imp == "risk"
                    else "⚪"
                )
                st.markdown(
                    f"{icon} **{s.get('name')}** — {s.get('value')}"
                )
            st.caption(
                "Short Score measures bearish thesis strength. "
                "High score = stronger case for shorting."
            )

        _completeness = (score or {}).get("signal_completeness") or {}
        if _completeness:
            _available = _completeness.get("available", [])
            _unavailable = _completeness.get("unavailable", [])
            _fallback = _completeness.get("fallback", [])
            _n = len(_available)
            _total = 11
            _color = (
                "🟢" if _n >= 8
                else "🟡" if _n >= 5
                else "🔴"
            )
            st.caption(
                f"{_color} Signal completeness: "
                f"**{_n}/{_total}** sources "
                f"returning real data"
            )
            if _unavailable:
                st.caption(
                    f"Unavailable: {', '.join(_unavailable)}"
                )
            if _fallback:
                st.caption(
                    f"Using fallback (0/neutral): {', '.join(_fallback)}"
                )
        else:
            quality = (score or {}).get("data_quality") or {}
            unavailable = [
                k for k, v in quality.items() if v == "unavailable"
            ]
            if unavailable:
                st.caption(
                    f"Data unavailable: {', '.join(unavailable)}. "
                    f"Score reflects available signals only."
                )

        try:
            from trading.data.sec_edgar import get_latest_filing, get_sec_signal

            _sec = get_sec_signal(sym)
            _filing = get_latest_filing(sym, "10-Q") or get_latest_filing(sym, "10-K")

            if _sec.get("sec_source") not in ("unavailable", "error", "no_filing"):
                st.markdown("**SEC Filing Analysis**")
                _label = str(_sec.get("sec_label", "neutral"))
                _icon = {
                    "positive": "🟢",
                    "negative": "🔴",
                    "neutral": "⬜",
                }.get(_label.lower(), "⬜")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Filing sentiment", f"{_icon} {_label.title()}")
                with col2:
                    if _filing:
                        st.metric(
                            "Latest filing",
                            str(_filing.get("form", "")),
                            delta=str(_filing.get("date", "")),
                            delta_color="off",
                        )
                themes = _sec.get("sec_themes") or []
                if themes:
                    st.caption("Key themes: " + " · ".join(str(t) for t in themes[:3]))
                st.caption(
                    f"Source: SEC EDGAR ({_sec.get('sec_source')})"
                )
        except Exception as e:
            st.caption(f"SEC data unavailable: {e}")

        try:
            from config.llm_config import llm_available
            from agents.llm.active_llm_calls import call_active_llm_simple

            _cmt_key = f"dd_ai_commentary_{sym}"
            _cmt_ts_key = f"dd_ai_commentary_ts_{sym}"
            _cmt_ttl = 3600.0
            _now_cm = time.time()
            _cached_cm = st.session_state.get(_cmt_key)
            _cm_age = _now_cm - float(
                st.session_state.get(_cmt_ts_key, 0.0)
            )

            if (
                llm_available()
                and hist is not None
                and not hist.empty
            ):
                if _cached_cm and _cm_age < _cmt_ttl:
                    st.markdown("**AI Commentary**")
                    st.markdown(str(_cached_cm))
                else:
                    _cm = {c.lower(): c for c in hist.columns}
                    _cc = _cm.get("close", hist.columns[0])
                    _last = float(hist[_cc].iloc[-1])
                    _chg_20d = (
                        float(
                            (
                                hist[_cc].iloc[-1]
                                / hist[_cc].iloc[-20]
                                - 1
                            )
                            * 100
                        )
                        if len(hist) >= 20
                        else 0.0
                    )

                    _prompt = (
                        f"You are a financial analyst. Give a 2-3 "
                        f"sentence commentary on {sym} for a trader. "
                        f"Current price: ${_last:.2f}. "
                        f"20-day change: {_chg_20d:+.1f}%. "
                        f"AI Score: "
                        f"{(score or {}).get('overall_score', 'N/A')}. "
                        f"Be concise and actionable."
                    )
                    _commentary = call_active_llm_simple(
                        _prompt,
                        max_tokens=150,
                    )
                    if _commentary:
                        _txt_cm = _commentary.strip()
                        st.session_state[_cmt_key] = _txt_cm
                        st.session_state[_cmt_ts_key] = _now_cm
                        st.markdown("**AI Commentary**")
                        st.markdown(_txt_cm)
        except Exception as _ce:
            logger.debug("Commentary: %s", _ce)

        tf, tn, tr, to, tp = st.tabs(
            ["Forecast", "News", "Risk", "Options", "Patterns"]
        )
        with tf:
            import time as _time_fc

            _fc_key = f"deep_dive_forecast_{sym}"
            _fc_ts_key = f"deep_dive_forecast_ts_{sym}"
            _cached_fc = st.session_state.get(_fc_key)
            _cached_fc_age = _time_fc.time() - st.session_state.get(
                _fc_ts_key, 0.0
            )
            forecast_result = None
            if _cached_fc is None or _cached_fc_age > 600:
                if st.button(
                    "📈 Generate Forecast",
                    key=f"dd_forecast_btn_{sym}",
                ):
                    with st.spinner("Running consensus forecast..."):
                        _fc = render_forecast(sym, hist, horizon=7)
                    if _fc:
                        st.session_state[_fc_key] = _fc
                        st.session_state[_fc_ts_key] = _time_fc.time()
                    forecast_result = _fc
                else:
                    st.info(
                        "Click **Generate Forecast** to run the multi-model "
                        "consensus.",
                        icon="📈",
                    )
            else:
                forecast_result = _cached_fc
                _cm = {c.lower(): c for c in hist.columns}
                _cc = _cm.get("close", hist.columns[0])
                last = float(hist[_cc].iloc[-1])
                cp = (_cached_fc or {}).get("consensus_price")
                direction = (_cached_fc or {}).get("direction", "—")
                conviction = (_cached_fc or {}).get("conviction", "—")
                models = (_cached_fc or {}).get("models_used") or []
                st.subheader("Consensus forecast")
                st.metric(
                    "Direction",
                    direction,
                    delta=f"{conviction} conviction",
                )
                if cp:
                    pct = (float(cp) - last) / last * 100 if last else 0
                    st.metric(
                        "Consensus price",
                        f"${float(cp):.2f}",
                        delta=f"{pct:+.1f}%",
                    )
                if models:
                    st.caption(
                        "Models: " + ", ".join(str(m) for m in models[:12])
                    )
                st.caption(
                    "Using cached forecast (refreshes every 10 min)."
                )
            _wf_conf = (forecast_result or {}).get("walk_forward_confidence")
            _wf_warn = (forecast_result or {}).get("walk_forward_warnings") or []
            if _wf_conf:
                _conf_color = {
                    "high": "🟢",
                    "medium": "🟡",
                    "low": "🔴",
                }.get(str(_wf_conf).lower(), "⬜")
                st.caption(
                    f"{_conf_color} Walk-forward validated confidence: "
                    f"{str(_wf_conf).upper()} — run Backtest → Walk-Forward to update."
                )
            else:
                st.caption(
                    "⬜ Walk-forward confidence: not yet validated. Run "
                    "Backtest → Walk-Forward tab to calibrate."
                )
            for w in _wf_warn:
                st.warning(w)
        with tn:
            render_news(sym)
        with tr:
            render_diagnostics(sym, hist)
        with to:
            st.subheader(f"Options Flow — {sym}")
            try:
                render_options(sym)
            except Exception as _oe:
                st.caption(f"Options unavailable: {_oe}")
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
