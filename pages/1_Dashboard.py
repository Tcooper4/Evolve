# -*- coding: utf-8 -*-
"""
Evolve Home — single scroll: pulse, briefing, watchlist, news, chat, deep dive.
"""
import logging
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st

from components.deep_dive import render_deep_dive
from components.theme import inject_theme, render_top_bar, keyboard_shortcut_js
from config.llm_config import get_llm_config
from config.user_store import load_user_preferences, save_user_preferences
from trading.data.price_cache import get_history, get_info, get_quote

logger = logging.getLogger(__name__)

try:
    st.markdown(keyboard_shortcut_js(), unsafe_allow_html=True)
except Exception as e:
    logger.warning("Dashboard: keyboard shortcut JS failed: %s", e)
inject_theme()
render_top_bar()


@st.cache_data(ttl=900)
def get_market_pulse() -> dict:
    tickers = {
        "SPY": "S&P 500",
        "^VIX": "VIX",
    }
    out = {}
    for tkr, label in tickers.items():
        try:
            q = get_quote(tkr)
            price = q.get("price")
            if price is None:
                h = get_history(tkr, period="1d")
                if not h.empty:
                    price = float(h["Close"].iloc[-1])
            prev = q.get("prev_close")
            if prev in (None, 0) and tkr != "^VIX":
                h2 = get_history(tkr, period="5d")
                if len(h2) >= 2:
                    prev = float(h2["Close"].iloc[-2])
            if price is None or prev in (None, 0):
                continue
            chg = (float(price) - float(prev)) / float(prev) * 100
            out[tkr] = {"label": label, "price": float(price), "chg": chg}
        except Exception:
            continue
    return out


def _set_deep_dive(symbol: str) -> None:
    st.session_state["deep_dive_ticker"] = symbol.strip().upper()
    st.session_state["analyze_ticker"] = st.session_state["deep_dive_ticker"]
    st.rerun()


def _score_color(val: float) -> str:
    if val >= 7:
        return "green"
    if val >= 6:
        return "orange"
    return "red"


def _home_platform_context() -> str:
    """Extra context for router + synthesis: focus ticker, briefing regime, top picks."""
    parts = []
    dd = st.session_state.get("deep_dive_ticker") or st.session_state.get(
        "analyze_ticker"
    )
    if dd:
        parts.append(f"User focus ticker (deep dive / Home): {str(dd).strip().upper()}")
    br = st.session_state.get("home_briefing_report") or {}
    reg = br.get("market_regime") or {}
    if reg:
        parts.append(
            f"Latest morning briefing regime: {reg.get('regime')} — "
            f"{reg.get('description', '')} "
            f"VIX={reg.get('vix_level')} SPY_trend={reg.get('spy_trend')}"
        )
    opps = br.get("top_opportunities") or []
    if opps:
        lines = [f"Top opportunities from cached briefing ({len(opps)}):"]
        for o in opps[:5]:
            lines.append(
                f"  - {o.get('symbol')}: AI score {o.get('ai_score')} "
                f"price {o.get('current_price')}"
            )
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _home_chat_reply(prompt: str):
    try:
        from agents.llm.tool_executor import execute_with_tools
        from trading.memory import get_memory_store
        from trading.services import chat_nl_service

        store = get_memory_store()
        try:
            store.ingest_preference_text(prompt, source="home_chat")
        except Exception as e:
            logger.warning("Home chat: preference ingest failed: %s", e)
        dd = st.session_state.get("deep_dive_ticker") or st.session_state.get(
            "home_chat_context_ticker"
        )
        if dd:
            prompt = f"[Context: focus ticker {dd}]\n{prompt}"
        if "chat_router" not in st.session_state:
            try:
                from trading.agents.enhanced_prompt_router import (
                    EnhancedPromptRouterAgent,
                )

                st.session_state.chat_router = EnhancedPromptRouterAgent()
            except Exception as e:
                logger.warning("Home chat: router failed: %s", e)
                st.session_state.chat_router = None
        router = st.session_state.get("chat_router")
        route_result = (
            chat_nl_service.parse_intent(router, prompt)
            if router
            else {"intent": "unknown", "args": {}}
        )
        intent = (
            route_result.get("intent", "unknown")
            if isinstance(route_result, dict)
            else getattr(route_result, "intent", "unknown")
        )
        memory_context = chat_nl_service.get_memory_context(store)
        agent_response = chat_nl_service.run_agent_action(prompt)
        context_block = chat_nl_service.build_context_block(
            memory_context,
            agent_response,
            intent=intent,
            store=store,
        )
        conv = [
            {"role": m["role"], "content": m.get("content", "")}
            for m in st.session_state.home_chat_messages[:-1]
        ]
        focus = ""
        if dd:
            focus = str(dd).strip().upper()
            if "(" in focus:
                focus = focus.split("(")[0].strip().upper()
        return execute_with_tools(
            user_message=prompt,
            context_block=context_block,
            conversation_messages=conv,
            system_prompt=chat_nl_service.EVOLVE_CHAT_SYSTEM_PROMPT,
            platform_context_suffix=_home_platform_context(),
            focus_symbol=focus or None,
            available_tools=[
                "scan_universe",
                "get_ai_score",
                "get_forecast",
                "get_news",
                "get_risk_metrics",
                "get_pattern_analysis",
                "run_backtest",
                "get_options_sentiment",
            ],
            max_tokens=2048,
        )
    except Exception as e:
        logger.exception("Home chat turn failed: %s", e)
        from agents.llm.tool_executor import ToolChatResult

        return ToolChatResult(text=f"Something went wrong: {e}", tool_captions=[])


session_id = st.session_state.get("evolve_session_id") or st.session_state.get(
    "session_id", ""
)
prefs = load_user_preferences(session_id) if session_id else {}
_prefs_done = bool(
    prefs.get("onboarding_done") or prefs.get("onboarding_completed")
)
if _prefs_done:
    st.session_state["onboarding_done"] = True
elif "onboarding_done" not in st.session_state:
    st.session_state["onboarding_done"] = False

if not st.session_state["onboarding_done"]:
    st.title("Welcome to Evolve")
    st.info(
        "Every morning Evolve scans thousands of stocks and shows you the best "
        "setups — no configuration needed."
    )
    st.info(
        "Tap any stock to see a full analysis with a plain-English buy/sell "
        "recommendation."
    )
    st.info(
        "Ask the chat anything — it knows the market, your watchlist, and can "
        "explain anything."
    )
    if st.button("Get started", type="primary"):
        st.session_state["onboarding_done"] = True
        if session_id:
            save_user_preferences(
                session_id, {**prefs, "onboarding_done": True}
            )
        st.rerun()
    st.stop()

st.title("Home")
st.caption(
    "Your trading copilot — market pulse and morning briefing first; "
    "watchlist, news, and chat; open a ticker for deep dive at the bottom."
)

try:
    _cfg = get_llm_config()
    _has_llm = _cfg.has_openai() or _cfg.has_anthropic()
    if not _has_llm:
        _has_llm = bool(
            st.session_state.get("user_key_OPENAI_API_KEY")
            or st.session_state.get("user_key_ANTHROPIC_API_KEY")
        )
    if not _has_llm:
        st.info(
            "Add an OpenAI or Anthropic key "
            "in Settings to enable the chat "
            "agent, AI commentary, and morning "
            "briefing narrative. "
            "All market data features are active.",
            icon="ℹ️",
        )
except Exception as e:
    logger.warning("Home LLM availability strip skipped: %s", e)

# --- Market pulse ---
st.subheader("Market pulse")
pulse = get_market_pulse()
c_sp, c_vx = st.columns(2)
with c_sp:
    spy = pulse.get("SPY")
    if spy:
        st.metric(
            "SPY",
            f"${spy['price']:.2f}",
            f"{spy['chg']:+.2f}%",
            help=spy["label"],
        )
    else:
        st.caption("Markets loading...")
with c_vx:
    vx = pulse.get("^VIX")
    if vx:
        st.metric("VIX", f"{vx['price']:.2f}", f"{vx['chg']:+.2f}%")
    else:
        st.caption("Markets loading...")

# --- Breaking news (lightweight — above briefing) ---
st.markdown("---")
st.subheader("Breaking news")
_news_cache_key = "home_news_summaries"
_news_cache_ts = "home_news_summaries_ts"
try:
    from trading.data.news_aggregator import (
        get_financial_headlines,
        get_walter_bloomberg_headlines,
    )

    items = get_walter_bloomberg_headlines(5) or get_financial_headlines(5) or []
    if (
        _news_cache_key not in st.session_state
        or time.time() - st.session_state.get(_news_cache_ts, 0) > 3600
    ):
        summaries = []
        try:
            from config.llm_config import llm_available, require_llm

            from agents.llm.active_llm_calls import call_active_llm_simple

            if not llm_available():
                require_llm("AI-enhanced headlines")
                summaries = [
                    {
                        "title": (it.get("title") or "")[:200],
                        "why": "",
                        "url": it.get("url", "") or it.get("link", ""),
                    }
                    for it in items[:5]
                    if (it.get("title") or "").strip()
                ]
            else:
                for it in items[:5]:
                    title = (it.get("title") or "")[:200]
                    if not title:
                        continue
                    why = ""
                    try:
                        why = call_active_llm_simple(
                            f"In one short sentence, why might this matter to traders: {title}",
                            max_tokens=80,
                        ).strip()
                    except Exception:
                        why = ""
                    summaries.append(
                        {
                            "title": title,
                            "why": why,
                            "url": it.get("url", "") or it.get("link", ""),
                        }
                    )
        except Exception as _ne:
            logger.warning("Home news: LLM summaries failed: %s", _ne)
            summaries = [
                {
                    "title": (it.get("title") or "")[:200],
                    "why": "",
                    "url": it.get("url", "") or it.get("link", ""),
                }
                for it in items[:5]
                if (it.get("title") or "").strip()
            ]
        try:
            st.session_state[_news_cache_key] = summaries
            st.session_state[_news_cache_ts] = time.time()
        except Exception as _se:
            logger.warning("Home news: session cache failed: %s", _se)
    news_with_summaries = st.session_state.get(_news_cache_key) or []
    for it in news_with_summaries:
        title = (it.get("title") or "")[:200]
        if not title:
            continue
        why = (it.get("why") or "").strip()
        url = it.get("url", "") or ""
        if url:
            st.markdown(
                f"**[{title}]({url})**"
            )
        else:
            st.markdown(f"**{title}**")
        if why:
            st.caption(f"Why it matters: {why}")
except Exception as e:
    st.caption(f"unavailable: {e}")


def _build_fallback_digest(scores: list) -> str:
    """Plain text digest when LLM is unavailable."""
    if not scores:
        return "No watchlist data."
    _top = scores[0]
    _bot = scores[-1]
    _bullish = [s for s in scores if s["score"] >= 6.5]
    _bearish = [s for s in scores if s["score"] <= 4.0]
    _lines = []
    if _bullish:
        _syms = ", ".join(s["symbol"] for s in _bullish[:3])
        _lines.append(f"**Strongest setups:** {_syms}")
    if _bearish:
        _syms = ", ".join(s["symbol"] for s in _bearish[:3])
        _lines.append(f"**Weak signals:** {_syms}")
    _lines.append(
        f"**Top pick:** {_top['symbol']} "
        f"(score {_top['score']:.1f}, grade {_top['grade']})"
    )
    return " · ".join(_lines) if _lines else "No clear setups."


# --- Watchlist (fragment — loads independently of briefing) ---
@st.fragment
def _render_watchlist():
    st.subheader("Your watchlist")
    MAX_WATCHLIST_SCORES = 5
    if "watchlist_load_all_scores" not in st.session_state:
        st.session_state["watchlist_load_all_scores"] = False
    try:
        from trading.analysis.ai_score import compute_ai_score
        from trading.data.watchlist import WatchlistManager

        wm = WatchlistManager()
        rows = wm.get_all() or []
        syms = [str(r.get("symbol", "")).upper() for r in rows if r.get("symbol")]
        if not syms:
            st.caption(
                "Your watchlist is empty. Search any ticker using the search bar above, "
                "open it, then click 'Add to watchlist' in the deep dive."
            )
        else:
            _load_all = bool(st.session_state.get("watchlist_load_all_scores"))
            _max_score = len(syms[:15]) if _load_all else min(
                MAX_WATCHLIST_SCORES, len(syms[:15])
            )
            if len(syms) > MAX_WATCHLIST_SCORES:
                _b1, _b2 = st.columns([3, 1])
                with _b1:
                    if not _load_all:
                        st.caption(
                            f"Showing AI scores for the first {MAX_WATCHLIST_SCORES} "
                            "tickers. Open a ticker for full analysis."
                        )
                with _b2:
                    if st.button("Load all scores", key="wl_load_all_scores"):
                        try:
                            st.session_state["watchlist_load_all_scores"] = True
                            st.rerun()
                        except Exception as _re:
                            logger.warning("watchlist load all: %s", _re)
            scored_count = 0
            for sym in syms[:15]:
                if scored_count >= _max_score:
                    break
                try:
                    q = get_quote(sym)
                    px = q.get("price")
                    prev = q.get("prev_close")
                    if px is None:
                        h = get_history(sym, period="5d")
                        if not h.empty:
                            px = float(h["Close"].iloc[-1])
                        if len(h) >= 2 and prev in (None, 0):
                            prev = float(h["Close"].iloc[-2])
                    dlt = None
                    if px is not None and prev not in (None, 0):
                        dlt = (float(px) - float(prev)) / float(prev) * 100
                    ai = compute_ai_score(sym)
                    try:
                        st.session_state[f"ai_cache_{sym}"] = ai
                    except Exception:
                        pass
                    asc = float(ai.get("overall_score", 0) or 0)
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        d_s = f"{dlt:+.2f}%" if dlt is not None else "—"
                        color = (
                            "green"
                            if dlt and dlt > 0
                            else "red"
                            if dlt and dlt < 0
                            else "gray"
                        )
                        px_s = f"${float(px):.2f}" if px is not None else "—"
                        st.markdown(
                            f"**{sym}** {px_s} :{color}[{d_s}] · "
                            f":{_score_color(asc)}[AI {asc:.1f}]"
                        )
                    with c2:
                        if st.button("View", key=f"wl_{sym}"):
                            _set_deep_dive(sym)
                    scored_count += 1
                except Exception as ex:
                    st.caption(f"unavailable: {ex}")

            # Watchlist AI Digest
            try:
                import time as _wt

                _dig_key = "wl_digest"
                _dig_ts_key = "wl_digest_ts"
                _dig_sym_key = "wl_digest_syms"
                _now_dig = _wt.time()
                _cached_dig = st.session_state.get(_dig_key)
                _cached_ts = st.session_state.get(_dig_ts_key, 0)
                _cached_syms = st.session_state.get(_dig_sym_key, [])

                _wl_scores = []
                for _s in syms[:15]:
                    _ai = st.session_state.get(f"ai_cache_{_s}")
                    if _ai and not _ai.get("error"):
                        _top_sig = [
                            sig.get("name", "")
                            + ": "
                            + str(sig.get("value", ""))
                            for sig in _ai.get("signals", [])
                            if sig.get("impact") in ("positive", "negative")
                        ][:3]
                        _wl_scores.append(
                            {
                                "symbol": _s,
                                "score": _ai.get(
                                    "overall_score",
                                    5.0,
                                ),
                                "grade": _ai.get("grade", "C"),
                                "top_signals": _top_sig,
                            }
                        )

                _stale = (
                    _now_dig - _cached_ts > 3600
                    or set(_cached_syms) != set(syms)
                )

                st.markdown("---")
                st.markdown("#### 🧠 Watchlist Digest")

                _col_d1, _col_d2 = st.columns([6, 1])
                with _col_d2:
                    _refresh = st.button(
                        "↻",
                        key="wl_digest_refresh",
                        help="Refresh digest",
                    )

                if _refresh:
                    st.session_state.pop(_dig_key, None)
                    _stale = True

                if not _stale and _cached_dig:
                    with _col_d1:
                        st.markdown(_cached_dig)
                elif _wl_scores:
                    with _col_d1:
                        with st.spinner("Generating digest..."):
                            try:
                                from config.llm_config import (
                                    llm_available,
                                    require_llm,
                                )

                                from agents.llm.active_llm_calls import (
                                    call_active_llm_simple,
                                )

                                _sorted = sorted(
                                    _wl_scores,
                                    key=lambda x: x["score"],
                                    reverse=True,
                                )
                                if not llm_available():
                                    require_llm("Watchlist digest")
                                    _digest = _build_fallback_digest(
                                        _sorted,
                                    )
                                else:
                                    _top3 = _sorted[:3]
                                    _prompt = (
                                        "You are a quant analyst. Write a "
                                        "3-4 sentence plain-English digest "
                                        "of this watchlist. Mention the "
                                        "strongest and weakest setups, any "
                                        "notable signals, and overall tone. "
                                        "Be direct and specific. No "
                                        "disclaimers.\n\n"
                                        "Watchlist scores:\n"
                                        f"{_sorted}\n\nTop signals:\n{_top3}"
                                    )
                                    _digest = call_active_llm_simple(
                                        _prompt,
                                        max_tokens=200,
                                    )
                                    if not _digest:
                                        _digest = _build_fallback_digest(
                                            _sorted,
                                        )
                            except Exception:
                                _digest = _build_fallback_digest(
                                    sorted(
                                        _wl_scores,
                                        key=lambda x: x["score"],
                                        reverse=True,
                                    ),
                                )

                            st.session_state[_dig_key] = _digest
                            st.session_state[_dig_ts_key] = _now_dig
                            st.session_state[_dig_sym_key] = list(syms)
                            st.markdown(_digest)
                else:
                    with _col_d1:
                        st.caption(
                            "Add tickers to your watchlist to see "
                            "your daily digest."
                        )
            except Exception as _de:
                st.caption(f"Digest unavailable: {_de}")
    except Exception as e:
        st.caption(f"unavailable: {e}")


_render_watchlist()

# --- Morning briefing (primary value — cached 30m) ---
st.markdown("---")
st.subheader("Morning briefing")
st.caption(
    "Briefing forecasts use a fast "
    "consensus model optimised for "
    "speed. For full precision, open "
    "any ticker in the Analyze page. "
    "Generate on demand below (cached 30 minutes)."
)


@st.fragment
def _render_briefing():
    _bkey = "home_briefing_report"
    _btkey = "home_briefing_ts"
    _bpkey = "home_briefing_prefs"

    def _load_briefing_prefs_home() -> dict:
        try:
            from config.user_store import load_user_preferences
            from utils.session_utils import get_stable_user_id

            _uid = st.session_state.get(
                "evolve_session_id"
            ) or get_stable_user_id()
            return load_user_preferences(_uid) or {}
        except Exception:
            return {}

    def _briefing_pref_fingerprint(_prefs: dict) -> str:
        """Invalidate home cache when Settings-saved prefs affecting the scan change."""
        sectors = _prefs.get("preferred_sectors") or []
        if isinstance(sectors, list):
            sectors = tuple(sorted(str(s) for s in sectors))
        else:
            sectors = ()
        return "|".join(
            [
                str(_prefs.get("briefing_universe", "")),
                str(float(_prefs.get("min_ai_score", 5.5))),
                str(bool(_prefs.get("watchlist_only", False))),
                str(sectors),
                str(_prefs.get("opportunity_direction", "")),
                str(_prefs.get("scoring_style", "")),
            ]
        )

    _now = time.time()
    _prefs_now = _load_briefing_prefs_home()
    _pref_hash = _briefing_pref_fingerprint(_prefs_now)
    _cached = st.session_state.get(_bkey)
    _cached_ts = st.session_state.get(_btkey, 0)
    _cached_prefs = st.session_state.get(_bpkey, "")
    _fresh = (
        _cached is not None
        and (_now - _cached_ts) < 1800
        and _cached_prefs == _pref_hash
    )

    def _run_briefing() -> None:
        _prefs = _load_briefing_prefs_home()

        _universe_pref = _prefs.get(
            "briefing_universe",
            "SP100 (balanced, ~60s)",
        )
        _universe_map = {
            "Top 25 (fastest, ~30s)": "sp100",
            "SP100 (balanced, ~60s)": "sp100",
            "SP500 (broadest, ~3min)": "sp500",
            "NASDAQ100 (tech-heavy)": "nasdaq100",
        }
        _universe = _universe_map.get(_universe_pref, "sp100")
        _min_score = float(_prefs.get("min_ai_score", 5.5))

        try:
            from agents.briefing.morning_briefing import MorningBriefing

            st.info(
                "Generating briefing: Quick Score on up to 50 tickers, "
                "then 5 fast models per top pick (~20–60s typical, 3 picks). "
                "News and watchlist above refresh independently.",
                icon="⏳",
            )
            _mb = MorningBriefing(
                universe=_universe,
                min_ai_score=_min_score,
            )
            progress = st.progress(0, text="Scanning universe (Quick Score)…")

            def _brief_progress(done: int, total: int) -> None:
                if total <= 0:
                    return
                progress.progress(
                    min(1.0, float(done) / float(total)),
                    text=f"Scanning universe… {done}/{total}",
                )

            try:
                report = _mb.generate(progress_callback=_brief_progress)
            finally:
                progress.empty()
            st.session_state[_bkey] = report
            st.session_state[_btkey] = time.time()
            st.session_state[_bpkey] = _briefing_pref_fingerprint(_prefs)
            st.rerun()
        except Exception as e:
            st.caption(f"unavailable: {e}")

    def _render_briefing_report(report: dict) -> None:
        reg = report.get("market_regime") or {}
        reg_lbl = reg.get("regime", "NEUTRAL")
        vix_l = reg.get("vix_level")
        reg_line = f"{reg_lbl.replace('_', '-').title()}"
        if vix_l is not None:
            reg_line += f" · VIX {vix_l:.1f}"
        st.markdown(f"**{reg_line}** — {reg.get('description', '')}")
        try:
            from datetime import datetime

            from trading.utils.time_utils import format_timestamp

            _ts = report.get("timestamp") or ""
            if _ts:
                _dt = datetime.fromisoformat(_ts.replace("Z", "+00:00"))
                st.caption(
                    f"Briefing generated: "
                    f"{format_timestamp(_dt, timezone='America/New_York')}"
                )
        except Exception:
            pass

        opps = report.get("top_opportunities") or []
        _portfolio = report.get("portfolio")
        if _portfolio and len(opps) >= 2:
            _sharpe = _portfolio.get("expected_sharpe")
            if _sharpe is not None:
                st.caption(
                    f"📐 Portfolio Sharpe: "
                    f"{float(_sharpe):.2f} | "
                    f"{_portfolio.get('note', '')}"
                )
        if opps:
            for opp in opps[:5]:
                sym = opp.get("symbol", "")
                if not sym:
                    continue
                info = {}
                try:
                    info = get_info(sym) or {}
                except Exception:
                    pass
                co = info.get("longName") or info.get("shortName") or ""
                sec = info.get("sector") or ""
                sc = float(opp.get("ai_score") or 0)
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    st.markdown(
                        f"**{sym}** · {co} · _{sec}_ — "
                        f":{_score_color(sc)}[AI {sc:.1f}]"
                    )
                    th = opp.get("thesis") or ""
                    if th:
                        st.caption(th)
                    fc = opp.get("forecast") or {}
                    if fc:
                        st.caption(
                            f"Entry **{opp.get('entry')}** → Target "
                            f"**{fc.get('consensus_price')}**"
                        )
                    if opp.get("risk_note"):
                        st.caption(opp["risk_note"])
                    _weight = opp.get("weight_pct")
                    if _weight:
                        st.caption(
                            f"Suggested allocation: **{_weight}** of portfolio "
                            f"(mean-variance optimized)"
                        )
                with col_b:
                    if st.button("Open", key=f"hb_{sym}"):
                        _set_deep_dive(sym)
        else:
            st.caption(
                "No opportunities passed the score threshold right now."
            )

    if not _fresh:
        if (
            _cached is not None
            and (_now - _cached_ts) < 1800
            and _cached_prefs != _pref_hash
        ):
            st.info(
                "Your briefing preferences were updated — "
                "generate a fresh briefing to apply them."
            )
        else:
            st.info(
                "Morning briefing not yet "
                "generated for this session."
            )
        if st.button(
            "🌅 Generate Morning Briefing",
            key="gen_briefing_btn",
            type="primary",
        ):
            st.session_state["_briefing_requested"] = True
            st.rerun()
        if st.session_state.get("_briefing_requested"):
            st.session_state.pop("_briefing_requested", None)
            _run_briefing()
        return

    _render_briefing_report(st.session_state[_bkey])
    if st.button(
        "🔄 Refresh Briefing",
        key="refresh_briefing_btn",
    ):
        st.session_state.pop(_bkey, None)
        st.session_state.pop(_btkey, None)
        st.session_state.pop(_bpkey, None)
        st.rerun()


_render_briefing()

# --- Chat ---
st.markdown("---")
st.subheader("Chat")
if "home_chat_messages" not in st.session_state:
    st.session_state.home_chat_messages = []

_pending = st.session_state.pop("home_chat_pending", None)
_inp = st.chat_input(
    "Ask anything about the market...",
    key="home_bottom_chat",
)
_prompt = (_inp or _pending or "").strip()
if _prompt:
    st.session_state.home_chat_messages.append(
        {"role": "user", "content": _prompt}
    )

for msg in st.session_state.home_chat_messages:
    with st.chat_message(msg.get("role", "user")):
        if msg.get("role") == "assistant":
            for _c in msg.get("tool_captions") or []:
                st.caption(_c)
        st.markdown(msg.get("content", ""))

if _prompt:
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            _chat_res = _home_chat_reply(_prompt)
            _reply = _chat_res.text
            for _c in _chat_res.tool_captions or []:
                st.caption(_c)
            st.markdown(_reply)
    st.session_state.home_chat_messages.append(
        {
            "role": "assistant",
            "content": _reply,
            "tool_captions": _chat_res.tool_captions or [],
        }
    )

# --- Deep dive (user-selected ticker — bottom of page) ---
dd = st.session_state.get("deep_dive_ticker")
if dd:
    st.markdown("---")
    render_deep_dive(str(dd))

try:
    from trading.services.alert_checker import check_alerts_for_user
    from utils.session_utils import get_stable_user_id

    _uid = get_stable_user_id()
    _triggered = check_alerts_for_user(_uid)
    if _triggered:
        for _alert in _triggered:
            st.warning(
                "Alert triggered: "
                f"{_alert.get('symbol', '')} "
                f"{_alert.get('condition', '')} "
                f"{_alert.get('threshold', '')}"
            )
except Exception as _e:
    logger.debug("Alert check failed: %s", _e)
