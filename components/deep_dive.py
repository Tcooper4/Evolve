# -*- coding: utf-8 -*-
"""Single-ticker deep analysis surface (scroll + 4 tabs)."""
import streamlit as st

from components.analyze_ai_score import get_ai_recommendation_dict, top_signals_summary
from components.analyze_chart import render_price_chart
from components.analyze_diagnostics import render_diagnostics
from components.analyze_forecast import render_forecast
from components.analyze_news import render_news
from components.analyze_options import render_options
from trading.data.price_cache import get_history, get_info, get_quote


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
    except Exception as e:
        st.caption(f"unavailable: {e}")
