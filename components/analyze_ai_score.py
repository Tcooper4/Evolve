# -*- coding: utf-8 -*-
"""AI Score panel and recommendation-style summary."""
from typing import Any, Dict, Optional

import streamlit as st

from components.analyze_common import _generate_recommendation, _news_sentiment_score


def render_ai_score(ticker: str, hist, *, trader_mode: str = "Short-term") -> None:
    """Compute AI Score, signals table, and buy/sell-style recommendation."""
    try:
        from trading.analysis.ai_score import compute_ai_score

        _sym = (ticker or "").strip().upper() or "AAPL"
        with st.spinner("Computing AI Score..."):
            score_result = compute_ai_score(_sym, hist)
        if score_result.get("error"):
            st.caption(str(score_result.get("error")))
            return
        st.markdown("### AI Score")
        _oc = float(score_result.get("overall_score", 0) or 0)
        _grade = score_result.get("grade", "—")
        st.metric("Overall", f"{_oc:.1f}/10", delta=_grade)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.caption(f"Technical: {float(score_result.get('technical_score', 0) or 0):.1f}")
        with c2:
            st.caption(f"Momentum: {float(score_result.get('momentum_score', 0) or 0):.1f}")
        with c3:
            st.caption(f"Sentiment: {float(score_result.get('sentiment_score', 0) or 0):.1f}")
        with c4:
            st.caption(f"Fundamental: {float(score_result.get('fundamental_score', 0) or 0):.1f}")
        summ = score_result.get("summary")
        if summ:
            st.caption(summ)
        news_score = _news_sentiment_score(_sym)
        st.caption(f"News sentiment score: {news_score:.1f}/10")
        forecast_result = None
        try:
            from trading.models.forecast_router import (
                ForecastRouter,
                get_router_singleton,
            )

            router = get_router_singleton()
            if hist is not None and not hist.empty:
                forecast_result = router.get_consensus_forecast(
                    data=hist, horizon=7, symbol=_sym
                )
        except Exception:
            forecast_result = None
        _rec = _generate_recommendation(
            _sym,
            score_result,
            forecast_result=forecast_result,
            trader_mode=trader_mode,
        )
        if _rec:
            _action = _rec.get("action", "HOLD")
            _color = (
                "#26a69a"
                if _action == "BUY"
                else "#ef5350"
                if _action == "SELL"
                else "#ff9800"
            )
            st.markdown(
                f'<div style="font-size:1.6rem;font-weight:700;color:{_color}">'
                f"{_action}</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"Conviction: **{_rec.get('conviction', '—')}** · "
                f"Entry {_rec.get('entry')} · Target {_rec.get('target')} · "
                f"Stop {_rec.get('stop')}"
            )
            if _rec.get("reasons"):
                for icon, txt in _rec["reasons"][:8]:
                    st.markdown(f"- {icon} {txt}")
        signals = score_result.get("signals") or []
        if signals:
            import pandas as pd

            sig_df = pd.DataFrame(signals)
            # Force string columns to prevent
            # PyArrow type conversion errors
            # on mixed-type values like
            # "88% OTC volume"
            for _col in ["value", "Value",
                         "description",
                         "Description",
                         "impact", "Impact",
                         "name", "Name"]:
                if _col in sig_df.columns:
                    sig_df[_col] = (
                        sig_df[_col]
                        .fillna("")
                        .astype(str)
                    )
            st.dataframe(
                sig_df,
                width="stretch",
                hide_index=True,
            )
    except Exception as e:
        st.caption(f"unavailable: {e}")


def get_ai_recommendation_dict(
    ticker: str, hist, *, trader_mode: str = "Short-term"
) -> Optional[Dict[str, Any]]:
    """Structured recommendation for Deep Dive header card (no Streamlit)."""
    try:
        from trading.analysis.ai_score import compute_ai_score
        from trading.models.forecast_router import (
            ForecastRouter,
            get_router_singleton,
        )

        _sym = (ticker or "").strip().upper() or "AAPL"
        score_result = compute_ai_score(_sym, hist)
        if score_result.get("error"):
            return None
        forecast_result = None
        try:
            router = get_router_singleton()
            if hist is not None and not hist.empty:
                forecast_result = router.get_consensus_forecast(
                    data=hist, horizon=7, symbol=_sym
                )
        except Exception:
            pass
        return _generate_recommendation(
            _sym,
            score_result,
            forecast_result=forecast_result,
            trader_mode=trader_mode,
        )
    except Exception:
        return None


def top_signals_summary(signals: list, limit: int = 5) -> str:
    parts = []
    for s in (signals or [])[:limit]:
        desc = s.get("description") or s.get("name") or ""
        if desc:
            parts.append(desc)
    return " ".join(parts) if parts else "No signal narrative available."
