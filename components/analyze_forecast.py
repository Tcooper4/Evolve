# -*- coding: utf-8 -*-
"""Consensus forecast panel for Deep Dive and compact Analyze-style views."""
import streamlit as st


def render_forecast(ticker: str, hist, *, horizon: int = 7) -> None:
    """Multi-model consensus summary (ForecastRouter)."""
    try:
        if hist is None or hist.empty:
            st.caption("Load price history to run a forecast.")
            return
        from trading.models.forecast_router import ForecastRouter

        router = ForecastRouter()
        fc = router.get_consensus_forecast(
            data=hist,
            horizon=int(horizon),
            symbol=str(ticker or "").strip().upper() or None,
        )
        if not fc or fc.get("error"):
            st.caption(str(fc.get("error", "Forecast unavailable.")))
            return
        cp = fc.get("consensus_price")
        direction = fc.get("direction", "—")
        conviction = fc.get("conviction", "—")
        models = fc.get("models_used") or []
        last = float(hist["Close"].iloc[-1])
        st.subheader("Consensus forecast")
        _wfc = fc.get("walk_forward_confidence")
        if _wfc:
            st.caption(f"Walk-forward cache confidence: **{_wfc}**")
        for _ww in fc.get("walk_forward_warnings") or []:
            st.warning(_ww)
        st.metric("Direction", direction, delta=f"{conviction} conviction")
        if cp:
            pct = (float(cp) - last) / last * 100 if last else 0
            st.metric("Consensus price", f"${float(cp):.2f}", delta=f"{pct:+.1f}%")
        if models:
            st.caption("Models: " + ", ".join(str(m) for m in models[:12]))
    except Exception as e:
        st.caption(f"unavailable: {e}")
