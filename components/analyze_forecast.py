# -*- coding: utf-8 -*-
"""Consensus forecast panel for Deep Dive and compact Analyze-style views."""
from typing import Any, Dict, Optional

import streamlit as st


def render_forecast(
    ticker: str, hist, *, horizon: int = 7
) -> Optional[Dict[str, Any]]:
    """Multi-model consensus summary (ForecastRouter). Returns forecast dict for callers."""
    try:
        if hist is None or hist.empty:
            st.caption("Load price history to run a forecast.")
            return None
        from trading.models.forecast_router import ForecastRouter

        router = ForecastRouter()
        fc = router.get_consensus_forecast(
            data=hist,
            horizon=int(horizon),
            symbol=str(ticker or "").strip().upper() or None,
        )
        if not fc or fc.get("error"):
            st.caption(str(fc.get("error", "Forecast unavailable.")))
            return None
        cp = fc.get("consensus_price")
        direction = fc.get("direction", "—")
        conviction = fc.get("conviction", "—")
        models = fc.get("models_used") or []
        _cm = {c.lower(): c for c in hist.columns}
        _cc = _cm.get("close", hist.columns[0])
        last = float(hist[_cc].iloc[-1])
        st.subheader("Consensus forecast")
        st.metric("Direction", direction, delta=f"{conviction} conviction")
        if cp:
            pct = (float(cp) - last) / last * 100 if last else 0
            st.metric("Consensus price", f"${float(cp):.2f}", delta=f"{pct:+.1f}%")
        if models:
            st.caption("Models: " + ", ".join(str(m) for m in models[:12]))
        return fc
    except Exception as e:
        st.caption(f"unavailable: {e}")
        return None
