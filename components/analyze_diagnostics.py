# -*- coding: utf-8 -*-
"""Econometric diagnostics (EconometricDiagnostics + quick stats)."""
import logging

import streamlit as st

logger = logging.getLogger(__name__)


def _render_risk_summary(ticker: str, hist) -> None:
    """
    Plain-English interpretation of key risk metrics for non-technical users.
    """
    if hist is None or hist.empty:
        return
    try:
        import numpy as np

        _cm = {c.lower(): c for c in hist.columns}
        _cc = _cm.get("close", hist.columns[0])
        _close = hist[_cc].dropna()
        _rets = _close.pct_change().dropna()

        _vol = float(_rets.std() * np.sqrt(252))

        _sma50 = (
            float(_close.iloc[-50:].mean())
            if len(_close) >= 50
            else None
        )
        _last = float(_close.iloc[-1])
        _trend = (
            "uptrend"
            if _sma50 and _last > _sma50
            else "downtrend"
            if _sma50
            else "unknown trend"
        )

        _mid = max(1, len(_close) // 2)
        _mean_1h = float(_close.iloc[:_mid].mean())
        _mean_2h = float(_close.iloc[_mid:].mean())
        _trending = (
            abs(_mean_2h - _mean_1h) / max(abs(_mean_1h), 1e-12) > 0.05
        )

        _roll_max = _close.cummax()
        _drawdown = float(
            ((_close - _roll_max) / _roll_max).min() * 100
        )

        _vol_plain = (
            "moves around a lot"
            if _vol > 0.4
            else "is fairly volatile"
            if _vol > 0.25
            else "moves at a normal pace"
            if _vol > 0.15
            else "is relatively stable"
        )
        _trend_plain = (
            "been going up recently"
            if _trend == "uptrend"
            else "been going down recently"
            if _trend == "downtrend"
            else "been moving sideways"
        )
        if _trending:
            _stat_plain = (
                "It has been trending in one direction, which means "
                "momentum strategies (following the trend) tend to work "
                "better here."
            )
        else:
            _stat_plain = (
                "It tends to bounce back after big moves up or down, "
                "which means buying dips or selling rallies can work well."
            )
        _dd_abs = abs(_drawdown)
        _dd_plain = (
            f"The worst it has dropped from a peak during this period was "
            f"**{_dd_abs:.0f}%** — "
            + (
                "that's a significant drop to be aware of."
                if _dd_abs > 20
                else "a moderate pullback."
                if _dd_abs > 10
                else "a relatively small pullback."
            )
        )
        parts = [
            f"**{ticker}** has {_trend_plain} and {_vol_plain} "
            f"({_vol * 100:.0f}% yearly swings). "
            f"{_stat_plain} {_dd_plain}"
        ]

        st.info(" ".join(parts), icon="📊")
    except Exception as e:
        logger.debug("Risk summary: %s", e)


def render_diagnostics(ticker: str, hist) -> None:
    try:
        _render_risk_summary(ticker, hist)
    except Exception:
        pass

    try:
        from trading.analysis.econometric_diagnostics import EconometricDiagnostics

        if hist is None or hist.empty:
            st.caption("No history for diagnostics.")
            return
        diag = EconometricDiagnostics(ticker, hist)
        diag.render_streamlit()
    except Exception as e:
        st.caption(f"unavailable: {e}")
