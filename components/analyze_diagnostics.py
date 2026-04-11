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
        _vol_label = (
            "very high"
            if _vol > 0.6
            else "high"
            if _vol > 0.4
            else "elevated"
            if _vol > 0.25
            else "normal"
            if _vol > 0.15
            else "low"
        )

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

        parts = []
        parts.append(
            f"**{ticker}** is in a **{_trend}** with **{_vol_label} volatility** "
            f"({(_vol * 100):.0f}% annualised)."
        )
        if _trending:
            parts.append(
                "The price series is **trending** (non-stationary) — momentum "
                "strategies may work better than mean-reversion."
            )
        else:
            parts.append(
                "The price series shows **mean-reverting** tendencies — range "
                "trading may be effective."
            )
        parts.append(
            f"Maximum drawdown over the period: **{_drawdown:.1f}%**."
        )

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

    st.markdown("---")
    st.caption("Quick checks (ADF, Ljung-Box, ARCH)")
    try:
        import numpy as np
        from trading.data.price_cache import get_history

        _ss_data = st.session_state.get(
            "analyze_forecast_data")
        if (_ss_data is not None
                and hasattr(_ss_data, "empty")
                and not _ss_data.empty):
            _dh = _ss_data
        elif (hist is not None
                and hasattr(hist, "empty")
                and not hist.empty):
            _dh = hist
        else:
            _dh = get_history(
                ticker, period="1y")
        if _dh is None or _dh.empty:
            st.caption("Load price data to see quick diagnostics.")
            return
        _close = _dh["Close"].dropna()
        _returns = _close.pct_change().dropna()
        st.markdown("**Stationarity (ADF)**")
        try:
            from statsmodels.tsa.stattools import adfuller

            _adf = adfuller(_close.values)
            _adf_p = _adf[1]
            _adf_stat = _adf[0]
            _col = "#26a69a" if _adf_p < 0.05 else "#ef5350"
            st.markdown(
                f'<span style="color:{_col}">ADF {_adf_stat:.4f} | p={_adf_p:.4f} | '
                f'{"Stationary" if _adf_p < 0.05 else "Non-stationary"}</span>',
                unsafe_allow_html=True,
            )
        except ImportError:
            st.caption("statsmodels not installed for ADF")

        st.markdown("**Ljung-Box**")
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox

            _lb = acorr_ljungbox(_returns, lags=[10], return_df=True)
            _lb_p = float(_lb["lb_pvalue"].iloc[0])
            _col = "#26a69a" if _lb_p > 0.05 else "#ef5350"
            st.markdown(
                f'<span style="color:{_col}">p={_lb_p:.4f} | '
                f'{"White noise" if _lb_p > 0.05 else "Autocorrelation"}</span>',
                unsafe_allow_html=True,
            )
        except ImportError:
            st.caption("statsmodels not installed for Ljung-Box")
    except Exception as e:
        st.caption(f"unavailable: {e}")
