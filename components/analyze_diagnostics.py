# -*- coding: utf-8 -*-
"""Econometric diagnostics (EconometricDiagnostics + quick stats)."""
import streamlit as st


def render_diagnostics(ticker: str, hist) -> None:
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
