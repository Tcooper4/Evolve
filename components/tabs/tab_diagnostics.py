# -*- coding: utf-8 -*-
"""Analyze page tab body (extracted; logic unchanged)."""
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as _components
from plotly.subplots import make_subplots

from components.analyze_common import (
    _extract_forecast_values,
    _generate_recommendation,
    _is_english,
    _news_sentiment_score,
)
from trading.data.earnings_calendar import get_upcoming_earnings
from trading.data.insider_flow import get_insider_flow
from trading.data.price_cache import get_history, get_info, get_news, get_quote

try:
    from utils.dataframe_utils import normalize_for_display
except ImportError:
    def normalize_for_display(df):
        return df

logger = logging.getLogger(__name__)


def render(
    ticker: str,
    hist,
    period: str,
    period_label: str,
    trader_mode: str,
    _interval: str,
    _tf_label: str,
    *,
    backend: dict,
) -> None:
    """Streamlit tab body (legacy Analyze)."""
    DataLoader = backend["DataLoader"]
    DataLoadRequest = backend["DataLoadRequest"]
    YFinanceProvider = backend["YFinanceProvider"]
    LSTMForecaster = backend["LSTMForecaster"]
    XGBoostModel = backend["XGBoostModel"]
    ProphetModel = backend["ProphetModel"]
    ARIMAModel = backend["ARIMAModel"]
    FeatureEngineering = backend["FeatureEngineering"]
    DataPreprocessor = backend["DataPreprocessor"]
    ModelSelectorAgent = backend["ModelSelectorAgent"]
    MarketAnalyzer = backend["MarketAnalyzer"]
    try:
        st.subheader("Model Diagnostics")
        st.caption("Econometric tests on price data")

        try:
            from trading.analysis.econometric_diagnostics import (
                EconometricDiagnostics,
            )
            if hist is not None and not hist.empty:
                diag = EconometricDiagnostics(ticker, hist)
                diag.render_streamlit()
            else:
                st.caption("Load a symbol to run diagnostics.")
        except Exception as e:
            st.caption(f"Diagnostics unavailable: {e}")

        st.markdown("---")
        st.caption("Quick checks (ADF, Ljung-Box, ARCH)")

        try:
            _dh = st.session_state.get("analyze_forecast_data") or get_history(ticker, period="1y")

            if _dh is not None and not _dh.empty:
                _close = _dh["Close"].dropna()
                _returns = _close.pct_change().dropna()

                st.markdown("**Stationarity Test (ADF)**")
                try:
                    from statsmodels.tsa.stattools import adfuller
                    _adf = adfuller(_close.values)
                    _adf_p = _adf[1]
                    _adf_stat = _adf[0]
                    _stat_color = "#26a69a" if _adf_p < 0.05 else "#ef5350"
                    st.markdown(
                        f'<span style="color:{_stat_color}">'
                        f'ADF Statistic: {_adf_stat:.4f} | '
                        f'p-value: {_adf_p:.4f} | '
                        f'{"Stationary" if _adf_p < 0.05 else "Non-stationary"}'
                        f'</span>',
                        unsafe_allow_html=True,
                    )
                except ImportError:
                    st.caption("statsmodels not installed for ADF")

                st.markdown("**White Noise Test (Ljung-Box)**")
                try:
                    from statsmodels.stats.diagnostic import acorr_ljungbox
                    _lb = acorr_ljungbox(_returns, lags=[10], return_df=True)
                    _lb_p = float(_lb["lb_pvalue"].iloc[0])
                    _wn_color = "#26a69a" if _lb_p > 0.05 else "#ef5350"
                    st.markdown(
                        f'<span style="color:{_wn_color}">'
                        f'Ljung-Box p-value: {_lb_p:.4f} | '
                        f'{"White noise ✓" if _lb_p > 0.05 else "Autocorrelation detected"}'
                        f'</span>',
                        unsafe_allow_html=True,
                    )
                except ImportError:
                    st.caption("statsmodels not installed for LB")

                st.markdown("**Volatility (ARCH Effect)**")
                try:
                    from statsmodels.stats.diagnostic import het_arch
                    _arch = het_arch(_returns.values)
                    _arch_p = _arch[1]
                    _arch_color = "#ff9800" if _arch_p < 0.05 else "#26a69a"
                    st.markdown(
                        f'<span style="color:{_arch_color}">'
                        f'ARCH p-value: {_arch_p:.4f} | '
                        f'{"Volatility clustering ⚠" if _arch_p < 0.05 else "No ARCH effect ✓"}'
                        f'</span>',
                        unsafe_allow_html=True,
                    )
                except ImportError:
                    st.caption("statsmodels not installed for ARCH")

                st.markdown("**Return Distribution**")
                from scipy import stats as _scipy_stats
                _skew = float(_scipy_stats.skew(_returns))
                _kurt = float(_scipy_stats.kurtosis(_returns))
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Skewness", f"{_skew:.3f}")
                with c2:
                    st.metric("Excess Kurtosis", f"{_kurt:.3f}")
                with c3:
                    _ann_vol = float(_returns.std() * np.sqrt(252) * 100)
                    st.metric("Ann. Volatility", f"{_ann_vol:.1f}%")
                with c4:
                    _sharpe = float(_returns.mean() / _returns.std() * np.sqrt(252)) if _returns.std() > 0 else 0.0
                    st.metric("Sharpe (approx)", f"{_sharpe:.2f}")

                st.caption(
                    "ADF: p<0.05 = stationary (good for ARIMA). "
                    "Ljung-Box: p>0.05 = white noise residuals (good model fit). "
                    "ARCH: p<0.05 = use GARCH for volatility modeling."
                )
            else:
                st.caption("Load price data (e.g. run Quick Forecast) to see diagnostics.")
        except Exception as _de:
            st.caption(f"Diagnostics unavailable: {_de}")
    except Exception as e:
        st.caption(f"Tab unavailable: {e}")
