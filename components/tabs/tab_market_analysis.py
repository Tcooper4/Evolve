# -*- coding: utf-8 -*-
"""Analyze page tab body (extracted; logic unchanged)."""
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
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
    score_mode: str = "Buy",
    scoring_style: str = "Balanced (default)",
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
        try:
            st.header("📊 Market Analysis")
            st.markdown("Rolling correlation vs SPY and volatility regime. Load data in Quick Forecast first.")
            # Self-fetch if session data missing
            if st.session_state.get("analyze_forecast_data") is None:
                try:
                    from trading.data.price_cache import get_history as _gh

                    _fetched = _gh(ticker, period="1y")
                    if not _fetched.empty:
                        st.session_state["analyze_forecast_data"] = _fetched
                        st.session_state["analyze_forecast_data_symbol"] = str(
                            ticker,
                        ).strip().upper()
                        st.session_state["analyze_symbol"] = ticker
                except Exception:
                    pass

            if st.session_state.get("analyze_forecast_data") is None:
                st.info(
                    "Enter a ticker above and "
                    "press Enter to load data.",
                    icon="📈",
                )
            else:
                _symbol = st.session_state.get("analyze_symbol", "Symbol")
                hist = st.session_state.get("analyze_forecast_data").copy()
                if "Close" not in hist.columns and "close" in hist.columns:
                    hist["Close"] = hist["close"]

                st.subheader(f"📊 {_symbol} Market Analysis")
                try:
                    import yfinance as yf
                    import plotly.graph_objects as go

                    _col1, _col2 = st.columns(2)
                    with _col1:
                        st.markdown("**Rolling 60-Day Correlation vs SPY**")
                        try:
                            _spy_df = get_history("SPY", period="1y")
                            if not _spy_df.empty and "Close" in _spy_df.columns:
                                _spy = _spy_df["Close"]
                            else:
                                import yfinance as yf
                                _spy = yf.Ticker("SPY").history(period="1y")["Close"]
                            _sym_ret = hist["Close"].pct_change().dropna()
                            _spy_ret = _spy.pct_change().dropna()
                            _sym_aligned, _spy_aligned = _sym_ret.align(_spy_ret, join="inner")
                            _rolling_corr = _sym_aligned.rolling(60).corr(_spy_aligned).dropna()
                            _fig_corr = go.Figure(
                                go.Scatter(
                                    x=_rolling_corr.index,
                                    y=_rolling_corr.values,
                                    mode="lines",
                                    line=dict(color="#00D4AA"),
                                    name="60d Correlation",
                                )
                            )
                            _fig_corr.add_hline(y=0, line_dash="dash", line_color="gray")
                            _fig_corr.update_layout(
                                template="plotly_dark",
                                height=250,
                                yaxis=dict(range=[-1, 1]),
                                margin=dict(l=20, r=20, t=20, b=20),
                            )
                            st.plotly_chart(_fig_corr, width='stretch')
                            _current_corr = float(_rolling_corr.iloc[-1])
                            st.caption(
                                f"Current 60d correlation with SPY: {_current_corr:.2f}"
                            )
                        except Exception as _e:
                            st.caption(f"Correlation unavailable: {_e}")

                    with _col2:
                        st.markdown("**Volatility Regime**")
                        try:
                            _close = hist["Close"].values.astype(float)
                            _returns = np.diff(_close) / _close[:-1]
                            _vol_20d = float(
                                np.std(_returns[-20:]) * np.sqrt(252) * 100
                            )
                            _vol_60d = (
                                float(np.std(_returns[-60:]) * np.sqrt(252) * 100)
                                if len(_returns) >= 60
                                else _vol_20d
                            )
                            _vol_1y = float(np.std(_returns) * np.sqrt(252) * 100)
                            _regime = (
                                "HIGH"
                                if _vol_20d > _vol_1y * 1.3
                                else "LOW"
                                if _vol_20d < _vol_1y * 0.7
                                else "NORMAL"
                            )
                            _c1, _c2, _c3 = st.columns(3)
                            _c1.metric("20d Vol", f"{_vol_20d:.1f}%")
                            _c2.metric("60d Vol", f"{_vol_60d:.1f}%")
                            _c3.metric("1Y Vol", f"{_vol_1y:.1f}%")
                            _color = {"HIGH": "🔴", "LOW": "🟢", "NORMAL": "🟡"}.get(
                                _regime, "⚪"
                            )
                            st.markdown(
                                f"**Volatility Regime: {_color} {_regime}**"
                            )
                            if _regime == "HIGH":
                                st.caption(
                                    "Current volatility significantly above 1-year average — elevated risk."
                                )
                            elif _regime == "LOW":
                                st.caption(
                                    "Current volatility below average — potential for mean reversion."
                                )
                        except Exception as _e:
                            st.caption(
                                f"Volatility analysis unavailable: {_e}"
                            )

                except Exception as _e:
                    st.error(
                        f"Market Analysis error: {type(_e).__name__}: {_e}"
                    )
                    import traceback
                    st.code(traceback.format_exc())
        except Exception as e:
            st.caption(f"Tab unavailable: {e}")
    except Exception as e:
        st.caption(f"Tab unavailable: {e}")
