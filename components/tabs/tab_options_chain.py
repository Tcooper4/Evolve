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
        st.subheader("Options & Short Data")

        try:
            import yfinance as yf

            _t = yf.Ticker(ticker)
            _info = _t.info

            # Short data
            _short_pct = _info.get("shortPercentOfFloat", 0) or 0
            _short_ratio = _info.get("shortRatio", 0) or 0

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(
                    "Short float %",
                    f"{_short_pct * 100:.1f}%",
                    help="% of float sold short. >20% = high short interest",
                )
            with c2:
                st.metric(
                    "Days to cover",
                    f"{_short_ratio:.1f}",
                    help=(
                        "Days to cover short positions at avg volume. "
                        ">5 = squeeze potential"
                    ),
                )
            with c3:
                _pc_key = f"put_call_{ticker}"
                _pc_val = st.session_state.get(_pc_key)
                if _pc_val is not None:
                    st.metric(
                        "Put/call ratio",
                        f"{_pc_val:.2f}",
                        help=">1 bearish, <1 bullish",
                    )
                else:
                    st.metric(
                        "Put/call ratio",
                        "Load chain →",
                        help=(
                            "Load options chain below "
                            "to calculate"
                        ),
                    )

            st.markdown("---")

            # Options chain
            if "options_loaded_" + ticker not in st.session_state:
                if st.button("Load options chain", key="load_options_btn"):
                    st.session_state["options_loaded_" + ticker] = True
                else:
                    st.caption("Click to load live options data.")
                    st.stop()

            _expiries = _t.options
            if _expiries:
                _sel_exp = st.selectbox(
                    "Expiry date",
                    _expiries[:8],
                    key="options_expiry",
                )
                _cache_key = f"options_chain_{ticker}_{_sel_exp}"
                if _cache_key not in st.session_state:
                    try:
                        import threading as _threading

                        _result = [None]

                        def _fetch_chain() -> None:
                            try:
                                _result[0] = _t.option_chain(_sel_exp)
                            except Exception as _e:
                                logger.warning(
                                    "Analyze: option_chain fetch failed: %s", _e
                                )

                        _th = _threading.Thread(target=_fetch_chain)
                        _th.start()
                        _th.join(timeout=8)
                        if _result[0] is not None:
                            st.session_state[_cache_key] = _result[0]
                    except Exception as _e:
                        logger.warning("Analyze: options chain fetch failed: %s", _e)

                _chain = st.session_state.get(_cache_key)
                if _chain is None:
                    st.warning("Options chain timed out. Try again.")
                    st.stop()
                _calls = _chain.calls
                _puts = _chain.puts

                # Put/call ratio from volumes (persist in session_state)
                try:
                    _cv = float(_chain.calls["volume"].fillna(0).sum())
                    _pv = float(_chain.puts["volume"].fillna(0).sum())
                    _pc = round(_pv / _cv, 2) if _cv > 0 else None
                    if _pc is not None:
                        st.session_state[f"put_call_{ticker}"] = _pc
                except Exception:
                    pass

                # IV rank approximation
                # Use ATM options (closest to current price) for IV calculation
                _cur_price = get_quote(ticker).get("price", 0)
                if _cur_price and not _calls.empty:
                    _calls = _calls.copy()
                    _puts = _puts.copy()
                    _calls["dist"] = abs(_calls["strike"] - _cur_price)
                    _atm_iv = float(
                        _calls.nsmallest(5, "dist")["impliedVolatility"].mean()
                    )
                    # Simple IV rank: compare to typical range (annualized)
                    _iv_pct = min(100, _atm_iv * 100)
                    _iv_color = (
                        "#ef5350"
                        if _iv_pct > 50
                        else "#ff9800"
                        if _iv_pct > 30
                        else "#26a69a"
                    )
                    st.markdown(
                        "**ATM IV (approx):** "
                        f'<span style="color:{_iv_color};font-weight:bold">'
                        f"{_iv_pct:.1f}%</span> annualized",
                        unsafe_allow_html=True,
                    )

                col_opt1, col_opt2 = st.columns(2)
                with col_opt1:
                    st.markdown("**Calls**")
                    _c_display = _calls[
                        [
                            "strike",
                            "lastPrice",
                            "impliedVolatility",
                            "volume",
                            "openInterest",
                            "inTheMoney",
                        ]
                    ].copy()
                    _c_display.columns = [
                        "Strike",
                        "Last",
                        "IV",
                        "Vol",
                        "OI",
                        "ITM",
                    ]
                    _c_display["IV"] = (_c_display["IV"] * 100).round(1).astype(str) + "%"
                    st.dataframe(
                        normalize_for_display(_c_display),
                        width='stretch',
                        height=300,
                        key="options_calls_table",
                    )
                with col_opt2:
                    st.markdown("**Puts**")
                    _p_display = _puts[
                        [
                            "strike",
                            "lastPrice",
                            "impliedVolatility",
                            "volume",
                            "openInterest",
                            "inTheMoney",
                        ]
                    ].copy()
                    _p_display.columns = [
                        "Strike",
                        "Last",
                        "IV",
                        "Vol",
                        "OI",
                        "ITM",
                    ]
                    _p_display["IV"] = (_p_display["IV"] * 100).round(1).astype(str) + "%"
                    st.dataframe(
                        normalize_for_display(_p_display),
                        width='stretch',
                        height=300,
                        key="options_puts_table",
                    )
            else:
                st.info("No options data available for " + ticker)
        except Exception as _oe:
            st.caption(f"Options data unavailable: {_oe}")
    except Exception as e:
        st.caption(f"Tab unavailable: {e}")
