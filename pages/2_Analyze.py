# -*- coding: utf-8 -*-
"""
Analyze Page — Forecasting & Market Analysis (reorganized from 2_Forecasting).
Ticker-driven analysis with theme, price_cache, period chart, and news sentiment.
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as _components
from plotly.subplots import make_subplots

from components.theme import inject_theme, render_top_bar, keyboard_shortcut_js
from components.news_candle_chart import render_news_candle_chart
from trading.data.price_cache import get_quote, get_history, get_info, get_news
from ui.page_assistant import render_page_assistant
from trading.data.earnings_calendar import get_upcoming_earnings
from trading.data.insider_flow import get_insider_flow

try:
    st.markdown(keyboard_shortcut_js(), unsafe_allow_html=True)
except Exception:
    pass
inject_theme()
render_top_bar()

logger = logging.getLogger(__name__)

# Backend: lazy load when Forecasting page is opened (not at app startup)
def _get_forecasting_backend():
    """Import and return backend modules/classes only when needed."""
    try:
        from trading.data.data_loader import DataLoader, DataLoadRequest
        from trading.data.providers.yfinance_provider import YFinanceProvider
        from trading.models.lstm_model import LSTMForecaster
        from trading.models.xgboost_model import XGBoostModel
        from trading.models.prophet_model import ProphetModel
        from trading.models.arima_model import ARIMAModel
        from trading.data.preprocessing import FeatureEngineering, DataPreprocessor
        from trading.agents.model_selector_agent import ModelSelectorAgent
        from trading.market.market_analyzer import MarketAnalyzer
        return {
            "DataLoader": DataLoader,
            "DataLoadRequest": DataLoadRequest,
            "YFinanceProvider": YFinanceProvider,
            "LSTMForecaster": LSTMForecaster,
            "XGBoostModel": XGBoostModel,
            "ProphetModel": ProphetModel,
            "ARIMAModel": ARIMAModel,
            "FeatureEngineering": FeatureEngineering,
            "DataPreprocessor": DataPreprocessor,
            "ModelSelectorAgent": ModelSelectorAgent,
            "MarketAnalyzer": MarketAnalyzer,
        }
    except Exception as e:
        logger.warning(f"Forecasting backend not available: {e}")
        return None

# Global page-level error boundary (outermost catch)
import os as _os
import runpy as _runpy
_guard_key = "EVOLVE_PAGE_GUARD_FORECASTING"
# if _os.environ.get(_guard_key) != "1":
#     _os.environ[_guard_key] = "1"
#     try:
#         _runpy.run_path(__file__, run_name="__main__")
#     except Exception as _page_error:
#         import traceback
#         st.error(f"⚠️ Page error: {type(_page_error).__name__}: {_page_error}")
#         with st.expander("Developer details"):
#             st.code(traceback.format_exc(), language="python")
#         st.info("Try refreshing the page or selecting a different symbol.")
#         st.stop()
#     finally:
#         _os.environ.pop(_guard_key, None)
#     st.stop()

# Lazy init: resolve backend when Forecasting page is first rendered (not at app startup)
if "forecasting_backend" not in st.session_state:
    st.session_state.forecasting_backend = _get_forecasting_backend()
_be = st.session_state.get("forecasting_backend")
if _be:
    DataLoader = _be["DataLoader"]
    DataLoadRequest = _be["DataLoadRequest"]
    YFinanceProvider = _be["YFinanceProvider"]
    LSTMForecaster = _be["LSTMForecaster"]
    XGBoostModel = _be["XGBoostModel"]
    ProphetModel = _be["ProphetModel"]
    ARIMAModel = _be["ARIMAModel"]
    FeatureEngineering = _be["FeatureEngineering"]
    DataPreprocessor = _be["DataPreprocessor"]
    ModelSelectorAgent = _be["ModelSelectorAgent"]
    MarketAnalyzer = _be["MarketAnalyzer"]
else:
    DataLoader = DataLoadRequest = YFinanceProvider = LSTMForecaster = XGBoostModel = None
    ProphetModel = ARIMAModel = FeatureEngineering = DataPreprocessor = ModelSelectorAgent = MarketAnalyzer = None

if not _be:
    st.error("Forecasting backend could not be loaded. Check logs and dependencies.")
    st.stop()

# Initialize session state variables (dict-style access is safer when imported outside streamlit)
if "forecast_data" not in st.session_state:
    st.session_state["forecast_data"] = None
if "selected_models" not in st.session_state:
    st.session_state["selected_models"] = []
if "ai_recommendation" not in st.session_state:
    st.session_state["ai_recommendation"] = None
if "comparison_results" not in st.session_state:
    st.session_state["comparison_results"] = None
if "market_regime" not in st.session_state:
    st.session_state["market_regime"] = None
if "symbol" not in st.session_state:
    st.session_state["symbol"] = None
if "forecast_horizon" not in st.session_state:
    st.session_state["forecast_horizon"] = 7


def _extract_forecast_values(result):
    """Extract forecast array from result (dict/list/array). Handles all result formats."""
    if result is None:
        return None
    if isinstance(result, (list, np.ndarray)):
        arr = np.asarray(result)
        return arr if arr.size > 0 else None
    if isinstance(result, dict):
        for key in ["forecast", "predictions", "values", "forecast_values", "consensus_forecast"]:
            val = result.get(key)
            if val is not None:
                arr = np.asarray(val)
                if arr.size > 0:
                    return arr
    return None


# Main page title
st.title("📈 Analyze")
st.markdown("Ticker analysis with forecasting, AI model selection, and market analysis")

# Ticker and controls at top
ticker = st.text_input(
    "Ticker",
    value=st.session_state.get("analyze_ticker", "AAPL"),
    key="analyze_ticker",
)
if not ticker or not ticker.strip():
    ticker = "AAPL"
else:
    ticker = ticker.strip().upper()

trader_mode = st.radio(
    "Mode",
    ["Short-term", "Long-term"],
    horizontal=True,
    key="analyze_trader_mode",
)

period_map = {
    "1D": "1d", "5D": "5d", "1M": "1mo", "3M": "3mo",
    "6M": "6mo", "1Y": "1y", "5Y": "5y",
}
period_labels = list(period_map.keys())
default_idx = 5 if "1Y" in period_labels else 0
period_label = st.radio(
    "Period",
    period_labels,
    horizontal=True,
    key="analyze_period",
    index=min(default_idx, len(period_labels) - 1),
)
period = period_map.get(period_label, "1y")

# Auto-load data when ticker changes (so Market Analysis, Monte Carlo, etc. have forecast_data without manual Load Data)
_cache_key = f"autoloaded_{ticker}_{period}"
if st.session_state.get("_last_autoload_key") != _cache_key:
    try:
        _auto_hist = get_history(ticker, period=period)
        if not _auto_hist.empty:
            st.session_state["forecast_data"] = _auto_hist
            st.session_state["_last_autoload_key"] = _cache_key
            st.session_state["symbol"] = ticker
    except Exception:
        pass

_st_ver = tuple(int(x) for x in st.__version__.split(".")[:2])

# Chart using price_cache
if period in ("1d", "5d"):
    _tf_options = ["1m", "5m", "15m", "30m", "1h"]
    _tf_default = "5m" if period == "1d" else "30m"
    _tf_label = st.radio(
        "Intraday interval",
        _tf_options,
        index=_tf_options.index(_tf_default),
        horizontal=True,
        key="analyze_intraday_tf",
    )
    _interval = _tf_label
else:
    _interval = "1d"

hist = get_history(ticker, period=period, interval=_interval)
if not hist.empty:
    try:
        chart_type = st.radio(
            "Chart type",
            ["Candle", "Line", "Area", "News + Volume"],
            horizontal=True,
            key="analyze_chart_type",
            index=0,
        )
        if chart_type == "News + Volume":
            with st.expander("News overlay settings", expanded=True):
                _news_lang = st.selectbox(
                    "News language",
                    ["English only", "All languages"],
                    index=0,
                    key="news_lang_filter",
                )
                _news_vol_thresh = st.slider(
                    "Min price move % to show news",
                    min_value=0.0,
                    max_value=5.0,
                    value=0.5,
                    step=0.1,
                    key="news_price_threshold",
                    help=(
                        "Only show news near bars where price moved at least this much"
                    ),
                )
            try:
                _nc_interval = (
                    "5m" if period == "1d"
                    else "1h" if period == "5d"
                    else "1d"
                )
                render_news_candle_chart(
                    symbol=ticker,
                    period=period,
                    interval=_nc_interval,
                    volume_threshold=1.5,
                    price_threshold=0.02,
                    show_annotations=True,
                )
            except Exception as e:
                st.caption(f"News chart unavailable: {e}")
        else:
            if period in ("1d", "5d"):
                from plotly.subplots import make_subplots as _make_subplots

                fig_chart = _make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.75, 0.25],
                )
                _intraday_mode = True
            else:
                fig_chart = go.Figure()
                _intraday_mode = False

            if _intraday_mode:
                if chart_type == "Candle":
                    fig_chart.add_trace(
                        go.Candlestick(
                            x=hist.index,
                            open=hist["Open"],
                            high=hist["High"],
                            low=hist["Low"],
                            close=hist["Close"],
                            increasing_line_color="#26a69a",
                            increasing_fillcolor="#26a69a",
                            decreasing_line_color="#ef5350",
                            decreasing_fillcolor="#ef5350",
                            name=ticker,
                            showlegend=False,
                        ),
                        row=1,
                        col=1,
                    )
                else:
                    fig_chart.add_trace(
                        go.Scatter(
                            x=hist.index,
                            y=hist["Close"],
                            mode="lines",
                            line=dict(color="#00d4ff", width=1.5),
                            name=ticker,
                            showlegend=False,
                        ),
                        row=1,
                        col=1,
                    )

                # VWAP
                try:
                    _vwap = (
                        (hist["Close"] * hist["Volume"]).cumsum()
                        / hist["Volume"].cumsum()
                    )
                    fig_chart.add_trace(
                        go.Scatter(
                            x=hist.index,
                            y=_vwap,
                            mode="lines",
                            line=dict(
                                color="#ff9800",
                                width=1,
                                dash="dot",
                            ),
                            name="VWAP",
                        ),
                        row=1,
                        col=1,
                    )
                except Exception:
                    pass

                # Volume bars
                try:
                    _vcols = [
                        "#26a69a"
                        if hist["Close"].iloc[i] >= hist["Open"].iloc[i]
                        else "#ef5350"
                        for i in range(len(hist))
                    ]
                    fig_chart.add_trace(
                        go.Bar(
                            x=hist.index,
                            y=hist["Volume"],
                            marker_color=_vcols,
                            name="Volume",
                            showlegend=False,
                        ),
                        row=2,
                        col=1,
                    )
                except Exception:
                    pass

                # Previous close
                try:
                    import yfinance as _yf2

                    _prev = _yf2.Ticker(ticker).history(
                        period="2d", interval="1d"
                    )
                    if len(_prev) >= 2:
                        _pc = float(_prev["Close"].iloc[-2])
                        fig_chart.add_hline(
                            y=_pc,
                            line_dash="dot",
                            line_color="#4a6080",
                            line_width=1,
                            annotation_text="prev close",
                            annotation_font_color="#4a6080",
                            annotation_font_size=10,
                            row=1,
                            col=1,
                        )
                except Exception:
                    pass

                # Current price hline (intraday)
                try:
                    _current = float(hist["Close"].dropna().iloc[-1])
                    if _current and _current == _current:
                        fig_chart.add_hline(
                            y=_current,
                            line_dash="dash",
                            line_color="#ef5350",
                            line_width=1,
                            annotation_text=f" ${_current:.2f}",
                            annotation_position="right",
                            annotation_font_color="#ef5350",
                            annotation_font_size=11,
                            row=1,
                            col=1,
                        )
                except Exception:
                    pass

                fig_chart.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0a0e1a",
                    plot_bgcolor="#0f1525",
                    font=dict(
                        family="'Courier New',monospace",
                        color="#e0e6f0",
                        size=11,
                    ),
                    title=dict(
                        text=f"{ticker} — {period_label} ({_tf_label})",
                        font=dict(color="#e0e6f0", size=14),
                        x=0,
                    ),
                    xaxis=dict(
                        gridcolor="#1a2535",
                        showgrid=True,
                        zeroline=False,
                        tickfont=dict(color="#4a6080", size=10),
                        rangeslider=dict(visible=False),
                        rangebreaks=(
                            [
                                dict(bounds=["sat", "mon"]),
                                dict(bounds=[16, 9.5], pattern="hour"),
                            ]
                            if period == "1d"
                            else [dict(bounds=["sat", "mon"])]
                        ),
                    ),
                    xaxis2=dict(
                        gridcolor="#1a2535",
                        showgrid=True,
                        zeroline=False,
                        tickfont=dict(color="#4a6080", size=10),
                        rangebreaks=(
                            [
                                dict(bounds=["sat", "mon"]),
                                dict(bounds=[16, 9.5], pattern="hour"),
                            ]
                            if period == "1d"
                            else [dict(bounds=["sat", "mon"])]
                        ),
                    ),
                    yaxis=dict(
                        gridcolor="#1a2535",
                        showgrid=True,
                        zeroline=False,
                        tickfont=dict(color="#4a6080", size=10),
                        tickprefix="$",
                        side="right",
                    ),
                    yaxis2=dict(
                        gridcolor="#1a2535",
                        showgrid=False,
                        zeroline=False,
                        tickfont=dict(color="#4a6080", size=9),
                        side="right",
                    ),
                    hovermode="x unified",
                    hoverlabel=dict(
                        bgcolor="#0f1525",
                        bordercolor="#1e2d45",
                        font=dict(color="#e0e6f0", size=11),
                    ),
                    margin=dict(l=0, r=60, t=30, b=20),
                    height=450,
                    showlegend=True,
                    legend=dict(
                        bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#4a6080", size=10),
                        x=0,
                        y=1.0,
                    ),
                )
            else:
                if chart_type == "Candle":
                    fig_chart = go.Figure(
                        data=[
                            go.Candlestick(
                                x=hist.index,
                                open=hist["Open"],
                                high=hist["High"],
                                low=hist["Low"],
                                close=hist["Close"],
                                increasing_line_color="#26a69a",
                                increasing_fillcolor="#26a69a",
                                decreasing_line_color="#ef5350",
                                decreasing_fillcolor="#ef5350",
                                name=ticker,
                            )
                        ]
                    )
                elif chart_type == "Area":
                    fig_chart = go.Figure(
                        data=[
                            go.Scatter(
                                x=hist.index,
                                y=hist["Close"],
                                fill="tozeroy",
                                fillcolor="rgba(0,212,255,0.1)",
                                line=dict(color="#00d4ff", width=2),
                                name=ticker,
                            )
                        ]
                    )
                else:  # Line
                    fig_chart = go.Figure(
                        data=[
                            go.Scatter(
                                x=hist.index,
                                y=hist["Close"],
                                mode="lines",
                                line=dict(color="#00d4ff", width=2),
                                name=ticker,
                            )
                        ]
                    )
                try:
                    _current = float(hist["Close"].dropna().iloc[-1])
                    if _current and _current == _current:
                        fig_chart.add_hline(
                            y=_current,
                            line_dash="dash",
                            line_color="#ef5350",
                            line_width=1,
                            annotation_text=f" ${_current:.2f}",
                            annotation_position="right",
                            annotation_font_color="#ef5350",
                            annotation_font_size=11,
                        )
                except Exception:
                    pass
                fig_chart.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0a0e1a",
                    plot_bgcolor="#0f1525",
                    font=dict(
                        family="'Courier New', monospace",
                        color="#e0e6f0",
                        size=11,
                    ),
                    title=dict(
                        text=f"{ticker} — {period_label}",
                        font=dict(color="#e0e6f0", size=14),
                        x=0,
                    ),
                    xaxis=dict(
                        title="Date",
                        gridcolor="#1a2535",
                        showgrid=True,
                        zeroline=False,
                        tickfont=dict(color="#4a6080", size=10),
                        rangeslider=dict(visible=False),
                        rangebreaks=(
                            [
                                dict(bounds=["sat", "mon"]),
                                dict(bounds=[16, 9.5], pattern="hour"),
                            ]
                            if period == "1d"
                            else [dict(bounds=["sat", "mon"])]
                        ),
                    ),
                    yaxis=dict(
                        title="Price ($)",
                        gridcolor="#1a2535",
                        showgrid=True,
                        zeroline=False,
                        tickfont=dict(color="#4a6080", size=10),
                        tickprefix="$",
                        side="right",
                    ),
                    hovermode="x unified",
                    hoverlabel=dict(
                        bgcolor="#0f1525",
                        bordercolor="#1e2d45",
                        font=dict(color="#e0e6f0", size=11),
                    ),
                    margin=dict(l=0, r=60, t=30, b=20),
                    height=380,
                    showlegend=False,
                )

            if _st_ver >= (1, 37) and period in ("1d", "5d"):

                @st.fragment(run_every=60)
                def _live_chart():
                    _tkr = st.session_state.get("analyze_ticker", ticker)
                    _per = period
                    _tf = st.session_state.get(
                        "analyze_intraday_tf",
                        "5m" if _per == "1d" else "30m",
                    )
                    _ct = st.session_state.get("analyze_chart_type", "Candle")

                    _hist = get_history(_tkr, period=_per, interval=_tf)
                    if _hist.empty:
                        st.caption("No intraday data.")
                        return

                    import datetime as _dt

                    _now = _dt.datetime.now().strftime("%H:%M:%S")
                    st.caption(f"Last updated: {_now}")

                    from plotly.subplots import make_subplots as _make_subplots

                    _fig = _make_subplots(
                        rows=2,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.75, 0.25],
                    )

                    if _ct == "Candle":
                        _fig.add_trace(
                            go.Candlestick(
                                x=_hist.index,
                                open=_hist["Open"],
                                high=_hist["High"],
                                low=_hist["Low"],
                                close=_hist["Close"],
                                increasing_line_color="#26a69a",
                                increasing_fillcolor="#26a69a",
                                decreasing_line_color="#ef5350",
                                decreasing_fillcolor="#ef5350",
                                name=_tkr,
                                showlegend=False,
                            ),
                            row=1,
                            col=1,
                        )
                    else:
                        _fig.add_trace(
                            go.Scatter(
                                x=_hist.index,
                                y=_hist["Close"],
                                mode="lines",
                                line=dict(color="#00d4ff", width=1.5),
                                name=_tkr,
                                showlegend=False,
                            ),
                            row=1,
                            col=1,
                        )

                    try:
                        _vwap_live = (
                            (_hist["Close"] * _hist["Volume"]).cumsum()
                            / _hist["Volume"].cumsum()
                        )
                        _fig.add_trace(
                            go.Scatter(
                                x=_hist.index,
                                y=_vwap_live,
                                mode="lines",
                                line=dict(
                                    color="#ff9800",
                                    width=1,
                                    dash="dot",
                                ),
                                name="VWAP",
                            ),
                            row=1,
                            col=1,
                        )
                    except Exception:
                        pass

                    try:
                        _vcols_live = [
                            "#26a69a"
                            if _hist["Close"].iloc[i] >= _hist["Open"].iloc[i]
                            else "#ef5350"
                            for i in range(len(_hist))
                        ]
                        _fig.add_trace(
                            go.Bar(
                                x=_hist.index,
                                y=_hist["Volume"],
                                marker_color=_vcols_live,
                                name="Volume",
                                showlegend=False,
                            ),
                            row=2,
                            col=1,
                        )
                    except Exception:
                        pass

                    try:
                        import yfinance as _yf_live

                        _prev_live = _yf_live.Ticker(_tkr).history(
                            period="2d", interval="1d"
                        )
                        if len(_prev_live) >= 2:
                            _pc_live = float(_prev_live["Close"].iloc[-2])
                            _fig.add_hline(
                                y=_pc_live,
                                line_dash="dot",
                                line_color="#4a6080",
                                line_width=1,
                                annotation_text="prev close",
                                annotation_font_color="#4a6080",
                                annotation_font_size=10,
                                row=1,
                                col=1,
                            )
                    except Exception:
                        pass

                    try:
                        _current_live = float(_hist["Close"].dropna().iloc[-1])
                        if _current_live and _current_live == _current_live:
                            _fig.add_hline(
                                y=_current_live,
                                line_dash="dash",
                                line_color="#ef5350",
                                line_width=1,
                                annotation_text=f" ${_current_live:.2f}",
                                annotation_position="right",
                                annotation_font_color="#ef5350",
                                annotation_font_size=11,
                                row=1,
                                col=1,
                            )
                    except Exception:
                        pass

                    _fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#0a0e1a",
                        plot_bgcolor="#0f1525",
                        font=dict(
                            family="'Courier New',monospace",
                            color="#e0e6f0",
                            size=11,
                        ),
                        title=dict(
                            text=f"{_tkr} — {_per} ({_tf})",
                            font=dict(color="#e0e6f0", size=14),
                            x=0,
                        ),
                        xaxis=dict(
                            gridcolor="#1a2535",
                            showgrid=True,
                            zeroline=False,
                            tickfont=dict(color="#4a6080", size=10),
                            rangeslider=dict(visible=False),
                        ),
                        xaxis2=dict(
                            gridcolor="#1a2535",
                            showgrid=True,
                            zeroline=False,
                            tickfont=dict(color="#4a6080", size=10),
                        ),
                        yaxis=dict(
                            gridcolor="#1a2535",
                            showgrid=True,
                            zeroline=False,
                            tickfont=dict(color="#4a6080", size=10),
                            tickprefix="$",
                            side="right",
                        ),
                        yaxis2=dict(
                            gridcolor="#1a2535",
                            showgrid=False,
                            zeroline=False,
                            tickfont=dict(color="#4a6080", size=9),
                            side="right",
                        ),
                        hovermode="x unified",
                        hoverlabel=dict(
                            bgcolor="#0f1525",
                            bordercolor="#1e2d45",
                            font=dict(color="#e0e6f0", size=11),
                        ),
                        margin=dict(l=0, r=60, t=30, b=20),
                        height=380,
                        showlegend=False,
                    )

                    st.plotly_chart(
                        _fig,
                        width='stretch',
                        key="analyze_main_chart_live",
                    )

                _live_chart()
            else:
                # News vlines on main chart (only for News+Vol chart type)
                if chart_type == "News + Volume":
                    try:
                        from trading.data.price_cache import get_news
                        _news_items = get_news(ticker)
                        _lang_filter = st.session_state.get(
                            "news_lang_filter", "English only"
                        )
                        _price_thresh = st.session_state.get(
                            "news_price_threshold", 0.5
                        )
                        _plotted = 0
                        if _news_items and not hist.empty:
                            from datetime import datetime

                            _prices = hist["Close"].dropna()

                            for _item in _news_items[:15]:
                                _content = _item.get("content") or {}
                                _title = (
                                    _item.get("title")
                                    or _content.get("title")
                                    or _content.get("summary")
                                    or ""
                                )
                                if not _title:
                                    continue

                                if (
                                    _lang_filter == "English only"
                                    and not _is_english(_title)
                                ):
                                    continue

                                _pub = (
                                    _item.get("providerPublishTime")
                                    or _content.get("pubDate")
                                    or _item.get("published")
                                )
                                if not _pub:
                                    continue
                                try:
                                    if isinstance(_pub, str):
                                        _dt = datetime.fromisoformat(
                                            _pub.replace("Z", "+00:00")
                                        )
                                        _pub_dt = _dt.replace(tzinfo=None)
                                        _pub_ts = _dt.timestamp()
                                    else:
                                        _pub_ts = float(_pub)
                                        _pub_dt = datetime.fromtimestamp(
                                            _pub_ts
                                        )
                                except Exception:
                                    continue

                                if len(hist.index) > 0:
                                    _idx_min = hist.index.min()
                                    _idx_max = hist.index.max()
                                    try:
                                        _idx_min = _idx_min.replace(tzinfo=None)
                                        _idx_max = _idx_max.replace(tzinfo=None)
                                    except Exception:
                                        pass
                                    if _pub_dt < _idx_min or _pub_dt > _idx_max:
                                        continue

                                if _price_thresh > 0:
                                    try:
                                        if hasattr(hist.index, "tz"):
                                            _index_dt = (
                                                hist.index.tz_localize(
                                                    None
                                                ).to_pydatetime()
                                            )
                                        else:
                                            _index_dt = (
                                                hist.index.to_pydatetime()
                                            )
                                        _diffs = abs(_index_dt - _pub_dt)
                                        _closest_idx = int(_diffs.argmin())
                                        if _closest_idx > 0:
                                            _p1 = float(
                                                _prices.iloc[_closest_idx]
                                            )
                                            _p0 = float(
                                                _prices.iloc[_closest_idx - 1]
                                            )
                                            if _p0 != 0:
                                                _move = abs(
                                                    (_p1 - _p0) / _p0 * 100
                                                )
                                                if _move < _price_thresh:
                                                    continue
                                    except Exception:
                                        pass

                                _pos_kw = [
                                    "beat",
                                    "surge",
                                    "raises",
                                    "upgrade",
                                    "strong",
                                    "growth",
                                ]
                                _neg_kw = [
                                    "miss",
                                    "falls",
                                    "cuts",
                                    "downgrade",
                                    "weak",
                                    "loss",
                                ]
                                _tl = _title.lower()
                                _pos = sum(
                                    1 for k in _pos_kw if k in _tl
                                )
                                _neg = sum(
                                    1 for k in _neg_kw if k in _tl
                                )
                                _ann_color = (
                                    "#26a69a"
                                    if _pos > _neg
                                    else "#ef5350"
                                    if _neg > _pos
                                    else "#ff9800"
                                )

                                _vline_kwargs = dict(
                                    x=_pub_dt,
                                    line_dash="dot",
                                    line_color=_ann_color,
                                    line_width=1,
                                    annotation_text="N",
                                    annotation_position="top",
                                    annotation_font_color=_ann_color,
                                    annotation_font_size=10,
                                )
                                if _intraday_mode:
                                    fig_chart.add_vline(
                                        row=1, col=1, **_vline_kwargs
                                    )
                                else:
                                    fig_chart.add_vline(**_vline_kwargs)
                                _plotted += 1

                        if _plotted == 0 and period in ("1d", "5d"):
                            st.caption(
                                "No recent English news found within chart timeframe."
                            )
                    except Exception:
                        pass

                st.plotly_chart(
                    fig_chart,
                    width='stretch',
                    key="analyze_main_chart",
                )
        if trader_mode == "Short-term":
            try:
                st.markdown("**Short-term Signals**")
                _hist_1d = get_history(ticker, period="1d", interval="5m")
                if not _hist_1d.empty:
                    _close = _hist_1d["Close"]
                    _vwap = (_hist_1d["Close"] * _hist_1d["Volume"]).cumsum() / _hist_1d["Volume"].cumsum()
                    _last = float(_close.iloc[-1])
                    _vwap_last = float(_vwap.iloc[-1])
                    _vwap_dev = ((_last - _vwap_last) / _vwap_last * 100)
                    _vol_avg = float(_hist_1d["Volume"].mean())
                    _vol_last = float(_hist_1d["Volume"].iloc[-1])
                    _vol_ratio = _vol_last / _vol_avg if _vol_avg > 0 else 1.0
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        _sign = "+" if _vwap_dev >= 0 else ""
                        st.metric(
                            "vs VWAP",
                            f"{_sign}{_vwap_dev:.2f}%",
                            help="Price deviation from Volume Weighted Avg Price",
                        )
                    with c2:
                        st.metric(
                            "Volume Ratio",
                            f"{_vol_ratio:.1f}x",
                            help="Current bar volume vs average intraday volume",
                        )
                    with c3:
                        _momentum = float(_close.pct_change(5).iloc[-1] * 100) if len(_close) > 5 else 0.0
                        _sign2 = "+" if _momentum >= 0 else ""
                        st.metric(
                            "5-bar Momentum",
                            f"{_sign2}{_momentum:.2f}%",
                            help="Price change over last 5 intraday bars",
                        )
            except Exception:
                pass
        elif trader_mode == "Long-term":
            try:
                st.markdown("**Value Signals**")
                _info = get_info(ticker)
                if _info:
                    _pe = _info.get("trailingPE") or _info.get("forwardPE")
                    _pb = _info.get("priceToBook")
                    _eps = _info.get("trailingEps") or _info.get("forwardEps")
                    _div = _info.get("dividendYield", 0) or 0
                    _52w_low = _info.get("fiftyTwoWeekLow")
                    _52w_high = _info.get("fiftyTwoWeekHigh")
                    _price = get_quote(ticker).get("price", 0)
                    _range_pct = None
                    if _52w_low and _52w_high and _price:
                        _range_pct = (_price - _52w_low) / (_52w_high - _52w_low) * 100
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric(
                            "P/E Ratio",
                            f"{_pe:.1f}x" if _pe else "N/A",
                            help="Lower = cheaper relative to earnings",
                        )
                    with c2:
                        st.metric(
                            "P/B Ratio",
                            f"{_pb:.2f}x" if _pb else "N/A",
                            help="Below 1.0 may indicate undervaluation",
                        )
                    with c3:
                        st.metric(
                            "Div Yield",
                            f"{_div * 100:.2f}%" if _div else "N/A",
                        )
                    with c4:
                        if _range_pct is not None:
                            st.metric(
                                "52w Position",
                                f"{_range_pct:.0f}%",
                                help="Where price sits in 52-week range. Low % = near lows",
                            )
                        else:
                            st.metric("52w Position", "N/A")
                    _signals = []
                    if _pe and _pe < 15:
                        _signals.append("P/E below 15")
                    if _pb and _pb < 1.5:
                        _signals.append("P/B below 1.5")
                    if _range_pct is not None and _range_pct < 25:
                        _signals.append("Near 52-week low")
                    if _signals:
                        st.success("Value signals: " + ", ".join(_signals))
            except Exception:
                pass
    except Exception as e:
        st.caption(f"Feature unavailable: {e}")

def _news_sentiment_score(ticker: str) -> float:
    """Return a 0–10 news score with recency decay for AI sentiment."""
    try:
        from trading.data.price_cache import get_news as _get_news
        import time

        items = _get_news(ticker)
        if not items:
            return 5.0
        pos_kw = [
            "beat",
            "surge",
            "raises",
            "upgrade",
            "strong",
            "growth",
            "record",
            "above",
            "buy",
            "bullish",
            "jumps",
            "soars",
        ]
        neg_kw = [
            "miss",
            "falls",
            "cuts",
            "downgrade",
            "weak",
            "below",
            "layoffs",
            "loss",
            "investigation",
            "sell",
            "bearish",
            "drops",
            "plunges",
        ]
        total_score = 0.0
        total_weight = 0.0
        now = time.time()
        for item in items[:10]:
            content = item.get("content") or {}
            raw_title = (
                item.get("title")
                or item.get("headline")
                or content.get("title")
                or content.get("summary")
                or ""
            )
            title = raw_title.lower()
            if not title:
                continue
            pub = (
                item.get("providerPublishTime")
                or item.get("published")
                or content.get("pubDate")
                or now
            )
            try:
                from datetime import datetime

                if isinstance(pub, str):
                    dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                    pub_ts = dt.timestamp()
                elif hasattr(pub, "timestamp"):
                    pub_ts = float(pub.timestamp())
                else:
                    pub_ts = float(pub)
            except Exception:
                pub_ts = now
            age_hours = max(0.0, (now - pub_ts) / 3600.0)
            if age_hours < 1:
                decay = 1.0
            elif age_hours < 6:
                decay = 0.7
            elif age_hours < 24:
                decay = 0.4
            else:
                decay = 0.1
            raw = sum(1 for k in pos_kw if k in title) - sum(
                1 for k in neg_kw if k in title
            )
            total_score += raw * decay
            total_weight += decay
        if total_weight == 0:
            return 5.0
        normalized = total_score / total_weight
        return round(min(10.0, max(0.0, 5.0 + normalized * 2.0)), 1)
    except Exception:
        return 5.0


def _is_english(text: str) -> bool:
    if not text:
        return False
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return ascii_count / len(text) > 0.8


def _generate_recommendation(
    ticker,
    ai_score_result,
    forecast_result=None,
    trader_mode="Short-term",
):
    try:
        # Entry is always current price from latest quote
        _quote = get_quote(ticker) if ticker else {}
        last_price = _quote.get("price") or _quote.get("regularMarketPrice") or 0.0
        try:
            last_price = float(last_price)
        except Exception:
            last_price = 0.0
        if last_price <= 0:
            # Fallback: no valid price, return neutral hold
            return {
                "action": "HOLD",
                "conviction": "LOW",
                "signal_score": 5.0,
                "fc_target": None,
                "fc_direction": "flat",
                "reasons": [("~", "Price unavailable — no trade recommendation")],
            }
        _entry = float(last_price)

        # Determine action from AI Score (not from consensus direction)
        _score = ai_score_result.get(
            "weighted_score", ai_score_result.get("overall_score", 5.0)
        )
        try:
            _score = float(_score)
        except Exception:
            _score = 5.0

        if _score >= 7.0:
            _action = "BUY"
            _conv = "HIGH"
        elif _score >= 6.0:
            _action = "BUY"
            _conv = "MEDIUM"
        elif _score <= 3.0:
            _action = "SELL"
            _conv = "HIGH"
        elif _score <= 4.0:
            _action = "SELL"
            _conv = "MEDIUM"
        else:
            _action = "HOLD"
            _conv = "LOW"

        # Get consensus forecast for context (direction/conviction may be absent)
        _consensus = st.session_state.get("current_forecast_result", {}) or (
            forecast_result or {}
        )
        _fc_arr = _consensus.get("forecast", [])
        _consensus_dir = _consensus.get("direction", "NEUTRAL")
        _consensus_conv = (
            _consensus.get("conviction")
            or _consensus.get("confidence_label", "LOW")
        )

        # Override action to HOLD if AI Score and consensus strongly conflict
        if (
            _action == "BUY"
            and _consensus_dir == "BEARISH"
            and str(_consensus_conv).upper() == "HIGH"
            and _score < 6.5
        ):
            _action = "HOLD"
            _conv = "LOW"
        elif (
            _action == "SELL"
            and _consensus_dir == "BULLISH"
            and str(_consensus_conv).upper() == "HIGH"
            and _score > 3.5
        ):
            _action = "HOLD"
            _conv = "LOW"

        # Set target and stop based on action direction
        # _vol from ai_score is annualized decimal (e.g. 0.2528 = 25.28%)
        # Convert to daily: ann_vol / sqrt(252)
        try:
            _ann_vol = max(float(ai_score_result.get("volatility", 0.15)), 0.05)
        except Exception:
            _ann_vol = 0.15
        _daily_vol = _ann_vol / np.sqrt(252)
        # 7-day expected move (1 std dev)
        _horizon_vol = _daily_vol * np.sqrt(7)
        # Target = 0.5 std dev move, cap 3%
        _target_pct = min(_horizon_vol * 0.5, 0.03)
        _target_pct = max(_target_pct, 0.005)
        # Stop = 0.75 std dev move, cap 2%
        _stop_pct = min(_horizon_vol * 0.75, 0.02)
        _stop_pct = max(_stop_pct, 0.005)

        if _action == "BUY":
            _target = _entry * (1 + _target_pct)
            _stop = _entry * (1 - _stop_pct)
        elif _action == "SELL":
            _target = _entry * (1 - _target_pct)
            _stop = _entry * (1 + _stop_pct)
        else:  # HOLD
            # Show a narrow range instead of directional target
            _target = _entry * 1.01
            _stop = _entry * 0.98
            _target_pct = 0.01
            _stop_pct = 0.02

        # R/R ratio and expected move
        _upside = abs(_target - _entry)
        _downside = abs(_entry - _stop)
        _rr = (_upside / _downside) if _downside > 0 else 1.0
        _pct_move = ((_target - _entry) / _entry * 100.0) if _entry else 0.0

        # Build reasoning bullets aligned with action/direction
        _reasons = []
        # AI Score strength
        if _score >= 6.0:
            _reasons.append(("+", f"AI Score bullish ({_score:.1f}/10)"))
        elif _score <= 4.0:
            _reasons.append(("-", f"AI Score bearish ({_score:.1f}/10)"))

        # Consensus direction
        if _consensus_dir == "BULLISH":
            _reasons.append(("+", "Forecast models project upside"))
        elif _consensus_dir == "BEARISH":
            _reasons.append(("-", "Forecast models project decline"))

        # Technical score
        tech = ai_score_result.get("technical_score", 5.0)
        try:
            tech = float(tech)
        except Exception:
            tech = 5.0
        if tech >= 7.0:
            _reasons.append(("+", "Technical trend is strong"))
        elif tech <= 4.0:
            _reasons.append(("-", "Technical trend is weak"))

        # Signal conflict warning
        if _action in ("BUY", "SELL"):
            if _action == "BUY" and _consensus_dir == "BEARISH":
                _reasons.append(
                    ("⚠", "AI Score bullish but models bearish — use caution")
                )
            elif _action == "SELL" and _consensus_dir == "BULLISH":
                _reasons.append(
                    ("⚠", "AI Score bearish but models bullish — use caution")
                )

        if not _reasons:
            _reasons.append(("~", "Signals are mixed — monitor closely"))

        # Limit to 3 reasons
        _reasons = _reasons[:3]

        # fc_target/fc_direction for backward compatibility (used by widget)
        if _action == "BUY":
            fc_direction = "up"
        elif _action == "SELL":
            fc_direction = "down"
        else:
            fc_direction = "flat"
        fc_target = _target

        return {
            "action": _action,
            "conviction": _conv,
            "signal_score": round(_score, 1),
            "fc_target": fc_target,
            "fc_direction": fc_direction,
            "reasons": _reasons,
            "entry": _entry,
            "target": _target,
            "stop": _stop,
            "target_pct": _target_pct,
            "stop_pct": _stop_pct,
            "risk_reward": _rr,
            "pct_move": _pct_move,
        }
    except Exception:
        return None


# News sentiment panel (HOT/POS/NEG/NEU labels)
try:
    news_items = get_news(ticker)
    if news_items:
        with st.expander("📰 News sentiment", expanded=False):
            pos_kw = ['beat','surge','raises','upgrade','strong',
                      'growth','record','above','buy','bullish']
            neg_kw = ['miss','falls','cuts','downgrade','weak',
                      'below','layoffs','investigation','sell',
                      'bearish','loss']
            for i, item in enumerate(news_items[:8]):
                content = item.get("content") or {}
                raw_title = (
                    item.get("title")
                    or item.get("headline")
                    or content.get("title")
                    or content.get("summary")
                    or ""
                )
                if not raw_title:
                    continue
                title_lower = raw_title.lower()
                score = 0
                score += sum(1 for k in pos_kw if k in title_lower)
                score -= sum(1 for k in neg_kw if k in title_lower)
                if score >= 3:
                    ns_label, ns_color = "HOT", "#ff9800"
                elif score >= 1:
                    ns_label, ns_color = "POS", "#26a69a"
                elif score <= -1:
                    ns_label, ns_color = "NEG", "#ef5350"
                else:
                    ns_label, ns_color = "NEU", "#4a6080"
                st.markdown(
                    f'<span style="color:{ns_color};font-size:10px;'
                    f'padding:1px 6px;border-radius:2px;'
                    f'background:{ns_color}22;font-family:monospace">'
                    f'{ns_label}</span> '
                    f'{raw_title[:80]}',
                    unsafe_allow_html=True
                )
except Exception as e:
    st.caption(f"Feature unavailable: {e}")

st.markdown("---")

# Create tabbed interface
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab_options, tab_insider, tab_earnings, tab_diag = st.tabs(
    [
        "🚀 Quick Forecast",
        "⚙️ Advanced Forecasting",
        "🤖 AI Model Selection",
        "📊 Model Comparison",
        "📈 Market Analysis",
        "🔗 Multi-Asset (GNN)",
        "🎲 Monte Carlo",
        "📊 Options & Short",
        "🕵️ Insider Flow",
        "📅 Earnings",
        "📐 Diagnostics",
    ]
)

# Quick Forecast: fast models only (ARIMA, XGBoost, Ridge). No walk-forward, no auto-select.
QUICK_FORECAST_MODELS = ["ARIMA", "XGBoost", "Ridge"]

with tab1:
    st.header("Quick Forecast")
    st.markdown("Generate fast forecasts with pre-configured models (ARIMA, XGBoost, Ridge)")
    
    # Data Loading Form
    with st.form("data_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbol = st.text_input(
                "Ticker Symbol",
                value="AAPL",
                help="Enter stock ticker (e.g., AAPL, MSFT, GOOGL)"
            ).upper()
        
        with col2:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now().date() - timedelta(days=365),
                min_value=datetime.now().date() - timedelta(days=365*10),
                max_value=datetime.now().date(),
                help="Historical data start. Cannot select a future date."
            )
        
        with col3:
            end_date = st.date_input(
                "End Date",
                value=datetime.now().date(),
                min_value=start_date,
                max_value=datetime.now().date(),
                help="End date for historical data. For forecasting future prices, use the 'Forecast Horizon (days)' slider — do not set end date in the future."
            )
        
        submitted = st.form_submit_button("📊 Load Data")
    
    if end_date > datetime.now().date():
        st.warning("End date is in the future. Historical data will only load up to today. Use the forecast horizon slider to project forward.")
    if submitted:
        # Validation
        if not symbol:
            st.error("Please enter a ticker symbol")
        elif start_date >= end_date:
            st.error("Start date must be before end date")
        elif (end_date - start_date).days < 30:
            st.error("Please select at least 30 days of data")
        else:
            try:
                with st.spinner(f"Fetching data for {symbol}..."):
                    # Initialize data loader
                    loader = DataLoader()
                    
                    # Create data load request
                    request = DataLoadRequest(
                        ticker=symbol,
                        start_date=start_date.strftime("%Y-%m-%d"),
                        end_date=end_date.strftime("%Y-%m-%d"),
                        interval="1d"
                    )
                    
                    # Load data
                    response = loader.load_market_data(request)
                    
                    if not response.success:
                        st.error(f"Error loading data: {response.message}")
                    elif response.data is None or len(response.data) < 30:
                        st.error(f"Insufficient data for {symbol}. Try different dates or symbol.")
                    else:
                        # Convert to standard format (lowercase column names)
                        data = response.data.copy()
                        data.columns = [col.lower() for col in data.columns]
                        
                        # Ensure we have 'close' column
                        if 'close' not in data.columns:
                            # Try to find close price column
                            close_col = None
                            for col in ['Close', 'close', 'CLOSE', 'price', 'Price']:
                                if col in data.columns:
                                    close_col = col
                                    break
                            if close_col:
                                data['close'] = data[close_col]
                            else:
                                st.error("Could not find close price column in data")
                                data = None
                        
                        if data is not None:
                            # Store in session state
                            st.session_state["forecast_data"] = data
                            st.session_state["symbol"] = symbol
                            # forecast_horizon is already in session_state (defaults to 7 if not set)
                            # No need to reassign it here unless we want to update it
                            
                            st.success(f"✅ Loaded {len(data)} days of data for {symbol}")
                            
                            # Show data quality metrics
                            try:
                                from src.utils.data_validation import DataValidator
                            except ImportError:
                                DataValidator = None
                            if DataValidator is not None:
                                try:
                                    validator = DataValidator()
                                    quality_metrics = validator.get_quality_metrics(data)
                                except Exception as qe:
                                    logger.debug("Data quality metrics failed: %s", qe)
                                    quality_metrics = None
                                if quality_metrics is not None:
                                    with st.expander("📊 Data Quality Metrics", expanded=False):
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("Completeness", f"{quality_metrics['completeness']:.1%}")
                                        with col2:
                                            st.metric("Missing Values", quality_metrics['missing_count'])
                                        with col3:
                                            st.metric("Outliers Detected", quality_metrics['outliers'])
                                        with col4:
                                            quality_score = quality_metrics['overall_quality']
                                            st.metric("Quality Score", f"{quality_score:.0f}/100")
                                        if quality_metrics['issues']:
                                            st.warning("⚠️ Data Quality Issues:")
                                            for issue in quality_metrics['issues']:
                                                st.write(f"• {issue}")
                        
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                st.info("Please check the ticker symbol and try again.")

    # Earnings proximity warning (forecasts may be less reliable near earnings)
    try:
        if st.session_state.get("forecast_data") is not None and symbol:
            _e = get_upcoming_earnings(symbol)
            if _e.get("is_within_window"):
                _d, _dt = _e["days_until"], _e["next_earnings_date"]
                _s = (
                    f" | Last surprise: {_e['last_eps_surprise_pct']:+.1f}%"
                    if _e.get("last_eps_surprise_pct") is not None
                    else "  "
                )
                st.warning(
                    f"Earnings in {_d} day{'s' if _d != 1 else ''} ({_dt}){_s} — "
                    "forecasts may be less reliable near earnings."
                )
    except Exception:
        pass

    # Display loaded data
    if st.session_state.get("forecast_data") is not None:
        data = st.session_state.get("forecast_data")
        # Normalize column case (yfinance uses Close, etc.)
        if "close" in data.columns and "Close" not in data.columns:
            data = data.rename(columns={"close": "Close"})
        if "Close" not in data.columns:
            data = data.copy()
            data["Close"] = data.iloc[:, 0]

        # Show data quality metrics (optional: src.utils.data_validation)
        try:
            from src.utils.data_validation import DataValidator
            validator = DataValidator()
            quality_metrics = validator.get_quality_metrics(data)
            with st.expander("📊 Data Quality Metrics", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Completeness", f"{quality_metrics['completeness']:.1%}")
                with col2:
                    st.metric("Missing Values", quality_metrics['missing_count'])
                with col3:
                    st.metric("Outliers Detected", quality_metrics['outliers'])
                with col4:
                    quality_score = quality_metrics['overall_quality']
                    st.metric("Quality Score", f"{quality_score:.0f}/100")
                if quality_metrics['issues']:
                    st.warning("⚠️ Data Quality Issues:")
                    for issue in quality_metrics['issues']:
                        st.write(f"• {issue}")
        except ImportError:
            pass
        except Exception as e:
            logger.debug("Data quality metrics unavailable: %s", e)
        
        st.markdown("---")
        st.subheader("📊 Data Preview")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Data Points", len(data))
        with col2:
            st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
        with col3:
            change = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
            st.metric("Period Return", f"{change:.2f}%")
        with col4:
            volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
            st.metric("Annualized Volatility", f"{volatility:.2f}%")
        
        # AI Score panel with trader-mode display weights and news score
        try:
            from trading.analysis.ai_score import compute_ai_score
            _sym = st.session_state.get("symbol") or symbol
            if _sym:
                _hist = data.copy()
                if "close" in _hist.columns and "Close" not in _hist.columns:
                    _hist = _hist.rename(columns={"close": "Close"})
                if "volume" in _hist.columns and "Volume" not in _hist.columns:
                    _hist = _hist.rename(columns={"volume": "Volume"})

                # Cache AI score per symbol for 5 minutes to avoid recompute on tab switch
                import time as _time

                _ai_score_key = f"ai_score_{_sym}"
                _ai_score_ts_key = f"ai_score_ts_{_sym}"
                _cached_score = st.session_state.get(_ai_score_key)
                _cached_ts = st.session_state.get(_ai_score_ts_key, 0.0)
                _score_age = _time.time() - _cached_ts

                if _cached_score is None or _score_age > 300:
                    with st.spinner("Computing AI Score..."):
                        score_result = compute_ai_score(_sym, _hist)
                    st.session_state[_ai_score_key] = score_result
                    # Also store under canonical key for cross-component access
                    st.session_state["ai_score_result"] = score_result
                    st.session_state[_ai_score_ts_key] = _time.time()
                else:
                    score_result = _cached_score
                if score_result.get("error") is None:
                    score = score_result["overall_score"]
                    grade = score_result["grade"]

                    # Trader-mode-specific display weights
                    if trader_mode == "Short-term":
                        display_weights = {
                            "technical": 0.45,
                            "momentum": 0.40,
                            "sentiment": 0.10,
                            "fundamental": 0.05,
                        }
                        mode_label = "Short-term weights"
                    else:
                        display_weights = {
                            "technical": 0.20,
                            "momentum": 0.20,
                            "sentiment": 0.15,
                            "fundamental": 0.45,
                        }
                        mode_label = "Long-term weights"

                    component_scores = {
                        "technical": score_result.get("technical_score", 0),
                        "momentum": score_result.get("momentum_score", 0),
                        "sentiment": score_result.get("sentiment_score", 0),
                        "fundamental": score_result.get("fundamental_score", 0),
                    }
                    weighted_score = sum(
                        component_scores[k] * display_weights[k] for k in display_weights
                    )
                    weighted_score = round(min(10.0, max(0.0, weighted_score)), 1)

                    # Show mode impact clearly (diff vs base score)
                    try:
                        base_score = float(score_result.get("overall_score", 0) or 0)
                        diff = round(weighted_score - base_score, 1)
                        diff_str = (
                            f"+{diff}" if diff > 0
                            else str(diff) if diff < 0
                            else "="
                        )
                        diff_color = (
                            "#26a69a" if diff > 0
                            else "#ef5350" if diff < 0
                            else "#4a6080"
                        )
                        st.markdown(
                            f'<span style="font-size:11px;color:{diff_color}">'
                            f"{trader_mode} view: {diff_str} vs base score"
                            f"</span>",
                            unsafe_allow_html=True,
                        )
                    except Exception:
                        pass

                    # News sentiment score feeding into sentiment view
                    news_score = _news_sentiment_score(_sym)
                    if news_score >= 7.5:
                        ns_label, ns_color = "HOT", "#ff9800"
                    elif news_score >= 6.0:
                        ns_label, ns_color = "POS", "#26a69a"
                    elif news_score <= 3.0:
                        ns_label, ns_color = "NEG", "#ef5350"
                    else:
                        ns_label, ns_color = "NEU", "#4a6080"

                    st.markdown("### 🤖 AI Score")
                    top_cols = st.columns([2, 2, 2, 2])
                    with top_cols[0]:
                        st.metric(
                            "Model Score",
                            f"{score}/10",
                            delta=grade,
                            delta_color="normal" if score >= 5 else "inverse",
                        )
                    with top_cols[1]:
                        st.metric("Weighted Score", f"{weighted_score}/10", help=mode_label)
                    with top_cols[2]:
                        st.markdown("**News Score**")
                        st.markdown(
                            f'<span style="font-family:monospace;font-size:14px;'
                            f'padding:3px 8px;border-radius:3px;'
                            f'background:{ns_color}22;color:{ns_color}">'
                            f'{news_score:.1f} · {ns_label}</span>',
                            unsafe_allow_html=True,
                        )
                    with top_cols[3]:
                        st.caption(mode_label)

                    # Component bars with weights
                    bar_rows = [
                        ("Technical", "technical"),
                        ("Momentum", "momentum"),
                        ("Sentiment", "sentiment"),
                        ("Fundamental", "fundamental"),
                    ]
                    for label, key_name in bar_rows:
                        val = float(component_scores.get(key_name, 0) or 0)
                        w = display_weights.get(key_name, 0)
                        pct = int(round(w * 100))
                        cols_row = st.columns([2, 5, 1])
                        with cols_row[0]:
                            st.markdown(f"**{label}**")
                        with cols_row[1]:
                            st.progress(min(1.0, max(0.0, val / 10.0)))
                        with cols_row[2]:
                            st.markdown(f"{val:.1f}  ({pct}%)")

                    st.caption(score_result["summary"])
                    with st.expander("What drives this score?", expanded=False):
                        signals = score_result.get("signals", [])
                        if signals:
                            by_impact = sorted(
                                signals,
                                key=lambda s: (
                                    s.get("impact") == "positive",
                                    s.get("impact") == "negative",
                                ),
                                reverse=True,
                            )
                            for sig in by_impact[:3]:
                                st.caption(
                                    f"• {sig.get('name', '')}: {sig.get('value', '')} — {sig.get('description', '')}"
                                )
                        else:
                            st.caption("No signal breakdown available.")
                    with st.expander("📊 Signal Breakdown", expanded=False):
                        signals = score_result.get("signals", [])
                        if signals:
                            sig_df = pd.DataFrame(
                                signals
                            )[["name", "value", "impact", "description"]]
                            sig_df.columns = [
                                "Signal",
                                "Value",
                                "Impact",
                                "Description",
                            ]

                            def _color_impact(val):
                                colors = {
                                    "positive": "background-color: #1a4a2a; color: #26a69a",
                                    "negative": "background-color: #3a1a1a; color: #ef5350",
                                    "neutral": "background-color: #3a2a0a; color: #ff9800",
                                }
                                return colors.get(val, "")

                            try:
                                styler = sig_df.style.applymap(
                                    _color_impact, subset=["Impact"]
                                )
                                st.dataframe(styler, width='stretch')
                            except Exception:
                                st.dataframe(sig_df, width='stretch')
                    # Recommendation panel
                    try:
                        _rec = _generate_recommendation(
                            ticker,
                            score_result,
                            st.session_state.get("current_forecast_result"),
                            trader_mode,
                        )
                        if _rec:
                            _action = _rec["action"]
                            _conv = _rec["conviction"]
                            _score = _rec["signal_score"]
                            _color = (
                                "#26a69a"
                                if "BUY" in _action
                                else "#ef5350"
                                if "SELL" in _action
                                else "#ff9800"
                            )
                            _reasons_html = ""
                            for _s, _t2 in _rec["reasons"]:
                                # _s is one of "+", "-", "⚠", or "~"
                                _icon = _s if _s in ("+", "-", "⚠") else "~"
                                if _icon == "+":
                                    _rc = "#26a60a"
                                elif _icon == "-":
                                    _rc = "#ef5350"
                                elif _icon == "⚠":
                                    _rc = "#ffb74d"
                                else:
                                    _rc = "#8899aa"
                                _reasons_html += (
                                    f'<div style="display:flex;gap:10px;'
                                    f'margin-bottom:6px">'
                                    f'<span style="color:{_rc};'
                                    f'font-family:monospace;font-weight:bold;'
                                    f'min-width:14px">{_icon}</span>'
                                    f'<span style="color:#c8d4e0;'
                                    f'font-size:13px">{_t2}</span></div>'
                                )
                            _mh = ""
                            _info_row = ""
                            if _rec.get("fc_target") is not None:
                                _entry = float(_rec.get("entry", 0.0))
                                _target = float(_rec.get("target", _rec["fc_target"]))
                                _stop = float(_rec.get("stop", _entry))
                                if _entry and _entry > 0:
                                    _pct2 = ((_target / _entry) - 1.0) * 100.0
                                    _stop2 = _stop
                                    _rr2 = float(_rec.get("risk_reward", 1.0))
                                    _tc2 = "#26a69a" if _pct2 > 0 else "#ef5350"

                                    # Trade horizon estimate (7 business days by default)
                                    try:
                                        from datetime import datetime as _dtime
                                        from datetime import timedelta as _tdelta

                                        _today = _dtime.now()
                                        _bdays = 0
                                        _end_dt = _today
                                        _fc_horizon = 7
                                        while _bdays < _fc_horizon:
                                            _end_dt += _tdelta(days=1)
                                            if _end_dt.weekday() < 5:
                                                _bdays += 1
                                        _horizon_str = _end_dt.strftime("%b %d")
                                    except Exception:
                                        _horizon_str = "~7 days"
                                        _fc_horizon = 7

                                    _conv_explain = {
                                        "HIGH": "Multiple models agree",
                                        "MEDIUM": "Models show mixed signals",
                                        "LOW": "Weak or conflicting signals",
                                    }.get(str(_conv).upper(), "Signal strength unknown")

                                    _mh = (
                                        f'<div style="display:grid;'
                                        f'grid-template-columns:repeat(3,1fr);'
                                        f'gap:1px;background:#1e2d45;'
                                        f'border-radius:4px;overflow:hidden;'
                                        f'margin-top:10px">'
                                        f'<div style="background:#0f1525;'
                                        f'padding:10px 14px">'
                                        f'<div style="font-size:10px;'
                                        f'color:#4a6080;'
                                        f'letter-spacing:1px;'
                                        f'margin-bottom:3px">'
                                        f'ENTRY</div>'
                                        f'<div style="font-size:15px;'
                                        f'font-weight:bold;'
                                        f'color:#e0e6f0">'
                                        f'${_entry:.2f}</div></div>'
                                        f'<div style="background:#0f1525;'
                                        f'padding:10px 14px">'
                                        f'<div style="font-size:10px;'
                                        f'color:#4a6080;'
                                        f'letter-spacing:1px;'
                                        f'margin-bottom:3px">'
                                        f'{"TARGET ↑" if "BUY" in _action else "TARGET ↓" if "SELL" in _action else "RANGE"}</div>'
                                        f'<div style="font-size:15px;'
                                        f'font-weight:bold;'
                                        f'color:{_tc2}">'
                                        f'${_rec["fc_target"]:.2f}</div>'
                                        f'<div style="font-size:11px;'
                                        f'color:{_tc2}">'
                                        f'{_pct2:+.1f}%</div></div>'
                                        f'<div style="background:#0f1525;'
                                        f'padding:10px 14px">'
                                        f'<div style="font-size:10px;'
                                        f'color:#4a6080;'
                                        f'letter-spacing:1px;'
                                        f'margin-bottom:3px">'
                                        f'STOP</div>'
                                        f'<div style="font-size:15px;'
                                        f'font-weight:bold;'
                                        f'color:#ef5350">'
                                        f'${_stop2:.2f}</div>'
                                        f'<div style="font-size:11px;'
                                        f'color:#ef5350">'
                                        f'R/R {_rr2:.1f}:1</div>'
                                        f'</div></div>'
                                    )

                                    _info_row = (
                                        f'<div style="padding:8px 16px;'
                                        f'border-top:1px solid #1e2d45;'
                                        f'display:flex;justify-content:space-between;'
                                        f'flex-wrap:wrap;gap:8px">'
                                        f'<span style="font-size:11px;'
                                        f'color:#4a6080">⏱ Until: {_horizon_str}</span>'
                                        f'<span style="font-size:11px;'
                                        f'color:#4a6080">📊 Move: {_pct2:+.1f}%</span>'
                                        f'<span style="font-size:11px;'
                                        f'color:#4a6080">❌ Cut if: &lt;${_stop2:.2f}</span>'
                                        f'<span style="font-size:11px;'
                                        f'color:#4a6080" title="{_conv_explain}">💡 {_conv_explain}</span>'
                                        f'</div>'
                                    )
                            _components.html(
                                f'<div style="background:#0a0e1a;border:1px solid '
                                f'#1e2d45;border-radius:6px;overflow:hidden;'
                                f'font-family:Courier New,monospace;'
                                f'margin:4px 0">'
                                f'<div style="padding:10px 16px;border-bottom:'
                                f'1px solid #1e2d45;display:flex;align-items:'
                                f'center;gap:14px">'
                                f'<span style="font-size:18px;font-weight:bold;'
                                f'color:{_color}">{_action}</span>'
                                f'<span style="font-size:11px;color:#4a6080;'
                                f'background:#0f1525;padding:2px 8px;border:1px '
                                f'solid #1e2d45;border-radius:3px">'
                                f'{_conv} conviction</span>'
                                f'<span style="font-size:11px;color:#4a6080;'
                                f'margin-left:auto">score {_score}/10</span>'
                                f'</div>'
                                f'<div style="padding:10px 16px">{_reasons_html}'
                                f'</div>{_mh}{_info_row}</div>',
                                height=300 if _mh else 160,
                            )
                    except Exception as _re:
                        st.caption(f"Recommendation unavailable: {_re}")
        except Exception as _e:
            st.caption(f"AI Score unavailable: {_e}")

        # Multi-timeframe chart
        with st.expander("📈 Multi-Timeframe Chart", expanded=False):
            try:
                from components.multi_timeframe_chart import render_multi_timeframe_chart
                _sym = st.session_state.get("symbol") or symbol
                render_multi_timeframe_chart(_sym, hist_daily=data)
            except Exception as _e:
                st.caption(f"Chart unavailable: {_e}")

        # Price chart - use advanced candlestick chart if OHLCV data available
        try:
            from utils.plotting_helper import create_candlestick_chart
            
            # Check if we have OHLCV data
            has_ohlcv = all(col in data.columns for col in ['open', 'high', 'low', 'close'])
            
            if has_ohlcv:
                fig = create_candlestick_chart(
                    data=data,
                    title=f"{st.session_state.symbol} Price History",
                    show_volume='volume' in data.columns,
                    show_ma=[20, 50, 200] if len(data) >= 200 else ([20, 50] if len(data) >= 50 else [20])
                )
            else:
                # Fallback to simple line chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='blue', width=2)
                ))
                fig.update_layout(
                    title=f"{st.session_state.symbol} Price History",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    hovermode='x unified',
                    height=400
                )
            
            st.plotly_chart(fig, width='stretch')
        except ImportError:
            # Fallback to basic chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            ))
            fig.update_layout(
                title=f"{st.session_state.symbol} Price History",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, width='stretch')
        
        # Data table (expandable)
        with st.expander("📋 View Full Data"):
            st.dataframe(data.tail(50))
        
        # Model Selection & Forecasting
        st.markdown("---")
        st.subheader("🎯 Generate Forecast")
        
        # Forecast horizon slider (outside form so it updates immediately)
        forecast_horizon = st.slider(
            "Forecast Horizon (days)",
            min_value=1,
            max_value=30,
            value=st.session_state.forecast_horizon,
            key="forecast_horizon_slider",
            help="Number of days to forecast into the future"
        )
        # Update session state immediately
        st.session_state.forecast_horizon = forecast_horizon
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            forecast_button = st.button(
                "🚀 Generate Forecast",
                type="primary",
                help="Run all models and show consensus forecast",
            )
        
        with col2:
            if forecast_button:
                try:
                    _denorm_price = None  # safe default before any conditional that might skip assignment
                    with st.spinner("Running consensus forecast (all models)..."):
                        data = st.session_state.get("forecast_data")
                        if data is None:
                            raise RuntimeError("No forecast_data in session_state; please load data first.")
                        data = data.copy()
                        # Normalize column case (yfinance uses Close, etc.)
                        if "close" in data.columns and "Close" not in data.columns:
                            data = data.rename(columns={"close": "Close"})
                        if "Close" not in data.columns:
                            data["Close"] = data.iloc[:, 0]
                        horizon = st.session_state.get("forecast_horizon", 7)
                        if not isinstance(data.index, pd.DatetimeIndex):
                            data.index = pd.to_datetime(data.index)
                        used_router = False
                        consensus = None
                        try:
                            from trading.models.forecast_router import ForecastRouter
                            import hashlib as _hashlib
                            import time as _time

                            _router = ForecastRouter()
                            _data_hash = _hashlib.md5(
                                str(data.index[-1]).encode()
                                + str(len(data)).encode()
                                + ticker.encode()
                            ).hexdigest()[:8]
                            _cons_key = f"consensus_{ticker}_{_data_hash}"
                            _cons_ts_key = f"consensus_ts_{ticker}_{_data_hash}"
                            _cached_cons = st.session_state.get(_cons_key)
                            _cached_cons_ts = st.session_state.get(_cons_ts_key, 0.0)
                            _cons_age = _time.time() - _cached_cons_ts

                            if _cached_cons is None or _cons_age > 600:
                                with st.spinner(
                                    "Running consensus forecast (all models)..."
                                ):
                                    consensus = _router.get_consensus_forecast(
                                        data, horizon=horizon
                                    )
                                st.session_state[_cons_key] = consensus
                                st.session_state[_cons_ts_key] = _time.time()
                            else:
                                consensus = _cached_cons
                                st.caption(
                                    "Using cached forecast (refreshes every 10 min)"
                                )
                            _used = consensus.get("models_used", []) if isinstance(consensus, dict) else []
                            _failed = consensus.get("models_failed", []) if isinstance(consensus, dict) else []
                            if _failed:
                                st.caption(
                                    f"Models available: {_used} | Unavailable: {_failed}"
                                )
                            if consensus and "error" not in consensus:
                                raw = consensus.get("consensus_forecast") or (consensus.get("consensus_price") and [consensus["consensus_price"]]) or []
                                forecast_values = np.asarray(raw, dtype="float64").ravel()
                                if forecast_values.size == 0 and consensus.get("price_targets"):
                                    import numpy as np
                                    pt = consensus["price_targets"]
                                    last_price = float(consensus.get("last_price") or 0)
                                    if pt and last_price:
                                        forecast_values = np.array([float(list(pt.values())[-1]) if pt else last_price])
                                    if forecast_values.size < horizon:
                                        forecast_values = np.resize(forecast_values, horizon)
                                if forecast_values.size < horizon:
                                    forecast_values = np.resize(forecast_values, horizon)
                                forecast_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=min(horizon, len(forecast_values)), freq="D")[:len(forecast_values)]
                                if hasattr(forecast_dates, "tz_localize"):
                                    try:
                                        forecast_dates = forecast_dates.tz_localize(None)
                                    except Exception:
                                        pass
                                st.session_state.current_forecast = pd.DataFrame({"forecast": forecast_values[:len(forecast_dates)]}, index=forecast_dates)
                                st.session_state.current_model = "Consensus"
                                st.session_state.current_forecast_result = {
                                    "forecast": np.asarray(consensus.get("consensus_forecast", [])).ravel().tolist(),
                                    "validation_mape": None,
                                    "in_sample_mape": None,
                                    "last_actual_price": consensus.get("last_price"),
                                    "confidence_label": consensus.get("conviction", "INSUFFICIENT"),
                                    "warnings": [],
                                    "lower_bound": consensus.get("lower_bound"),
                                    "upper_bound": consensus.get("upper_bound"),
                                }
                                st.session_state["forecast_debug"] = consensus.get("forecast_debug", {})
                                used_router = True
                                st.success("✅ Consensus forecast generated")
                                if st.session_state.get("forecast_debug"):
                                    with st.expander("🔍 Forecast debug (per-model std/mean/size)", expanded=False):
                                        st.json(st.session_state["forecast_debug"])
                        except Exception as _ce:
                            logger.debug("Consensus forecast failed: %s", _ce)
                            st.caption(f"Consensus error: {_ce}")
                            consensus = None
                        if not used_router:
                            _pt = consensus.get("price_targets") if isinstance(consensus, dict) else None
                            _failed = consensus.get("models_failed") if isinstance(consensus, dict) else None
                            if consensus and (_pt or _failed):
                                pt = _pt or {}
                                failed = _failed or []
                                n_ok = len(pt)
                                total = n_ok + len(failed)
                                st.caption(f"Partial consensus — {n_ok} of {total} models")
                            else:
                                st.error("Consensus forecast failed. Load data and try again, or use Tab 2 for a single-model forecast.")
                        # Quick Forecast now uses only the consensus router; single-model flows live in Tab 2.
                        # If consensus fails, we surface the above error and do not attempt any model initialization here.
                        if used_router:
                            # Store full forecast result for confidence intervals if available
                            forecast_result = st.session_state.get("current_forecast_result")
                            if forecast_result is not None:
                                # Postprocess forecast
                                try:
                                    from trading.forecasting.forecast_postprocessor import ForecastPostprocessor

                                    postprocessor = ForecastPostprocessor()

                                    # Extract forecast values (handles forecast, predictions, values, forecast_values, consensus_forecast)
                                    forecast_vals = _extract_forecast_values(forecast_result)
                                    if forecast_vals is not None:
                                        forecast_vals = forecast_vals.tolist() if hasattr(forecast_vals, 'tolist') else list(forecast_vals)
                                    else:
                                        forecast_vals = []

                                    # Postprocess forecast
                                    processed_forecast = postprocessor.process(
                                        forecast=forecast_vals,
                                        historical_data=data,
                                        apply_smoothing=True,
                                        remove_outliers=True,
                                        ensure_realistic_bounds=True
                                    )

                                    # Update forecast_result with processed version
                                    if isinstance(forecast_result, dict):
                                        forecast_result['forecast'] = processed_forecast['values']
                                        forecast_result['postprocessing_notes'] = processed_forecast.get('notes', [])
                                    else:
                                        forecast_result = processed_forecast['values']

                                    # Show what was done
                                    if processed_forecast.get('modifications'):
                                        with st.expander("⚙️ Forecast Postprocessing", expanded=False):
                                            st.write("**Modifications applied:**")
                                            for mod in processed_forecast['modifications']:
                                                st.write(f"• {mod}")

                                    # Update session state with processed forecast
                                    st.session_state.current_forecast_result = forecast_result
                                except ImportError:
                                    pass  # Silently fail if postprocessor not available
                                except Exception as e:
                                    logger.warning(f"Forecast postprocessing failed: {e}")
                            
                            # Log model performance
                            if 'model_log' in st.session_state and 'perf_logger' in st.session_state:
                                try:
                                    import time
                                    train_time = time.time() - time.time()  # Placeholder - would need actual timing
                                    
                                    # Calculate basic metrics if possible
                                    r2_score = 0.0  # Placeholder - would need actual evaluation
                                    rmse = 0.0
                                    mae = 0.0
                                    
                                    st.session_state.model_log.log_training(
                                        model_name=selected_model,
                                        model_version='1.0',
                                        training_data_size=len(data),
                                        training_time=train_time,
                                        parameters={},
                                        metrics={
                                            'r2_score': r2_score,
                                            'rmse': rmse,
                                            'mae': mae
                                        }
                                    )
                                    
                                    st.session_state.perf_logger.log_performance(
                                        model_name=selected_model,
                                        metric_name='accuracy',
                                        metric_value=r2_score,
                                        timestamp=datetime.now()
                                    )
                                except Exception as e:
                                    pass  # Silently fail if logging not available
                            
                            # Extract forecast values (robust for all result formats)
                            forecast_values = _extract_forecast_values(forecast_result)
                            if isinstance(forecast_result, dict):
                                forecast_dates = forecast_result.get('dates', pd.date_range(
                                    start=data.index[-1] + timedelta(days=1),
                                    periods=horizon,
                                    freq='D'
                                ))
                            else:
                                forecast_dates = pd.date_range(
                                    start=data.index[-1] + timedelta(days=1),
                                    periods=horizon,
                                    freq='D'
                                )
                            if forecast_values is not None:
                                forecast_values = np.asarray(forecast_values).ravel()
                            
                            # Debug: Check what we got
                            if forecast_values is None or (hasattr(forecast_values, '__len__') and len(forecast_values) == 0):
                                st.warning(f"⚠️ Forecast returned empty values. Result type: {type(forecast_result)}")
                                if isinstance(forecast_result, dict):
                                    st.write("Forecast result keys:", list(forecast_result.keys()))
                                # Use last known price as fallback
                                last_price = data['Close'].iloc[-1] if 'Close' in data.columns else data.iloc[-1, 0]
                                forecast_values = np.full(horizon, float(last_price))
                            
                            # Ensure forecast_values is array-like
                            if isinstance(forecast_values, (list, np.ndarray)):
                                forecast_values = np.array(forecast_values).flatten()
                                # Check for NaN/None values
                                if np.any(np.isnan(forecast_values)) or np.any(forecast_values == None):
                                    st.warning("⚠️ Forecast contains NaN/None values. Replacing with last known price.")
                                    last_price = float(data['Close'].iloc[-1] if 'Close' in data.columns else data.iloc[-1, 0])
                                    forecast_values = np.where(
                                        np.isnan(forecast_values) | (forecast_values == None),
                                        last_price,
                                        forecast_values
                                    )
                            else:
                                # Single value case
                                if forecast_values is None or (isinstance(forecast_values, float) and np.isnan(forecast_values)):
                                    last_price = float(data['Close'].iloc[-1] if 'Close' in data.columns else data.iloc[-1, 0])
                                    forecast_values = np.full(horizon, last_price)
                                else:
                                    forecast_values = np.array([float(forecast_values)] * horizon)
                            # Denormalize: model was trained on price/last_price
                            if _denorm_price and _denorm_price != 1.0:
                                forecast_values = np.asarray(forecast_values, dtype=float) * _denorm_price
                            
                            # Create forecast DataFrame
                            forecast_df = pd.DataFrame({
                                'forecast': forecast_values
                            }, index=forecast_dates[:len(forecast_values)])
                            
                            # Store in session state
                            st.session_state.current_forecast = forecast_df
                            selected_model = st.session_state.get("selected_model", "consensus")
                            st.session_state.current_model = selected_model
                            model = st.session_state.get("current_model_instance", None)
                            st.session_state.current_model_instance = model  # Store model instance for explainability
                            
                            # Situational awareness: write to MemoryStore for Chat context (quality gate)
                            try:
                                from trading.memory import get_memory_store
                                from trading.memory.memory_store import MemoryType
                                _store = get_memory_store()
                                _vals = forecast_values if hasattr(forecast_values, "__len__") else [float(forecast_values)]
                                _first = _vals[0] if len(_vals) > 0 else None
                                _last = _vals[-1] if len(_vals) > 0 else None
                                try:
                                    _all_same = len(_vals) > 1 and (np.unique(np.asarray(_vals).flatten()).size <= 1)
                                except Exception:
                                    _all_same = False
                                _degenerate = _first is None or _last is None or _all_same
                                if _first is not None or _last is not None:
                                    _conf = None
                                    if isinstance(forecast_result, dict):
                                        if "lower_bound" in forecast_result and "upper_bound" in forecast_result:
                                            _conf = {"lower": forecast_result["lower_bound"], "upper": forecast_result["upper_bound"]}
                                    _store.add(
                                        MemoryType.LONG_TERM,
                                        namespace="forecasts",
                                        value={
                                            "symbol": st.session_state.get("symbol", ""),
                                            "model_name": selected_model,
                                            "horizon": horizon,
                                            "forecast_first": _first,
                                            "forecast_last": _last,
                                            "confidence": _conf,
                                            "timestamp": datetime.utcnow().isoformat(),
                                            **({"model_failed": True} if _degenerate else {}),
                                        },
                                        category="results",
                                    )
                            except Exception:
                                pass
                            
                            st.success(f"✅ Forecast generated using {selected_model}")
                            
                            # Natural Language Insights
                            st.markdown("---")
                            st.subheader("💬 Natural Language Insights")
                            
                            if st.button("Generate Plain English Explanation", key="generate_nlg_insights"):
                                try:
                                    from nlp.natural_language_insights import NaturalLanguageInsights
                                    
                                    nlg = NaturalLanguageInsights()
                                    
                                    # Prepare forecast result for insights
                                    forecast_for_insights = {
                                        'forecast': forecast_df['forecast'].values.tolist() if hasattr(forecast_df['forecast'].values, 'tolist') else list(forecast_df['forecast'].values),
                                        'dates': forecast_df.index.tolist() if hasattr(forecast_df.index, 'tolist') else list(forecast_df.index),
                                        'model_type': selected_model,
                                        'symbol': st.session_state.symbol
                                    }
                                    
                                    # Add confidence intervals if available
                                    if isinstance(forecast_result, dict):
                                        if 'lower_bound' in forecast_result:
                                            forecast_for_insights['lower_bound'] = forecast_result['lower_bound']
                                        if 'upper_bound' in forecast_result:
                                            forecast_for_insights['upper_bound'] = forecast_result['upper_bound']
                                    
                                    # Generate explanation
                                    insights = nlg.generate_forecast_insights(
                                        forecast=forecast_for_insights,
                                        historical_data=data,
                                        model_type=selected_model,
                                        symbol=st.session_state.symbol
                                    )
                                    
                                    # Display insights
                                    st.info(insights['summary'])
                                    
                                    with st.expander("📊 Detailed Analysis", expanded=False):
                                        st.write("**Trend Analysis:**")
                                        st.write(insights['trend_analysis'])
                                        
                                        st.write("**Key Factors:**")
                                        for factor in insights['key_factors']:
                                            st.write(f"• {factor}")
                                        
                                        st.write("**Confidence Assessment:**")
                                        st.write(insights['confidence_explanation'])
                                        
                                        st.write("**Recommendations:**")
                                        for rec in insights['recommendations']:
                                            st.write(f"• {rec}")
                                
                                except ImportError:
                                    st.error("Natural Language Insights not available")
                                except Exception as e:
                                    st.error(f"Error generating insights: {e}")
                                    import traceback
                                    if st.checkbox("Show technical details", key="nl_insights_trace"):
                                        st.code(traceback.format_exc())
                            
                            # AI Commentary Service
                            if 'commentary_service' in st.session_state:
                                st.markdown("---")
                                st.subheader("💬 AI Commentary")
                                
                                if st.button("Generate Commentary", key="generate_forecast_commentary"):
                                    commentary_service = st.session_state.commentary_service
                                    
                                    with st.spinner("Generating commentary..."):
                                        try:
                                            # Prepare forecast result for commentary
                                            forecast_for_commentary = {
                                                'forecast': forecast_df['forecast'].values.tolist() if hasattr(forecast_df['forecast'].values, 'tolist') else list(forecast_df['forecast'].values),
                                                'dates': forecast_df.index.tolist() if hasattr(forecast_df.index, 'tolist') else list(forecast_df.index),
                                                'model_type': selected_model,
                                                'symbol': st.session_state.symbol
                                            }
                                            
                                            # Add confidence intervals if available
                                            if isinstance(forecast_result, dict):
                                                if 'lower_bound' in forecast_result:
                                                    forecast_for_commentary['lower_bound'] = forecast_result['lower_bound']
                                                if 'upper_bound' in forecast_result:
                                                    forecast_for_commentary['upper_bound'] = forecast_result['upper_bound']
                                            
                                            commentary = commentary_service.generate_forecast_commentary(
                                                symbol=st.session_state.symbol,
                                                forecast_result=forecast_for_commentary,
                                                model_type=selected_model,
                                                historical_data=data
                                            )
                                            
                                            st.info(commentary.get('summary', 'Commentary generated'))
                                            
                                            with st.expander("📊 Detailed Analysis", expanded=False):
                                                if 'trend_analysis' in commentary:
                                                    st.write("**Trend Analysis:**")
                                                    st.write(commentary['trend_analysis'])
                                                
                                                if 'insights' in commentary and commentary['insights']:
                                                    st.write("**Key Insights:**")
                                                    for insight in commentary['insights']:
                                                        st.write(f"• {insight}")
                                                
                                                if 'warnings' in commentary and commentary['warnings']:
                                                    st.write("**Warnings:**")
                                                    for warning in commentary['warnings']:
                                                        st.warning(f"⚠️ {warning}")
                                        except Exception as e:
                                            st.error(f"Error generating commentary: {e}")
                                            import traceback
                                            st.code(traceback.format_exc())
                
                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.info("Try adjusting the date range or selecting a different model.")
            
            # Display consensus view using the ForecastRouter
            try:
                hist_data_cons = st.session_state.get("forecast_data")
                if hist_data_cons is not None and len(hist_data_cons) >= 2:
                    from trading.models.forecast_router import ForecastRouter

                    router = ForecastRouter()
                    horizon = st.session_state.get("forecast_horizon", 7)
                    import hashlib as _hashlib
                    import time as _time

                    try:
                        _data_hash = _hashlib.md5(
                            str(hist_data_cons.index[-1]).encode()
                            + str(len(hist_data_cons)).encode()
                            + ticker.encode()
                        ).hexdigest()[:8]
                        _cons_key = f"consensus_{ticker}_{_data_hash}"
                        _cons_ts_key = f"consensus_ts_{ticker}_{_data_hash}"
                        _cached_cons = st.session_state.get(_cons_key)
                        _cached_cons_ts = st.session_state.get(_cons_ts_key, 0.0)
                        _cons_age = _time.time() - _cached_cons_ts

                        if _cached_cons is None or _cons_age > 600:
                            consensus = router.get_consensus_forecast(
                                data=hist_data_cons,
                                horizon=horizon,
                            )
                            st.session_state[_cons_key] = consensus
                            st.session_state[_cons_ts_key] = _time.time()
                        else:
                            consensus = _cached_cons
                            st.caption(
                                "Using cached forecast (refreshes every 10 min)"
                            )
                    except Exception as _ce:
                        st.caption(f"Consensus error: {_ce}")
                        consensus = {"error": str(_ce)}
                    _used = consensus.get("models_used", []) if isinstance(consensus, dict) else []
                    _failed = consensus.get("models_failed", []) if isinstance(consensus, dict) else []
                    if _failed:
                        st.caption(
                            f"Models available: {_used} | Unavailable: {_failed}"
                        )
                    if "error" not in consensus:
                        raw = consensus.get("consensus_forecast") or (
                            consensus.get("consensus_price")
                            and [consensus["consensus_price"]]
                        ) or []
                        if not raw and (
                            consensus.get("price_targets") or consensus.get("models_failed")
                        ):
                            pt = consensus.get("price_targets") or {}
                            failed = consensus.get("models_failed") or []
                            n_ok, total = len(pt), len(pt) + len(failed)
                            st.caption(f"Partial consensus — {n_ok} of {total} models")

                        with st.expander("🎯 Model Consensus", expanded=True):
                            direction = consensus.get("direction", "NEUTRAL")
                            conviction = consensus.get("conviction", "INSUFFICIENT")
                            last_price = float(consensus.get("last_price") or 0.0)
                            consensus_price = float(
                                consensus.get("consensus_price")
                                or consensus.get("consensus_forecast", [0.0])[-1]
                            )
                            price_targets = consensus.get("price_targets") or {}
                            models_failed = consensus.get("models_failed") or []

                            # Row 1: direction, conviction, consensus price
                            col1, col2, col3 = st.columns(3)
                            direction_emoji = (
                                "📈"
                                if direction == "BULLISH"
                                else "📉"
                                if direction == "BEARISH"
                                else "➡️"
                            )
                            with col1:
                                col1.metric(
                                    "Consensus Direction",
                                    f"{direction_emoji} {direction}",
                                )
                            with col2:
                                col2.metric("Conviction", conviction)
                            with col3:
                                delta_pct = (
                                    (consensus_price / last_price - 1.0) * 100.0
                                    if last_price
                                    else 0.0
                                )
                                col3.metric(
                                    f"Consensus Price (Day {horizon})",
                                    f"${consensus_price:.2f}",
                                    delta=f"{delta_pct:+.2f}%",
                                )

                            # Signal synthesis: reconcile consensus, AI Score, and outlier exclusions
                            _consensus_dir = direction
                            _conv = conviction
                            # Try canonical key first, then dynamic key, then any ai_score key
                            _symbol = st.session_state.get("symbol", "")
                            _ai_result = (
                                st.session_state.get("ai_score_result")
                                or st.session_state.get(f"ai_score_{_symbol}")
                                or {}
                            )
                            _excluded = consensus.get("models_excluded", {}) or {}

                            try:
                                _ai_val_f = float(
                                    _ai_result.get("weighted_score")
                                    or _ai_result.get("overall_score")
                                    or _ai_result.get("model_score")
                                    or 5.0)
                            except Exception:
                                _ai_val_f = 5.0
                            if _ai_val_f == 5.0 and _ai_result:
                                import logging as _lg
                                _lg.getLogger(__name__).warning(
                                    "AI score result found but score keys missing: %s",
                                    list(_ai_result.keys()))

                            _signals = []
                            if _consensus_dir == "BULLISH":
                                _signals.append("models lean bullish")
                            elif _consensus_dir == "BEARISH":
                                _signals.append("models lean bearish")
                            else:
                                _signals.append("models are mixed")

                            if _ai_val_f >= 7.0:
                                _signals.append("AI Score is strong")
                            elif _ai_val_f >= 5.5:
                                _signals.append("AI Score is neutral")
                            else:
                                _signals.append("AI Score is weak")

                            _excl_str = ""
                            if _excluded:
                                _excl_names = ", ".join(
                                    f"{k} ({v.get('deviation_std', 0)}σ)"
                                    for k, v in _excluded.items()
                                )
                                _excl_str = f" ⚠️ Outlier excluded: {_excl_names}."

                            _synthesis = (
                                f"**Signal synthesis:** "
                                f"{' · '.join(_signals)}."
                                f"{_excl_str} "
                                "Consensus and Monte Carlo measure different things — "
                                "point forecast vs probability distribution. "
                                "Use Monte Carlo for risk sizing and consensus for direction bias."
                            )
                            st.info(_synthesis)

                            # Row 2: per-model price targets table
                            if price_targets:
                                targets_df = pd.DataFrame(
                                    [
                                        {
                                            "Model": k,
                                            "Day 7 Price": f"${v:.2f}",
                                            "vs Current": (
                                                f"{((v/last_price)-1)*100:.1f}%"
                                                if last_price
                                                else "N/A"
                                            ),
                                        }
                                        for k, v in price_targets.items()
                                    ]
                                )
                                st.dataframe(
                                    targets_df,
                                    width='stretch',
                                    hide_index=True,
                                )

                            # Row 3: failed models (optional)
                            if models_failed:
                                with st.expander(
                                    f"⚠️ {len(models_failed)} model(s) excluded",
                                    expanded=False,
                                ):
                                    for f in models_failed:
                                        st.caption(f"• {f}")
                    else:
                        st.warning("Consensus forecast temporarily unavailable.")
            except Exception:
                try:
                    if isinstance(consensus, dict) and consensus.get("error"):
                        st.warning("Consensus forecast temporarily unavailable.")
                except NameError:
                    st.warning("Consensus forecast temporarily unavailable.")

            # Display forecast using UI components
            if st.session_state.get('current_forecast') is not None:
                st.markdown(f"**Forecast using {st.session_state.current_model}**")

                try:
                    from trading.ui.forecast_components import render_forecast_results, render_confidence_metrics
                    
                    hist_data = st.session_state.get("forecast_data")
                    forecast_df = st.session_state.current_forecast
                    forecast_result = st.session_state.get('current_forecast_result', {})
                    
                    # Prepare forecast data for component
                    forecast_data = {
                        'dates': forecast_df.index.tolist() if hasattr(forecast_df.index, 'tolist') else list(forecast_df.index),
                        'forecast': forecast_df['forecast'].values.tolist() if hasattr(forecast_df['forecast'].values, 'tolist') else list(forecast_df['forecast'].values),
                        'model_name': st.session_state.current_model
                    }
                    if isinstance(forecast_result, dict):
                        if 'lower_bound' in forecast_result:
                            forecast_data['lower_bound'] = forecast_result['lower_bound']
                        if 'upper_bound' in forecast_result:
                            forecast_data['upper_bound'] = forecast_result['upper_bound']
                        if 'confidence' in forecast_result:
                            forecast_data['confidence'] = forecast_result['confidence']
                        if forecast_result.get('validation_mape') is not None:
                            forecast_data['validation_mape'] = forecast_result['validation_mape']
                        if forecast_result.get('confidence_label'):
                            forecast_data['confidence_label'] = forecast_result['confidence_label']
                        if forecast_result.get('last_actual_price') is not None:
                            forecast_data['last_actual_price'] = forecast_result['last_actual_price']
                    if forecast_data.get('last_actual_price') is None and hist_data is not None and len(hist_data) > 0:
                        close_col = 'close' if 'close' in hist_data.columns else (hist_data.columns[0] if len(hist_data.columns) > 0 else None)
                        if close_col:
                            forecast_data['last_actual_price'] = float(hist_data[close_col].iloc[-1])
                    
                    # Show validation MAPE, confidence label, last actual vs first forecast
                    if forecast_data.get('validation_mape') is not None or forecast_data.get('last_actual_price') is not None:
                        with st.expander("📐 Accuracy & continuity", expanded=True):
                            mape_val = forecast_data.get('validation_mape') or forecast_data.get('in_sample_mape')
                            if mape_val is not None:
                                st.metric("Validation / in-sample MAPE", f"{mape_val:.2f}%")
                            if forecast_data.get('confidence_label'):
                                st.metric("Confidence (based on MAPE)", forecast_data['confidence_label'])
                            if mape_val is not None:
                                confidence_pct = max(0.0, min(100.0, 100.0 - float(mape_val)))
                                st.metric("Confidence score", f"{confidence_pct:.0f}%")
                            last_act = forecast_data.get('last_actual_price')
                            fvals = forecast_data.get('forecast', [])
                            first_fc = fvals[0] if fvals else None
                            if last_act is not None and first_fc is not None:
                                st.write("**Continuity:** Last actual price **${:.2f}** → First forecast **${:.2f}**".format(float(last_act), float(first_fc)))
                    
                    # Render forecast results
                    render_forecast_results(
                        forecast=forecast_data,
                        historical_data=hist_data,
                        symbol=st.session_state.symbol,
                        show_chart=True,
                        show_table=True
                    )
                    
                    # Show confidence metrics
                    if isinstance(forecast_result, dict) and ('lower_bound' in forecast_result or 'confidence' in forecast_result):
                        render_confidence_metrics(forecast_data)
                    # News sentiment overlay below forecast results
                    try:
                        _ns = _news_sentiment_score(ticker)
                        if _ns >= 7.5:
                            _ns_label, _ns_color = "HOT", "#ff9800"
                            _ns_text = "Breaking positive news may accelerate move"
                        elif _ns >= 6.0:
                            _ns_label, _ns_color = "POS", "#26a69a"
                            _ns_text = "Positive news sentiment supports forecast"
                        elif _ns <= 3.0:
                            _ns_label, _ns_color = "NEG", "#ef5350"
                            _ns_text = "Negative news may create headwinds"
                        else:
                            _ns_label, _ns_color = "NEU", "#4a6080"
                            _ns_text = "News sentiment is neutral"
                        st.markdown(
                            f'<div style="padding:8px 12px;'
                            f'background:#0f1525;border-left:3px solid '
                            f'{_ns_color};border-radius:0 4px 4px 0;'
                            f'margin-top:8px">'
                            f'<span style="color:{_ns_color};'
                            f'font-weight:bold;font-size:11px">'
                            f'NEWS {_ns_label}</span>'
                            f'<span style="color:#8899aa;font-size:11px;'
                            f'margin-left:8px">{_ns_text}</span>'
                            f'<span style="color:#4a6080;font-size:10px;'
                            f'float:right">score: {_ns:.1f}/10</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    except Exception:
                        pass
                    
                except ImportError:
                    # Fallback to original display code
                    from utils.plotting_helper import create_forecast_chart
                    
                    hist_data = st.session_state.get("forecast_data")
                    forecast_df = st.session_state.current_forecast
                    forecast_result = st.session_state.get('current_forecast_result', {})
                    
                    confidence_intervals = None
                    if isinstance(forecast_result, dict) and 'lower_bound' in forecast_result and 'upper_bound' in forecast_result:
                        confidence_intervals = {
                            'lower': forecast_result['lower_bound'],
                            'upper': forecast_result['upper_bound']
                        }
                    
                    fig = create_forecast_chart(
                        historical=hist_data,
                        forecast=forecast_df['forecast'].values,
                        forecast_dates=forecast_df.index,
                        confidence_intervals=confidence_intervals,
                        title=f"{st.session_state.symbol} - Historical & Forecast"
                    )
                    st.plotly_chart(fig, width='stretch')
                    
                    # Forecast table (format dates as YYYY-MM-DD, no timezone)
                    st.markdown("**Forecast Values:**")
                    display_df = forecast_df.copy()
                    try:
                        _idx = pd.to_datetime(display_df.index)
                        if hasattr(_idx, "tz") and _idx.tz is not None:
                            _idx = _idx.tz_localize(None)
                        display_df.index = _idx.strftime("%Y-%m-%d")
                    except Exception:
                        pass
                    display_df["forecast"] = display_df["forecast"].apply(
                        lambda x: f"${x:.2f}"
                        if isinstance(x, (int, float)) and not pd.isna(x)
                        else "—"
                    )
                    display_df.rename(columns={"forecast": "Forecast"}, inplace=True)

                    # Add confidence bounds if available
                    _lb = None
                    _ub = None
                    _forecast_result = st.session_state.get("current_forecast_result")
                    if isinstance(_forecast_result, dict):
                        _lb = _forecast_result.get("lower_bound")
                        _ub = _forecast_result.get("upper_bound")
                    _n = len(display_df)
                    display_df["Lower Bound"] = (
                        [f"${v:.2f}" for v in _lb[:_n]]
                        if _lb and len(_lb) >= _n
                        else ["—"] * _n
                    )
                    display_df["Upper Bound"] = (
                        [f"${v:.2f}" for v in _ub[:_n]]
                        if _ub and len(_ub) >= _n
                        else ["—"] * _n
                    )
                    st.dataframe(display_df, width="stretch")
                
                # Download button
                csv = forecast_df.to_csv()
                st.download_button(
                    label="📥 Download Forecast CSV",
                    data=csv,
                    file_name=f"{st.session_state.symbol}_forecast.csv",
                    mime="text/csv"
                )
                
                # Add explainability section
                st.markdown("---")
                st.subheader("🔍 Model Explainability")
                
                with st.expander("View Feature Importance & Explanations", expanded=False):
                    try:
                        from trading.models.forecast_explainability import (
                            IntelligentForecastExplainability,
                        )
                        ForecastExplainability = IntelligentForecastExplainability
                        
                        # Display cached explanation so it persists across reruns
                        if st.session_state.get('forecast_explanation') is not None:
                            cached = st.session_state.forecast_explanation
                            if cached.get('success') and cached.get('explanation'):
                                expl_obj = cached['explanation']
                                if hasattr(expl_obj, 'feature_importance') and expl_obj.feature_importance:
                                    st.write("**Feature Importance:**")
                                    importance_data = expl_obj.feature_importance
                                    if isinstance(importance_data, dict):
                                        importance_df = pd.DataFrame(
                                            list(importance_data.items()),
                                            columns=['Feature', 'Importance']
                                        ).sort_values('Importance', ascending=False)
                                        import plotly.express as px
                                        fig = px.bar(
                                            importance_df, x='Importance', y='Feature',
                                            orientation='h', title='Feature Importance (SHAP values)'
                                        )
                                        st.plotly_chart(fig)
                                if hasattr(expl_obj, 'explanation_text') and expl_obj.explanation_text:
                                    st.write("**Explanation:**")
                                    st.write(expl_obj.explanation_text)
                            if st.button("Clear explanation", key="clear_explanation_quick"):
                                st.session_state.forecast_explanation = None
                                st.rerun()
                        
                        if st.button("Generate Explanation", key="generate_explanation"):
                            with st.spinner("Analyzing model predictions..."):
                                try:
                                    explainer = ForecastExplainability()
                                    model = st.session_state.get('current_model_instance')
                                    forecast_result = st.session_state.get('current_forecast_result', {})
                                    
                                    _fv = _extract_forecast_values(forecast_result)
                                    if _fv is not None and _fv.size > 0:
                                        forecast_value = float(np.asarray(_fv).flat[0])
                                    elif isinstance(forecast_result, (int, float)):
                                        forecast_value = float(forecast_result)
                                    else:
                                        forecast_value = float(data['Close'].iloc[-1])
                                    
                                    features = data.copy()
                                    target_history = features['Close'] if 'Close' in features.columns else (features['close'] if 'close' in features.columns else features.iloc[:, 0])
                                    
                                    try:
                                        if hasattr(explainer, 'generate_forecast_explanation'):
                                            explanation = explainer.generate_forecast_explanation(
                                                model=model,
                                                X=features,
                                                forecast_value=forecast_value,
                                                forecast_horizon=st.session_state.forecast_horizon,
                                                actual_values=target_history,
                                            )
                                            st.session_state.forecast_explanation = {"success": True, "explanation": explanation}
                                        elif hasattr(explainer, 'explain_forecast'):
                                            explanation = explainer.explain_forecast(
                                                forecast_id=f"forecast_{st.session_state.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                                symbol=st.session_state.symbol,
                                                forecast_value=forecast_value,
                                                model=model,
                                                features=features,
                                                target_history=target_history,
                                                horizon=st.session_state.forecast_horizon,
                                            )
                                            st.session_state.forecast_explanation = explanation
                                        else:
                                            st.caption("Model explainability unavailable for this model type.")
                                            st.rerun()
                                            explanation = None
                                        if explanation is not None:
                                            st.success("✅ Explanation generated")
                                            st.rerun()
                                    except Exception:
                                        st.caption("Model explainability unavailable for this model type.")
                                except Exception as e:
                                    st.error(f"Error generating explanation: {e}")
                                    import traceback
                                    st.code(traceback.format_exc())
                    
                    except ImportError as e:
                        st.caption(f"Feature unavailable: Forecast explainability requires SHAP (pip install shap). Details: {e}")
                    except Exception as e:
                        st.caption(f"Feature unavailable: {e}")
                
                # AI Commentary section
                st.markdown("---")
                st.subheader("🤖 AI Market Commentary")
                
                if st.button("Generate AI Commentary", key="generate_commentary"):
                    try:
                        from agents.llm.agent import get_prompt_agent
                        agent = get_prompt_agent()
                        model_name = st.session_state.get("current_model", "Unknown")
                        symbol = st.session_state.get("symbol", "Unknown")
                        last_price = float(data["Close"].iloc[-1]) if "Close" in data.columns else (float(data["close"].iloc[-1]) if "close" in data.columns else 0.0)
                        fcast = _extract_forecast_values(forecast_result)
                        fcast = fcast if fcast is not None and fcast.size > 0 else np.array([])
                        forecast_mean = float(np.mean(fcast)) if fcast.size > 0 else last_price
                        pct_change = ((forecast_mean - last_price) / last_price * 100) if last_price else 0.0
                        if agent:
                            with st.spinner("AI is analyzing the forecast..."):
                                commentary_prompt = (
                                    f"Provide 3 sentences of market commentary on the {model_name} forecast for {symbol}: "
                                    f"last price ${last_price:.2f}, 7-day forecast ${forecast_mean:.2f} ({pct_change:+.1f}%). Be specific and concise."
                                )
                                response = agent.process_prompt(commentary_prompt)
                                commentary = response.message if hasattr(response, "message") else str(response)
                                st.write(commentary)
                        else:
                            st.info("AI commentary requires a valid API key in .env")
                    except Exception as e:
                        st.info(f"AI commentary unavailable: {e}")

with tab2:
    st.header("Advanced Forecasting")
    st.markdown("Full model configuration with hyperparameter tuning and feature engineering")
    
    if st.session_state.get("forecast_data") is None:
        st.warning("⚠️ Please load data first in the Quick Forecast tab")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Configuration")
            
            # Model selection using registry
            try:
                from trading.models.model_registry import get_registry
                
                registry = get_registry()
                
                # Get all single-asset models (includes advanced ones, excludes GNN)
                available_models = registry.get_advanced_models()
                
                if not available_models:
                    st.warning("⚠️ No models available. Using default models.")
                    available_models = ["LSTM", "XGBoost", "Prophet", "ARIMA"]
                
                # Create friendly names
                model_display_names = {
                    'LSTM': 'LSTM (Deep Learning)',
                    'XGBoost': 'XGBoost (Gradient Boosting)',
                    'Prophet': 'Prophet (Seasonality)',
                    'ARIMA': 'ARIMA (Statistical)',
                    'Ensemble': 'Ensemble (Multi-Model)',
                    'TCN': 'TCN (Temporal Conv)',
                    'GARCH': 'GARCH (Volatility)',
                    'Autoformer': 'Autoformer (Advanced Transformer)',
                    'CatBoost': 'CatBoost (Categorical Boosting)',
                    'Ridge': 'Ridge (Linear Baseline)'
                }
                
                display_names = [model_display_names.get(m, m) for m in available_models if m in model_display_names or m not in model_display_names]
                
                selected_display = st.selectbox(
                    "Select Model Type",
                    display_names if display_names else available_models,
                    help="Advanced forecasting with all available models"
                )
                
                # Convert back to model name
                if display_names:
                    model_type = [k for k, v in model_display_names.items() if v == selected_display]
                    model_type = model_type[0] if model_type else selected_display
                else:
                    model_type = selected_display
                
                # Show model info
                model_info = registry.get_model_info(model_type)
                if model_info:
                    st.info(f"ℹ️ {model_info.get('description', 'No description')}")
                    
                    # Show requirements
                    reqs = []
                    if model_info.get('requires_gpu'):
                        reqs.append("🖥️ GPU recommended")
                    if model_info.get('min_data_points'):
                        reqs.append(f"📊 Min {model_info['min_data_points']} data points")
                    
                    if reqs:
                        st.caption(" • ".join(reqs))
                
                # Auto-select best model (Advanced only; runs all models, slow)
                if st.button("🔬 Auto-select best model", key="auto_select_best_advanced"):
                    data_for_best = st.session_state.get("forecast_data")
                    if data_for_best is not None and len(data_for_best) >= 60:
                        try:
                            from trading.models.forecast_router import ForecastRouter
                            _router = ForecastRouter()
                            _horizon = st.session_state.get("forecast_horizon", 7)
                            _symbol = st.session_state.get("symbol", "")
                            _df = data_for_best.copy()
                            if "Close" in _df.columns and "close" not in _df.columns:
                                _df = _df.rename(columns={"Close": "close", "Open": "open", "High": "high", "Low": "low", "Volume": "volume"})
                            best_key, best_result = _router.select_best_model(_df, _horizon, symbol=_symbol or None)
                            display_map = {"lstm": "LSTM", "xgboost": "XGBoost", "prophet": "Prophet", "arima": "ARIMA", "ridge": "Ridge", "garch": "GARCH", "tcn": "TCN", "catboost": "CatBoost", "ensemble": "Ensemble", "transformer": "Transformer", "autoformer": "Autoformer", "hybrid": "Hybrid"}
                            best_display = display_map.get(best_key, best_key.capitalize())
                            st.session_state["advanced_forecast_best_model"] = best_display
                            st.session_state["advanced_best_model_scores"] = best_result
                            st.success(f"Best model: **{best_display}** (validation MAPE: {best_result.get('validation_mape') or 'N/A'}%)")
                        except Exception as e:
                            st.warning(f"Auto-select failed: {e}")
                    else:
                        st.warning("Load at least 60 rows of data first in Quick Forecast tab.")
                if st.session_state.get("advanced_best_model_scores"):
                    with st.expander("📊 Model validation scores (walk-forward)", expanded=False):
                        scores = st.session_state["advanced_best_model_scores"].get("scores", [])
                        if scores is not None and len(scores) > 0:
                            import pandas as pd
                            tab_df = pd.DataFrame(scores)
                            st.dataframe(tab_df, width='stretch')
            except Exception as e:
                st.warning(f"⚠️ Could not load model registry: {e}. Using default models.")
                model_type = st.selectbox(
                    "Model Type",
                    ["LSTM", "XGBoost", "Prophet", "ARIMA", "Ridge", "TCN", "GARCH", "CatBoost", "Ensemble"]
                )
                model_info = {}
            
            st.markdown("---")
            st.markdown(f"**{model_type} Parameters**")
            
            # Model-specific hyperparameters
            params = {}
            
            if model_type == "LSTM":
                params['hidden_dim'] = st.slider("Hidden Dimension", 16, 256, 64, 16)
                params['num_layers'] = st.slider("Number of Layers", 1, 5, 2)
                params['dropout'] = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
                params['learning_rate'] = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
                params['sequence_length'] = st.slider("Sequence Length", 30, 120, 60, 10)
            
            elif model_type == "XGBoost":
                params['n_estimators'] = st.slider("Number of Trees", 50, 500, 100, 50)
                params['max_depth'] = st.slider("Max Depth", 3, 15, 5, 1)
                params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
                params['subsample'] = st.slider("Subsample Ratio", 0.5, 1.0, 0.8, 0.1)
                params['colsample_bytree'] = st.slider("Feature Sampling", 0.5, 1.0, 0.8, 0.1)
            
            elif model_type == "Prophet":
                params['changepoint_prior_scale'] = st.slider("Changepoint Prior Scale", 0.001, 0.5, 0.05, 0.001)
                params['seasonality_prior_scale'] = st.slider("Seasonality Prior Scale", 0.01, 10.0, 10.0, 0.1)
                params['holidays_prior_scale'] = st.slider("Holidays Prior Scale", 0.01, 10.0, 10.0, 0.1)
                params['seasonality_mode'] = st.selectbox("Seasonality Mode", ['additive', 'multiplicative'])
            
            elif model_type == "ARIMA":
                p = st.slider("AR Order (p)", 0, 10, 5, 1)
                d = st.slider("Differencing (d)", 0, 2, 1, 1)
                q = st.slider("MA Order (q)", 0, 10, 0, 1)
                params['order'] = (p, d, q)
                params['use_auto_arima'] = st.checkbox("Use Auto ARIMA", value=True)
            
            elif model_type == "GARCH":
                st.markdown("**GARCH Parameters:**")
                params['p'] = st.slider("p (AR order)", 1, 5, 1, 1)
                params['q'] = st.slider("q (MA order)", 1, 5, 1, 1)
            
            elif model_type == "Autoformer":
                st.markdown("**Autoformer Parameters:**")
                params['seq_len'] = st.slider("Sequence Length", 30, 120, 60, 10)
                params['pred_len'] = st.session_state.forecast_horizon
                params['d_model'] = st.slider("Model Dimension", 64, 512, 128, 32)
            
            elif model_type == "CatBoost":
                st.markdown("**CatBoost Parameters:**")
                params['iterations'] = st.slider("Iterations", 100, 1000, 500, 50)
                params['depth'] = st.slider("Tree Depth", 3, 10, 6, 1)
                params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.1, 0.03, 0.01)
            
            elif model_type == "TCN":
                st.markdown("**TCN Parameters:**")
                params['num_channels'] = [64, 128, 256]  # Can be made configurable
                params['kernel_size'] = st.slider("Kernel Size", 2, 5, 3, 1)
                params['dropout'] = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
            
            elif model_type == "Ridge":
                st.markdown("**Ridge Parameters:**")
                params['alpha'] = st.slider("Regularization (alpha)", 0.01, 10.0, 1.0, 0.1)
            
            st.markdown("---")
            st.subheader("🔧 Feature Engineering")
            
            use_technical = st.checkbox("Add Technical Indicators", value=False)
            indicators = []
            if use_technical:
                indicators = st.multiselect(
                    "Select Indicators",
                    ["SMA", "EMA", "RSI", "MACD", "Bollinger Bands", "ATR"],
                    default=["SMA", "RSI"]
                )
            
            use_lags = st.checkbox("Add Lag Features", value=False)
            lag_periods = []
            if use_lags:
                lag_periods = st.multiselect(
                    "Lag Periods",
                    [1, 2, 3, 5, 7, 14, 21, 30],
                    default=[1, 7, 14]
                )
            
            use_macro_features = st.checkbox(
                "Include macro-economic features",
                value=False,
                help="Add features like interest rates, inflation, GDP growth"
            )
            
            macro_features_list = []
            if use_macro_features:
                macro_features_list = st.multiselect(
                    "Select Macro Features",
                    ["interest_rate", "inflation", "gdp_growth", "unemployment"],
                    default=["interest_rate", "inflation"],
                    help="Choose which macroeconomic indicators to include"
                )
            
            normalize = st.checkbox("Normalize Data", value=True)
            
            train_button = st.button("🚀 Train Model", type="primary")
        
        with col2:
            st.subheader("📊 Results")
            
            if train_button:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Prepare data
                    data = st.session_state.get("forecast_data")
                    if data is None:
                        raise RuntimeError("No forecast_data in session_state; please load data first.")
                    data = data.copy()
                    # Normalize column case (yfinance uses Close, etc.)
                    if "close" in data.columns and "Close" not in data.columns:
                        data["Close"] = data["close"]
                    if "Close" not in data.columns:
                        data["Close"] = data.iloc[:, 0]
                    # Ensure proper column names (capitalize for FeatureEngineering)
                    if "close" in data.columns:
                        data["Close"] = data["close"]
                    if "open" in data.columns:
                        data['Open'] = data['open']
                    if 'high' in data.columns:
                        data['High'] = data['high']
                    if 'low' in data.columns:
                        data['Low'] = data['low']
                    if 'volume' in data.columns:
                        data['Volume'] = data['volume']
                    
                    # Ensure we have required columns
                    if 'Open' not in data.columns:
                        data['Open'] = data['Close']
                    if 'High' not in data.columns:
                        data['High'] = data['Close']
                    if 'Low' not in data.columns:
                        data['Low'] = data['Close']
                    if 'Volume' not in data.columns:
                        data['Volume'] = 1000000
                    
                    # Feature engineering
                    if use_technical:
                        status_text.text("Adding technical indicators...")
                        progress_bar.progress(0.2)
                        fe = FeatureEngineering()
                        
                        if "SMA" in indicators or "EMA" in indicators:
                            ma_features = fe.calculate_moving_averages(data)
                            data = pd.concat([data, ma_features], axis=1)
                        
                        if "RSI" in indicators:
                            rsi_features = fe.calculate_rsi(data)
                            data = pd.concat([data, rsi_features], axis=1)
                        
                        if "MACD" in indicators:
                            macd_features = fe.calculate_macd(data)
                            data = pd.concat([data, macd_features], axis=1)
                        
                        if "Bollinger Bands" in indicators:
                            bb_features = fe.calculate_bollinger_bands(data)
                            data = pd.concat([data, bb_features], axis=1)
                    
                    if use_lags:
                        status_text.text("Adding lag features...")
                        progress_bar.progress(0.4)
                        for lag in lag_periods:
                            data[f'lag_{lag}'] = data['Close'].shift(lag)
                    
                    # Macro feature engineering
                    if use_macro_features and macro_features_list:
                        status_text.text("Adding macro-economic features...")
                        progress_bar.progress(0.5)
                        try:
                            from trading.feature_engineering.macro_feature_engineering import MacroFeatureEngineer
                            
                            engineer = MacroFeatureEngineer()
                            
                            # Add macro features to data (enrich_trading_data adds FRED/World Bank features)
                            data_with_macro = engineer.enrich_trading_data(
                                data,
                                include_sentiment=('sentiment' in (macro_features_list or []))
                            )
                            
                            # Show what was added
                            new_cols = set(data_with_macro.columns) - set(data.columns)
                            if new_cols:
                                st.write("**Macro features added:**")
                                for col in new_cols:
                                    st.write(f"• {col}")
                            
                            # Use data_with_macro for training
                            data = data_with_macro
                        except ImportError:
                            st.warning("Macro Feature Engineer not available. Skipping macro features.")
                        except Exception as e:
                            st.warning(f"Could not add macro features: {e}. Continuing without them.")
                    
                    # Remove NaN
                    data = data.dropna()
                    
                    if len(data) < 30:
                        st.error("Not enough data after feature engineering. Try fewer features or more historical data.")
                        progress_bar.empty()
                        status_text.empty()
                    else:
                        # Normalize
                        if normalize:
                            status_text.text("Normalizing data...")
                            progress_bar.progress(0.6)
                            preprocessor = DataPreprocessor()
                            # Normalize only numeric columns
                            numeric_cols = data.select_dtypes(include=[np.number]).columns
                            for col in numeric_cols:
                                if col != 'Close':  # Don't normalize target yet
                                    data[col] = (data[col] - data[col].mean()) / (data[col].std() + 1e-8)
                        
                        # Train model
                        status_text.text(f"Training {model_type}...")
                        progress_bar.progress(0.8)
                        
                        # Create model using registry
                        try:
                            from trading.models.model_registry import get_registry
                            registry = get_registry()
                            ModelClass = registry.get(model_type)
                            
                            if ModelClass is None:
                                st.error(f"Model {model_type} not available")
                                model = None
                            else:
                                # Create model config
                                model_config = {
                                    "target_column": "close" if "close" in data.columns else "Close",
                                    "feature_columns": list(data.columns) if len(data.columns) > 0 else ["close"]
                                }
                                
                                # Add model-specific parameters
                                if model_type == "LSTM":
                                    model_config.update({
                                        "sequence_length": params.get('sequence_length', 60),
                                        "hidden_size": params.get('hidden_dim', 64),
                                        "num_layers": params.get('num_layers', 2),
                                        "dropout": params.get('dropout', 0.2),
                                        "learning_rate": params.get('learning_rate', 0.001),
                                        "input_size": len(data.columns) if len(data.columns) > 0 else 1
                                    })
                                elif model_type == "XGBoost":
                                    model_config.update({
                                        "n_estimators": params.get('n_estimators', 100),
                                        "max_depth": params.get('max_depth', 5),
                                        "learning_rate": params.get('learning_rate', 0.1),
                                        "subsample": params.get('subsample', 0.8),
                                        "colsample_bytree": params.get('colsample_bytree', 0.8)
                                    })
                                elif model_type == "Prophet":
                                    model_config.update({
                                        "date_column": "ds",
                                        "prophet_params": {
                                            "changepoint_prior_scale": params.get('changepoint_prior_scale', 0.05),
                                            "seasonality_prior_scale": params.get('seasonality_prior_scale', 10.0),
                                            "holidays_prior_scale": params.get('holidays_prior_scale', 10.0),
                                            "seasonality_mode": params.get('seasonality_mode', 'additive')
                                        }
                                    })
                                elif model_type == "ARIMA":
                                    model_config.update({
                                        "order": params.get('order', (5, 1, 0)),
                                        "use_auto_arima": params.get('use_auto_arima', True)
                                    })
                                elif model_type == "GARCH":
                                    model_config.update({
                                        "p": params.get('p', 1),
                                        "q": params.get('q', 1)
                                    })
                                elif model_type == "Autoformer":
                                    model_config.update({
                                        "seq_len": params.get('seq_len', 60),
                                        "pred_len": params.get('pred_len', st.session_state.forecast_horizon),
                                        "d_model": params.get('d_model', 128)
                                    })
                                elif model_type == "CatBoost":
                                    model_config.update({
                                        "iterations": params.get('iterations', 500),
                                        "depth": params.get('depth', 6),
                                        "learning_rate": params.get('learning_rate', 0.03)
                                    })
                                elif model_type == "TCN":
                                    model_config.update({
                                        "num_channels": params.get('num_channels', [64, 128, 256]),
                                        "kernel_size": params.get('kernel_size', 3),
                                        "dropout": params.get('dropout', 0.2)
                                    })
                                elif model_type == "Ridge":
                                    model_config.update({
                                        "alpha": params.get('alpha', 1.0)
                                    })
                                
                                model = ModelClass(model_config)
                        except Exception as e:
                            st.warning(f"Could not use model registry: {e}. Using direct imports.")
                            # Fallback to direct imports
                            model_config = {
                                "target_column": "close" if "close" in data.columns else "Close"
                            }
                            
                            if model_type == "LSTM":
                                model_config.update({
                                    "sequence_length": params.get('sequence_length', 60),
                                    "hidden_size": params.get('hidden_dim', 64),
                                    "num_layers": params.get('num_layers', 2),
                                    "dropout": params.get('dropout', 0.2),
                                    "learning_rate": params.get('learning_rate', 0.001),
                                    "feature_columns": list(data.columns) if len(data.columns) > 0 else ["close"],
                                    "input_size": len(data.columns) if len(data.columns) > 0 else 1
                                })
                                model = LSTMForecaster(model_config)
                            elif model_type == "XGBoost":
                                model_config.update({
                                    "n_estimators": params.get('n_estimators', 100),
                                    "max_depth": params.get('max_depth', 5),
                                    "learning_rate": params.get('learning_rate', 0.1),
                                    "subsample": params.get('subsample', 0.8),
                                    "colsample_bytree": params.get('colsample_bytree', 0.8)
                                })
                                model = XGBoostModel(model_config)
                            elif model_type == "Prophet":
                                model_config.update({
                                    "date_column": "ds",
                                    "prophet_params": {
                                        "changepoint_prior_scale": params.get('changepoint_prior_scale', 0.05),
                                        "seasonality_prior_scale": params.get('seasonality_prior_scale', 10.0),
                                        "holidays_prior_scale": params.get('holidays_prior_scale', 10.0),
                                        "seasonality_mode": params.get('seasonality_mode', 'additive')
                                    }
                                })
                                model = ProphetModel(model_config)
                            elif model_type == "ARIMA":
                                model_config.update({
                                    "order": params.get('order', (5, 1, 0)),
                                    "use_auto_arima": params.get('use_auto_arima', True)
                                })
                                model = ARIMAModel(model_config)
                            else:
                                st.error(f"Model {model_type} not available with direct imports. Please use model registry.")
                                model = None
                        
                        # Train model
                        if model_type == "Prophet":
                            train_df = pd.DataFrame({
                                'ds': data.index,
                                'y': data[model_config["target_column"]]
                            })
                            fit_result = model.fit(train_df)
                        elif model_type == "ARIMA":
                            fit_result = model.fit(data[model_config["target_column"]])
                        else:
                            # Most regression models accept (data, target); some ignore target
                            fit_result = model.fit(data, data[model_config["target_column"]])

                        # Validate that model is actually fitted before forecasting
                        fit_success = True
                        if isinstance(fit_result, dict) and "success" in fit_result:
                            fit_success = bool(fit_result.get("success", True))
                        is_fitted_flag = True
                        if hasattr(model, "is_fitted"):
                            is_fitted_flag = bool(getattr(model, "is_fitted"))
                        elif hasattr(model, "model"):
                            is_fitted_flag = getattr(model, "model") is not None

                        if not (fit_success and is_fitted_flag):
                            detail = ""
                            if isinstance(fit_result, dict):
                                detail = fit_result.get("error") or str(fit_result)
                            st.error(f"{model_type} training did not complete successfully. Details: {detail}")
                            st.stop()
                        
                        # Generate forecast - try with uncertainty if available; support both forecast() and predict() (e.g. HybridModel)
                        progress_bar.progress(0.9)
                        horizon = st.session_state.forecast_horizon
                        if hasattr(model, 'forecast_with_uncertainty'):
                            try:
                                forecast_result = model.forecast_with_uncertainty(data, horizon=horizon, num_samples=100)
                            except Exception as e:
                                st.warning(f"Could not generate forecast with uncertainty: {e}. Using standard forecast.")
                                if hasattr(model, 'forecast'):
                                    forecast_result = model.forecast(data, horizon=horizon)
                                else:
                                    preds = model.predict(data) if hasattr(model, 'predict') else np.array([])
                                    forecast_result = {"forecast": preds[-horizon:].tolist() if len(preds) >= horizon else preds.tolist(), "horizon": horizon}
                        elif hasattr(model, 'forecast'):
                            forecast_result = model.forecast(data, horizon=horizon)
                        elif hasattr(model, 'predict'):
                            preds = np.asarray(model.predict(data))
                            preds = preds.astype("float64").ravel()
                            if preds.size == 0:
                                forecast_result = {"forecast": [], "horizon": horizon}
                            else:
                                tail = preds[-horizon:] if preds.size >= horizon else preds
                                forecast_result = {"forecast": tail.tolist(), "horizon": horizon}
                        else:
                            forecast_result = {"forecast": [], "horizon": horizon}
                        
                        # Store full forecast result for confidence intervals
                        st.session_state.current_forecast_result = forecast_result
                        model = st.session_state.get("current_model_instance", None)
                        st.session_state.current_model_instance = model  # Store model instance for explainability
                        
                        # Postprocess forecast
                        try:
                            from trading.forecasting.forecast_postprocessor import ForecastPostprocessor
                            
                            postprocessor = ForecastPostprocessor()
                            
                            # Extract forecast values (handles all result formats)
                            forecast_vals = _extract_forecast_values(forecast_result)
                            if forecast_vals is not None:
                                forecast_vals = forecast_vals.tolist() if hasattr(forecast_vals, 'tolist') else list(forecast_vals)
                            else:
                                forecast_vals = []
                            
                            # Postprocess forecast
                            processed_forecast = postprocessor.process(
                                forecast=forecast_vals,
                                historical_data=data,
                                apply_smoothing=True,
                                remove_outliers=True,
                                ensure_realistic_bounds=True
                            )
                            
                            # Update forecast_result with processed version
                            if isinstance(forecast_result, dict):
                                forecast_result['forecast'] = processed_forecast['values']
                                forecast_result['postprocessing_notes'] = processed_forecast.get('notes', [])
                            else:
                                forecast_result = processed_forecast['values']
                            
                            # Update session state
                            st.session_state.current_forecast_result = forecast_result
                        except ImportError:
                            pass  # Silently fail if postprocessor not available
                        except Exception as e:
                            logger.warning(f"Forecast postprocessing failed: {e}")
                        
                        progress_bar.progress(1.0)
                        status_text.text("Complete!")
                        
                        st.success("✅ Model trained successfully!")
                        
                        # Extract and validate forecast values (robust for all result formats)
                        forecast_values = _extract_forecast_values(forecast_result)
                        if forecast_values is not None:
                            forecast_values = np.asarray(forecast_values).ravel()
                        if isinstance(forecast_result, dict):
                            forecast_dates = forecast_result.get('dates', pd.date_range(
                                start=data.index[-1] + timedelta(days=1),
                                periods=st.session_state.forecast_horizon,
                                freq='D'
                            ))
                        else:
                            forecast_dates = pd.date_range(
                                start=data.index[-1] + timedelta(days=1),
                                periods=st.session_state.forecast_horizon,
                                freq='D'
                            )

                        if isinstance(forecast_values, (list, np.ndarray)):
                            forecast_array = np.array(
                                [v for v in forecast_values if v is not None],
                                dtype="float64"
                            ).flatten()
                        elif forecast_values is None:
                            forecast_array = np.array([], dtype="float64")
                        else:
                            forecast_array = np.array([float(forecast_values)], dtype="float64")

                        if forecast_array.size == 0:
                            st.error(f"{model_type} returned an empty forecast. Please check the model configuration.")
                            st.stop()

                        # Sanity-check price range relative to last known price
                        # Allow forecasts in the range [0.4 * last_price, 2.0 * last_price]
                        last_price_series = None
                        if "close" in data.columns:
                            last_price_series = data["close"]
                        elif "Close" in data.columns:
                            last_price_series = data["Close"]

                        if last_price_series is not None and not last_price_series.empty:
                            last_price = float(last_price_series.iloc[-1])
                            lower_bound = 0.4 * last_price
                            upper_bound = 2.0 * last_price
                            if not np.all((forecast_array >= lower_bound) & (forecast_array <= upper_bound)):
                                st.error(
                                    f"{model_type} produced forecasts outside the expected range "
                                    f"[{lower_bound:.2f}, {upper_bound:.2f}] based on last close "
                                    f"({last_price:.2f}). Please review the model configuration and data scaling."
                                )
                                st.stop()
                        
                        # Store forecast
                        forecast_df = pd.DataFrame(
                            {'forecast': forecast_array},
                            index=forecast_dates[: len(forecast_array)],
                        )
                        
                        st.session_state.advanced_forecast = forecast_df
                        st.session_state.advanced_model = model_type
                        
                        # Display forecast chart
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data[model_config["target_column"]],
                            mode='lines',
                            name='Historical',
                            line=dict(color='blue', width=2)
                        ))
                        fig.add_trace(go.Scatter(
                            x=forecast_df.index,
                            y=forecast_df['forecast'],
                            mode='lines+markers',
                            name='Forecast',
                            line=dict(color='red', width=2, dash='dash'),
                            marker=dict(size=8)
                        ))
                        fig.update_layout(
                            title=f"{st.session_state.symbol} - Advanced Forecast ({model_type})",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            hovermode='x unified',
                            height=500
                        )
                        st.plotly_chart(fig)
                        
                        # Display forecast table
                        st.markdown("**Forecast Values:**")
                        display_df = forecast_df.copy()
                        try:
                            _idx = pd.to_datetime(display_df.index)
                            if hasattr(_idx, "tz") and _idx.tz is not None:
                                _idx = _idx.tz_localize(None)
                            display_df.index = _idx.strftime("%Y-%m-%d")
                        except Exception:
                            pass
                        display_df["forecast"] = display_df["forecast"].apply(
                            lambda x: f"${x:.2f}"
                            if isinstance(x, (int, float)) and not pd.isna(x)
                            else "—"
                        )
                        display_df.rename(columns={"forecast": "Forecast"}, inplace=True)

                        # Add bounds if available
                        _lb = None
                        _ub = None
                        if isinstance(forecast_result, dict):
                            _lb = forecast_result.get("lower_bound")
                            _ub = forecast_result.get("upper_bound")
                        _n = len(display_df)
                        display_df["Lower Bound"] = (
                            [f"${v:.2f}" for v in _lb[:_n]]
                            if _lb and len(_lb) >= _n
                            else ["—"] * _n
                        )
                        display_df["Upper Bound"] = (
                            [f"${v:.2f}" for v in _ub[:_n]]
                            if _ub and len(_ub) >= _n
                            else ["—"] * _n
                        )

                        st.dataframe(display_df, width="stretch")
                        
                        # Add explainability section
                        st.markdown("---")
                        st.subheader("🔍 Model Explainability")
                        
                        with st.expander("📊 View Feature Importance & Explanations", expanded=False):
                            st.write("Understand what drives the model's predictions")
                            
                            # Display cached explanation (persist across reruns)
                            if st.session_state.get('forecast_explanation_tab2') is not None:
                                cached = st.session_state.forecast_explanation_tab2
                                if cached.get('success') and cached.get('explanation'):
                                    expl_obj = cached['explanation']
                                    if hasattr(expl_obj, 'feature_importance') and expl_obj.feature_importance:
                                        st.write("**📊 Feature Importance:**")
                                        importance_data = expl_obj.feature_importance
                                        if isinstance(importance_data, dict):
                                            importance_df = pd.DataFrame(
                                                importance_data.items(),
                                                columns=['Feature', 'Importance']
                                            ).sort_values('Importance', ascending=False)
                                            fig_imp = px.bar(
                                                importance_df, x='Importance', y='Feature',
                                                orientation='h', title='What Drives the Predictions?',
                                                labels={'Importance': 'Impact on Prediction'}
                                            )
                                            st.plotly_chart(fig_imp)
                                    if hasattr(expl_obj, 'explanation_text') and expl_obj.explanation_text:
                                        st.write("**💬 Plain English Explanation:**")
                                        st.info(expl_obj.explanation_text)
                                if st.button("Clear explanation", key="clear_explanation_tab2"):
                                    st.session_state.forecast_explanation_tab2 = None
                            
                            if st.button("Generate Explanation", key="explain_btn_tab2"):
                                try:
                                    with st.spinner("Analyzing model predictions..."):
                                        from trading.models.forecast_explainability import ForecastExplainability
                                        explainer = ForecastExplainability()
                                        fv = _extract_forecast_values(forecast_result)
                                        forecast_value = float(np.asarray(fv).flat[0]) if fv is not None and np.asarray(fv).size > 0 else float(data[model_config["target_column"]].iloc[-1])
                                        features = data.copy()
                                        target_history = features[model_config["target_column"]] if model_config["target_column"] in features.columns else features.iloc[:, 0]
                                        try:
                                            explanation = explainer.explain_forecast(
                                                forecast_id=f"forecast_{st.session_state.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                                symbol=st.session_state.symbol,
                                                forecast_value=forecast_value,
                                                model=model,
                                                features=features,
                                                target_history=target_history,
                                                horizon=st.session_state.forecast_horizon
                                            )
                                            # Persist explanation in session state so it survives reruns
                                            st.session_state.forecast_explanation_tab2 = {
                                                "success": True,
                                                "explanation": explanation,
                                            }
                                            st.success("✅ Explanation generated successfully!")
                                        except Exception:
                                            st.caption("Model explainability unavailable for this model type.")
                                except ImportError:
                                    st.warning("⚠️ Forecast explainability requires SHAP library")
                                    st.code("pip install shap")
                                except Exception as e:
                                    st.error(f"Error generating explanation: {e}")
                
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                finally:
                    progress_bar.empty()
                    status_text.empty()
            
            # Display previous forecast if exists
            if st.session_state.get('advanced_forecast') is not None:
                st.markdown("---")
                st.markdown(f"**Previous Forecast ({st.session_state.get('advanced_model', 'Unknown')})**")
                prev_forecast = st.session_state.advanced_forecast
                st.dataframe(prev_forecast.tail(10))

with tab3:
    try:
        st.header("AI Model Selection")
        st.markdown("""
        🤖 Let AI analyze your data and recommend the best forecasting model.
        
        The AI considers:
        - Data characteristics (trend, seasonality, volatility)
        - Historical model performance
        - Forecast horizon
        - Computational efficiency
        """)
        
        if st.session_state.get("forecast_data") is None:
            st.warning("⚠️ Please load data first in the Quick Forecast tab")
        else:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📊 Data Analysis")
                
                analyze_button = st.button(
                    "🔍 Analyze Data & Recommend Model",
                    type="primary"
                )
                
                if analyze_button:
                    with st.spinner("AI analyzing data characteristics..."):
                        try:
                            # Use ModelSelectorAgent if available; otherwise fallback to simple recommendation
                            try:
                                from trading.agents.model_selector_agent import ModelSelectorAgent
                                agent = ModelSelectorAgent()
                            except Exception:
                                agent = None
                            
                            # Detect market regime from data
                            fd = st.session_state.get("forecast_data")
                            price_data = fd["close"].values if hasattr(fd, "columns") and "close" in fd.columns else (
                                fd.get("close")
                                if isinstance(fd, dict)
                                else (list(fd.values())[0] if isinstance(fd, dict) and fd else None)
                            )
                            if price_data is None:
                                price_data = np.array([])
                            if hasattr(price_data, 'tolist'):
                                price_data = np.asarray(price_data).ravel()
                            horizon = st.session_state.forecast_horizon
                            
                            # Determine horizon enum
                            from trading.agents.model_selector_agent import ForecastingHorizon, MarketRegime
                            
                            if horizon <= 7:
                                horizon_enum = ForecastingHorizon.SHORT_TERM
                            elif horizon <= 30:
                                horizon_enum = ForecastingHorizon.MEDIUM_TERM
                            else:
                                horizon_enum = ForecastingHorizon.LONG_TERM
                            
                            # Simple market regime detection
                            price_trend = 0.0
                            volatility = 0.0
                            regime_enum = MarketRegime.SIDEWAYS
                            if len(price_data) >= 2:
                                price_arr = np.asarray(price_data).astype(float)
                                price_trend = (price_arr[-1] - price_arr[0]) / (price_arr[0] or 1)
                                volatility = np.nanstd(price_arr)
                                if price_trend > 0.05:
                                    regime_enum = MarketRegime.TRENDING_UP
                                elif price_trend < -0.05:
                                    regime_enum = MarketRegime.TRENDING_DOWN
                                elif volatility > (np.nanmean(price_arr) * 0.1):
                                    regime_enum = MarketRegime.VOLATILE
                                else:
                                    regime_enum = MarketRegime.SIDEWAYS
                            
                            if agent is not None:
                                # Use agent's select_model method
                                selected_model_id, confidence = agent.select_model(
                                    horizon=horizon_enum,
                                    market_regime=regime_enum,
                                    data_length=len(price_data),
                                    required_features=[],
                                    performance_weight=0.6,
                                    capability_weight=0.4
                                )
                                recommendations = agent.get_model_recommendations(
                                    horizon_enum, regime_enum, top_k=3
                                )
                            else:
                                selected_model_id = "xgboost_medium"
                                confidence = 0.75
                                recommendations = [
                                    {"model_type": "xgboost", "model_id": "xgboost_medium", "total_score": 0.8},
                                    {"model_type": "lstm", "model_id": "lstm_medium", "total_score": 0.7},
                                    {"model_type": "prophet", "model_id": "prophet_medium", "total_score": 0.65},
                                ]
                            
                            # Map model types to display names
                            model_name_map = {
                                'lstm': 'LSTM (Deep Learning)',
                                'xgboost': 'XGBoost (Gradient Boosting)',
                                'prophet': 'Prophet (Facebook)',
                                'arima': 'ARIMA (Statistical)',
                                'transformer': 'Transformer (Attention)',
                                'ensemble': 'Ensemble (Multiple Models)'
                            }
                            
                            # Extract model type from selected_model_id
                            model_type = selected_model_id.split('_')[0] if '_' in selected_model_id else selected_model_id
                            display_name = model_name_map.get(model_type.lower(), selected_model_id)
                            
                            # Create recommendation dict
                            recommendation = {
                                'model_name': display_name,
                                'confidence': confidence,
                                'reasoning': f"Selected based on data characteristics and forecast horizon of {horizon} days. Market regime: {regime_enum.value}.",
                                'data_characteristics': {
                                    'Data Points': len(price_data),
                                    'Forecast Horizon': f"{horizon} days",
                                    'Volatility': f"{volatility:.2f}",
                                    'Trend': f"{price_trend*100:.2f}%",
                                    'Market Regime': regime_enum.value
                                },
                                'alternatives': [
                                    {
                                        'model_name': model_name_map.get(rec.get('model_type', '').lower(), rec.get('model_id', '')),
                                        'confidence': rec.get('total_score', 0.5),
                                        'reason': f"Score: {rec.get('total_score', 0):.2f}"
                                    }
                                    for rec in recommendations[:3]
                                ] if recommendations else []
                            }
                            
                            st.session_state.ai_recommendation = recommendation
                        
                        except Exception as e:
                            st.error(f"Tab error: {type(e).__name__}: {e}")
                            import traceback
                            st.code(traceback.format_exc(), language="python")
            
            with col2:
                st.subheader("💡 AI Recommendation")
                
                if st.session_state.get('ai_recommendation'):
                    rec = st.session_state.ai_recommendation
                    
                    # Main recommendation
                    st.success(f"**Recommended Model: {rec['model_name']}**")
                    
                    # Confidence
                    confidence_pct = rec.get('confidence', 0.75) * 100
                    st.metric("Confidence", f"{confidence_pct:.1f}%")
                    
                    # Progress bar for confidence
                    st.progress(rec.get('confidence', 0.75))
                    
                    # Reasoning
                    with st.expander("🧠 Why this model?", expanded=True):
                        st.markdown(rec.get('reasoning', 'Model selected based on data analysis'))
                        
                        if 'data_characteristics' in rec:
                            st.markdown("**Data Characteristics:**")
                            chars = rec['data_characteristics']
                            for key, value in chars.items():
                                st.text(f"• {key}: {value}")
                    
                    # Alternatives
                    if rec.get('alternatives'):
                        with st.expander("🔄 Alternative Models"):
                            for alt in rec['alternatives']:
                                st.markdown(f"**{alt['model_name']}**")
                                st.caption(f"Confidence: {alt.get('confidence', 0)*100:.1f}%")
                                st.caption(alt.get('reason', ''))
                                st.markdown("---")
                    
                    # Action buttons
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        if st.button("✅ Use Recommended Model"):
                            st.session_state.selected_model = rec['model_name']
                            st.success(f"Selected: {rec['model_name']}")
                            st.info("Go to Quick Forecast tab to generate forecast")
                    
                    with col_b:
                        if st.button("🔄 Choose Different Model"):
                            st.session_state.show_override = True
                    
                    if st.session_state.get('show_override'):
                        st.markdown("**Override AI Selection:**")
                        override = st.selectbox(
                            "Select model:",
                            ["LSTM (Deep Learning)", "XGBoost (Gradient Boosting)", 
                             "Prophet (Facebook)", "ARIMA (Statistical)"]
                        )
                        if st.button("Confirm Override"):
                            st.session_state.selected_model = override
                            st.session_state.show_override = False
                            st.success(f"Selected: {override}")
            
            # Hybrid Model Selector Section
            st.markdown("---")
            st.subheader("🎯 Hybrid Model Selection")
            
            st.write("""
            Hybrid selection combines multiple model types and automatically 
            switches between them based on market conditions.
            """)
            
            if st.button("Use Hybrid Selector"):
                try:
                    from trading.forecasting.hybrid_model_selector import HybridModelSelector
                    
                    data = st.session_state.get("forecast_data")
                    selector = HybridModelSelector()
                    
                    with st.spinner("Analyzing market conditions and selecting models..."):
                        # Detect market regime: HybridModelSelector has no detect_market_regime; use MarketAnalyzer or default
                        regime = "neutral"
                        try:
                            from trading.market.market_analyzer import MarketAnalyzer
                            import pandas as pd
                            df = data.copy() if hasattr(data, 'copy') else pd.DataFrame(data)
                            if 'close' in df.columns and 'Close' not in df.columns:
                                df['Close'] = df['close']
                            if 'Close' in df.columns:
                                analyzer = MarketAnalyzer()
                                trend_result = analyzer.detect_market_regime(df, "trend")
                                regime = trend_result.get("regime", "neutral")
                        except Exception as e:
                            st.caption(f"Regime detection fell back to neutral: {e}")
                        
                        st.info(f"📊 Detected market regime: **{regime}**")
                        
                        # HybridModelSelector has select_best_model(model_scores, metric), not select_models_for_regime
                        # Show a simple recommendation based on regime
                        recommended = [
                            {"name": "XGBoost", "reason": "Robust across regimes"},
                            {"name": "LSTM", "reason": "Captures nonlinear patterns"},
                            {"name": "Prophet", "reason": "Good for trend and seasonality"},
                        ]
                        selected_models = recommended
                        
                        st.write("**Recommended models for current conditions:**")
                        for i, model_config in enumerate(selected_models, 1):
                            st.write(f"{i}. {model_config['name']} - {model_config['reason']}")
                        
                        # Store selected models
                        st.session_state.hybrid_selected_models = selected_models
                        st.session_state.hybrid_regime = regime
                        
                        st.caption("To train an ensemble, use the Model Comparison tab and select multiple models.")
                
                except ImportError:
                    st.error("Hybrid Model Selector not available")
                except Exception as e:
                    st.error(f"Error using hybrid selector: {e}")
                    import traceback
                    st.code(traceback.format_exc())

            # Model comparison table — run all registered models via ForecastRouter
            st.markdown("---")
            st.subheader("📋 Model Comparison Table")
            st.caption("Run each registered model and compare MAPE / 7-day forecast. Load data in Quick Forecast first.")
            if st.session_state.get("forecast_data") is not None:
                _hist = st.session_state.get("forecast_data").copy()
                if "Close" not in _hist.columns and "close" in _hist.columns:
                    _hist["Close"] = _hist["close"]
                _horizon = st.session_state.get("forecast_horizon", 7)
                try:
                    from trading.models.forecast_router import ForecastRouter
                    from trading.models.model_registry import get_registry

                    _registry = get_registry()
                    _router = ForecastRouter()
                    _model_names = (
                        _registry.list_models()
                        if hasattr(_registry, "list_models")
                        else list(getattr(_registry, "_models", {}).keys())
                    )
                    if not _model_names:
                        _model_names = list(_router.model_registry.keys())

                    _comparison_rows = []
                    _progress = st.progress(0.0, text="Running model comparison...")

                    for _i, _model_name in enumerate(_model_names):
                        _pct = (_i + 1) / max(len(_model_names), 1)
                        _progress.progress(_pct, text=f"Testing {_model_name}...")
                        try:
                            _result = _router.get_forecast(
                                _hist,
                                model_type=_model_name.lower(),
                                horizon=_horizon,
                                run_walk_forward=False,
                            )
                            _fc = _extract_forecast_values(_result)
                            _fc_valid = _fc is not None and _fc.size > 0
                            _mape = (
                                _result.get("validation_mape")
                                or _result.get("in_sample_mape")
                                or _result.get("mape")
                                or _result.get("score")
                                or _result.get("error_pct")
                            )
                            _last_fc = (
                                round(float(_fc.flat[-1]), 2) if _fc_valid else None
                            )

                            _comparison_rows.append({
                                "Model": _model_name,
                                "Status": "✅" if _fc_valid else "⚠️",
                                "MAPE": (
                                    round(float(_mape), 2)
                                    if _mape is not None
                                    else None
                                ),
                                "7d Forecast": (
                                    f"${_last_fc:.2f}" if _last_fc else "—"
                                ),
                                "Notes": _result.get("error", "") or "",
                            })
                        except Exception as _e:
                            _comparison_rows.append({
                                "Model": _model_name,
                                "Status": "❌",
                                "MAPE": None,
                                "7d Forecast": "—",
                                "Notes": f"{type(_e).__name__}: {str(_e)[:80]}",
                            })

                    _progress.empty()
                    if _comparison_rows:
                        _df = pd.DataFrame(_comparison_rows)
                        st.dataframe(_df, width='stretch')
                        _working = sum(
                            1 for r in _comparison_rows if r["Status"] == "✅"
                        )
                        st.caption(
                            f"{_working}/{len(_comparison_rows)} models produced valid forecasts"
                        )
                except Exception as _e:
                    st.error(f"Comparison failed: {type(_e).__name__}: {_e}")
                    import traceback
                    st.code(traceback.format_exc())

    except Exception as e:
        st.error(f"Tab error: {type(e).__name__}: {e}")
        import traceback
        st.code(traceback.format_exc(), language="python")

with tab4:
    try:
        st.header("Model Comparison")
        st.markdown("Compare multiple models side-by-side on a single chart.")
        if st.session_state.get("forecast_data") is None:
            st.warning("⚠️ Please load data first in the Quick Forecast tab")
        else:
            from trading.models.forecast_router import ForecastRouter
            import plotly.graph_objects as go

            _router = ForecastRouter()
            _hist = st.session_state.get("forecast_data").copy()
            if "Close" not in _hist.columns and "close" in _hist.columns:
                _hist["Close"] = _hist["close"]
            if "close" not in _hist.columns and "Close" in _hist.columns:
                _hist["close"] = _hist["Close"]
            _symbol = st.session_state.get("symbol", "Symbol")

            _models_to_compare = st.multiselect(
                "Select models to compare",
                options=["arima", "xgboost", "lstm", "prophet", "catboost", "ridge", "tcn", "ensemble"],
                default=["arima", "xgboost", "lstm"],
                key="model_comparison_select",
            )
            _horizon = st.slider(
                "Forecast horizon (days)", 5, 30, 7, key="comparison_horizon"
            )

            if st.button("Compare Models", key="run_comparison"):
                _fig = go.Figure()
                _last_price = float(_hist["Close"].iloc[-1])
                _ylen = min(21, len(_hist))
                _y_hist = _hist["Close"].values[-_ylen:].tolist()
                _x_hist = list(range(-_ylen + 1, 1))

                _fig.add_trace(
                    go.Scatter(
                        x=_x_hist,
                        y=_y_hist,
                        mode="lines",
                        name="Historical",
                        line=dict(color="white", width=2),
                    )
                )
                _colors = [
                    "#00D4AA",
                    "#FF6B6B",
                    "#4FC3F7",
                    "#FFA500",
                    "#B39DDB",
                    "#80CBC4",
                    "#FFCC02",
                    "#EF9A9A",
                ]
                for _ci, _mn in enumerate(_models_to_compare):
                    try:
                        _r = _router.get_forecast(
                            _hist, model_type=_mn, horizon=_horizon, run_walk_forward=False
                        )
                        _fc = _extract_forecast_values(_r)
                        if _fc is None or _fc.size == 0:
                            st.caption(f"{_mn}: no forecast output")
                            continue
                        _fc = [float(v) for v in _fc]
                        if not all(
                            _last_price * 0.5 < v < _last_price * 2.0 for v in _fc
                        ):
                            st.caption(f"{_mn}: forecast out of range, skipped")
                            continue
                        _fig.add_trace(
                            go.Scatter(
                                x=list(range(1, len(_fc) + 1)),
                                y=_fc,
                                mode="lines+markers",
                                name=_mn.upper(),
                                line=dict(
                                    color=_colors[_ci % len(_colors)], width=1.5
                                ),
                                marker=dict(size=4),
                            )
                        )
                    except Exception as _e:
                        st.caption(f"{_mn}: {type(_e).__name__}")

                _fig.add_vline(
                    x=0, line_dash="dash", line_color="gray", annotation_text="Today"
                )
                _fig.add_hline(
                    y=_last_price,
                    line_dash="dot",
                    line_color="gray",
                    annotation_text=f"Last: ${_last_price:.2f}",
                )
                _fig.update_layout(
                    title=f"{_symbol} — Model Forecast Comparison",
                    template="plotly_dark",
                    xaxis_title="Days (negative=historical, positive=forecast)",
                    yaxis_title="Price ($)",
                    height=450,
                )
                st.plotly_chart(_fig, width='stretch')
    except Exception as e:
        st.error(f"Tab error: {type(e).__name__}: {e}")
        import traceback
        st.code(traceback.format_exc(), language="python")

with tab5:
    try:
        st.header("📊 Market Analysis")
        st.markdown("Rolling correlation vs SPY and volatility regime. Load data in Quick Forecast first.")
        if st.session_state.get("forecast_data") is None:
            st.warning("⚠️ Please load data first in the Quick Forecast tab")
        else:
            _symbol = st.session_state.get("symbol", "Symbol")
            hist = st.session_state.get("forecast_data").copy()
            if "Close" not in hist.columns and "close" in hist.columns:
                hist["Close"] = hist["close"]

            st.subheader(f"📊 {_symbol} Market Analysis")
            try:
                import yfinance as yf
                import plotly.graph_objects as go
                import numpy as np

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
        st.error(f"Tab error: {type(e).__name__}: {e}")
        import traceback
        st.code(traceback.format_exc(), language="python")

with tab6:
    try:
        st.header("🔗 Multi-Asset Forecasting with Graph Neural Networks")
        st.markdown("""
        GNN models relationships between correlated assets. Perfect for:
        - **Portfolio-level forecasting** - Predict entire portfolio together
        - **Sector analysis** - Model how sector stocks move together
        - **Market contagion** - Understand how shocks spread
        - **Relationship-based predictions** - Use asset correlations for better forecasts
        """)
        
        st.info("💡 GNN requires 3-20 correlated assets. It won't work with single tickers.")
        
        # Multi-ticker input
        st.subheader("📊 Step 1: Select Multiple Assets")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            tickers_input = st.text_area(
                "Enter ticker symbols (one per line)",
                value="AAPL\nMSFT\nGOOGL\nAMZN\nMETA",
                height=150,
                help="Enter 3-20 correlated stocks. Tech stocks, bank stocks, etc."
            )
            
            tickers = [t.strip().upper() for t in tickers_input.split('\n') if t.strip()]
            
            if len(tickers) < 3:
                st.error("❌ GNN requires at least 3 assets")
            elif len(tickers) > 20:
                st.warning("⚠️ Too many assets (max 20). Performance may be slow.")
            else:
                st.success(f"✅ {len(tickers)} assets selected")
        
        with col2:
            st.write("**Settings:**")
            
            correlation_threshold = st.slider(
                "Correlation Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Assets with correlation > this will be connected in graph"
            )
            
            gnn_horizon = st.slider(
                "Forecast Days",
                min_value=1,
                max_value=30,
                value=7
            )
            
            gnn_epochs = st.slider(
                "Training Epochs",
                min_value=20,
                max_value=100,
                value=50,
                help="More epochs = better accuracy but slower"
            )
        
        # Load multi-asset data
        if st.button("📥 Load Multi-Asset Data", type="primary", width='stretch'):
            if len(tickers) < 3:
                st.error("Please select at least 3 assets")
            elif len(tickers) > 20:
                st.error("Please limit to 20 assets for performance")
            else:
                try:
                    with st.spinner(f"Loading data for {len(tickers)} assets..."):
                        from trading.data.data_loader import DataLoader, DataLoadRequest
                        from datetime import datetime, timedelta
                        
                        loader = DataLoader()
                        
                        # Load data for each ticker
                        multi_asset_data = {}
                        failed_tickers = []
                        
                        progress_bar = st.progress(0)
                        
                        for i, ticker in enumerate(tickers):
                            try:
                                request = DataLoadRequest(
                                    ticker=ticker,
                                    start_date=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                                    end_date=datetime.now().strftime("%Y-%m-%d"),
                                    interval="1d"
                                )
                                response = loader.load_market_data(request)
                                
                                if response.success and response.data is not None:
                                    # Get close prices
                                    data = response.data
                                    if 'close' in data.columns:
                                        multi_asset_data[ticker] = data['close']
                                    elif 'Close' in data.columns:
                                        multi_asset_data[ticker] = data['Close']
                                else:
                                    failed_tickers.append(ticker)
                            except Exception as e:
                                logger.error(f"Error loading {ticker}: {e}")
                                failed_tickers.append(ticker)
                            
                            progress_bar.progress((i + 1) / len(tickers))
                        
                        progress_bar.empty()
                        
                        if len(multi_asset_data) < 3:
                            st.error(f"Could not load enough assets. Failed: {', '.join(failed_tickers)}")
                        else:
                            # Combine into DataFrame
                            multi_df = pd.DataFrame(multi_asset_data)
                            
                            # Align dates (drop rows with any NaN)
                            multi_df = multi_df.dropna()
                            
                            if len(multi_df) < 100:
                                st.error("Insufficient overlapping data. Try different tickers or longer date range.")
                            else:
                                st.session_state.gnn_data = multi_df
                                st.session_state.gnn_tickers = list(multi_df.columns)
                                
                                st.success(f"✅ Loaded {len(multi_df)} days of data for {len(multi_df.columns)} assets")
                                
                                if failed_tickers:
                                    st.warning(f"⚠️ Could not load: {', '.join(failed_tickers)}")
                
                except Exception as e:
                    st.error(f"Error loading data: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Display correlation matrix and train GNN
        if 'gnn_data' in st.session_state and st.session_state.gnn_data is not None:
            st.markdown("---")
            st.subheader("📊 Step 2: View Asset Correlations")
            
            multi_df = st.session_state.gnn_data
            
            # Calculate correlation matrix
            corr_matrix = multi_df.corr()
            
            # Plot heatmap
            fig_corr = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                color_continuous_scale='RdBu_r',
                zmin=-1,
                zmax=1,
                title=f"Asset Correlation Matrix ({len(multi_df.columns)} assets)"
            )
            
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr)
            
            # Show graph connections
            num_connections = (corr_matrix.abs() > correlation_threshold).sum().sum() - len(multi_df.columns)
            st.info(f"🔗 Graph will have {num_connections // 2} edges (connections) based on {correlation_threshold:.0%} threshold")
            
            # Generate forecast
            st.markdown("---")
            st.subheader("🚀 Step 3: Generate GNN Forecast")
            
            target_asset = st.selectbox(
                "Select primary asset to forecast",
                options=st.session_state.gnn_tickers,
                help="GNN will forecast this asset using all connected assets"
            )
            
            if st.button("🔮 Train GNN & Generate Forecast", type="primary", width='stretch'):
                try:
                    # Type guards: ensure scalars (sliders can sometimes be dict in edge cases)
                    _gnn_epochs = gnn_epochs
                    _gnn_horizon = gnn_horizon
                    _correlation_threshold = correlation_threshold
                    if isinstance(_gnn_epochs, dict):
                        _gnn_epochs = int(_gnn_epochs.get("value", _gnn_epochs.get("epochs", 50)))
                    else:
                        _gnn_epochs = int(_gnn_epochs)
                    if isinstance(_gnn_horizon, dict):
                        _gnn_horizon = int(_gnn_horizon.get("value", _gnn_horizon.get("horizon", 7)))
                    else:
                        _gnn_horizon = int(_gnn_horizon)
                    if isinstance(_correlation_threshold, dict):
                        _correlation_threshold = float(_correlation_threshold.get("value", _correlation_threshold.get("threshold", 0.5)))
                    else:
                        _correlation_threshold = float(_correlation_threshold)
                    with st.spinner(f"Training Graph Neural Network on {len(multi_df.columns)} assets..."):
                        from trading.models.advanced.gnn.gnn_model import GNNForecaster
                        
                        # Progress indicator
                        progress_text = st.empty()
                        
                        # Initialize GNN
                        progress_text.text("Initializing GNN model...")
                        gnn = GNNForecaster(
                            num_assets=len(multi_df.columns),
                            hidden_size=64,
                            num_layers=2,
                            seq_length=30,
                            correlation_threshold=_correlation_threshold
                        )
                        
                        # Train
                        progress_text.text(f"Training for {_gnn_epochs} epochs...")
                        gnn.fit(multi_df, epochs=_gnn_epochs, batch_size=32)
                        
                        # Generate forecast
                        progress_text.text("Generating forecast...")
                        forecast_result = gnn.forecast(
                            multi_df,
                            horizon=_gnn_horizon,
                            target_asset=target_asset
                        )
                        
                        progress_text.empty()
                        st.success("✅ GNN forecast generated!")
                        
                        # Display forecast chart
                        st.subheader(f"📈 {target_asset} Forecast")
                        
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=multi_df.index,
                            y=multi_df[target_asset],
                            name='Historical',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Forecast
                        fig.add_trace(go.Scatter(
                            x=forecast_result['dates'],
                            y=forecast_result['forecast'],
                            name='GNN Forecast',
                            line=dict(color='red', width=2, dash='dash')
                        ))
                        
                        fig.update_layout(
                            title=f"GNN Multi-Asset Forecast for {target_asset}",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            hovermode='x unified',
                            height=500
                        )
                        
                        st.plotly_chart(fig)
                        
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Forecast Horizon", f"{_gnn_horizon} days")
                        with col2:
                            st.metric("Assets Used", len(multi_df.columns))
                        with col3:
                            avg_conf = forecast_result['confidence'].mean()
                            st.metric("Avg Confidence", f"{avg_conf:.1%}")
                        with col4:
                            last_price = multi_df[target_asset].iloc[-1]
                            forecast_return = ((forecast_result['forecast'][-1] / last_price) - 1) * 100
                            st.metric("Forecast Return", f"{forecast_return:+.2f}%")
                        
                        # Show relationship matrix
                        st.markdown("---")
                        st.subheader("🔗 Learned Asset Relationships")
                        
                        relationship_matrix = gnn.get_relationship_matrix()
                        
                        fig_rel = px.imshow(
                            relationship_matrix,
                            labels=dict(color="Connection Strength"),
                            x=multi_df.columns,
                            y=multi_df.columns,
                            color_continuous_scale='Viridis',
                            title="GNN Asset Relationship Graph (1 = connected, 0 = not connected)"
                        )
                        
                        fig_rel.update_layout(height=500)
                        st.plotly_chart(fig_rel)
                        
                        # Forecast table
                        with st.expander("📋 View Forecast Data"):
                            forecast_df = pd.DataFrame({
                                'Date': forecast_result['dates'],
                                'Forecast': forecast_result['forecast'],
                                'Confidence': forecast_result['confidence']
                            })
                            _lb = forecast_result.get("lower_bound") if isinstance(forecast_result, dict) else None
                            _ub = forecast_result.get("upper_bound") if isinstance(forecast_result, dict) else None
                            _n = len(forecast_df)
                            forecast_df["Lower Bound"] = [
                                f"${v:.2f}" for v in _lb[:_n]
                            ] if _lb and len(_lb) >= _n else ["—"] * _n
                            forecast_df["Upper Bound"] = [
                                f"${v:.2f}" for v in _ub[:_n]
                            ] if _ub and len(_ub) >= _n else ["—"] * _n
                            st.dataframe(forecast_df)
                
                except ImportError:
                    st.error("❌ GNN model not available. Make sure it has been recreated using the prompts.")
                except Exception as e:
                    import traceback
                    st.error(f"GNN error: {e}")
                    with st.expander("Traceback"):
                        st.code(traceback.format_exc(), language="python")
    except Exception as e:
        st.error(f"Tab error: {type(e).__name__}: {e}")
        import traceback
        st.code(traceback.format_exc(), language="python")

with tab7:
    try:
        st.header("🎲 Monte Carlo Price Simulation")
        st.markdown("Self-contained monte_carlo simulation: percentile fan chart and P(price > today). Load data in Quick Forecast first.")
        if st.session_state.get("forecast_data") is None:
            st.warning("⚠️ Please load data first in the Quick Forecast tab")
        else:
            _symbol = st.session_state.get("symbol", "Symbol")
            hist = st.session_state.get("forecast_data").copy()
            if "Close" not in hist.columns and "close" in hist.columns:
                hist["Close"] = hist["close"]

            _mc_sims = st.slider(
                "Simulations", 100, 2000, 500, step=100, key="mc_sims"
            )
            _mc_horizon = st.slider(
                "Horizon (days)", 5, 90, 30, key="mc_horizon"
            )

            if st.button("Run Monte Carlo", key="run_mc"):
                try:
                    import numpy as np
                    import plotly.graph_objects as go

                    _close = hist["Close"].values.astype(float)
                    _returns = np.diff(_close) / _close[:-1]
                    _mu = float(np.mean(_returns))
                    _sigma = float(np.std(_returns))
                    _last = float(_close[-1])

                    _rng = np.random.default_rng(42)
                    _sims = _rng.normal(_mu, _sigma, (_mc_sims, _mc_horizon))
                    _paths = np.zeros((_mc_sims, _mc_horizon + 1))
                    _paths[:, 0] = _last
                    for _t in range(_mc_horizon):
                        _paths[:, _t + 1] = _paths[:, _t] * (1 + _sims[:, _t])

                    _final = _paths[:, -1]
                    _p5, _p25, _p50, _p75, _p95 = np.percentile(
                        _final, [5, 25, 50, 75, 95]
                    )
                    _days = list(range(_mc_horizon + 1))

                    _fig = go.Figure()
                    _p5_path = np.percentile(_paths, 5, axis=0)
                    _p95_path = np.percentile(_paths, 95, axis=0)
                    _p25_path = np.percentile(_paths, 25, axis=0)
                    _p75_path = np.percentile(_paths, 75, axis=0)
                    _median_path = np.percentile(_paths, 50, axis=0)

                    _fig.add_trace(
                        go.Scatter(
                            x=_days,
                            y=_p95_path,
                            mode="lines",
                            line=dict(color="rgba(0,212,170,0.2)"),
                            name="95th pct",
                        )
                    )
                    _fig.add_trace(
                        go.Scatter(
                            x=_days,
                            y=_p5_path,
                            mode="lines",
                            fill="tonexty",
                            fillcolor="rgba(0,212,170,0.1)",
                            line=dict(color="rgba(0,212,170,0.2)"),
                            name="5th pct",
                        )
                    )
                    _fig.add_trace(
                        go.Scatter(
                            x=_days,
                            y=_p75_path,
                            mode="lines",
                            line=dict(color="rgba(0,212,170,0.4)"),
                            name="75th pct",
                        )
                    )
                    _fig.add_trace(
                        go.Scatter(
                            x=_days,
                            y=_p25_path,
                            mode="lines",
                            fill="tonexty",
                            fillcolor="rgba(0,212,170,0.2)",
                            line=dict(color="rgba(0,212,170,0.4)"),
                            name="25th pct",
                        )
                    )
                    _fig.add_trace(
                        go.Scatter(
                            x=_days,
                            y=_median_path,
                            mode="lines",
                            line=dict(color="#00D4AA", width=2.5),
                            name="Median",
                        )
                    )
                    _fig.add_hline(
                        y=_last,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text=f"Today: ${_last:.2f}",
                    )

                    _fig.update_layout(
                        title=f"{_symbol} — {_mc_sims} Path Monte Carlo ({_mc_horizon}d)",
                        template="plotly_dark",
                        height=400,
                        xaxis_title="Days",
                        yaxis_title="Price ($)",
                    )
                    st.plotly_chart(_fig, width='stretch')

                    _c1, _c2, _c3, _c4, _c5 = st.columns(5)
                    _c1.metric(
                        "5th Pct",
                        f"${_p5:.2f}",
                        f"{(_p5 / _last - 1) * 100:+.1f}%",
                    )
                    _c2.metric(
                        "25th Pct",
                        f"${_p25:.2f}",
                        f"{(_p25 / _last - 1) * 100:+.1f}%",
                    )
                    _c3.metric(
                        "Median",
                        f"${_p50:.2f}",
                        f"{(_p50 / _last - 1) * 100:+.1f}%",
                    )
                    _c4.metric(
                        "75th Pct",
                        f"${_p75:.2f}",
                        f"{(_p75 / _last - 1) * 100:+.1f}%",
                    )
                    _c5.metric(
                        "95th Pct",
                        f"${_p95:.2f}",
                        f"{(_p95 / _last - 1) * 100:+.1f}%",
                    )
                    _prob_up = float(np.mean(_final > _last) * 100)
                    st.metric(
                        f"P(price > ${_last:.0f} in {_mc_horizon}d)",
                        f"{_prob_up:.1f}%",
                    )

                except Exception as _e:
                    st.error(
                        f"Monte Carlo error: {type(_e).__name__}: {_e}"
                    )
                    import traceback
                    st.code(traceback.format_exc())
    except Exception as e:
        st.error(f"Tab error: {type(e).__name__}: {e}")
        import traceback
        st.code(traceback.format_exc(), language="python")

with tab_options:
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
                        except Exception:
                            pass

                    _th = _threading.Thread(target=_fetch_chain)
                    _th.start()
                    _th.join(timeout=8)
                    if _result[0] is not None:
                        st.session_state[_cache_key] = _result[0]
                except Exception:
                    pass

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
                    _c_display,
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
                    _p_display,
                    width='stretch',
                    height=300,
                    key="options_puts_table",
                )
        else:
            st.info("No options data available for " + ticker)
    except Exception as _oe:
        st.caption(f"Options data unavailable: {_oe}")

with tab_insider:
    try:
        st.header("Insider Flow")
        if st.session_state.get("forecast_data") is None or not st.session_state.get(
            "symbol"
        ):
            st.info("Load data in the Quick Forecast tab to see insider activity.")
        else:
            symbol = st.session_state.get("symbol")
            insider = get_insider_flow(symbol)

            c1, c2, c3 = st.columns(3)
            c1.metric("Insider Buys (90d)", insider.get("buy_count", 0))
            c2.metric("Insider Sells (90d)", insider.get("sell_count", 0))
            color = {
                "INSIDER_BUYING": "green",
                "INSIDER_SELLING": "red",
                "MIXED": "orange",
            }.get(insider.get("signal"), "gray")
            c3.markdown(
                f"**Signal:** :{color}[{insider.get('signal', 'NO_ACTIVITY').replace('_', ' ').title()}]"
            )

            txns = insider.get("transactions", [])
            if txns:
                st.subheader(f"Recent Insider Transactions for {symbol}")
                df_txn = pd.DataFrame(txns)[
                    [
                        "date",
                        "insider",
                        "title",
                        "transaction_type",
                        "shares",
                        "value",
                        "is_buy",
                    ]
                ]
                st.dataframe(df_txn, width='stretch')
            else:
                st.info("No insider transactions found in the last 90 days.")
    except Exception as e:
        st.error(f"Insider Flow tab error: {e}")

with tab_earnings:
    try:
        from trading.data.earnings_reaction import get_earnings_reactions
        import plotly.graph_objects as go

        symbol = st.session_state.get("symbol") or "AAPL"
        with st.spinner("Loading earnings history..."):
            er = get_earnings_reactions(symbol)

        if er.get("error") and not er["reactions"]:
            st.info(f"Earnings data unavailable: {er['error']}")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg 1-Day Move", f"±{er['avg_move_1d']:.1f}%")
            c2.metric("EPS Beat Rate", f"{er['beat_rate']:.0f}%")
            c3.metric(
                "Positive Reaction",
                f"{er['positive_reaction_rate']:.0f}%",
                help="% of beats where stock rose the next day",
            )
            c4.metric(
                "Typical Range",
                f"{er['typical_range'][0]:+.1f}% to {er['typical_range'][1]:+.1f}%",
            )

            ne = er.get("next_earnings")
            if ne and ne.get("next_earnings_date"):
                days = ne.get("days_until", "?")
                st.info(
                    f"📅 Next earnings: **{ne['next_earnings_date']}** "
                    f"({days} days away) | "
                    f"EPS est: ${ne.get('eps_estimate') or '?'}"
                )

            if er["reactions"]:
                df_r = pd.DataFrame(er["reactions"])
                fig = go.Figure()
                colors = [
                    "#00D4AA" if d == "UP" else "#FF4B4B"
                    for d in df_r["direction"]
                ]
                fig.add_trace(
                    go.Bar(
                        x=df_r["date"],
                        y=df_r["move_1d"],
                        marker_color=colors,
                        text=[
                            f"{v:+.1f}%"
                            for v in df_r["move_1d"].fillna(0)
                        ],
                        textposition="outside",
                        name="1-Day Move",
                    )
                )
                fig.add_hline(y=0, line_color="gray", line_dash="dash")
                fig.update_layout(
                    title=f"{symbol} — Earnings Day Price Reactions",
                    template="plotly_dark",
                    xaxis_title="Earnings Date",
                    yaxis_title="1-Day Price Change (%)",
                    height=350,
                )
                st.plotly_chart(fig, width='stretch')

                with st.expander("Historical earnings detail"):
                    st.dataframe(
                        df_r[
                            [
                                "date",
                                "eps_estimate",
                                "eps_actual",
                                "surprise_pct",
                                "move_1d",
                                "move_3d",
                                "move_5d",
                            ]
                        ],
                        width='stretch',
                    )
    except Exception as e:
        st.error(f"Earnings tab error: {type(e).__name__}: {e}")
        import traceback
        st.code(traceback.format_exc())

with tab_diag:
    st.subheader("Model Diagnostics")
    st.caption("Econometric tests on price data")

    try:
        _dh = st.session_state.get("forecast_data") or get_history(ticker, period="1y")

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

render_page_assistant("Analyze")

