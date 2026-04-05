# -*- coding: utf-8 -*-
"""
Analyze Page — Forecasting & Market Analysis.
Thin orchestrator: chart, news strip, and tabbed sections live in components/.
"""

import logging

import streamlit as st

from components.analyze_chart import render_price_chart
from components.analyze_news import render_news_overlay_strip
from components.analyze_tabs_sections import render_tabbed_analyze_sections
from components.theme import inject_theme, render_top_bar, keyboard_shortcut_js
from trading.data.price_cache import get_history
from ui.page_assistant import render_page_assistant

try:
    st.markdown(keyboard_shortcut_js(), unsafe_allow_html=True)
except Exception as _e:
    logging.getLogger(__name__).warning(
        "Analyze: keyboard shortcut JS injection failed: %s", _e
    )
inject_theme()
render_top_bar()

logger = logging.getLogger(__name__)


def _get_forecasting_backend():
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
        logger.warning("Forecasting backend not available: %s", e)
        return None


if "forecasting_backend" not in st.session_state:
    st.session_state.forecasting_backend = _get_forecasting_backend()
_be = st.session_state.get("forecasting_backend")
if not _be:
    st.error("Forecasting backend could not be loaded. Check logs and dependencies.")
    st.stop()

if "analyze_forecast_data" not in st.session_state:
    st.session_state["analyze_forecast_data"] = None
if "analyze_selected_models" not in st.session_state:
    st.session_state["analyze_selected_models"] = []
if "analyze_ai_recommendation" not in st.session_state:
    st.session_state["analyze_ai_recommendation"] = None
if "comparison_results" not in st.session_state:
    st.session_state["comparison_results"] = None
if "analyze_market_regime" not in st.session_state:
    st.session_state["analyze_market_regime"] = None
if "analyze_symbol" not in st.session_state:
    st.session_state["analyze_symbol"] = None
if "analyze_forecast_horizon" not in st.session_state:
    st.session_state["analyze_forecast_horizon"] = 7

c1, c2, c3 = st.columns([2, 2, 2])
with c1:
    st.markdown("### Analyze")
    st.caption("Single-stock deep analysis")
with c2:
    ticker = st.text_input(
        "Ticker",
        value=st.session_state.get("analyze_ticker", "AAPL"),
        key="analyze_ticker",
        label_visibility="collapsed",
        placeholder="Enter ticker...",
    )
with c3:
    trader_mode = st.radio(
        "Mode",
        ["Short-term", "Long-term"],
        horizontal=True,
        key="analyze_trader_mode",
        label_visibility="collapsed",
    )

if not ticker or not ticker.strip():
    ticker = "AAPL"
else:
    ticker = ticker.strip().upper()

period_map = {
    "1D": "1d",
    "5D": "5d",
    "1M": "1mo",
    "3M": "3mo",
    "6M": "6mo",
    "1Y": "1y",
    "5Y": "5y",
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

_cache_key = f"autoloaded_{ticker}_{period}"
_existing_data = st.session_state.get("analyze_forecast_data")
_needs_load = (
    st.session_state.get("_last_autoload_key") != _cache_key
    or _existing_data is None
    or (hasattr(_existing_data, "empty") and _existing_data.empty)
)
if _needs_load:
    try:
        _auto_hist = get_history(ticker, period=period)
        if not _auto_hist.empty:
            st.session_state["analyze_forecast_data"] = _auto_hist
            st.session_state["_last_autoload_key"] = _cache_key
            st.session_state["analyze_symbol"] = ticker
    except Exception as _e:
        logger.warning("Analyze: history load failed for %s: %s", ticker, _e)
        st.caption(f"⚠️ Could not load price history: {_e}")

# Auto-run consensus forecast on page load
# so all tabs are populated without button press
_auto_forecast_key = f"auto_forecast_{ticker}_{period}"
_auto_forecast_ts_key = f"auto_forecast_ts_{ticker}_{period}"
_auto_forecast_age = __import__("time").time() - st.session_state.get(
    _auto_forecast_ts_key, 0
)
_auto_hist = st.session_state.get("analyze_forecast_data")

if (
    st.session_state.get(_auto_forecast_key) is None or _auto_forecast_age > 600
) and _auto_hist is not None and not getattr(_auto_hist, "empty", True):
    try:
        from trading.models.forecast_router import get_router_singleton

        import hashlib as _hl

        _dh = _auto_hist.copy()
        _col_map = {c.lower(): c for c in _dh.columns}
        if "close" not in _dh.columns and "Close" in _dh.columns:
            _dh["close"] = _dh["Close"]
        _data_hash = _hl.md5(str(_dh.shape).encode()).hexdigest()[:8]
        _cons_key = f"consensus_{ticker}_{_data_hash}"
        _cons_ts_key = f"consensus_ts_{ticker}_{_data_hash}"
        _cons_age = __import__("time").time() - st.session_state.get(
            _cons_ts_key, 0
        )
        if st.session_state.get(_cons_key) is None or _cons_age > 600:
            with st.spinner("Loading forecast data..."):
                _router = get_router_singleton()
                _horizon = st.session_state.get("analyze_forecast_horizon", 7)
                _consensus = _router.get_consensus_forecast(
                    _dh,
                    horizon=_horizon,
                    symbol=ticker,
                )
                if _consensus and not _consensus.get("error"):
                    st.session_state[_cons_key] = _consensus
                    st.session_state[_cons_ts_key] = __import__("time").time()
                    st.session_state[_auto_forecast_key] = True
                    st.session_state[_auto_forecast_ts_key] = __import__(
                        "time"
                    ).time()
                    # Populate keys that tabs depend on
                    import pandas as _pd

                    _fc_arr = _consensus.get("consensus_forecast", [])
                    if _fc_arr:
                        _last_date = _dh.index[-1]
                        _fc_dates = _pd.date_range(
                            start=_last_date,
                            periods=len(_fc_arr) + 1,
                            freq="B",
                        )[1:]
                        st.session_state["current_forecast"] = _pd.DataFrame(
                            {"forecast": _fc_arr[: len(_fc_dates)]},
                            index=_fc_dates,
                        )
                        st.session_state["current_model"] = "Consensus"
                        _last_price = float(
                            _dh["close"].iloc[-1]
                            if "close" in _dh.columns
                            else _dh.iloc[-1, 0]
                        )
                        st.session_state["current_forecast_result"] = {
                            "forecast": _fc_arr,
                            "last_actual_price": _last_price,
                            "confidence_label": _consensus.get(
                                "confidence_label", "Medium"
                            ),
                            "warnings": _consensus.get("warnings", []),
                        }
    except Exception as _afe:
        logger.warning(
            "Auto-forecast failed for %s: %s",
            ticker,
            _afe,
        )

_st_ver = tuple(int(x) for x in st.__version__.split(".")[:2])

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
    _tf_label = "1d"

hist = get_history(ticker, period=period, interval=_interval)
if not hist.empty:
    render_price_chart(
        ticker,
        hist,
        period=period,
        period_label=period_label,
        _interval=_interval,
        _tf_label=_tf_label,
        trader_mode=trader_mode,
        st_ver=_st_ver,
    )

render_news_overlay_strip(ticker)

render_tabbed_analyze_sections(
    ticker=ticker,
    hist=hist,
    period=period,
    period_label=period_label,
    trader_mode=trader_mode,
    _interval=_interval,
    _tf_label=_tf_label,
)

render_page_assistant("Analyze")
