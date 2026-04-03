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
        try:
            st.header("Model Comparison")
            st.markdown("Compare multiple models side-by-side on a single chart.")
            if st.session_state.get("analyze_forecast_data") is None:
                st.warning("⚠️ Please load data first in the Quick Forecast tab")
            else:
                from trading.models.forecast_router import ForecastRouter
                import plotly.graph_objects as go

                _router = ForecastRouter()
                _hist = st.session_state.get("analyze_forecast_data").copy()
                if "Close" not in _hist.columns and "close" in _hist.columns:
                    _hist["Close"] = _hist["close"]
                if "close" not in _hist.columns and "Close" in _hist.columns:
                    _hist["close"] = _hist["Close"]
                _symbol = st.session_state.get("analyze_symbol", "Symbol")

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
    except Exception as e:
        st.caption(f"Tab unavailable: {e}")
