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
            st.header("🎲 Monte Carlo Price Simulation")
            st.markdown("Self-contained monte_carlo simulation: percentile fan chart and P(price > today). Load data in Quick Forecast first.")
            if st.session_state.get("analyze_forecast_data") is None:
                st.warning("⚠️ Please load data first in the Quick Forecast tab")
            else:
                _symbol = st.session_state.get("analyze_symbol", "Symbol")
                hist = st.session_state.get("analyze_forecast_data").copy()
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

                        try:
                            from trading.backtesting.monte_carlo import (
                                MonteCarloSimulator,
                            )

                            _mcs = MonteCarloSimulator()
                            _ret_s = pd.Series(_returns)
                            _boot = _mcs.simulate_portfolio_paths(
                                _ret_s,
                                initial_capital=_last,
                                n_simulations=min(500, _mc_sims),
                            )
                            st.caption(
                                "MonteCarloSimulator (historical bootstrap): "
                                f"{_boot.shape[0]} paths × {_boot.shape[1]} steps"
                            )
                        except Exception as _me:
                            st.caption(f"Bootstrap path supplement: {_me}")

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
    except Exception as e:
        import traceback
        st.error(f"Tab error: {type(e).__name__}: {e}")
        st.code(traceback.format_exc())
