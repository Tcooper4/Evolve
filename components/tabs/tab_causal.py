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
        st.markdown("---")
        st.subheader("🔬 Causal Analysis")
        try:
            import sys as _sys
            from pathlib import Path as _Path

            _aroot = _Path(__file__).resolve().parents[2]
            _cap = _aroot / "_archive" / "causal"
            _causal_ok = False
            CausalModel = None
            if _cap.is_dir():
                _p = str(_cap)
                if _p not in _sys.path:
                    _sys.path.insert(0, _p)
                try:
                    from causal_model import CausalModel as _CM

                    CausalModel = _CM
                    _causal_ok = True
                except Exception as _ce:
                    logger.warning("Analyze: causal import failed: %s", _ce)
            if not _causal_ok or CausalModel is None:
                st.info(
                    "Causal analysis module not yet available. "
                    "Install dependencies (e.g. networkx) to enable."
                )
            else:
                st.caption(
                    "Exploratory causal graph from price/volume columns "
                    "(archive module)."
                )
                try:
                    _dh = st.session_state.get("analyze_forecast_data") or get_history(
                        ticker, period="1y"
                    )
                    if _dh is None or _dh.empty:
                        st.caption("Load a symbol to build a causal graph.")
                    else:
                        _feat = _dh.select_dtypes(include=[np.number]).dropna(
                            axis=0, how="any"
                        )
                        if _feat.shape[1] < 2:
                            st.caption("Need at least two numeric series for causal view.")
                        else:
                            _cols = list(_feat.columns[:6])
                            _sub = _feat[_cols].copy()
                            _cmodel = CausalModel()
                            _g = _cmodel.build_causal_graph(
                                _sub,
                                treatment_vars=_cols[:1],
                                outcome_vars=_cols[1:2],
                                confounders=_cols[2:] or None,
                            )
                            st.write(
                                f"Nodes: **{_g.number_of_nodes()}**, "
                                f"edges: **{_g.number_of_edges()}**"
                            )
                            if _g.number_of_edges():
                                _edges = list(_g.edges(data=True))[:20]
                                st.dataframe(
                                    normalize_for_display(
                                        pd.DataFrame(
                                            [
                                                {
                                                    "from": a,
                                                    "to": b,
                                                    "weight": d.get("weight"),
                                                }
                                                for a, b, d in _edges
                                            ]
                                        )
                                    ),
                                    width="stretch",
                                )
                except Exception as _cae:
                    st.caption(f"Causal analysis unavailable: {_cae}")
        except Exception as _cae2:
            st.caption(f"Causal section error: {_cae2}")
    except Exception as e:
        st.caption(f"Tab unavailable: {e}")
