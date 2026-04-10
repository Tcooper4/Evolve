# -*- coding: utf-8 -*-
"""Insider Flow tab (legacy `with tab_insider:`). Named tab_scanner_signals in module layout."""
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
            st.header("Insider Flow")
            if st.session_state.get("analyze_forecast_data") is None or not st.session_state.get(
                "analyze_symbol"
            ):
                st.info(
                    "Open the **Forecast** section above to load price data first."
                )
            else:
                symbol = st.session_state.get("analyze_symbol")
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
                    st.dataframe(normalize_for_display(df_txn), width='stretch')
                else:
                    st.info("No insider transactions found in the last 90 days.")
        except Exception as e:
            st.error(f"Insider Flow tab error: {e}")
    except Exception as e:
        st.caption(f"Tab unavailable: {e}")
