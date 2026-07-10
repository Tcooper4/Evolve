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
            from trading.data.earnings_reaction import get_earnings_reactions
            import plotly.graph_objects as go

            symbol = st.session_state.get("analyze_symbol") or "AAPL"
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
                            normalize_for_display(
                                df_r[
                                    [
                                        c
                                        for c in [
                                            "date",
                                            "timing",
                                            "eps_estimate",
                                            "eps_actual",
                                            "surprise_pct",
                                            "move_1d",
                                            "move_3d",
                                            "move_5d",
                                        ]
                                        if c in df_r.columns
                                    ]
                                ]
                            ),
                            width='stretch',
                        )
        except Exception as e:
            st.error(f"Earnings tab error: {type(e).__name__}: {e}")
            import traceback
            st.code(traceback.format_exc())
    except Exception as e:
        st.caption(f"Tab unavailable: {e}")
