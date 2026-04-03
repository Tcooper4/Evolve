# -*- coding: utf-8 -*-
"""One-off: split pages/2_Analyze.py into components (run from repo root)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / ".cache" / "_analyze_split_source.py"
if not SRC.exists():
    SRC = ROOT / "pages" / "2_Analyze.py"
lines = SRC.read_text(encoding="utf-8", errors="replace").splitlines()

# Line numbers are 1-based inclusive
common_slice = lines[111:127]  # _extract_forecast_values
common_slice2 = lines[1091:1375]  # _news_sentiment_score through _generate_recommendation

# Inside outer try: line 199 chart_type … through line 1088 (before except)
chart_body = lines[198:1088]

tabs_body = lines[1421:5282]  # st.markdown("---") through end of tab_diag

# Lines 1379–1418: body inside news try (before except)
news_inner = lines[1378:1418]


def indent(block: list[str,], spaces: int = 4) -> str:
    p = " " * spaces
    return "\n".join(p + (ln if ln.strip() else "") for ln in block)


COMMON_HEADER = '''# -*- coding: utf-8 -*-
"""Shared helpers for Analyze / Deep Dive (extracted from legacy Analyze page)."""
import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


'''

COMMON_FOOTER = '''
'''

chart_header = '''# -*- coding: utf-8 -*-
"""Price chart and short-term / value signal strip for Analyze."""
import logging

import plotly.graph_objects as go
import streamlit as st

from components.analyze_common import _is_english
from components.news_candle_chart import render_news_candle_chart
from trading.data.price_cache import get_history, get_info, get_quote

logger = logging.getLogger(__name__)


def render_price_chart(
    ticker: str,
    hist,
    *,
    period: str,
    period_label: str,
    _interval: str,
    _tf_label: str,
    trader_mode: str,
    st_ver: tuple,
) -> None:
    """Full chart UI (candle/line/area/news, intraday live fragment, pattern detector)."""
    if hist is None or hist.empty:
        st.caption("No price history for this symbol and period.")
        return
    try:
'''

chart_footer = '''
    except Exception as e:
        st.caption(f"unavailable: {e}")
'''

tabs_header = '''# -*- coding: utf-8 -*-
"""
All Analyze page tabs (legacy body). Kept sequential for Streamlit tab context.
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


def render_tabbed_analyze_sections(
    ticker: str,
    hist,
    period: str,
    period_label: str,
    trader_mode: str,
    _interval: str,
    _tf_label: str,
) -> None:
    _be = st.session_state.get("forecasting_backend")
    if not _be:
        st.error("Forecasting backend could not be loaded.")
        return
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
'''

tabs_footer = "\n"

news_header = '''# -*- coding: utf-8 -*-
"""News headline sentiment strip + Deep Dive news."""
import streamlit as st
from trading.data.price_cache import get_news


def render_analyze_headline_news_panel(ticker: str) -> None:
    """Hot/pos/neg labels for recent headlines (legacy Analyze strip)."""
    try:
'''

news_footer = '''
    except Exception as e:
        st.caption(f"unavailable: {e}")
'''

# --- write analyze_common.py
common_txt = COMMON_HEADER + "\n".join(common_slice) + "\n\n" + "\n".join(common_slice2) + COMMON_FOOTER
(ROOT / "components" / "analyze_common.py").write_text(common_txt, encoding="utf-8")

# --- write analyze_chart.py
chart_txt = chart_header + "\n".join(chart_body) + chart_footer
(ROOT / "components" / "analyze_chart.py").write_text(chart_txt, encoding="utf-8")

# --- write analyze_tabs_sections.py (name avoids user alias conflict)
tabs_txt = tabs_header + indent(tabs_body, 4) + tabs_footer
(ROOT / "components" / "analyze_tabs_sections.py").write_text(tabs_txt, encoding="utf-8")

# --- write analyze_news_headline.py
news_txt = news_header + indent(news_inner, 8) + news_footer
(ROOT / "components" / "analyze_news_headline.py").write_text(news_txt, encoding="utf-8")

print("Wrote analyze_common.py, analyze_chart.py, analyze_tabs_sections.py, analyze_news_headline.py")
