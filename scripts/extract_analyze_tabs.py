# -*- coding: utf-8 -*-
"""
Split monolithic Analyze tab source into components/tabs/*.py.

Re-extract: save the pre-refactor file as .cache/_analyze_tabs_monolith.py
(see git history), then set SRC below, or temporarily restore the monolith.

Run: .\evolve_venv\Scripts\python.exe scripts\extract_analyze_tabs.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_CACHED = ROOT / ".cache" / "_analyze_tabs_monolith.py"
SRC = _CACHED if _CACHED.exists() else ROOT / "components" / "analyze_tabs_sections.py"
TABS = ROOT / "components" / "tabs"
TABS.mkdir(parents=True, exist_ok=True)

lines = SRC.read_text(encoding="utf-8", errors="replace").splitlines()

# (start, end_exclusive) 0-based indices into `lines` (body inside each `with tab*:`)
SLICES = {
    "tab_quick_forecast": (81, 1619),
    "tab_backtester": (1620, 2376),
    "tab_ai_model_selection": (2377, 2728),
    "tab_model_comparison": (2729, 2834),
    "tab_market_analysis": (2835, 2945),
    "tab_multi_asset_gnn": (2946, 3261),
    "tab_monte_carlo": (3262, 3419),
    "tab_options_chain": (3420, 3610),
    # Insider Flow tab — file named per UX spec (insider / flow signals)
    "tab_scanner_signals": (3611, 3653),
    "tab_earnings": (3654, 3739),
    "tab_diagnostics": (3740, 3841),
    "tab_causal": (3841, 3920),
}

HEADER = '''# -*- coding: utf-8 -*-
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


'''


def undent_one_with_level(block: list[str]) -> str:
    """Remove exactly one Streamlit `with tab:` indent level (4 spaces)."""
    out = []
    for ln in block:
        if len(ln) >= 4 and ln.startswith("    "):
            out.append(ln[4:])
        else:
            out.append(ln)
    return "\n".join(out)


def indent_block(text: str, spaces: int = 4) -> str:
    pad = " " * spaces
    return "\n".join(pad + ln if ln.strip() else "" for ln in text.splitlines())


def fix_causal_path(text: str) -> str:
    return text.replace(
        "_aroot = _Path(__file__).resolve().parent.parent",
        "_aroot = _Path(__file__).resolve().parents[2]",
    )


for name, (a, b) in SLICES.items():
    body_lines = lines[a:b]
    body = undent_one_with_level(body_lines)
    if name == "tab_causal":
        body = fix_causal_path(body)

    inner = indent_block(body, 4)

    fn = f'''def render(
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
'''

    if name == "tab_quick_forecast":
        inner = (
            "        QUICK_FORECAST_MODELS = [\"ARIMA\", \"XGBoost\", \"Ridge\"]\n"
            + inner
        )

    file_txt = (
        HEADER
        + fn
        + inner
        + "\n    except Exception as e:\n"
        + '        st.caption(f"Tab unavailable: {e}")\n'
    )

    out_path = TABS / f"{name}.py"
    out_path.write_text(file_txt, encoding="utf-8")
    print("Wrote", out_path.name, len(file_txt.splitlines()), "lines")

(TABS / "__init__.py").write_text(
    '"""Tab modules for Analyze (see analyze_tabs_sections)."""\n',
    encoding="utf-8",
)
print("Done.")
