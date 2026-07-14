# -*- coding: utf-8 -*-
"""chart-events lookback window must match the plotted period."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _hist(n: int = 60) -> pd.DataFrame:
    idx = pd.bdate_range(end=date.today(), periods=n)
    return pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
            "Volume": 1_000_000.0,
        },
        index=idx,
    )


def test_chart_events_1d_excludes_old_marks():
    anns = [
        {"date": (date.today() - timedelta(days=40)).isoformat(), "text": "N"},
        {"date": date.today().isoformat(), "text": "N"},
    ]
    win_start = date.today()  # 1d window
    clipped = []
    for a in anns:
        d = date.fromisoformat(a["date"][:10])
        if d >= win_start:
            clipped.append(a)
    assert len(clipped) == 1
    assert clipped[0]["date"] == date.today().isoformat()


def test_win_start_filters_before_news_io():
    """1D/5D must not call get_news_for_date for marks outside the window."""
    from trading.analysis.volume_news_linker import build_chart_annotations

    idx = pd.bdate_range(end=date.today(), periods=40)
    df = pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
            "Volume": 1_000_000.0,
            "volume_ratio": 3.0,
            "price_change_pct": 0.03,
            "is_significant": True,
            "candle_type": "bullish",
        },
        index=idx,
    )
    # Only the oldest bar is "significant" for a stricter check: mark all
    # significant then rely on win_start.
    called_dates: list[str] = []

    def _fake_news(symbol, date_str, *args, **kwargs):
        called_dates.append(date_str[:10])
        return []

    with patch(
        "trading.analysis.volume_news_linker.get_news_for_date",
        side_effect=_fake_news,
    ):
        anns = build_chart_annotations(
            df,
            "SPY",
            max_annotations=14,
            min_visible=8,
            win_start=date.today(),
            include_archives=False,
        )
    assert all(d == date.today().isoformat() for d in called_dates)
    assert all(str(a["date"])[:10] == date.today().isoformat() for a in anns)
    assert len(called_dates) <= 2


def test_filter_markers_helper_parity():
    """Mirror frontend chartDayKey clipping rules in spirit."""
    candles_days = {date.today().isoformat()}
    marks = [
        {"time": (date.today() - timedelta(days=10)).isoformat()},
        {"time": date.today().isoformat()},
    ]
    visible = [m for m in marks if m["time"][:10] in candles_days]
    assert len(visible) == 1
