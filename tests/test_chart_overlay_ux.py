# -*- coding: utf-8 -*-
"""Notable news fill + strategy price overlay series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.analysis.strategy_chart_overlay import price_overlay_series
from trading.analysis.volume_news_linker import (
    build_chart_annotations,
    detect_significant_candles,
)


def test_notable_fills_when_full_spikes_sparse(monkeypatch):
    monkeypatch.setattr(
        "trading.analysis.volume_news_linker.get_news_for_date",
        lambda *a, **k: [],
    )
    idx = pd.bdate_range("2025-01-01", periods=80)
    close = np.full(80, 100.0)
    vol = np.full(80, 1_000_000.0)
    # Spaced so rolling avg stays ~1e6; 1.6× + 1.6% = notable, not full spike
    for i in (15, 30, 45, 60):
        vol[i] = 1_650_000.0
        close[i] = 101.7
        close[i + 1] = 100.0  # snap back so neighbors stay flat
    df = pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": vol},
        index=idx,
    )
    tagged = detect_significant_candles(df)
    anns = build_chart_annotations(tagged, "TEST", max_annotations=8, min_visible=4)
    assert len(anns) >= 4
    assert all(a["tier"] in ("notable", "event_move") for a in anns)
    assert all(a["text"] in ("n", "E") for a in anns)


def test_price_overlay_series_sma():
    idx = pd.bdate_range("2025-01-01", periods=10)
    df = pd.DataFrame(
        {
            "close": np.linspace(100, 110, 10),
            "short_sma": np.linspace(99, 109, 10),
            "long_sma": np.linspace(98, 108, 10),
            "signal": 0,
        },
        index=idx,
    )
    series = price_overlay_series(df, "SMAStrategy")
    assert {s["id"] for s in series} == {"short_sma", "long_sma"}
    assert all(s["style"] == "dotted" for s in series)
    assert all(len(s["points"]) == 10 for s in series)


def test_price_overlay_series_skips_rsi():
    idx = pd.bdate_range("2025-01-01", periods=5)
    df = pd.DataFrame({"close": [1, 2, 3, 4, 5], "rsi": [30, 40, 50, 60, 70]}, index=idx)
    assert price_overlay_series(df, "RSIStrategy") == []
