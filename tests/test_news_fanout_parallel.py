# -*- coding: utf-8 -*-
"""Parallel per-mark news/GDELT fan-out for chart annotations."""

from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd
import pytest

from trading.analysis import volume_news_linker as VNL
from trading.analysis.volume_news_linker import (
    LINK_NEWS_TIMEOUT,
    LINK_SAME_DAY,
    annotations_from_rows,
    build_chart_annotations,
)


def _marked_frame(n_marks: int = 5, total: int = 40) -> pd.DataFrame:
    """Daily bars with ``n_marks`` significant sessions needing news I/O."""
    idx = pd.bdate_range(end=date.today() - timedelta(days=3), periods=total)
    df = pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
            "Volume": 1_000_000.0,
            "volume_ratio": 1.0,
            "price_change_pct": 0.0,
            "is_significant": False,
            "candle_type": "bullish",
        },
        index=idx,
    )
    # Space marks so select_chart_event_rows keeps them all
    step = max(1, total // (n_marks + 1))
    for i in range(n_marks):
        pos = step * (i + 1)
        if pos >= total:
            pos = total - 1 - i
        df.iloc[pos, df.columns.get_loc("volume_ratio")] = 3.0
        df.iloc[pos, df.columns.get_loc("price_change_pct")] = 0.03
        df.iloc[pos, df.columns.get_loc("is_significant")] = True
    return df


def test_parallel_fanout_wall_clock_near_one_lookup(monkeypatch):
    """N marks with slow lookups finish near one lookup, not N× serial.

    Implementation note: fan-out uses ``ThreadPoolExecutor`` + ``wait(timeout)``
    (same concurrency idea as ``asyncio.gather`` + ``to_thread``) so the
    wall-clock budget can abandon without waiting on hung GDELT threads.
    """
    n_marks = 5
    lookup_s = 0.40
    call_dates: list[str] = []

    def _slow_news(symbol, date_str, *args, **kwargs):
        call_dates.append(str(date_str)[:10])
        time.sleep(lookup_s)
        return [
            {
                "title": f"Headline for {date_str}",
                "source": "test",
                "link_quality": LINK_SAME_DAY,
                "date_confirmed": True,
            }
        ]

    monkeypatch.setattr(VNL, "get_news_for_date", _slow_news)
    monkeypatch.setattr(VNL, "NEWS_FANOUT_BUDGET_S", 30.0)

    df = _marked_frame(n_marks=n_marks)
    t0 = time.perf_counter()
    anns = build_chart_annotations(
        df, "TEST", max_annotations=n_marks, min_visible=n_marks, include_archives=True
    )
    elapsed = time.perf_counter() - t0

    assert len(call_dates) == n_marks
    assert len(anns) == n_marks
    # Coverage: every mark got a real headline
    assert all(a.get("news") for a in anns)
    assert all(a.get("link_quality") == LINK_SAME_DAY for a in anns)
    # Wall clock ≈ one lookup (parallel), not the serial sum
    serial_floor = lookup_s * n_marks
    assert elapsed < serial_floor * 0.55, (
        f"expected parallel ~{lookup_s:.2f}s, got {elapsed:.2f}s "
        f"(serial would be ~{serial_floor:.2f}s)"
    )
    assert elapsed >= lookup_s * 0.75


def test_fanout_budget_timeout_volume_only_honest(monkeypatch):
    """Pathological hang → volume marks kept + news_lookup_timeout flag."""

    def _hang(symbol, date_str, *args, **kwargs):
        time.sleep(2.0)
        return [{"title": "should not appear", "link_quality": LINK_SAME_DAY}]

    monkeypatch.setattr(VNL, "get_news_for_date", _hang)
    monkeypatch.setattr(VNL, "NEWS_FANOUT_BUDGET_S", 0.25)

    df = _marked_frame(n_marks=4)
    t0 = time.perf_counter()
    anns = build_chart_annotations(
        df, "TEST", max_annotations=4, min_visible=4, include_archives=True
    )
    elapsed = time.perf_counter() - t0

    assert len(anns) == 4  # marks not dropped
    assert elapsed < 1.5  # budget tripped, not 4×2s
    assert all(a.get("link_quality") == LINK_NEWS_TIMEOUT for a in anns)
    assert all(not (a.get("news") or []) for a in anns)
    assert all(a.get("date_confirmed") is False for a in anns)
    assert all(
        "timed out" in str(a.get("hover") or "").lower() for a in anns
    )


def test_deconflict_still_avoids_duplicate_titles(monkeypatch):
    """Parallel fetch + serial claim must not reuse the same headline."""

    def _shared(symbol, date_str, *args, **kwargs):
        return [
            {
                "title": "SHARED MOVE STORY",
                "source": "wire",
                "link_quality": LINK_SAME_DAY,
                "date_confirmed": True,
            },
            {
                "title": f"Unique {date_str}",
                "source": "wire",
                "link_quality": LINK_SAME_DAY,
                "date_confirmed": True,
            },
        ]

    monkeypatch.setattr(VNL, "get_news_for_date", _shared)
    df = _marked_frame(n_marks=3)
    anns = build_chart_annotations(df, "TEST", max_annotations=3, min_visible=3)
    titles = []
    for a in anns:
        for n in a.get("news") or []:
            titles.append(str(n.get("title")))
    assert titles.count("SHARED MOVE STORY") == 1
    assert len(anns) == 3


def test_annotations_from_rows_skip_news_no_network(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("skip_news must not call get_news_for_date")

    monkeypatch.setattr(VNL, "get_news_for_date", _boom)
    idx = pd.bdate_range(end=date.today(), periods=5)
    rows = []
    for i, d in enumerate(idx):
        row = pd.Series(
            {
                "volume_ratio": 3.0,
                "price_change_pct": 0.03,
                "candle_type": "bullish",
                "High": 101.0,
                "close": 100.0,
            }
        )
        rows.append((d, row, "significant"))
    anns = annotations_from_rows(rows, "X", skip_news=True)
    assert len(anns) == 5
    assert all(not a.get("news") for a in anns)
