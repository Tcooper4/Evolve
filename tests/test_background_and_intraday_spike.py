# -*- coding: utf-8 -*-
"""Phase 3 — background jobs + provisional intraday volume spikes."""

from __future__ import annotations

from datetime import datetime, time

import numpy as np
import pandas as pd
import pytest
import pytz

from trading.analysis.intraday_volume_spike import (
    attach_news_honesty,
    evaluate_provisional_spike,
    prior_20d_avg_volume,
)
from trading.analysis.volume_news_linker import meets_spike_thresholds
from trading.services.background_jobs import (
    background_jobs_enabled,
    is_us_equity_session_open,
    username_from_session_id,
)


class TestSpikeThresholdsShared:
    def test_vol_and_move(self):
        assert meets_spike_thresholds(2.0, 0.02) is True
        assert meets_spike_thresholds(2.0, 0.019) is False

    def test_extreme_vol_alone(self):
        assert meets_spike_thresholds(3.0, 0.0) is True
        assert meets_spike_thresholds(2.9, 0.0) is False


class TestBackgroundHelpers:
    def test_kill_switch(self, monkeypatch):
        monkeypatch.setenv("EVOLVE_BACKGROUND_JOBS", "1")
        assert background_jobs_enabled() is True
        monkeypatch.setenv("EVOLVE_BACKGROUND_JOBS", "0")
        assert background_jobs_enabled() is False
        monkeypatch.setenv("EVOLVE_BACKGROUND_JOBS", "false")
        assert background_jobs_enabled() is False

    def test_username_from_session(self):
        assert username_from_session_id("user:alice") == "alice"
        assert username_from_session_id("local") == "local"

    def test_market_hours_weekday_open_vs_weekend(self):
        tz = pytz.timezone("America/New_York")
        # Wednesday 10:00 ET — regular session
        wed = tz.localize(datetime(2026, 7, 8, 10, 0, 0))
        assert is_us_equity_session_open(wed) is True
        # Saturday 10:00 ET
        sat = tz.localize(datetime(2026, 7, 11, 10, 0, 0))
        assert is_us_equity_session_open(sat) is False
        # Wednesday 2:00 ET — overnight
        night = tz.localize(datetime(2026, 7, 8, 2, 0, 0))
        assert is_us_equity_session_open(night) is False


def _make_hist(n_days: int = 25, avg_vol: float = 1_000_000.0) -> pd.DataFrame:
    idx = pd.bdate_range("2026-06-01", periods=n_days, freq="B")
    close = 100 + np.arange(n_days, dtype=float) * 0.1
    return pd.DataFrame(
        {
            "Open": close - 0.2,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": np.full(n_days, avg_vol),
        },
        index=idx,
    )


class TestProvisionalIntradaySpike:
    def test_prior_avg_excludes_today(self):
        hist = _make_hist(25, avg_vol=1_000_000.0)
        # Inflate "today" if present as last row — as_of is day after last bar
        as_of = datetime(2026, 7, 6, 15, 0, 0)  # after last business day in range
        # Force last index date to be "today" relative to as_of
        last = hist.index[-1]
        as_of = datetime.combine(last.date(), time(15, 0))
        hist.loc[hist.index[-1], "Volume"] = 50_000_000.0
        avg = prior_20d_avg_volume(hist, as_of=as_of)
        assert avg is not None
        # Should be ~1e6 from prior days, not polluted by today's 50M
        assert abs(avg - 1_000_000.0) < 1.0

    def test_triggers_on_2x_and_2pct(self):
        hist = _make_hist(25, avg_vol=1_000_000.0)
        as_of = datetime.combine(hist.index[-1].date(), time(15, 0))
        # Exclude using as_of = last bar date → last bar is "today" in hist;
        # evaluator excludes today from baseline, so use volume 2.1M vs prior days.
        event = evaluate_provisional_spike(
            hist,
            live_price=104.0,
            live_volume=2_100_000.0,
            as_of=as_of,
            prev_close=100.0,
        )
        assert event is not None
        assert event["provisional"] is True
        assert event["active"] is True
        assert event["text"] == "LIVE"
        assert event["color"] == "#F5A623"
        assert abs(event["volume_ratio"] - 2.1) < 0.05
        assert abs(event["price_change_pct"] - 0.04) < 1e-9

    def test_no_trigger_below_threshold(self):
        hist = _make_hist(25, avg_vol=1_000_000.0)
        as_of = datetime.combine(hist.index[-1].date(), time(15, 0))
        event = evaluate_provisional_spike(
            hist,
            live_price=100.5,  # +0.5%
            live_volume=1_500_000.0,  # 1.5x — below 2x and below 3x
            as_of=as_of,
            prev_close=100.0,
        )
        assert event is None

    def test_extreme_3x_triggers_without_move(self):
        hist = _make_hist(25, avg_vol=1_000_000.0)
        as_of = datetime.combine(hist.index[-1].date(), time(15, 0))
        event = evaluate_provisional_spike(
            hist,
            live_price=100.1,
            live_volume=3_000_000.0,
            as_of=as_of,
            prev_close=100.0,
        )
        assert event is not None
        assert event["provisional"] is True

    def test_provisional_structurally_distinct(self):
        hist = _make_hist(25)
        as_of = datetime.combine(hist.index[-1].date(), time(15, 0))
        event = evaluate_provisional_spike(
            hist,
            live_price=103.0,
            live_volume=2_200_000.0,
            as_of=as_of,
            day_open=100.0,
        )
        assert event is not None
        # Confirmed EOD markers do not carry provisional=True / LIVE text
        assert event.get("provisional") is True
        assert event.get("text") == "LIVE"
        assert "shape" in event and event["shape"] == "circle"

    def test_news_honesty_fallback_flag(self):
        core = {
            "provisional": True,
            "title": "Provisional live volume spike",
            "text": "LIVE",
        }
        tagged = attach_news_honesty(core, [{
            "title": "Old article",
            "link_quality": "fallback_recent",
            "date_confirmed": False,
        }])
        assert tagged["link_quality"] == "fallback_recent"
        assert tagged["date_confirmed"] is False
        assert "may not be same-day" in tagged["title"].lower()
