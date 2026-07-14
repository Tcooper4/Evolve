# -*- coding: utf-8 -*-
"""Hand-verifiable tests for strategy chart overlay markers."""

from __future__ import annotations

import pandas as pd
import pytest

from trading.analysis.strategy_chart_overlay import (
    DEFAULT_STRATEGY_OVERLAY_ENABLED,
    DISCLOSURE,
    LABEL_BUY,
    LABEL_SELL,
    reference_levels_from_last_bar,
    signal_events_from_series,
)


class TestSignalEventsHand:
    def test_known_crossings_emit_buy_then_sell(self):
        # Constructed: flat → buy on day 3 → hold → sell on day 6 → flat
        dates = pd.date_range("2024-01-01", periods=8, freq="B")
        signals = [0, 0, 1, 1, 1, -1, 0, 0]
        closes = [100, 101, 102, 103, 104, 99, 98, 97]
        events = signal_events_from_series(dates, signals, closes)
        assert len(events) == 2
        assert events[0]["side"] == "buy"
        assert events[0]["time"] == "2024-01-03"
        assert events[0]["price"] == pytest.approx(102.0)
        assert events[0]["label"] == LABEL_BUY
        assert events[0]["gamma_tag"] is None
        assert events[1]["side"] == "sell"
        assert events[1]["time"] == "2024-01-08"  # 6th business day
        assert events[1]["price"] == pytest.approx(99.0)
        assert events[1]["label"] == LABEL_SELL

    def test_all_zero_yields_no_events(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        assert signal_events_from_series(dates, [0, 0, 0, 0, 0]) == []

    def test_flip_without_flat_emits_both(self):
        dates = ["2024-02-01", "2024-02-02", "2024-02-03"]
        events = signal_events_from_series(dates, [1, -1, 1], [10, 11, 12])
        assert [e["side"] for e in events] == ["buy", "sell", "buy"]
        assert [e["time"] for e in events] == dates

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="mismatch"):
            signal_events_from_series(["a", "b"], [1])


class TestReferenceLevels:
    def test_bollinger_levels_from_last_bar(self):
        df = pd.DataFrame({
            "signal": [0, 1],
            "upper_band": [110.0, 112.5],
            "middle_band": [100.0, 101.0],
            "lower_band": [90.0, 89.5],
        })
        out = reference_levels_from_last_bar(df, "BollingerStrategy")
        keys = {lvl["key"] for lvl in out["levels"]}
        assert keys == {"upper_band", "middle_band", "lower_band"}
        assert "not a live trade" in (out["note"] or "").lower()


class TestPolicy:
    def test_default_off(self):
        assert DEFAULT_STRATEGY_OVERLAY_ENABLED is False

    def test_disclosure_mentions_research_guide(self):
        assert "research guide" in DISCLOSURE.lower() or "backtest" in DISCLOSURE.lower()
        assert "not trade instructions" in DISCLOSURE.lower()
