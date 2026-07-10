# -*- coding: utf-8 -*-
"""Tests for the earnings-reaction window resolution (formerly the flagged
d0 anomaly in the audit tracker).

The reaction day is inferred from overnight gaps: BMO announcements gap
into d0's open, AMC announcements gap into d1's open. Inconclusive gaps
fall back to the legacy conservative window (announcement guaranteed
inside), labeled "unknown". All tests run the real helper on synthetic
price histories - no network.
"""

import numpy as np
import pandas as pd
import pytest

from trading.data.earnings_reaction import _reaction_windows


def _flat_history(n=40, price=100.0, start="2026-01-05"):
    idx = pd.date_range(start, periods=n, freq="B")
    return pd.DataFrame(
        {"Open": price, "High": price, "Low": price, "Close": price,
         "Volume": 1_000_000.0},
        index=idx,
    )


class TestReactionWindows:
    def test_amc_announcement_detected_and_measured(self):
        """Earnings on day X after the close: price jumps at X+1's open.
        d0 (=X) close is the correct baseline; 1-day move is X+1's close
        vs X's close."""
        hist = _flat_history()
        x = 15  # announcement day (a trading day)
        hist.iloc[x + 1:, hist.columns.get_loc("Open")] = 110.0
        hist.iloc[x + 1:, hist.columns.get_loc("Close")] = 110.0
        hist.iloc[x + 1:, hist.columns.get_loc("High")] = 110.0
        hist.iloc[x + 1:, hist.columns.get_loc("Low")] = 110.0

        w = _reaction_windows(hist, hist.index[x])
        assert w["timing"] == "AMC"
        assert w["reaction_date"] == str(hist.index[x + 1].date())
        assert w["move_1d"] == pytest.approx(10.0)
        assert w["move_3d"] == pytest.approx(10.0)  # flat after the jump

    def test_bmo_announcement_detected_and_measured(self):
        """Earnings on day X before the open: price gaps down at X's own
        open. Pre-earnings close is the baseline; 1-day move is X's close
        vs X-1's close."""
        hist = _flat_history()
        x = 15
        hist.iloc[x:, hist.columns.get_loc("Open")] = 90.0
        hist.iloc[x:, hist.columns.get_loc("Close")] = 90.0
        hist.iloc[x:, hist.columns.get_loc("High")] = 90.0
        hist.iloc[x:, hist.columns.get_loc("Low")] = 90.0

        w = _reaction_windows(hist, hist.index[x])
        assert w["timing"] == "BMO"
        assert w["reaction_date"] == str(hist.index[x].date())
        assert w["move_1d"] == pytest.approx(-10.0)

    def test_inconclusive_gaps_fall_back_to_legacy_window(self):
        """No meaningful gap on either side: timing unknown, and the move
        matches the legacy convention exactly (close[d1] vs close[d-1])."""
        hist = _flat_history()
        x = 15
        # Gentle drift, no gap: closes creep up 0.1/day with opens equal
        # to prior closes.
        drift = 100.0 + 0.1 * np.arange(len(hist))
        hist["Close"] = drift
        hist["Open"] = np.r_[drift[0], drift[:-1]]
        hist["High"] = drift + 0.05
        hist["Low"] = drift - 0.05

        w = _reaction_windows(hist, hist.index[x])
        assert w["timing"] == "unknown"
        legacy = (hist["Close"].iloc[x + 1] / hist["Close"].iloc[x - 1] - 1) * 100
        assert w["move_1d"] == pytest.approx(legacy)

    def test_weekend_announcement_date_maps_to_next_session(self):
        """An earnings date falling on a Saturday resolves d0 to Monday;
        a gap into Monday's open reads as BMO relative to Friday's close."""
        hist = _flat_history()
        # Find a Friday, announce Saturday, gap Monday.
        fridays = [i for i, ts in enumerate(hist.index) if ts.weekday() == 4]
        f = fridays[2]
        saturday = hist.index[f] + pd.Timedelta(days=1)
        hist.iloc[f + 1:, hist.columns.get_loc("Open")] = 108.0
        hist.iloc[f + 1:, hist.columns.get_loc("Close")] = 108.0

        w = _reaction_windows(hist, pd.Timestamp(saturday))
        assert w["timing"] == "BMO"
        assert w["reaction_date"] == str(hist.index[f + 1].date())
        assert w["move_1d"] == pytest.approx(8.0)

    def test_missing_open_column_degrades_to_unknown(self):
        hist = _flat_history().drop(columns=["Open"])
        w = _reaction_windows(hist, hist.index[15])
        assert w["timing"] == "unknown"
        assert w["move_1d"] is not None

    def test_insufficient_history_returns_empty(self):
        hist = _flat_history(n=8)
        # Date near the end: fewer than 6 forward sessions.
        assert _reaction_windows(hist, hist.index[5]) == {}
        # Date before any history: no past sessions.
        assert _reaction_windows(
            hist, hist.index[0] - pd.Timedelta(days=5)
        ) == {}
