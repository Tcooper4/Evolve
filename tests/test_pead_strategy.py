# -*- coding: utf-8 -*-
"""Hand-verifiable PEAD trade simulation + predeclared trial lock."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.backtesting.pead_strategy import (
    PREDECLARED_TRIALS,
    PeadParams,
    equity_round_trip_cost_fraction,
    simulate_event_trades,
    simulate_pead_trade_return,
)


class TestPredeclaredTrialLock:
    def test_exactly_four_theory_trials(self):
        assert len(PREDECLARED_TRIALS) == 4
        holds = {t["hold_days"] for t in PREDECLARED_TRIALS}
        lags = {t["entry_lag"] for t in PREDECLARED_TRIALS}
        assert holds == {20, 40}
        assert lags == {0, 1}


class TestSimulatePeadTradeHand:
    def test_known_drift_positive_pnl_sign_and_magnitude(self):
        # 60 business days; reaction at day 10; entry lag 0; hold 20
        idx = pd.bdate_range("2024-01-02", periods=60)
        closes = pd.Series(100.0, index=idx)
        # From entry (day 10) to exit (day 30): +10% drift
        closes.iloc[10:31] = np.linspace(100.0, 110.0, 21)
        closes.iloc[31:] = 110.0
        cost = 0.0034  # fixed for hand check
        out = simulate_pead_trade_return(
            closes,
            idx[10],
            hold_days=20,
            entry_lag=0,
            cost_rt=cost,
        )
        assert out is not None
        assert out["raw_return"] == pytest.approx(0.10, abs=1e-6)
        assert out["pnl_return"] == pytest.approx(0.10 - cost, abs=1e-6)
        assert out["pnl_return"] > 0

    def test_entry_lag_shifts_entry(self):
        idx = pd.bdate_range("2024-01-02", periods=40)
        closes = pd.Series(np.arange(40, dtype=float) + 100.0, index=idx)
        a = simulate_pead_trade_return(
            closes, idx[5], hold_days=5, entry_lag=0, cost_rt=0.0
        )
        b = simulate_pead_trade_return(
            closes, idx[5], hold_days=5, entry_lag=1, cost_rt=0.0
        )
        assert a is not None and b is not None
        assert a["entry_date"] != b["entry_date"]
        assert float(b["entry_px"]) == float(a["entry_px"]) + 1.0

    def test_retail_cost_positive(self):
        assert equity_round_trip_cost_fraction() == pytest.approx(0.0034)


class TestPurgedEventSplitCalendar:
    def test_embargo_is_calendar_not_event_count(self):
        from trading.backtesting.pead_strategy import _purged_event_split

        # 20 events weekly; purge 10 bdays should keep many test events
        events = []
        start = pd.Timestamp("2023-01-02")
        for i in range(20):
            events.append({
                "symbol": "X",
                "reaction_date": str((start + pd.tseries.offsets.BDay(i * 5)).date()),
                "surprise_pct": 1.0,
            })
        train, test, train_end = _purged_event_split(events, purge_days=10, train_frac=0.5)
        assert train_end == 10
        assert len(train) == 10
        assert len(test) >= 5  # calendar purge, not skip 10 events

    def test_pools_positive_surprise_only_via_events(self):
        idx = pd.bdate_range("2024-01-02", periods=50)
        closes = pd.Series(100.0, index=idx)
        # Jump after entry window: entry at idx[10]=100, exit idx[15]=108
        closes.iloc[11:] = 108.0
        events = [
            {
                "symbol": "TEST",
                "reaction_date": str(idx[10].date()),
                "surprise_pct": 5.0,
            }
        ]
        sim = simulate_event_trades(
            events,
            PeadParams(hold_days=5, entry_lag=0),
            closes_by_symbol={"TEST": closes},
            cost_rt=0.0,
        )
        assert sim["n_trades"] == 1
        assert sim["trades"][0]["pnl_return"] > 0
        assert sim["trades"][0]["pnl_return"] == pytest.approx(0.08)