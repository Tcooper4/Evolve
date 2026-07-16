# -*- coding: utf-8 -*-
"""Hand-verifiable tests for momentum edge adapter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.research.momentum_edge import (
    EQUITY_BASKET,
    ETF_BASKET,
    PREDECLARED_TRIALS,
    TRIAL_JUSTIFICATION,
    formation_return,
    tsmom_signal_series,
)


class TestTrialLock:
    def test_two_formations_only(self):
        assert len(PREDECLARED_TRIALS) == 2
        labels = {t.label for t in PREDECLARED_TRIALS}
        assert labels == {"form_12_1", "form_3_1"}
        assert "Jegadeesh" in TRIAL_JUSTIFICATION
        assert "Moskowitz" in TRIAL_JUSTIFICATION
        assert len(ETF_BASKET) == 3
        assert len(EQUITY_BASKET) == 10


class TestFormationReturnHand:
    def test_skip_month_geometry(self):
        # close grows 1 per day: formation from t-5 to t-2 = 3/0 issues —
        # use explicit levels
        idx = pd.bdate_range("2020-01-02", periods=20)
        close = pd.Series(np.arange(100, 120, dtype=float), index=idx)
        # lookback=5, skip=2 → close[t-2]/close[t-5]-1
        form = formation_return(close, lookback=5, skip=2)
        # at i=10: close[8]/close[5]-1 = 108/105 - 1
        assert form.iloc[10] == pytest.approx(108 / 105 - 1.0)


class TestTSMOMSignal:
    def test_positive_formation_gives_long(self):
        n = 300
        idx = pd.bdate_range("2018-01-02", periods=n)
        # Steady uptrend → positive formation → +1
        close = pd.Series(100 * (1.001 ** np.arange(n)), index=idx)
        hist = pd.DataFrame({"Close": close})
        sig = tsmom_signal_series(
            hist, lookback=63, skip=21, step_size=21, min_bars=80
        )
        fired = sig.dropna()
        assert len(fired) >= 3
        assert set(fired.unique()).issubset({1.0, -1.0})
        assert (fired == 1.0).mean() > 0.8
