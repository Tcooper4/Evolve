# -*- coding: utf-8 -*-
"""Hand-verifiable tests for risk-parity portfolio edge adapter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.research.risk_parity_edge import (
    BASKET,
    HOLD_DAYS,
    LOOKBACK_COV,
    PREDECLARED_TRIALS,
    STEP_SIZE,
    TRIAL_JUSTIFICATION,
    build_portfolio_nav,
    equal_weights,
    portfolio_key,
    portfolio_signal_series,
    risk_parity_weights,
)


class TestTrialLock:
    def test_three_methods_fixed_basket(self):
        assert len(PREDECLARED_TRIALS) == 3
        labels = {t.label for t in PREDECLARED_TRIALS}
        assert labels == {"equal_weight", "risk_parity", "conditional_vol"}
        assert BASKET == ("SPY", "QQQ", "IWM", "EFA", "TLT")
        assert HOLD_DAYS == STEP_SIZE == 21
        assert LOOKBACK_COV == 252
        assert "1/N" in TRIAL_JUSTIFICATION or "equal_weight" in TRIAL_JUSTIFICATION
        assert "DeMiguel" in TRIAL_JUSTIFICATION or "1/N" in TRIAL_JUSTIFICATION


class TestWeightsHand:
    def test_equal_weights_sum_to_one(self):
        w = equal_weights(["A", "B", "C", "D"])
        assert abs(sum(w.values()) - 1.0) < 1e-12
        assert all(abs(v - 0.25) < 1e-12 for v in w.values())

    def test_risk_parity_prefers_low_vol(self):
        rng = np.random.default_rng(0)
        n = 300
        idx = pd.bdate_range("2019-01-02", periods=n)
        # Asset L low vol, H high vol — RP should weight L higher
        low = rng.normal(0.0003, 0.005, n)
        high = rng.normal(0.0003, 0.03, n)
        rets = pd.DataFrame({"L": low, "H": high}, index=idx)
        w = risk_parity_weights(rets)
        assert w["L"] > w["H"]
        assert abs(sum(w.values()) - 1.0) < 1e-6


class TestNavAndSignal:
    def test_nav_positive_and_signal_routes_by_method(self):
        n = 400
        idx = pd.bdate_range("2018-01-02", periods=n)
        rng = np.random.default_rng(1)
        panel = pd.DataFrame(
            {
                "SPY": 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, n)),
                "QQQ": 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n)),
                "IWM": 100 * np.cumprod(1 + rng.normal(0.0003, 0.015, n)),
            },
            index=idx,
        )
        nav = build_portfolio_nav(panel, method="equal_weight", lookback=60, step_size=21)
        assert not nav.empty
        assert (nav["Close"] > 0).all()

        params = {"method": "equal_weight", "lookback": 60, "step_size": 21}
        sig_match = portfolio_signal_series(
            portfolio_key("equal_weight"), nav, params
        )
        sig_miss = portfolio_signal_series(
            portfolio_key("risk_parity"), nav, params
        )
        assert sig_match.dropna().shape[0] >= 3
        assert set(sig_match.dropna().unique()) == {1.0}
        assert sig_miss.dropna().empty
