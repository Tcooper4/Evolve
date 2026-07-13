# -*- coding: utf-8 -*-
"""Hand-verifiable tests for trading.data.options_skew."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from trading.data.options_skew import (
    DATA_DISCLOSURE,
    FLAT_DIFF,
    WING_ELEVATED,
    check_same_day_catalysts,
    classify_skew_shape,
    compute_vertical_skew,
)


def _chain(spot: float, rows: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    rows keys are strikes; values are (call_iv, put_iv).
    """
    strikes = sorted(rows.keys())
    calls = pd.DataFrame({
        "strike": strikes,
        "impliedVolatility": [rows[k][0] for k in strikes],
        "openInterest": [100] * len(strikes),
    })
    puts = pd.DataFrame({
        "strike": strikes,
        "impliedVolatility": [rows[k][1] for k in strikes],
        "openInterest": [100] * len(strikes),
    })
    return calls, puts


class TestClassifyShapeHand:
    def test_flat(self):
        # put=call=atm → flat
        out = classify_skew_shape(0.20, 0.20, 0.20)
        assert out["shape"] == "flat"
        assert abs(out["skew_diff"]) < FLAT_DIFF

    def test_put_smirk(self):
        # OTM put 28%, OTM call 18%, ATM 20% → put skew
        out = classify_skew_shape(0.28, 0.18, 0.20)
        assert out["shape"] == "put_smirk"
        assert out["skew_diff"] == pytest.approx(0.10)

    def test_call_smirk(self):
        out = classify_skew_shape(0.18, 0.28, 0.20)
        assert out["shape"] == "call_smirk"
        assert out["skew_diff"] == pytest.approx(-0.10)

    def test_smile_both_wings_elevated(self):
        # ATM 20%, both wings 24% → smile (diff 0)
        out = classify_skew_shape(0.24, 0.24, 0.20)
        assert out["put_wing"] >= WING_ELEVATED
        assert out["call_wing"] >= WING_ELEVATED
        assert out["shape"] == "smile"


class TestComputeVerticalSkew:
    def test_put_smirk_from_constructed_chain(self):
        """
        Spot=100, ±5% → put target 95, call target 105.
        IV: 95p=0.30, 100=0.20, 105c=0.18 → put_smirk, diff=+0.12
        """
        calls, puts = _chain(100.0, {
            95.0: (0.22, 0.30),
            100.0: (0.20, 0.20),
            105.0: (0.18, 0.22),
        })
        out = compute_vertical_skew(calls, puts, 100.0, otm_pct=0.05)
        assert out["success"] is True
        assert out["put_strike"] == pytest.approx(95.0)
        assert out["call_strike"] == pytest.approx(105.0)
        assert out["put_otm_iv"] == pytest.approx(0.30)
        assert out["call_otm_iv"] == pytest.approx(0.18)
        assert out["skew_diff"] == pytest.approx(0.12)
        assert out["shape"] == "put_smirk"
        assert out["delayed_data"] is True
        assert "OPRA" in out["disclosure"] or "delayed" in out["disclosure"].lower()

    def test_smile_from_constructed_chain(self):
        calls, puts = _chain(100.0, {
            95.0: (0.22, 0.25),
            100.0: (0.20, 0.20),
            105.0: (0.25, 0.22),
        })
        out = compute_vertical_skew(calls, puts, 100.0, otm_pct=0.05)
        assert out["shape"] == "smile"
        assert out["put_otm_iv"] == pytest.approx(0.25)
        assert out["call_otm_iv"] == pytest.approx(0.25)

    def test_flat_from_constructed_chain(self):
        calls, puts = _chain(100.0, {
            95.0: (0.20, 0.205),
            100.0: (0.20, 0.20),
            105.0: (0.202, 0.20),
        })
        out = compute_vertical_skew(calls, puts, 100.0, otm_pct=0.05)
        assert out["shape"] == "flat"
        assert abs(out["skew_diff"]) < FLAT_DIFF


class TestEventContext:
    def test_same_day_earnings_is_event_driven(self):
        def earn(_sym):
            return {
                "next_earnings_date": "2026-07-13",
                "days_until": 0,
            }

        def macro(days_ahead=1):
            return {"success": True, "events": []}

        ctx = check_same_day_catalysts(
            "AAPL",
            as_of=date(2026, 7, 13),
            earnings_fn=earn,
            macro_fn=macro,
        )
        assert ctx["interpretation"] == "event_driven"
        assert ctx["same_day_catalyst"] is True
        assert ctx["catalysts"][0]["type"] == "earnings"

    def test_same_day_fomc_is_event_driven(self):
        def earn(_sym):
            return {"days_until": 10, "next_earnings_date": "2099-01-01"}

        def macro(days_ahead=1):
            return {
                "success": True,
                "events": [{
                    "date": "2026-07-13",
                    "name": "FOMC Meeting",
                    "type": "rates",
                    "days_until": 0,
                }],
            }

        ctx = check_same_day_catalysts(
            "SPY",
            as_of=date(2026, 7, 13),
            earnings_fn=earn,
            macro_fn=macro,
        )
        assert ctx["interpretation"] == "event_driven"
        assert any(c["name"] == "FOMC Meeting" for c in ctx["catalysts"])

    def test_no_catalyst_found(self):
        def earn(_sym):
            return {"days_until": 12, "next_earnings_date": "2026-07-25"}

        def macro(days_ahead=1):
            return {"success": True, "events": []}

        ctx = check_same_day_catalysts(
            "MSFT",
            as_of=date(2026, 7, 13),
            earnings_fn=earn,
            macro_fn=macro,
        )
        assert ctx["interpretation"] == "no_same_day_catalyst_found"
        assert ctx["same_day_catalyst"] is False

    def test_could_not_check_when_both_fail(self):
        def earn(_sym):
            return {"error": "down", "days_until": None, "next_earnings_date": None}

        def macro(days_ahead=1):
            return {"success": False, "events": [], "error": "down"}

        ctx = check_same_day_catalysts(
            "X",
            as_of=date(2026, 7, 13),
            earnings_fn=earn,
            macro_fn=macro,
        )
        assert ctx["interpretation"] == "could_not_check"
        assert "could not check" in ctx["note"].lower()


class TestDisclosure:
    def test_constant_mentions_delayed(self):
        assert "delayed" in DATA_DISCLOSURE.lower()
        assert "OPRA" in DATA_DISCLOSURE
