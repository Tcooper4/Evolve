# -*- coding: utf-8 -*-
"""Hand-verifiable options transaction cost magnitudes (%% of mid, not bps)."""

from __future__ import annotations

import pytest

from trading.backtesting.backtester import DEFAULT_SPREAD
from trading.backtesting.cost_model import get_options_retail_cost_config, get_retail_cost_config
from trading.backtesting.options_cost_model import (
    ATM_LIQUID_FLOOR,
    NEAR_EXPIRY_MULT,
    OTM_PER_5PCT,
    equity_vs_options_spread_ratio,
    estimate_option_half_spread,
    modeled_half_spread_fraction,
    observed_half_spread_fraction,
)


class TestObservedBidAsk:
    def test_hand_half_spread_from_quotes(self):
        # bid=1.00, ask=1.10 → mid=1.05, full spread=0.10, half=0.10/(2*1.05)
        half = observed_half_spread_fraction(1.00, 1.10)
        assert half == pytest.approx(0.10 / (2.0 * 1.05))
        assert half > ATM_LIQUID_FLOOR * 0.1  # still pct-of-mid scale


class TestModeledMagnitudes:
    def test_atm_floor_is_percentage_points_not_bps(self):
        assert ATM_LIQUID_FLOOR >= 0.02  # ≥2% of mid
        assert ATM_LIQUID_FLOOR < 0.10
        # vs equity default
        ratio = equity_vs_options_spread_ratio(DEFAULT_SPREAD, ATM_LIQUID_FLOOR)
        assert ratio >= 20.0
        assert ratio == pytest.approx(ATM_LIQUID_FLOOR / DEFAULT_SPREAD)

    def test_atm_modeled(self):
        m = modeled_half_spread_fraction(spot=100.0, strike=100.0, dte=7.0)
        assert m["source"] == "modeled"
        assert m["half_spread_fraction"] == pytest.approx(ATM_LIQUID_FLOOR)
        assert m["round_trip_fraction"] == pytest.approx(2.0 * ATM_LIQUID_FLOOR)

    def test_otm_wider_than_atm(self):
        atm = modeled_half_spread_fraction(spot=100.0, strike=100.0, dte=7.0)
        # 10% OTM → moneyness 0.10 → +2 * OTM_PER_5PCT
        otm = modeled_half_spread_fraction(spot=100.0, strike=110.0, dte=7.0)
        expected = ATM_LIQUID_FLOOR + OTM_PER_5PCT * (0.10 / 0.05)
        assert otm["half_spread_fraction"] == pytest.approx(expected)
        assert otm["half_spread_fraction"] > atm["half_spread_fraction"]

    def test_near_expiry_wider(self):
        calm = modeled_half_spread_fraction(spot=100.0, strike=100.0, dte=10.0)
        zero_dte = modeled_half_spread_fraction(
            spot=100.0, strike=100.0, dte=0.0, minutes_to_close=15.0
        )
        assert zero_dte["near_expiry_or_close"] is True
        assert zero_dte["half_spread_fraction"] == pytest.approx(
            ATM_LIQUID_FLOOR * NEAR_EXPIRY_MULT
        )
        assert zero_dte["half_spread_fraction"] > calm["half_spread_fraction"]


class TestEstimatePrefersObserved:
    def test_bid_ask_beats_model(self):
        out = estimate_option_half_spread(
            spot=100.0,
            strike=100.0,
            bid=2.0,
            ask=2.20,
            dte=0.0,
        )
        assert out["source"] == "observed_bid_ask"
        assert out["half_spread_fraction"] == pytest.approx(
            0.20 / (2.0 * 2.10)
        )

    def test_model_when_no_quotes(self):
        out = estimate_option_half_spread(spot=100.0, strike=105.0, dte=5.0)
        assert out["success"] is True
        assert out["source"] == "modeled"


class TestPresetsUnchangedEquityDefault:
    def test_equity_retail_still_5bps(self):
        assert get_retail_cost_config().spread_rate == pytest.approx(0.0005)
        assert DEFAULT_SPREAD == pytest.approx(0.0005)

    def test_options_preset_uses_atm_floor(self):
        cfg = get_options_retail_cost_config()
        assert cfg.spread_rate == pytest.approx(ATM_LIQUID_FLOOR)
        assert cfg.min_spread == pytest.approx(ATM_LIQUID_FLOOR)
