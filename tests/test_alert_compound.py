# -*- coding: utf-8 -*-
"""Phase 2 — compound (AND) confirming-factor alerts."""

from __future__ import annotations

import pytest

from trading.services.alert_checker import (
    CONFIRM_RSI_GE,
    CONFIRM_RSI_LE,
    CONFIRM_VOLUME_GE,
    check_alerts_for_user,
    compound_should_fire,
    confirming_factor_met,
    parse_confirm_factor,
)


class TestCompoundTruthTable:
    """Hand-verifiable AND semantics."""

    @pytest.mark.parametrize(
        "primary,confirm_type,confirm_ok,expect",
        [
            (True, None, None, True),          # single-factor: primary alone
            (False, None, None, False),
            (True, CONFIRM_VOLUME_GE, True, True),   # both true
            (True, CONFIRM_VOLUME_GE, False, False),  # only primary
            (False, CONFIRM_VOLUME_GE, True, False),  # only confirm
            (False, CONFIRM_VOLUME_GE, False, False),
            (True, CONFIRM_VOLUME_GE, None, False),   # missing metric = no fire
            (True, CONFIRM_RSI_LE, True, True),
            (True, CONFIRM_RSI_LE, False, False),
        ],
    )
    def test_compound_should_fire(self, primary, confirm_type, confirm_ok, expect):
        assert compound_should_fire(primary, confirm_type, confirm_ok) is expect

    def test_confirming_factor_volume(self):
        ok, extras = confirming_factor_met(
            CONFIRM_VOLUME_GE, 2.0, volume_ratio=2.1
        )
        assert ok is True and extras["volume_ratio"] == 2.1
        ok2, _ = confirming_factor_met(
            CONFIRM_VOLUME_GE, 2.0, volume_ratio=1.5
        )
        assert ok2 is False

    def test_confirming_factor_rsi(self):
        assert confirming_factor_met(CONFIRM_RSI_LE, 30.0, rsi=28.0)[0] is True
        assert confirming_factor_met(CONFIRM_RSI_LE, 30.0, rsi=35.0)[0] is False
        assert confirming_factor_met(CONFIRM_RSI_GE, 70.0, rsi=72.0)[0] is True
        assert confirming_factor_met(CONFIRM_RSI_GE, 70.0, rsi=60.0)[0] is False

    def test_parse_confirm_opt_in_only(self):
        assert parse_confirm_factor({"condition": "price_above"}) is None
        assert parse_confirm_factor({"confirm": "", "confirm_threshold": 2}) is None
        assert parse_confirm_factor(
            {"confirm": "volume_ge", "confirm_threshold": 2.5}
        ) == ("volume_ge", 2.5)


class TestCompoundInChecker:
    def test_both_factors_required_to_fire(self, monkeypatch):
        store = {
            "evolve_alerts": [
                {
                    "id": "c1",
                    "symbol": "SPY",
                    "condition": "price_above",
                    "threshold": 100.0,
                    "confirm": "volume_ge",
                    "confirm_threshold": 2.0,
                    "status": "active",
                }
            ]
        }
        import config.user_store as us
        import trading.data.price_cache as pc
        import trading.services.alert_checker as ac
        from trading.utils.time_utils import MarketHours

        monkeypatch.setattr(us, "load_user_preferences", lambda _s: dict(store))
        monkeypatch.setattr(
            us,
            "save_user_preferences",
            lambda _s, p: (store.clear(), store.update(p)),
        )
        monkeypatch.setattr(pc, "get_quote", lambda _s: {"price": 105.0})
        monkeypatch.setattr(MarketHours, "is_market_open", lambda self, dt=None: True)

        # Primary true, confirm false → no fire
        monkeypatch.setattr(ac, "fetch_volume_ratio", lambda _s, hist=None: 1.2)
        assert check_alerts_for_user("user:test") == []
        assert store["evolve_alerts"][0]["status"] == "active"

        # Both true → fire once
        monkeypatch.setattr(ac, "fetch_volume_ratio", lambda _s, hist=None: 2.5)
        fired = check_alerts_for_user("user:test")
        assert len(fired) == 1
        assert store["evolve_alerts"][0]["status"] == "triggered"

    def test_legacy_alert_without_confirm_still_single_factor(self, monkeypatch):
        store = {
            "evolve_alerts": [
                {
                    "id": "legacy",
                    "symbol": "QQQ",
                    "condition": "price_below",
                    "threshold": 200.0,
                }
            ]
        }
        import config.user_store as us
        import trading.data.price_cache as pc
        import trading.services.alert_checker as ac
        from trading.utils.time_utils import MarketHours

        monkeypatch.setattr(us, "load_user_preferences", lambda _s: dict(store))
        monkeypatch.setattr(
            us,
            "save_user_preferences",
            lambda _s, p: (store.clear(), store.update(p)),
        )
        monkeypatch.setattr(pc, "get_quote", lambda _s: {"price": 150.0})
        monkeypatch.setattr(MarketHours, "is_market_open", lambda self, dt=None: True)
        calls = {"n": 0}

        def _vol(_s, hist=None):
            calls["n"] += 1
            return 0.1

        monkeypatch.setattr(ac, "fetch_volume_ratio", _vol)
        fired = check_alerts_for_user("user:test")
        assert len(fired) == 1
        assert calls["n"] == 0  # no confirm I/O for legacy alerts
