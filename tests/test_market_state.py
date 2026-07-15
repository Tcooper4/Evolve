# -*- coding: utf-8 -*-
"""Hand-verifiable market-state severity + composite labeling."""

from __future__ import annotations

from trading.analysis.market_state import (
    DISCLOSURE,
    SEVERITY_HIGH,
    compose_market_state,
    event_severity_score,
    get_market_state,
    score_headlines_severity,
)
from trading.data.news_aggregator import source_reputation_factor
from trading.services.alert_push_policy import (
    MODE_WATCH,
    AlertPushRateLimiter,
    should_push_alert_notification,
)
import pytest


class TestEventSeverityHand:
    def test_lone_low_rep_mild_below_corroborated_wire(self):
        """Same |sentiment|; Seeking Alpha alone << multi-reputable wires."""
        mag = 0.40
        low = event_severity_score(
            mag, source_reputation_factor("Seeking Alpha"), corroboration_count=1
        )
        high = event_severity_score(
            mag, source_reputation_factor("Reuters"), corroboration_count=3
        )
        # Exact hand values: rep SA=0.35-0.18=0.17; Reuters=0.35+0.25=0.60
        assert abs(source_reputation_factor("Seeking Alpha") - 0.17) < 1e-9
        assert abs(source_reputation_factor("Reuters") - 0.60) < 1e-9
        assert low == pytest.approx(0.40 * 0.17 * 1.0)
        assert high == pytest.approx(0.40 * 0.60 * 1.5)
        assert high > low * 4  # materially higher

    def test_batch_corroboration_raises_severity(self):
        mild = score_headlines_severity(
            [{"title": "Markets steady", "source": "Motley Fool"}],
            [0.35],
        )
        hot = score_headlines_severity(
            [
                {"title": "Fed emergency", "source": "Reuters"},
                {"title": "Fed emergency", "source": "Bloomberg"},
                {"title": "Fed emergency", "source": "CNBC"},
            ],
            [0.35, 0.35, 0.35],
        )
        assert hot["severity"] > mild["severity"]
        assert hot["corroboration_count"] == 3
        assert mild["corroboration_count"] == 1


class TestComposeMarketState:
    def test_short_gamma_high_news_high_vol_critical_label(self):
        c = compose_market_state(
            gex_regime="short_gamma",
            event_severity=SEVERITY_HIGH,
            volatility_regime="high",
        )
        assert c["level"] == "critical"
        assert "dealers short gamma" in c["label"]
        assert "high-severity breaking news" in c["label"]
        assert "elevated realized vol" in c["label"]
        assert c["push_priority"] is True

    def test_long_gamma_quiet_news_normal_vol_calm(self):
        c = compose_market_state(
            gex_regime="long_gamma",
            event_severity=0.05,
            volatility_regime="medium",
        )
        assert c["level"] == "calm"
        assert "dealers long gamma" in c["label"]
        assert "no major catalysts" in c["label"]
        assert c["push_priority"] is False

    def test_injected_get_market_state_matches_compose(self):
        out = get_market_state(
            "SPY",
            gex={"success": True, "regime_short": "near_flip"},
            headlines=[
                {"title": "CPI shock", "source": "Bloomberg"},
                {"title": "CPI shock", "source": "Reuters"},
            ],
            sentiment_scores=[0.8, 0.8],
            volatility_regime="high",
        )
        assert out["success"] is True
        assert out["predicts_direction"] is False
        assert out["disclosure"] == DISCLOSURE
        assert "near gamma flip" in out["label"]
        assert out["level"] in ("elevated", "critical")


class TestWatchPushPriority:
    def test_elevated_state_allows_watch_push_via_same_limiter(self):
        lim = AlertPushRateLimiter(max_per_symbol_hour=2, window_seconds=3600)
        row = {"symbol": "SPY", "mode": MODE_WATCH}
        # Without priority → watch stays silent
        assert should_push_alert_notification(
            "u", row, limiter=lim, now=1.0, market_state=None
        ) == (False, "watch")
        ok, reason = should_push_alert_notification(
            "u",
            row,
            limiter=lim,
            now=2.0,
            market_state={"level": "elevated", "push_priority": True},
        )
        assert ok is True and reason == "ok_priority"
        # Same limiter consumes quota
        ok2, reason2 = should_push_alert_notification(
            "u",
            row,
            limiter=lim,
            now=3.0,
            market_state={"level": "elevated", "push_priority": True},
        )
        assert ok2 is True and reason2 == "ok_priority"
        ok3, reason3 = should_push_alert_notification(
            "u",
            row,
            limiter=lim,
            now=4.0,
            market_state={"level": "elevated", "push_priority": True},
        )
        assert ok3 is False and reason3 == "rate_limited"
