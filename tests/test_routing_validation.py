# -*- coding: utf-8 -*-
"""Phase 3 routing validation + walk-forward purge gap tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.models.routing_validation import (
    CANDIDATE_RULES,
    DEFAULT_MIN_IMPROVEMENT,
    LIVE_FEATURE_ROUTING_ENABLED,
    SymbolRuleResult,
    live_forecast_policy,
    make_synthetic_universe,
    models_for_rule,
    stylized_forecast_fn,
    summarize_universe,
    validate_routing_universe,
)
from trading.validation.walk_forward_utils import WalkForwardValidator


class TestWalkForwardPurge:
    def test_purge_separates_train_and_test(self):
        n = 200
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        close = 100 + np.arange(n, dtype=float)
        df = pd.DataFrame({
            "Open": close, "High": close, "Low": close, "Close": close, "Volume": 1e6,
        }, index=idx)

        import trading.validation.walk_forward_utils as WF

        original = WF.WalkForwardValidator._generate_forecasts

        def _patched(self, train_data, test_data, close_col, horizon):
            assert test_data.index[0] > train_data.index[-1]
            preds, actuals, anchors = [], [], []
            for i in range(0, len(test_data), horizon):
                ctx = train_data if i == 0 else pd.concat([train_data, test_data.iloc[:i]])
                last = float(ctx[close_col].iloc[-1])
                slice_ = test_data.iloc[i: i + horizon][close_col].tolist()
                for j, a in enumerate(slice_):
                    anchors.append(last if j == 0 else float(slice_[j - 1]))
                    preds.append(last)
                    actuals.append(float(a))
            return preds, actuals, anchors

        wfv = WalkForwardValidator(model_name="ridge", symbol="T")
        WF.WalkForwardValidator._generate_forecasts = _patched
        try:
            res = wfv.run(
                df, train_window=60, test_window=20, step_size=20, horizon=5, purge=5,
            )
        finally:
            WF.WalkForwardValidator._generate_forecasts = original

        assert res.windows, "expected at least one purged window"
        for w in res.windows:
            assert w.test_start > w.train_end
        assert res.model_performance.get("purge") == 5


class TestRoutingRulesUnit:
    def test_models_for_rule_noisy_prefers_arima(self):
        rule = next(r for r in CANDIDATE_RULES if r.rule_id == "arima_when_noisy")
        feats = {"trend_strength": 0.1, "noise_entropy": 0.8}
        assert models_for_rule(rule, feats) == ["arima", "ridge"]
        assert models_for_rule(rule, {"trend_strength": 0.9, "noise_entropy": 0.2}) == [
            "arima", "xgboost", "ridge", "catboost", "prophet",
        ]

    def test_synthetic_universe_ledger_and_gate(self):
        series = make_synthetic_universe(n=180, seed=1)
        report = validate_routing_universe(
            series,
            forecast_fn=stylized_forecast_fn,
            min_improvement=DEFAULT_MIN_IMPROVEMENT,
            horizon=5,
            train_window=70,
            test_window=25,
            step_size=20,
        )
        assert report["success"]
        summary = report["summary"]
        assert report["n_symbols"] == 3
        assert len(report["results"]) == 3 * len(CANDIDATE_RULES)
        for rule in summary["rules"]:
            if not rule["broad_win"]:
                assert rule["recommend_always_on"] is False
        assert set(summary["always_on_candidates"]) == {
            r["rule_id"] for r in summary["rules"] if r["recommend_always_on"]
        }


class TestSummarizeGate:
    def test_broad_win_requires_fraction_and_margin(self):
        rows = [
            SymbolRuleResult("A", "r1", 0.5, 0.60, 0.10, True, False, True, "ok"),
            SymbolRuleResult("B", "r1", 0.5, 0.60, 0.10, True, False, True, "ok"),
            SymbolRuleResult("C", "r1", 0.5, 0.40, -0.10, False, True, False, "hurt"),
        ]
        # 2/3 helped, mean delta = 0.033 < 0.05 → no broad win
        s = summarize_universe(rows, min_improvement=0.05, min_help_fraction=0.67)
        r1 = s["rules"][0]
        assert r1["helped"] == 2 and r1["hurt"] == 1
        assert r1["broad_win"] is False
        assert s["always_on_candidates"] == []

        rows2 = [
            SymbolRuleResult(sym, "r1", 0.5, 0.65, 0.15, True, False, True, "ok")
            for sym in ("A", "B", "C")
        ]
        s2 = summarize_universe(rows2, min_improvement=0.05, min_help_fraction=0.67)
        assert s2["rules"][0]["broad_win"] is True
        assert "r1" in s2["always_on_candidates"]


class TestLiveForecastPolicy:
    def _df(self, n: int) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        close = 100 + np.linspace(0, 1, n)
        return pd.DataFrame({"Close": close}, index=idx)

    def test_feature_routing_disabled_by_default(self):
        assert LIVE_FEATURE_ROUTING_ENABLED is False
        policy = live_forecast_policy(self._df(200), n_assets=1)
        assert policy["feature_routing_applied"] is False
        assert policy["active_rule"] is None
        assert policy["always_on_rules"] == []
        assert "No feature-routing rule is live" in policy["justification"]

    def test_short_history_drops_high_min_data_models(self):
        # Ridge min=30, ARIMA/XGBoost min=50 — 40 points keeps Ridge from that set
        policy = live_forecast_policy(
            self._df(40),
            requested_models=["ridge", "arima", "xgboost"],
            n_assets=1,
        )
        assert "ridge" in policy["models"]
        assert "arima" in policy["excluded_ineligible"]
        assert "xgboost" in policy["excluded_ineligible"]

    def test_apply_feature_rules_true_still_noop_without_always_on(self):
        policy = live_forecast_policy(
            self._df(200),
            requested_models=["arima", "xgboost", "ridge"],
            apply_feature_rules=True,
        )
        assert policy["feature_routing_applied"] is False
        assert set(policy["models"]) <= {
            "arima", "xgboost", "ridge", "catboost", "prophet", "garch",
        }
