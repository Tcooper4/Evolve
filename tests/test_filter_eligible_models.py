# -*- coding: utf-8 -*-
"""Phase 2: registry metadata eligibility + honest GNN asset guard.

filter_eligible_models is inclusion only — not weights or ranking.
"""

from __future__ import annotations

import pandas as pd
import pytest

from trading.models.model_registry import ModelRegistry, filter_eligible_models


class _TinyRegistry(ModelRegistry):
    """Skip real model imports — register metadata stubs only."""

    def _register_default_models(self):
        class _Stub:
            pass

        self.register("Ridge", _Stub, {
            "type": "single_asset",
            "complexity": "low",
            "min_data_points": 30,
        })
        self.register("ARIMA", _Stub, {
            "type": "single_asset",
            "complexity": "low",
            "min_data_points": 50,
        })
        self.register("XGBoost", _Stub, {
            "type": "single_asset",
            "complexity": "medium",
            "min_data_points": 50,
        })
        self.register("LSTM", _Stub, {
            "type": "single_asset",
            "complexity": "high",
            "min_data_points": 100,
        })
        self.register("GNN", _Stub, {
            "type": "multi_asset",
            "complexity": "high",
            "min_data_points": 100,
            "min_assets": 3,
            "max_assets": 20,
        })


@pytest.fixture()
def reg():
    return _TinyRegistry()


class TestFilterEligibleModels:
    def test_min_data_points_gate(self, reg):
        # 40 points: Ridge (30) in; ARIMA/XGBoost (50) out
        out = filter_eligible_models({}, available_data_points=40, n_assets=1, registry=reg)
        assert out == ["Ridge"]

    def test_enough_data_includes_single_asset_peers(self, reg):
        out = filter_eligible_models({}, available_data_points=120, n_assets=1, registry=reg)
        assert "Ridge" in out and "ARIMA" in out and "XGBoost" in out and "LSTM" in out
        assert "GNN" not in out  # multi-asset needs ≥3

    def test_gnn_needs_three_assets(self, reg):
        two = filter_eligible_models({}, available_data_points=200, n_assets=2, registry=reg)
        assert "GNN" not in two
        three = filter_eligible_models({}, available_data_points=200, n_assets=3, registry=reg)
        assert "GNN" in three
        # single-asset models remain eligible alongside GNN
        assert "XGBoost" in three

    def test_gnn_max_assets(self, reg):
        out = filter_eligible_models({}, available_data_points=200, n_assets=25, registry=reg)
        assert "GNN" not in out

    def test_max_complexity_ceiling(self, reg):
        out = filter_eligible_models(
            {"max_complexity": "low"},
            available_data_points=200,
            n_assets=1,
            registry=reg,
        )
        assert set(out) == {"ARIMA", "Ridge"}

    def test_phase1_features_do_not_route(self, reg):
        # Eligibility must ignore trend/seasonality — those are Phase 3
        a = filter_eligible_models(
            {"trend_strength": 0.99, "seasonality_strength": 0.01},
            available_data_points=200,
            n_assets=1,
            registry=reg,
        )
        b = filter_eligible_models(
            {"trend_strength": 0.01, "seasonality_strength": 0.9},
            available_data_points=200,
            n_assets=1,
            registry=reg,
        )
        assert a == b

    def test_zero_assets_excludes_all(self, reg):
        assert filter_eligible_models({}, available_data_points=500, n_assets=0, registry=reg) == []


class TestGnnAssetGuard:
    def test_init_rejects_under_three(self):
        torch = pytest.importorskip("torch")
        from trading.models.advanced.gnn.gnn_model import GNNForecaster, TORCH_AVAILABLE
        if not TORCH_AVAILABLE:
            pytest.skip("torch unavailable")
        with pytest.raises(ValueError, match="at least 3 assets"):
            GNNForecaster(num_assets=2)

    def test_fit_rejects_two_column_frame(self):
        torch = pytest.importorskip("torch")
        from trading.models.advanced.gnn.gnn_model import GNNForecaster, TORCH_AVAILABLE
        if not TORCH_AVAILABLE:
            pytest.skip("torch unavailable")
        model = GNNForecaster(num_assets=3)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="at least 3 assets"):
            model.fit(df, epochs=1)

    def test_router_no_longer_inflates_single_ticker(self):
        from trading.models.forecast_router import ForecastRouter

        fr = ForecastRouter()
        # Ensure the gnn branch is reachable even if discovery skipped GNN
        fr.model_registry["gnn"] = object
        df = pd.DataFrame({"Close": [100.0 + i * 0.1 for i in range(80)]})
        with pytest.raises(ValueError, match="at least 3 assets"):
            fr._get_cached_trained_model(df, "gnn", run_walk_forward=False)
