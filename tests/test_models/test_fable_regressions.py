# -*- coding: utf-8 -*-
"""Regression tests for the Fable-session torch and risk-metrics fixes.

Torch-dependent tests skip cleanly when torch is unavailable.
"""

import numpy as np
import pandas as pd
import pytest


class TestRiskMetricsDegeneracy:
    """utils/risk_metrics.compute_performance_metrics previously reported
    Sharpe ≈ -31,500,000 on a flat series (risk-free drag divided by an
    epsilon) and NaN everywhere on an empty series."""

    def _idx(self, n=250):
        return pd.date_range("2025-01-01", periods=n, freq="B")

    def test_flat_series_ratios_are_zero(self):
        from utils.risk_metrics import compute_performance_metrics

        m = compute_performance_metrics(pd.Series(0.0, index=self._idx()))
        assert m.sharpe_ratio == 0.0
        assert m.sortino_ratio == 0.0
        assert m.calmar_ratio == 0.0
        assert m.max_drawdown == 0.0

    def test_empty_and_all_nan_series(self):
        from utils.risk_metrics import compute_performance_metrics

        for s in (pd.Series(dtype=float), pd.Series(np.nan, index=self._idx())):
            m = compute_performance_metrics(s)
            assert m.sharpe_ratio == 0.0
            assert np.isfinite(m.total_return)

    def test_normal_series_matches_manual_math(self):
        from utils.risk_metrics import compute_performance_metrics

        rng = np.random.default_rng(1)
        r = pd.Series(rng.normal(0.0006, 0.01, 250), index=self._idx())
        m = compute_performance_metrics(r)
        daily_rf = 0.05 / 252
        ex = r - daily_rf
        manual_sharpe = float(ex.mean() / ex.std() * np.sqrt(252))
        # Function rounds to 3 decimals on output.
        assert m.sharpe_ratio == pytest.approx(manual_sharpe, abs=5e-4)
        cum = (1 + r).cumprod()
        manual_dd = float((cum / cum.cummax() - 1).min())
        assert m.max_drawdown == pytest.approx(manual_dd, abs=5e-4)


class TestRiskMetricsEngineConsolidation:
    """RiskMetricsEngine now lives in utils/risk_metrics (single canonical
    module); the old trading/backtesting/risk_metrics.py is gone."""

    def test_engine_importable_from_canonical_home(self):
        from utils.risk_metrics import RiskMetric, RiskMetricsEngine

        rng = np.random.default_rng(2)
        r = pd.Series(rng.normal(0.0005, 0.01, 300))
        eng = RiskMetricsEngine()
        metrics = eng.calculate(r)
        assert "sharpe_ratio" in metrics and np.isfinite(metrics["sharpe_ratio"])
        assert eng.get_metric(r, RiskMetric.SHARPE) == metrics["sharpe_ratio"]

    def test_old_module_is_gone(self):
        with pytest.raises(ModuleNotFoundError):
            import trading.backtesting.risk_metrics  # noqa: F401


class TestTransformerEncoder:
    """The Transformer's EncoderWithDropout crashed on every forward pass
    with masking enabled (the default) - nn.Sequential can't take a mask
    kwarg - and reused one encoder-layer instance across all depths
    (shared weights)."""

    def _df(self, n=200):
        rng = np.random.default_rng(11)
        idx = pd.date_range("2024-06-01", periods=n, freq="B")
        close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
        return pd.DataFrame(
            {"open": close, "high": close * 1.004, "low": close * 0.996,
             "close": close, "volume": np.full(n, 3e6)},
            index=idx,
        )

    def test_fit_predict_with_default_masking(self):
        pytest.importorskip("torch")
        from trading.models.advanced.transformer.time_series_transformer import (
            TransformerForecaster,
        )

        m = TransformerForecaster(
            {"sequence_length": 10, "d_model": 16, "nhead": 2,
             "num_layers": 2, "feature_columns": ["close", "volume"],
             "target_column": "close", "epochs": 1, "dropout": 0.1}
        )
        m.fit(self._df(), save_checkpoint=False)
        preds = np.asarray(m.predict(self._df()))
        assert preds.size > 0 and np.isfinite(preds).all()

    def test_encoder_layers_do_not_share_weights(self):
        pytest.importorskip("torch")
        from trading.models.advanced.transformer.time_series_transformer import (
            TransformerForecaster,
        )

        m = TransformerForecaster(
            {"sequence_length": 10, "d_model": 16, "nhead": 2,
             "num_layers": 2, "feature_columns": ["close", "volume"],
             "target_column": "close", "epochs": 1, "dropout": 0.1}
        )
        m.fit(self._df(120), save_checkpoint=False)
        p0 = next(iter(m.transformer_encoder.layers[0].parameters()))
        p1 = next(iter(m.transformer_encoder.layers[1].parameters()))
        assert p0.data_ptr() != p1.data_ptr()
