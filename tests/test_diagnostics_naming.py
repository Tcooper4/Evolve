# -*- coding: utf-8 -*-
"""Phase 2 — Causal mislabeling fix (Path A: rename to Diagnostics)."""

from __future__ import annotations

import inspect
from pathlib import Path

from trading.analysis import econometric_diagnostics as ed

ROOT = Path(__file__).resolve().parents[1]


class TestPathADiagnosticsHonesty:
    def test_module_doc_no_longer_claims_granger_as_feature(self):
        src = inspect.getsource(ed)
        assert "Granger causality (vs SPY)" not in src
        # Honest negative claim is fine / required
        assert "does **not** implement" in src or "Not included" in src

    def test_constructor_no_benchmark_data(self):
        sig = inspect.signature(ed.EconometricDiagnostics.__init__)
        assert "benchmark_data" not in sig.parameters

    def test_run_all_keys_have_no_granger(self):
        import numpy as np
        import pandas as pd

        idx = pd.bdate_range("2024-01-02", periods=120)
        close = 100 * np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, size=120))
        df = pd.DataFrame({"Close": close, "Volume": 1e6}, index=idx)
        out = ed.EconometricDiagnostics("TEST", df).run_all()
        blob = str(out).lower()
        assert "granger" not in blob

    def test_shipped_ui_and_route_copy_drop_causal_tab_label(self):
        analyze = (ROOT / "web" / "frontend" / "src" / "Analyze.tsx").read_text(
            encoding="utf-8", errors="replace"
        )
        assert ">Causal<" not in analyze
        assert ">Diagnostics<" in analyze
        assert "not Granger" in analyze or "Not Granger" in analyze

        routes = (ROOT / "web" / "backend" / "parity_routes.py").read_text(
            encoding="utf-8", errors="replace"
        )
        assert "/api/diagnostics/{symbol}" in routes
        assert "Not Granger causality" in routes or "not Granger" in routes
        assert "benchmark_data=spy" not in routes
        assert "test_granger_causality" not in routes

        api = (ROOT / "web" / "frontend" / "src" / "api.ts").read_text(
            encoding="utf-8", errors="replace"
        )
        assert "getDiagnostics" in api
        assert "/api/diagnostics/" in api
