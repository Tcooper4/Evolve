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
        # The honesty guarantee ("not causal analysis") now comes from the
        # backend's real disclosure field (see test_diagnostics_disclosure_
        # field_present_and_honest below) and is rendered dynamically -
        # Analyze.tsx just needs to actually render whatever the backend
        # supplies, not hardcode the disclaimer as static text.
        assert "diagnostics.disclosure" in analyze

        routes = (ROOT / "web" / "backend" / "parity_routes.py").read_text(
            encoding="utf-8", errors="replace"
        )
        assert "/api/diagnostics/{symbol}" in routes
        assert "Not Granger causality" in routes or "not Granger" in routes
        assert "benchmark_data=spy" not in routes
        assert "test_granger_causality" not in routes

    def test_diagnostics_disclosure_field_present_and_honest(self):
        """The causality-honesty guarantee must survive independently of
        the plain-language pass on the seven statistical findings - this
        is what test_shipped_ui_and_route_copy_drop_causal_tab_label
        actually depends on now."""
        import numpy as np
        import pandas as pd

        from trading.analysis.econometric_diagnostics import (
            EconometricDiagnostics,
        )

        rng = np.random.default_rng(3)
        idx = pd.date_range("2025-01-01", periods=200, freq="B")
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
        df = pd.DataFrame({"Close": close}, index=idx)
        r = EconometricDiagnostics("TEST", df).run_all()

        disclosure = r.get("disclosure")
        assert isinstance(disclosure, str) and disclosure
        # Same substance as the original honesty fix (this shows patterns,
        # not proof of causation) - now in the plain-language voice the
        # rest of this module uses, not jargon.
        assert "causes another" in disclosure or "cause" in disclosure.lower()
        # Must not silently reintroduce the jargon it replaced.
        assert "granger" not in disclosure.lower()
        assert "causal" not in disclosure.lower()

        api = (ROOT / "web" / "frontend" / "src" / "api.ts").read_text(
            encoding="utf-8", errors="replace"
        )
        assert "getDiagnostics" in api
        assert "/api/diagnostics/" in api
