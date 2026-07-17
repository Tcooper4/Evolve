# -*- coding: utf-8 -*-
"""Dashboard first-paint must not wait on chart-events.

The real sequencing lives in ``web/frontend/src/dashboardLoad.ts``.
This test drives that module via Node (strip-types) with a hanging
chart-events double and asserts critical paint returns promptly.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "web" / "frontend"
SELFTEST = FRONTEND / "src" / "dashboardLoad.selftest.ts"
DASHBOARD = FRONTEND / "src" / "Dashboard.tsx"
LOAD_MOD = FRONTEND / "src" / "dashboardLoad.ts"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_hanging_chart_events_does_not_block_critical_paint():
    assert SELFTEST.is_file()
    proc = subprocess.run(
        ["node", "--experimental-strip-types", str(SELFTEST.name)],
        cwd=str(FRONTEND / "src"),
        capture_output=True,
        text=True,
        timeout=30,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode == 0, (
        f"selftest failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "PASS" in proc.stdout


def test_dashboard_wires_split_load():
    """Hard load must go through runDashboardLoad (critical then overlay)."""
    dash = DASHBOARD.read_text(encoding="utf-8", errors="replace")
    load = LOAD_MOD.read_text(encoding="utf-8", errors="replace")
    assert "runDashboardLoad" in dash
    assert "from \"./dashboardLoad\"" in dash or "from './dashboardLoad'" in dash
    # Critical path must not await chart-events in the same Promise.all
    assert "getChartEvents(symbol, period)" in load
    assert "hooks.onCritical" in load
    assert "hooks.onOverlay" in load
    # Soft refresh still skips overlay fetchers
    assert "if (soft)" in load
