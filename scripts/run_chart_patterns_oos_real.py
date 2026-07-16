# -*- coding: utf-8 -*-
"""Run Phase-2 chart-pattern OOS → data/chart_patterns_oos_real.json."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading.research.chart_pattern_edge import run_chart_patterns_oos_real  # noqa: E402


def main() -> int:
    report = run_chart_patterns_oos_real()
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
