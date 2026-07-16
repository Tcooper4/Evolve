# -*- coding: utf-8 -*-
"""Run Phase-5 scanner criteria OOS → data/scanner_criteria_oos_real.json."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading.research.scanner_criteria_edge import (  # noqa: E402
    run_scanner_criteria_oos_real,
)


def main() -> int:
    report = run_scanner_criteria_oos_real()
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
