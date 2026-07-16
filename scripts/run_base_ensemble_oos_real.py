# -*- coding: utf-8 -*-
"""Run Phase-1 BASE ensemble absolute-edge OOS → data/base_ensemble_oos_real.json."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading.research.base_ensemble_edge import run_base_ensemble_oos_real  # noqa: E402


def main() -> int:
    report = run_base_ensemble_oos_real()
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
