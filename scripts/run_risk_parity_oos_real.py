# -*- coding: utf-8 -*-
"""Run risk-parity vs 1/N vs conditional-vol OOS → data/risk_parity_oos_real.json."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading.research.risk_parity_edge import run_risk_parity_oos_real  # noqa: E402


def main() -> int:
    report = run_risk_parity_oos_real()
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
