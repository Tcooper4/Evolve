# -*- coding: utf-8 -*-
"""Run market-state validity availability audit → data/market_state_validity_real.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading.research.market_state_validity import (  # noqa: E402
    run_market_state_validity_real,
)


def main() -> int:
    report = run_market_state_validity_real()
    print(
        "REAL COUNTS:",
        json.dumps(report.get("real_counts"), indent=2),
        flush=True,
    )
    print(
        "AVAILABILITY REASON:",
        (report.get("availability") or {}).get("reason"),
        flush=True,
    )
    print("STATUS:", report.get("status"), flush=True)
    print("wrote", report.get("wrote"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
