# -*- coding: utf-8 -*-
"""Run Phase-5 IV-skew availability audit → data/skew_signal_oos_real.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading.research.skew_signal_edge import run_skew_signal_oos_real  # noqa: E402


def main() -> int:
    report = run_skew_signal_oos_real()
    print(
        "SKEW AVAILABILITY:",
        json.dumps(report.get("availability"), indent=2),
        flush=True,
    )
    print("STATUS:", report.get("status"), flush=True)
    print("NOTE:", report.get("note"), flush=True)
    print("wrote", report.get("wrote"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
