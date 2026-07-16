# -*- coding: utf-8 -*-
"""Run Phase-3 GEX availability audit → data/gex_signal_oos_real.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading.research.gex_signal_edge import (  # noqa: E402
    audit_gex_snapshot_availability,
    run_gex_signal_oos_real,
)


def main() -> int:
    audit = audit_gex_snapshot_availability()
    print("GEX SNAPSHOT AUDIT:", json.dumps(audit, indent=2), flush=True)
    report = run_gex_signal_oos_real()
    print("STATUS:", report.get("status"), flush=True)
    print("NOTE:", report.get("note"), flush=True)
    print("wrote", report.get("wrote"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
