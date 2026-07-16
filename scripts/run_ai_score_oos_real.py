# -*- coding: utf-8 -*-
"""Run Phase-4 AI Score availability re-check → data/ai_score_oos_real.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading.research.ai_score_edge import run_ai_score_oos_real  # noqa: E402


def main() -> int:
    report = run_ai_score_oos_real(run_backfill=True)
    print(
        "AI SCORE AVAILABILITY:",
        json.dumps(report.get("availability"), indent=2),
        flush=True,
    )
    print("STATUS:", report.get("status"), flush=True)
    print("NOTE:", report.get("note"), flush=True)
    print("wrote", report.get("wrote"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
