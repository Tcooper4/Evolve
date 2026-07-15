# -*- coding: utf-8 -*-
"""Run real PEAD purged OOS (Phase 1 research pass)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading.backtesting.pead_strategy import (  # noqa: E402
    PREDECLARED_TRIALS,
    TRIAL_JUSTIFICATION,
    run_pead_oos_real,
)


def main() -> int:
    print("PREDECLARED_TRIALS (locked before run):", flush=True)
    print(json.dumps(list(PREDECLARED_TRIALS), indent=2), flush=True)
    print("JUSTIFICATION:", TRIAL_JUSTIFICATION, flush=True)

    out_path = ROOT / "data" / "pead_oos_real.json"
    report = run_pead_oos_real(out_path=str(out_path), run_midcap=True)

    summary = {
        "success": report.get("success"),
        "recommend_live": report.get("recommend_live"),
        "note": report.get("note"),
        "large_cap": {
            "n_events": (report.get("large_cap") or {}).get(
                "n_events_positive_surprise"
            ),
            "champion": (report.get("large_cap") or {}).get("champion"),
            "oos_stats": ((report.get("large_cap") or {}).get("test") or {}).get(
                "stats"
            ),
            "dsr": (
                ((report.get("large_cap") or {}).get("deflated_sharpe") or {}).get(
                    "deflated_sharpe"
                )
            ),
            "recommend_live": (report.get("large_cap") or {}).get("recommend_live"),
            "error": (report.get("large_cap") or {}).get("error"),
        },
        "mid_less_covered": {
            "n_events": (report.get("mid_less_covered") or {}).get(
                "n_events_positive_surprise"
            ),
            "champion": (report.get("mid_less_covered") or {}).get("champion"),
            "oos_stats": (
                (report.get("mid_less_covered") or {}).get("test") or {}
            ).get("stats"),
            "dsr": (
                (
                    (report.get("mid_less_covered") or {}).get("deflated_sharpe") or {}
                ).get("deflated_sharpe")
            ),
            "recommend_live": (report.get("mid_less_covered") or {}).get(
                "recommend_live"
            ),
            "error": (report.get("mid_less_covered") or {}).get("error"),
        },
        "wrote": str(out_path),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
