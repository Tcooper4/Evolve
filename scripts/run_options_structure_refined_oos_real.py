# -*- coding: utf-8 -*-
"""Refined 1-DTE options-structure OOS — narrow literature grid + ETF basket.

PREDECLARED before any DSR inspection (do not expand after peeking):
  grid = (0.20, 0.05), (0.25, 0.05), (0.20, 0.06)   # N=3 vs legacy 12
  universe = SPY, QQQ, IWM (pooled)
  dte = 1 (daily close proxy for short-dated; not true 0DTE)

Justification: project defaults (0.20Δ / 5% wing) + one adjacent delta
inside the VRP/practitioner 0.15–0.30Δ band + one slightly wider wing
used in prior research guides — not a post-hoc subset of the old 4×3 grid.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading.backtesting.options_strategy_backtest import (  # noqa: E402
    DISCLOSURE,
    run_options_structure_oos_pooled,
)

# --- locked BEFORE run ---
BASKET = ("SPY", "QQQ", "IWM")
STRATEGIES = ("put_credit_spread", "iron_condor")
DTE = 1
EXIT_DTE_FLOOR = 0
PERIOD = "max"
REFINED_GRID: Tuple[Tuple[float, float], ...] = (
    (0.20, 0.05),
    (0.25, 0.05),
    (0.20, 0.06),
)
GRID_JUSTIFICATION = (
    "Three literature/practitioner-anchored trials only (not the legacy "
    "4x3=12 grid): (1) short_delta=0.20 wing=5% -- Evolve DEFAULT / research "
    "default; (2) short_delta=0.25 wing=5% -- interior of the commonly cited "
    "0.15-0.30 delta VRP / short-premium band; (3) short_delta=0.20 wing=6% -- "
    "same default delta with a modestly wider defined-risk wing already "
    "inside prior guide ranges. Chosen before inspecting DSR; N_trials=3 "
    "lowers selection burden vs the prior 12-trial sweep that left PCS "
    "DSR~0.59-0.85."
)

DISCLOSURE_REFINED = (
    DISCLOSURE
    + " REFINED 1-DTE basket run: pooled SPY/QQQ/IWM, narrow predeclared "
    "grid (N=3), daily-close 1-DTE PROXY (not true 0DTE). VIX30 remains a "
    "rougher IV proxy for short-dated structures."
)


def main() -> int:
    print("PREDECLARED GRID:", REFINED_GRID, flush=True)
    print("JUSTIFICATION:", GRID_JUSTIFICATION, flush=True)
    print("BASKET:", BASKET, flush=True)

    report: Dict[str, Any] = {
        "success": True,
        "basket": list(BASKET),
        "period": PERIOD,
        "dte": DTE,
        "actual_tested": "1DTE_daily_close_pooled",
        "proxy_for": "0DTE",
        "predeclared_grid": [list(x) for x in REFINED_GRID],
        "n_trials": len(REFINED_GRID),
        "grid_justification": GRID_JUSTIFICATION,
        "ordering_note": (
            "Grid and basket were fixed in this script before the run; "
            "not selected after inspecting DSR."
        ),
        "disclosure": DISCLOSURE_REFINED,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategies": {},
        "recommend_live": False,
    }

    any_ok = False
    for strat in STRATEGIES:
        print(f"=== pooled {strat} ===", flush=True)
        out = run_options_structure_oos_pooled(
            BASKET,
            strategy=strat,  # type: ignore[arg-type]
            period=PERIOD,
            dte=DTE,
            exit_dte_floor=EXIT_DTE_FLOOR,
            apply_costs=True,
            grid=REFINED_GRID,
            grid_justification=GRID_JUSTIFICATION,
        )
        out["disclosure"] = DISCLOSURE_REFINED
        report["strategies"][strat] = out
        if out.get("success"):
            any_ok = True
        dsr = (out.get("deflated_sharpe") or {})
        print(
            json.dumps(
                {
                    "strategy": strat,
                    "champion": out.get("champion"),
                    "oos_stats": (out.get("test") or {}).get("stats"),
                    "n_trades": (out.get("test") or {}).get("n_trades"),
                    "deflated_sharpe": dsr.get("deflated_sharpe")
                    if isinstance(dsr, dict)
                    else dsr,
                    "recommend_live": out.get("recommend_live"),
                    "n_trials": out.get("n_trials"),
                    "error": out.get("error"),
                    "note": out.get("note"),
                },
                indent=2,
            ),
            flush=True,
        )

    report["success"] = any_ok
    report["recommend_live"] = any(
        bool((report["strategies"][s] or {}).get("recommend_live"))
        for s in STRATEGIES
    )
    report["note"] = (
        "At least one structure cleared OOS+DSR — still research-only."
        if report["recommend_live"]
        else (
            "Null / not significant on refined 1-DTE pooled OOS+DSR — "
            "leave research-only (acceptable; do not re-run grids to chase)."
        )
    )

    path = ROOT / "data" / "options_structure_refined_oos_real.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {path}", flush=True)
    return 0 if any_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
