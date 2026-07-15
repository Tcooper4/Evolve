# -*- coding: utf-8 -*-
"""Call credit spread OOS — own structure, same narrow grid as Phase 2.

PREDECLARED before any DSR inspection:
  grid = (0.20, 0.05), (0.25, 0.05), (0.20, 0.06)   # N=3
  horizons:
    - 37 DTE on SPY (matches Phase-1 SPY structure OOS horizon)
    - 1 DTE pooled SPY/QQQ/IWM (matches Phase-2 refined basket)

Honest comparison target: put_credit_spread under identical grids / baskets.
Equity index skew usually pads put premium vs calls at matched |delta|;
BS+VIX flat-IV model will understate that asymmetry — noted in disclosure.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading.backtesting.options_strategy_backtest import (  # noqa: E402
    DISCLOSURE,
    run_options_structure_oos_pooled,
)

# --- locked BEFORE run ---
REFINED_GRID: Tuple[Tuple[float, float], ...] = (
    (0.20, 0.05),
    (0.25, 0.05),
    (0.20, 0.06),
)
GRID_JUSTIFICATION = (
    "Same Phase-2 predeclared N=3 literature grid: (0.20, 0.05) project "
    "default; (0.25, 0.05) interior 0.15-0.30 delta short-premium band; "
    "(0.20, 0.06) default delta with modestly wider wing. Not expanded "
    "after inspecting Phase-2 PCS/IC nulls."
)
PERIOD = "max"
STRATEGY = "call_credit_spread"


def _compact(out: Dict[str, Any]) -> Dict[str, Any]:
    """Drop trial bulk if huge; keep champion + OOS + DSR."""
    keep = dict(out)
    # trials are small (N=3) — retain for audit
    return keep


def _dsr_val(out: Dict[str, Any]) -> Optional[float]:
    dsr = out.get("deflated_sharpe")
    if isinstance(dsr, dict):
        v = dsr.get("deflated_sharpe")
        return float(v) if v is not None else None
    return None


def _load_prior_pcs() -> Dict[str, Any]:
    """Pull prior PCS numbers for side-by-side (no re-run)."""
    prior: Dict[str, Any] = {}
    p37 = ROOT / "data" / "options_structure_oos_real.json"
    p1 = ROOT / "data" / "options_structure_refined_oos_real.json"
    try:
        if p37.is_file():
            blob = json.loads(p37.read_text(encoding="utf-8", errors="replace"))
            pcs = (blob.get("strategies") or {}).get("put_credit_spread") or {}
            sweep = pcs.get("sweep") or pcs
            prior["spy_37dte_full_grid"] = {
                "source": str(p37.name),
                "n_trials": sweep.get("n_trials"),
                "champion": sweep.get("champion"),
                "oos_stats": (sweep.get("test") or {}).get("stats")
                or sweep.get("oos_stats"),
                "deflated_sharpe": _dsr_val(sweep)
                if isinstance(sweep.get("deflated_sharpe"), dict)
                else sweep.get("deflated_sharpe"),
                "recommend_live": sweep.get("recommend_live"),
                "note": (
                    "Prior SPY 37DTE PCS used legacy 12-trial cartesian grid — "
                    "not identical selection burden to this N=3 CCS run."
                ),
            }
    except Exception as e:
        prior["spy_37dte_load_error"] = str(e)
    try:
        if p1.is_file():
            blob = json.loads(p1.read_text(encoding="utf-8", errors="replace"))
            pcs = (blob.get("strategies") or {}).get("put_credit_spread") or {}
            prior["pooled_1dte_narrow_grid"] = {
                "source": str(p1.name),
                "basket": blob.get("basket"),
                "n_trials": pcs.get("n_trials") or blob.get("n_trials"),
                "champion": pcs.get("champion"),
                "oos_stats": (pcs.get("test") or {}).get("stats"),
                "deflated_sharpe": _dsr_val(pcs),
                "recommend_live": pcs.get("recommend_live"),
                "note": (
                    "Same basket + same N=3 grid as this CCS 1DTE leg — "
                    "fairest PCS comparison."
                ),
            }
    except Exception as e:
        prior["pooled_1dte_load_error"] = str(e)
    return prior


def main() -> int:
    print("PREDECLARED GRID:", REFINED_GRID, flush=True)
    print("JUSTIFICATION:", GRID_JUSTIFICATION, flush=True)
    print("STRATEGY:", STRATEGY, flush=True)

    disclosure = (
        DISCLOSURE
        + " CALL CREDIT as its own structure (not buried inside IC). "
        "BS+VIX is flat-IV across strikes — equity put skew is NOT priced; "
        "real markets usually pad OTM puts vs calls at matched |delta|, so "
        "CCS vs PCS here understates the typical skew asymmetry favoring "
        "short puts. 1DTE leg is daily-close PROXY (not true 0DTE)."
    )

    report: Dict[str, Any] = {
        "success": True,
        "strategy": STRATEGY,
        "period": PERIOD,
        "predeclared_grid": [list(x) for x in REFINED_GRID],
        "n_trials": len(REFINED_GRID),
        "grid_justification": GRID_JUSTIFICATION,
        "ordering_note": (
            "Grid fixed before this run; identical to Phase-2 refined set."
        ),
        "disclosure": disclosure,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "horizons": {},
        "prior_put_credit_for_comparison": _load_prior_pcs(),
        "recommend_live": False,
    }

    horizons = (
        {
            "key": "spy_37dte",
            "symbols": ("SPY",),
            "dte": 37,
            "exit_dte_floor": 8,
            "label": "37DTE_SPY_narrow_grid",
        },
        {
            "key": "pooled_1dte",
            "symbols": ("SPY", "QQQ", "IWM"),
            "dte": 1,
            "exit_dte_floor": 0,
            "label": "1DTE_daily_close_pooled",
        },
    )

    any_ok = False
    for h in horizons:
        print(f"=== {STRATEGY} {h['label']} ===", flush=True)
        out = run_options_structure_oos_pooled(
            h["symbols"],
            strategy=STRATEGY,  # type: ignore[arg-type]
            period=PERIOD,
            dte=int(h["dte"]),
            exit_dte_floor=int(h["exit_dte_floor"]),
            apply_costs=True,
            grid=REFINED_GRID,
            grid_justification=GRID_JUSTIFICATION,
        )
        out["disclosure"] = disclosure
        out["horizon_label"] = h["label"]
        report["horizons"][h["key"]] = _compact(out)
        if out.get("success"):
            any_ok = True
        dsr = out.get("deflated_sharpe") or {}
        print(
            json.dumps(
                {
                    "horizon": h["label"],
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
        bool((report["horizons"][k] or {}).get("recommend_live"))
        for k in report["horizons"]
    )

    # Honest asymmetry caption from available numbers
    pcs_1 = (report["prior_put_credit_for_comparison"] or {}).get(
        "pooled_1dte_narrow_grid"
    ) or {}
    ccs_1 = report["horizons"].get("pooled_1dte") or {}
    pcs_sr = ((pcs_1.get("oos_stats") or {}) or {}).get("sharpe")
    ccs_sr = ((ccs_1.get("test") or {}).get("stats") or {}).get("sharpe")
    report["skew_asymmetry_note"] = (
        "Under flat BS+VIX, CCS and PCS should look similar; any OOS gap "
        "is mostly path / cost noise, not equity skew. Real OPRA fills "
        "would typically favor short puts (higher OTM put IV). "
        f"This run — CCS 1DTE OOS Sharpe={ccs_sr}; prior PCS 1DTE OOS "
        f"Sharpe={pcs_sr}. Neither clears DSR>=0.95 live bar unless "
        f"recommend_live flags say otherwise (CCS live="
        f"{report['recommend_live']})."
    )
    report["note"] = (
        "At least one CCS horizon cleared OOS+DSR — still research-only."
        if report["recommend_live"]
        else (
            "Null / not significant on CCS OOS+DSR across tested horizons — "
            "leave research-only (acceptable; do not re-grid to chase)."
        )
    )

    path = ROOT / "data" / "call_credit_oos_real.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {path}", flush=True)
    print(report["skew_asymmetry_note"], flush=True)
    return 0 if any_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
