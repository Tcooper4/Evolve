# -*- coding: utf-8 -*-
"""IV-percentile-filtered put credit OOS — NEW hypothesis, not a re-grid.

---------------------------------------------------------------------------
PREDECLARED BEFORE ANY DSR INSPECTION
---------------------------------------------------------------------------
Hypothesis (Bakshi & Kapadia 2003, RFS): the volatility risk premium
captured by short-vol / delta-hedged structures is larger in elevated-vol
regimes. Therefore put credit spreads that previously failed unfiltered
OOS may show stronger edge when entry is restricted to elevated IV.

This is a *regime-filter* hypothesis on the same structure+basket+grid
already tested — not expanding the delta/wing grid to chase DSR.

Locked settings (identical to refined pooled PCS where applicable):
  basket     = SPY, QQQ, IWM (pooled)
  dte        = 1 (daily-close 1DTE proxy)
  grid N=3   = (0.20, 0.05), (0.25, 0.05), (0.20, 0.06)
  strategy   = put_credit_spread only (closest prior DSR)

IV entry gate (ONE threshold — do not retune after peeking):
  min_iv_percentile = 0.50
  lookback          = 252  (reuse options_vix_sizing.VIX_LOOKBACK)

Justification for 0.50: midpoint of the practitioner iron-condor /
short-premium entry band commonly cited at the 40th–60th IV percentile;
above-median trailing VIX is the minimal "elevated" cut consistent with
that band without introducing a second free parameter.

If null: the regime-filter idea, though academically motivated, does
not rescue this structure under these locked settings — stop; do not
try 0.40 / 0.60 / 0.67 afterward to chase significance.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading.backtesting.options_strategy_backtest import (  # noqa: E402
    DISCLOSURE,
    run_options_structure_oos_pooled,
)
from trading.portfolio.options_vix_sizing import VIX_LOOKBACK  # noqa: E402

# --- locked BEFORE run ---
BASKET = ("SPY", "QQQ", "IWM")
STRATEGY = "put_credit_spread"
DTE = 1
EXIT_DTE_FLOOR = 0
PERIOD = "max"
REFINED_GRID: Tuple[Tuple[float, float], ...] = (
    (0.20, 0.05),
    (0.25, 0.05),
    (0.20, 0.06),
)
MIN_IV_PERCENTILE = 0.50
IV_LOOKBACK = int(VIX_LOOKBACK)

GRID_JUSTIFICATION = (
    "Same Phase-2 refined N=3 delta/wing grid as options_structure_refined "
    "(not expanded). This run tests a NEW entry-regime hypothesis only."
)
IV_GATE_JUSTIFICATION = (
    "Single predeclared threshold min_iv_percentile=0.50 (lookback=252 via "
    "vix_trailing_percentile): midpoint of the practitioner 40–60 IV-percentile "
    "short-premium entry band; Bakshi & Kapadia (2003) predict larger VRP in "
    "elevated-vol regimes. Not retuned after DSR."
)
HYPOTHESIS_NOTE = (
    "NEW hypothesis test (IV-regime filtered entry on put credit), NOT a "
    "re-grid of the prior unfiltered PCS OOS. A null here means the "
    "regime-filter idea does not rescue this structure — do not chase "
    "alternate thresholds on the same window."
)

DISCLOSURE_IV = (
    DISCLOSURE
    + " IV-FILTERED PCS: causal trailing VIX percentile gate "
    f"(min={MIN_IV_PERCENTILE}, lookback={IV_LOOKBACK}) before entry. "
    "1DTE daily-close proxy; pooled SPY/QQQ/IWM; same N=3 structure grid."
)


def main() -> int:
    print("HYPOTHESIS:", HYPOTHESIS_NOTE, flush=True)
    print("PREDECLARED GRID:", REFINED_GRID, flush=True)
    print("IV GATE:", MIN_IV_PERCENTILE, "lookback", IV_LOOKBACK, flush=True)
    print("IV JUSTIFICATION:", IV_GATE_JUSTIFICATION, flush=True)

    out = run_options_structure_oos_pooled(
        BASKET,
        strategy=STRATEGY,  # type: ignore[arg-type]
        period=PERIOD,
        dte=DTE,
        exit_dte_floor=EXIT_DTE_FLOOR,
        apply_costs=True,
        grid=REFINED_GRID,
        grid_justification=GRID_JUSTIFICATION,
        min_iv_percentile=MIN_IV_PERCENTILE,
        iv_percentile_lookback=IV_LOOKBACK,
    )
    out["disclosure"] = DISCLOSURE_IV
    out["hypothesis"] = HYPOTHESIS_NOTE
    out["iv_gate_justification"] = IV_GATE_JUSTIFICATION
    out["ordering_note"] = (
        "Grid, basket, and IV threshold fixed before this run; threshold "
        "will not be re-tuned after inspecting DSR."
    )

    report: Dict[str, Any] = {
        "success": bool(out.get("success")),
        "signal_name": "put_credit_iv_filtered",
        "basket": list(BASKET),
        "period": PERIOD,
        "dte": DTE,
        "actual_tested": "1DTE_daily_close_pooled_iv_filtered",
        "proxy_for": "0DTE",
        "strategy": STRATEGY,
        "predeclared_grid": [list(x) for x in REFINED_GRID],
        "n_trials": len(REFINED_GRID),
        "grid_justification": GRID_JUSTIFICATION,
        "min_iv_percentile": MIN_IV_PERCENTILE,
        "iv_percentile_lookback": IV_LOOKBACK,
        "iv_gate_justification": IV_GATE_JUSTIFICATION,
        "hypothesis": HYPOTHESIS_NOTE,
        "ordering_note": out["ordering_note"],
        "disclosure": DISCLOSURE_IV,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "result": out,
        "recommend_live": bool(out.get("recommend_live")),
        "note": out.get("note"),
    }

    dsr = out.get("deflated_sharpe") or {}
    print(
        json.dumps(
            {
                "champion": out.get("champion"),
                "oos_stats": (out.get("test") or {}).get("stats"),
                "n_trades": (out.get("test") or {}).get("n_trades"),
                "deflated_sharpe": dsr.get("deflated_sharpe")
                if isinstance(dsr, dict)
                else dsr,
                "recommend_live": out.get("recommend_live"),
                "error": out.get("error"),
                "note": out.get("note"),
            },
            indent=2,
        ),
        flush=True,
    )

    path = ROOT / "data" / "options_iv_filtered_oos_real.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"wrote {path}", flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
