# -*- coding: utf-8 -*-
"""Run real options-structure OOS (Phase 5 research pass)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading.backtesting.options_strategy_backtest import (  # noqa: E402
    DISCLOSURE,
    run_options_structure_backtest,
)

# Longest history reasonably available via free yfinance.
PERIOD = "max"
SYMBOL = "SPY"
STRATEGIES = ("iron_condor", "put_credit_spread")


def _summarize(out: Dict[str, Any]) -> Dict[str, Any]:
    test = out.get("test") or {}
    stats = test.get("stats") if isinstance(test, dict) else None
    if stats is None and out.get("method") == "fixed_params":
        stats = out.get("stats")
    dsr = out.get("deflated_sharpe") or {}
    champ = out.get("champion")
    return {
        "success": out.get("success"),
        "method": out.get("method"),
        "error": out.get("error"),
        "n_trades": (test.get("n_trades") if isinstance(test, dict) else None)
        or out.get("n_trades"),
        "champion": champ,
        "oos_stats": stats,
        "deflated_sharpe": dsr.get("deflated_sharpe") if isinstance(dsr, dict) else dsr,
        "dsr_detail": dsr if isinstance(dsr, dict) else None,
        "recommend_live": out.get("recommend_live"),
        "note": out.get("note"),
        "purge_days": out.get("purge_days"),
        "n_trials": out.get("n_trials"),
    }


def main() -> int:
    report: Dict[str, Any] = {
        "success": True,
        "symbol": SYMBOL,
        "period": PERIOD,
        "limitation": (
            "SPY used as longest liquid ETF proxy for defined-risk short-premium "
            "style (iron condor / put credit). Not a fill-level historical options "
            "replay — BS+VIX IV proxy + modeled half-spread costs only. "
            "Single-name equity underlyings may differ in IV / liquidity."
        ),
        "disclosure": DISCLOSURE,
        "base_params": {
            "dte": 37,
            "short_delta": 0.20,
            "wing_pct": 0.05,
            "profit_take": 0.50,
            "exit_dte_floor": 8,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategies": {},
    }

    any_ok = False
    for strat in STRATEGIES:
        print(f"=== {strat} fixed ===", flush=True)
        fixed = run_options_structure_backtest(
            SYMBOL,
            strategy=strat,  # type: ignore[arg-type]
            period=PERIOD,
            sweep=False,
            apply_costs=True,
        )
        print(f"=== {strat} sweep OOS+DSR ===", flush=True)
        sweep = run_options_structure_backtest(
            SYMBOL,
            strategy=strat,  # type: ignore[arg-type]
            period=PERIOD,
            sweep=True,
            apply_costs=True,
        )
        # Drop bulky trade lists from archive JSON; keep stats + champion.
        fixed_compact = {k: v for k, v in fixed.items() if k != "trades"}
        if isinstance(fixed_compact.get("stats"), dict):
            pass
        sweep_compact = dict(sweep)
        if isinstance(sweep_compact.get("test"), dict):
            sweep_compact["test"] = {
                k: v for k, v in sweep_compact["test"].items() if k != "trades"
            }
        if "trials" in sweep_compact:
            # Keep trial scores for DSR auditability; drop nested trade noise.
            trials = []
            for t in sweep_compact.get("trials") or []:
                trials.append({
                    "short_delta": t.get("short_delta"),
                    "wing_pct": t.get("wing_pct"),
                    "n_trades": t.get("n_trades"),
                    "score": t.get("score"),
                    "train_stats": t.get("train_stats"),
                })
            sweep_compact["trials"] = trials

        entry = {
            "fixed": fixed_compact,
            "sweep_oos": sweep_compact,
            "summary": _summarize(sweep),
        }
        report["strategies"][strat] = entry
        if fixed.get("success") or sweep.get("success"):
            any_ok = True
        print(
            json.dumps(
                {
                    "strategy": strat,
                    "fixed_n": fixed.get("n_trades"),
                    "fixed_sharpe": (fixed.get("stats") or {}).get("sharpe"),
                    **_summarize(sweep),
                },
                indent=2,
            ),
            flush=True,
        )

    report["success"] = any_ok
    # Overall live flag: never true unless a strategy clears its own bar.
    report["recommend_live"] = any(
        bool((report["strategies"][s].get("sweep_oos") or {}).get("recommend_live"))
        for s in STRATEGIES
    )
    report["note"] = (
        "At least one structure cleared OOS+DSR live bar — still research-only wiring."
        if report["recommend_live"]
        else (
            "Null / not significant on OOS+DSR for iron_condor and put_credit_spread "
            "on SPY — leave research-only (acceptable expected outcome)."
        )
    )

    out_path = ROOT / "data" / "options_structure_oos_real.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    return 0 if any_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
