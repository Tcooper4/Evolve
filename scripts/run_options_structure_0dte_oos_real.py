# -*- coding: utf-8 -*-
"""1-DTE proxy for short-dated / 0DTE-style structures (Phase 2).

Phase 1 finding: true 0DTE is not representable on the daily-close
simulator (dte=0 is degenerate). Closest honest reuse of existing
machinery is 1-DTE entry→next-day exit. This is a PROXY for 0DTE, not
0DTE itself — see DISCLOSURE_0DTE_PROXY.
"""

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

PERIOD = "max"
SYMBOL = "SPY"
STRATEGIES = ("iron_condor", "put_credit_spread")
# Phase 1: closest non-degenerate daily proxy for short-dated premium.
DTE = 1
EXIT_DTE_FLOOR = 0

DISCLOSURE_0DTE_PROXY = (
    DISCLOSURE
    + " ADDITIONAL CAVEAT (short-dated / 0DTE-style run): this result is a "
    "1-DTE daily-bar PROXY (enter day t, exit day t+1) — NOT true same-session "
    "0DTE. Daily closes cannot see intraday theta acceleration or gamma pin. "
    "VIX is a 30-day constant-maturity IV proxy; using it for 0DTE/1-DTE is a "
    "materially rougher approximation than for ~37 DTE, because near-term / "
    "same-day implied vol can diverge sharply from the 30-day read around "
    "intraday catalysts. Do not interpret as a fill-level 0DTE backtest."
)


def _summarize(out: Dict[str, Any]) -> Dict[str, Any]:
    test = out.get("test") or {}
    stats = test.get("stats") if isinstance(test, dict) else None
    if stats is None and out.get("method") == "fixed_params":
        stats = out.get("stats")
    dsr = out.get("deflated_sharpe") or {}
    return {
        "success": out.get("success"),
        "method": out.get("method"),
        "error": out.get("error"),
        "n_trades": (test.get("n_trades") if isinstance(test, dict) else None)
        or out.get("n_trades"),
        "champion": out.get("champion"),
        "oos_stats": stats,
        "deflated_sharpe": dsr.get("deflated_sharpe") if isinstance(dsr, dict) else dsr,
        "dsr_detail": dsr if isinstance(dsr, dict) else None,
        "recommend_live": out.get("recommend_live"),
        "note": out.get("note"),
        "purge_days": out.get("purge_days"),
        "n_trials": out.get("n_trials"),
    }


def _stamp_disclosure(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    out["disclosure"] = DISCLOSURE_0DTE_PROXY
    return out


def main() -> int:
    report: Dict[str, Any] = {
        "success": True,
        "symbol": SYMBOL,
        "period": PERIOD,
        "proxy_for": "0DTE",
        "actual_tested": "1DTE_daily_close",
        "phase1_finding": (
            "True 0DTE not representable on daily-close simulator (dte=0 "
            "degenerates to zero entry credit / same-bar exit). 1-DTE is the "
            "closest honest reuse of existing purge/sweep/DSR machinery."
        ),
        "limitation": (
            "SPY longest-history liquid ETF proxy. 1-DTE daily close→close "
            "structure marks via BS + VIX30 — not SPX 0DTE fills, not intraday "
            "theta/gamma path. See disclosure."
        ),
        "disclosure": DISCLOSURE_0DTE_PROXY,
        "base_params": {
            "dte": DTE,
            "short_delta": 0.20,
            "wing_pct": 0.05,
            "profit_take": 0.50,
            "exit_dte_floor": EXIT_DTE_FLOOR,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategies": {},
    }

    any_ok = False
    for strat in STRATEGIES:
        print(f"=== {strat} fixed 1DTE ===", flush=True)
        fixed = run_options_structure_backtest(
            SYMBOL,
            strategy=strat,  # type: ignore[arg-type]
            period=PERIOD,
            dte=DTE,
            exit_dte_floor=EXIT_DTE_FLOOR,
            sweep=False,
            apply_costs=True,
        )
        print(f"=== {strat} sweep OOS+DSR 1DTE ===", flush=True)
        sweep = run_options_structure_backtest(
            SYMBOL,
            strategy=strat,  # type: ignore[arg-type]
            period=PERIOD,
            dte=DTE,
            exit_dte_floor=EXIT_DTE_FLOOR,
            sweep=True,
            apply_costs=True,
        )

        fixed_compact = _stamp_disclosure(
            {k: v for k, v in fixed.items() if k != "trades"}
        )
        sweep_compact = _stamp_disclosure(dict(sweep))
        if isinstance(sweep_compact.get("test"), dict):
            sweep_compact["test"] = {
                k: v for k, v in sweep_compact["test"].items() if k != "trades"
            }
        if "trials" in sweep_compact:
            sweep_compact["trials"] = [
                {
                    "short_delta": t.get("short_delta"),
                    "wing_pct": t.get("wing_pct"),
                    "n_trades": t.get("n_trades"),
                    "score": t.get("score"),
                    "train_stats": t.get("train_stats"),
                }
                for t in (sweep_compact.get("trials") or [])
            ]

        report["strategies"][strat] = {
            "fixed": fixed_compact,
            "sweep_oos": sweep_compact,
            "summary": _summarize(sweep),
        }
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
    report["recommend_live"] = any(
        bool((report["strategies"][s].get("sweep_oos") or {}).get("recommend_live"))
        for s in STRATEGIES
    )
    report["note"] = (
        "At least one 1-DTE proxy structure cleared OOS+DSR — still research-only; "
        "not true 0DTE."
        if report["recommend_live"]
        else (
            "Null / not significant on OOS+DSR for 1-DTE iron_condor and "
            "put_credit_spread proxies on SPY — leave research-only "
            "(acceptable expected outcome). Not a verdict on true 0DTE."
        )
    )

    out_path = ROOT / "data" / "options_structure_0dte_oos_real.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    return 0 if any_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
