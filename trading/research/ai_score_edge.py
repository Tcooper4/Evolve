# -*- coding: utf-8 -*-
"""AI Score composite edge — Phase 4 of the signal-edge program.

---------------------------------------------------------------------------
DATA-AVAILABILITY FIRST (same discipline as IC audit / GEX Phase 3)
---------------------------------------------------------------------------
Scores land in ``data/signal_scores.db`` via ``signal_score_store``.
A filled observation is a row with ``return_7d IS NOT NULL`` (7-day
forward return backfilled by ``fill_forward_returns``).

Prior audit (``data/ic_score_audit.json``): 0 symbols with 30+ filled
pairs. This module re-runs that census. If still below the locked floor,
write an honest ``insufficient_data`` deferral — do not force OOS.

---------------------------------------------------------------------------
LOCKED FLOORS (do not lower after peeking)
---------------------------------------------------------------------------
  MIN_ROWS_PER_SYMBOL = 30   # matches get_dimension_scores_and_returns / IC
  GLOBAL_MIN_ROWS     = 100  # matches get_global_dimension_scores fallback

enough_for_oos := (≥1 symbol with 30+ filled) OR (global filled ≥ 100)

---------------------------------------------------------------------------
PREDECLARED (for when data eventually clears)
---------------------------------------------------------------------------
Targets: composite ``overall`` and dimensions technical/momentum/
sentiment/fundamental vs forward 7d return (native store horizon).
Trial set locked in ``TRIAL_JUSTIFICATION_WHEN_READY`` — not evaluated
on thin data.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]

MIN_ROWS_PER_SYMBOL = 30
GLOBAL_MIN_ROWS = 100
MIN_SYMBOLS_WITH_FLOOR = 1

DISCLOSURE = (
    "AI Score signal research: requires filled score→return_7d pairs in "
    "data/signal_scores.db. Prior IC audit found 0 symbols at 30+ filled; "
    "this phase re-checks. Null / deferred is expected until calendar time "
    "accumulates. recommend_live never auto-wires."
)

TRIAL_JUSTIFICATION_WHEN_READY = (
    "When data clears the floor: trials will be predeclared over score "
    "sources {overall, technical, momentum, sentiment, fundamental} with "
    "target=forward_return horizon=7 (native store fill) and purge=7. "
    "Universe = symbols clearing 30 filled pairs (or global pool if that "
    "path activates first). Floors 30/100 locked to IC activation gates — "
    "not lowered after inspecting thin samples."
)


def _db_path() -> Path:
    from trading.analysis.signal_score_store import _DB_PATH

    return Path(_DB_PATH)


def audit_ai_score_availability(*, run_backfill: bool = True) -> Dict[str, Any]:
    """Re-run the IC-style filled-pair census."""
    db = _db_path()
    audit: Dict[str, Any] = {
        "db_path": str(db),
        "db_exists": bool(db.exists()),
        "min_rows_per_symbol_required": MIN_ROWS_PER_SYMBOL,
        "global_min_rows_required": GLOBAL_MIN_ROWS,
        "just_filled": 0,
        "total_rows": 0,
        "return_7d_filled": 0,
        "return_7d_still_null_too_young": 0,
        "symbols": 0,
        "symbols_clearing_min_rows_30": 0,
        "symbols_meeting_floor": [],
        "global_pool_clears_min_rows_100": False,
        "max_filled_per_symbol": 0,
        "top10_by_filled": [],
        "enough_for_oos": False,
        "ic_active": False,
    }

    if not db.exists():
        audit["reason"] = (
            "signal_scores.db does not exist — zero filled score→return pairs. "
            "Edge test deferred."
        )
        return audit

    if run_backfill:
        try:
            from trading.analysis.signal_score_store import fill_forward_returns

            audit["just_filled"] = int(fill_forward_returns() or 0)
        except Exception as e:
            audit["backfill_error"] = str(e)

    cutoff = time.time() - 7 * 86400
    try:
        with sqlite3.connect(str(db)) as conn:
            audit["total_rows"] = int(
                conn.execute("SELECT COUNT(*) FROM signal_scores").fetchone()[0]
            )
            audit["return_7d_filled"] = int(
                conn.execute(
                    "SELECT COUNT(*) FROM signal_scores WHERE return_7d IS NOT NULL"
                ).fetchone()[0]
            )
            audit["return_7d_still_null_too_young"] = int(
                conn.execute(
                    "SELECT COUNT(*) FROM signal_scores "
                    "WHERE return_7d IS NULL AND scored_at >= ?",
                    (cutoff,),
                ).fetchone()[0]
            )
            per = conn.execute(
                """
                SELECT symbol,
                       SUM(CASE WHEN return_7d IS NOT NULL THEN 1 ELSE 0 END) AS n_filled,
                       COUNT(*) AS n_rows
                FROM signal_scores
                GROUP BY symbol
                ORDER BY n_filled DESC, n_rows DESC
                """
            ).fetchall()
    except Exception as e:
        audit["reason"] = f"DB read failed: {e}"
        return audit

    audit["symbols"] = len(per)
    meeting = [str(s) for s, nf, _nr in per if int(nf) >= MIN_ROWS_PER_SYMBOL]
    audit["symbols_meeting_floor"] = meeting
    audit["symbols_clearing_min_rows_30"] = len(meeting)
    audit["max_filled_per_symbol"] = int(per[0][1]) if per else 0
    audit["top10_by_filled"] = [
        {"symbol": str(s), "n_filled": int(nf), "n_rows": int(nr)}
        for s, nf, nr in per[:10]
    ]
    audit["global_pool_clears_min_rows_100"] = (
        audit["return_7d_filled"] >= GLOBAL_MIN_ROWS
    )
    audit["ic_active"] = (
        len(meeting) >= MIN_SYMBOLS_WITH_FLOOR
        or audit["global_pool_clears_min_rows_100"]
    )
    audit["enough_for_oos"] = bool(audit["ic_active"])

    if audit["enough_for_oos"]:
        audit["reason"] = (
            f"Floor cleared: {len(meeting)} symbol(s) with "
            f">={MIN_ROWS_PER_SYMBOL} filled and/or global filled "
            f"{audit['return_7d_filled']} >= {GLOBAL_MIN_ROWS}."
        )
    else:
        audit["reason"] = (
            f"NOT enough real data for AI Score edge OOS. "
            f"{len(meeting)} symbols at >={MIN_ROWS_PER_SYMBOL} filled "
            f"(max per symbol={audit['max_filled_per_symbol']}); "
            f"global filled {audit['return_7d_filled']} < {GLOBAL_MIN_ROWS}. "
            "Calendar-time accumulation — not a bug. Edge test deferred."
        )
    return audit


def run_ai_score_oos_real(
    *,
    out_path: Optional[str] = None,
    run_backfill: bool = True,
) -> Dict[str, Any]:
    """Audit first; harness OOS only when the store clears the locked floor."""
    audit = audit_ai_score_availability(run_backfill=run_backfill)
    report: Dict[str, Any] = {
        "success": True,
        "signal_name": "ai_score_composite",
        "disclosure": DISCLOSURE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ordering_note": (
            "Availability floors (30/symbol, 100 global) match the prior IC "
            "audit and were not lowered after this re-check."
        ),
        "trial_justification_when_ready": TRIAL_JUSTIFICATION_WHEN_READY,
        "prior_audit_reference": "data/ic_score_audit.json",
        "availability": audit,
        "recommend_live": False,
        "oos": None,
    }

    if not audit.get("enough_for_oos"):
        report["status"] = "insufficient_data"
        report["deferred"] = True
        report["note"] = audit.get("reason")
    else:
        report["status"] = "data_ready_oos_not_yet_wired"
        report["deferred"] = True
        report["note"] = (
            "Filled-pair floor cleared for the first time. This installment "
            "ships the availability re-check + deferred artifact; wire "
            "run_signal_edge_oos on overall/dimension scores in a follow-up "
            "once the clear is confirmed stable — do not rush a borderline "
            "sample."
        )

    path = Path(out_path) if out_path else ROOT / "data" / "ai_score_oos_real.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["wrote"] = str(path)

    # Also refresh the compact IC audit sidecar for Settings parity
    try:
        sidecar = {
            "timestamp": report["timestamp"],
            "symbols_clearing_min_rows_30": audit.get("symbols_clearing_min_rows_30"),
            "return_7d_filled_after_backfill": audit.get("return_7d_filled"),
            "return_7d_still_null_too_young": audit.get(
                "return_7d_still_null_too_young"
            ),
            "max_filled_per_symbol_approx": audit.get("max_filled_per_symbol"),
            "thresholds": {
                "per_symbol_min_rows": MIN_ROWS_PER_SYMBOL,
                "global_min_rows": GLOBAL_MIN_ROWS,
            },
            "ic_active": audit.get("ic_active"),
            "finding": audit.get("reason"),
            "source": "trading.research.ai_score_edge.audit_ai_score_availability",
        }
        (ROOT / "data" / "ic_score_audit.json").write_text(
            json.dumps(sidecar, indent=2), encoding="utf-8"
        )
        report["ic_audit_sidecar"] = str(ROOT / "data" / "ic_score_audit.json")
    except Exception as e:
        report["ic_audit_sidecar_error"] = str(e)

    return report


__all__ = [
    "MIN_ROWS_PER_SYMBOL",
    "GLOBAL_MIN_ROWS",
    "DISCLOSURE",
    "TRIAL_JUSTIFICATION_WHEN_READY",
    "audit_ai_score_availability",
    "run_ai_score_oos_real",
]
