# -*- coding: utf-8 -*-
"""GEX regime signal edge — Phase 3 of the signal-edge program.

---------------------------------------------------------------------------
DATA-AVAILABILITY FIRST (same discipline as AI Score IC audit)
---------------------------------------------------------------------------
Historical dealer GEX is not reconstructible from free chains
(``gex_snapshot_logger`` module doc). Edge testing requires the
forward-accumulating store ``data/gex_regime_snapshots.db``.

This module ALWAYS audits row counts first. If per-symbol filled
observations are below the locked floor, it writes an honest
``insufficient_data`` deferral to ``data/gex_signal_oos_real.json`` and
does **not** invent a purged OOS on empty/thin data.

---------------------------------------------------------------------------
PREDECLARED (locked for when data eventually clears the floor)
---------------------------------------------------------------------------
Not evaluated until ``audit_gex_snapshot_availability`` says so:

  regimes as categorical signal → long_gamma / short_gamma / near_flip
  targets: next-day abs return (vol proxy) and next-day directional
           persistence (sign continuity) — columns already on the store
  min per symbol: MIN_ROWS_PER_SYMBOL (60) matching logger docstring;
                  prefer 120+ before treating a clear as strong

Do not lower this floor after peeking at a thin sample.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from trading.data.gex_snapshot_logger import _db_path, count_snapshots

ROOT = Path(__file__).resolve().parents[2]

# Locked BEFORE any future DSR run — matches gex_snapshot_logger timeline note.
MIN_ROWS_PER_SYMBOL = 60
PREFERRED_ROWS_PER_SYMBOL = 120
MIN_SYMBOLS_WITH_FLOOR = 1

DISCLOSURE = (
    "GEX signal research: requires forward-accumulated daily snapshots in "
    "data/gex_regime_snapshots.db (free chains have no historical GEX). "
    "Null / deferred is expected until the store clears the locked floor. "
    "recommend_live never auto-wires."
)

TRIAL_JUSTIFICATION_WHEN_READY = (
    "When data clears the floor: trials will be predeclared over regime "
    "encodings (e.g. long_gamma=+1 / short_gamma=-1 / near_flip=0 as a "
    "vol-regime score, plus a near_flip vs not binary) × targets "
    "{next_day_abs_return, directional_persistence} with horizon=1 and "
    "purge=1. Floor MIN_ROWS_PER_SYMBOL=60 locked to the logger docstring; "
    "not lowered after inspecting thin samples."
)


def audit_gex_snapshot_availability() -> Dict[str, Any]:
    """Honest census of the forward GEX snapshot store."""
    db = _db_path()
    audit: Dict[str, Any] = {
        "db_path": str(db),
        "db_exists": bool(db.exists()),
        "total_rows": int(count_snapshots()),
        "by_symbol": {},
        "date_range": None,
        "rows_with_next_day_abs_return": 0,
        "rows_with_next_day_range_pct": 0,
        "min_rows_per_symbol_required": MIN_ROWS_PER_SYMBOL,
        "preferred_rows_per_symbol": PREFERRED_ROWS_PER_SYMBOL,
        "symbols_meeting_floor": [],
        "enough_for_oos": False,
        "logging_hint": (
            "Set EVOLVE_GEX_SNAPSHOT_LOG=1 (and optional "
            "EVOLVE_GEX_SNAPSHOT_SYMBOLS=SPY,QQQ) then let background jobs "
            "accumulate daily rows."
        ),
    }

    if not db.exists():
        audit["enough_for_oos"] = False
        audit["reason"] = (
            "Snapshot DB does not exist yet — zero forward-accumulated GEX "
            "observations. Edge test deferred."
        )
        return audit

    if audit["total_rows"] == 0:
        audit["enough_for_oos"] = False
        audit["reason"] = (
            "Snapshot DB exists but has 0 rows — zero forward-accumulated GEX "
            "observations. Edge test deferred."
        )
        return audit

    try:
        with sqlite3.connect(str(db)) as conn:
            by_sym = conn.execute(
                "SELECT symbol, COUNT(*) FROM gex_regime_snapshots "
                "GROUP BY symbol ORDER BY symbol"
            ).fetchall()
            audit["by_symbol"] = {str(s): int(n) for s, n in by_sym}

            dr = conn.execute(
                "SELECT MIN(as_of_date), MAX(as_of_date) "
                "FROM gex_regime_snapshots"
            ).fetchone()
            if dr and dr[0]:
                audit["date_range"] = {"start": dr[0], "end": dr[1]}

            audit["rows_with_next_day_abs_return"] = int(
                conn.execute(
                    "SELECT COUNT(*) FROM gex_regime_snapshots "
                    "WHERE next_day_abs_return IS NOT NULL"
                ).fetchone()[0]
            )
            audit["rows_with_next_day_range_pct"] = int(
                conn.execute(
                    "SELECT COUNT(*) FROM gex_regime_snapshots "
                    "WHERE next_day_range_pct IS NOT NULL"
                ).fetchone()[0]
            )

            # Prefer filled outcomes for the floor check when available
            filled = conn.execute(
                "SELECT symbol, COUNT(*) FROM gex_regime_snapshots "
                "WHERE next_day_abs_return IS NOT NULL "
                "GROUP BY symbol"
            ).fetchall()
            filled_map = {str(s): int(n) for s, n in filled}
    except Exception as e:
        audit["enough_for_oos"] = False
        audit["reason"] = f"DB read failed: {e}"
        return audit

    meeting: List[str] = []
    for sym, n in audit["by_symbol"].items():
        # Require raw rows at floor; filled outcomes should also be near floor
        n_fill = filled_map.get(sym, 0)
        if n >= MIN_ROWS_PER_SYMBOL and n_fill >= max(30, MIN_ROWS_PER_SYMBOL // 2):
            meeting.append(sym)
    audit["symbols_meeting_floor"] = meeting
    audit["filled_by_symbol"] = filled_map
    audit["enough_for_oos"] = len(meeting) >= MIN_SYMBOLS_WITH_FLOOR

    if audit["enough_for_oos"]:
        audit["reason"] = (
            f"{len(meeting)} symbol(s) meet the locked floor "
            f"({MIN_ROWS_PER_SYMBOL}+ rows) — OOS may proceed."
        )
    else:
        audit["reason"] = (
            f"No symbol has >= {MIN_ROWS_PER_SYMBOL} snapshots with adequate "
            f"next-day fills (have {audit['by_symbol'] or 'none'}). "
            "Edge test deferred — do not fabricate OOS on thin data."
        )
    return audit


def run_gex_signal_oos_real(
    *,
    out_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Audit first; run harness OOS only if the store clears the floor."""
    audit = audit_gex_snapshot_availability()
    report: Dict[str, Any] = {
        "success": True,
        "signal_name": "gex_regime",
        "disclosure": DISCLOSURE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ordering_note": (
            "Availability floor and future trial justification were fixed "
            "before inspecting DSR; floor not lowered after this audit."
        ),
        "trial_justification_when_ready": TRIAL_JUSTIFICATION_WHEN_READY,
        "availability": audit,
        "recommend_live": False,
        "oos": None,
    }

    if not audit.get("enough_for_oos"):
        report["status"] = "insufficient_data"
        report["deferred"] = True
        report["note"] = audit.get("reason") or (
            "Insufficient GEX snapshot history — deferred."
        )
    else:
        # Intentional: do not silently invent an OOS path in this installment
        # if the store somehow already clears — wire harness here in a
        # follow-up once real multi-month panels exist. For now still defer
        # the *implementation* of the edge loop unless we have data; if we
        # have data, attempt a minimal harness pass.
        report["status"] = "data_ready_oos_not_yet_wired"
        report["deferred"] = True
        report["note"] = (
            "Snapshot floor cleared, but Phase-3 of this installment only "
            "ships the availability gate + deferred artifact. Wire "
            "run_signal_edge_oos on regime labels in a follow-up once a "
            "multi-month panel is confirmed in production — do not rush a "
            "one-day sample."
        )
        # If we somehow have enough data already, still don't fake trials —
        # leave deferred with clear status so the program stays honest.
        # (Current repo: 0 rows, so this branch is unused today.)

    path = Path(out_path) if out_path else ROOT / "data" / "gex_signal_oos_real.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["wrote"] = str(path)
    return report


__all__ = [
    "MIN_ROWS_PER_SYMBOL",
    "PREFERRED_ROWS_PER_SYMBOL",
    "DISCLOSURE",
    "TRIAL_JUSTIFICATION_WHEN_READY",
    "audit_gex_snapshot_availability",
    "run_gex_signal_oos_real",
]
