# -*- coding: utf-8 -*-
"""Market-state synthesis validity — situational-awareness check (Phase 3).

---------------------------------------------------------------------------
DATA-AVAILABILITY FIRST
---------------------------------------------------------------------------
``compose_market_state`` / ``get_market_state`` are live composites of:
  GEX regime + news event-severity + statistical vol regime.

Historical dealer GEX and free-chain skew are not reconstructible
(``gex_snapshot_logger`` / ``skew_snapshot_logger``). News severity is
also live-only without a headline archive. This module audits the
forward snapshot stores first. If under the locked floor, write an
honest ``insufficient_data`` deferral — do **not** invent a validity
panel from live-only reads.

---------------------------------------------------------------------------
PREDECLARED (locked for when data eventually clears)
---------------------------------------------------------------------------
This is NOT a directional prediction test (the module explicitly forbids
that). When data clears:

  Bucket A: level in {elevated, critical}  ("stressed")
  Bucket B: level in {calm, watchful}      ("quiet")

  Targets (risk, not direction):
    T1  forward_realized_vol horizon=5
    T2  forward_realized_vol horizon=21
    T3  forward abs drawdown over 21d hold

  Validity claim to test: stressed days precede *rougher* subsequent
  conditions (higher fwd vol / deeper abs DD) than quiet days —
  situational awareness, not alpha.

Floor: MIN_ROWS = 60 filled GEX snapshot rows with next-day outcomes
(match GEX logger); prefer skew snapshots too but GEX is the binding
gate for the composite's mechanical leg.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from trading.data.gex_snapshot_logger import (
    _db_path as gex_db_path,
    count_snapshots as count_gex_snapshots,
)
from trading.data.skew_snapshot_logger import (
    _db_path as skew_db_path,
    count_snapshots as count_skew_snapshots,
)

ROOT = Path(__file__).resolve().parents[2]

MIN_ROWS_GEX = 60
PREFERRED_ROWS_GEX = 120
MIN_ROWS_SKEW = 60  # informational; not required alone to clear

DISCLOSURE = (
    "Market-state validity research: situational-awareness check only "
    "(elevated/critical vs calm/watchful → subsequent realized vol / "
    "drawdown risk). Not a directional prediction test. Requires "
    "forward-accumulated GEX (and ideally skew) snapshots — live "
    "get_market_state cannot be replayed historically. recommend_live "
    "never auto-wires."
)

TRIAL_JUSTIFICATION_WHEN_READY = (
    "When data clears: three validity contrasts only — stressed "
    "(elevated|critical) vs quiet (calm|watchful) on targets "
    "{forward_realized_vol h=5, forward_realized_vol h=21, "
    "forward_abs_drawdown h=21}. Universe SPY (primary composite symbol). "
    "No directional return target. Floor MIN_ROWS_GEX=60 locked before "
    "any comparison; not lowered after inspecting thin samples."
)


def audit_market_state_data_availability() -> Dict[str, Any]:
    """Honest census of inputs needed to validate market-state labels."""
    gex_db = gex_db_path()
    skew_db = skew_db_path()
    gex_n = int(count_gex_snapshots())
    skew_n = int(count_skew_snapshots())

    audit: Dict[str, Any] = {
        "gex": {
            "db_path": str(gex_db),
            "db_exists": bool(gex_db.exists()),
            "total_rows": gex_n,
            "min_rows_required": MIN_ROWS_GEX,
            "preferred_rows": PREFERRED_ROWS_GEX,
            "logging_hint": (
                "Set EVOLVE_GEX_SNAPSHOT_LOG=1 "
                "(EVOLVE_GEX_SNAPSHOT_SYMBOLS=SPY,QQQ)."
            ),
        },
        "skew": {
            "db_path": str(skew_db),
            "db_exists": bool(skew_db.exists()),
            "total_rows": skew_n,
            "min_rows_required": MIN_ROWS_SKEW,
            "logging_hint": (
                "Set EVOLVE_SKEW_SNAPSHOT_LOG=1 "
                "(EVOLVE_SKEW_SNAPSHOT_SYMBOLS=SPY,QQQ)."
            ),
        },
        "news_severity_archive": {
            "available": False,
            "note": (
                "Event-severity leg is live-only (breaking headlines + "
                "FinBERT/VADER). No durable dated severity panel exists "
                "yet — even with GEX filled, full composite replay needs "
                "a future news-severity snapshot or a reduced test using "
                "GEX+vol legs only (predeclare that reduction before running)."
            ),
        },
        "market_state_snapshot_store": {
            "exists": False,
            "note": (
                "No dedicated market_state daily snapshot DB — composite "
                "is computed live via get_market_state."
            ),
        },
        "enough_for_validity_oos": False,
        "predeclared_when_ready": {
            "stressed_levels": ["elevated", "critical"],
            "quiet_levels": ["calm", "watchful"],
            "targets": [
                {"kind": "forward_realized_vol", "horizon": 5},
                {"kind": "forward_realized_vol", "horizon": 21},
                {"kind": "forward_abs_drawdown", "horizon": 21},
            ],
            "directional_return_target": False,
        },
        "trial_justification_when_ready": TRIAL_JUSTIFICATION_WHEN_READY,
    }

    # Fill GEX outcome census when DB has rows
    if gex_db.exists() and gex_n > 0:
        try:
            with sqlite3.connect(str(gex_db)) as conn:
                by_sym = conn.execute(
                    "SELECT symbol, COUNT(*) FROM gex_regime_snapshots "
                    "GROUP BY symbol"
                ).fetchall()
                audit["gex"]["by_symbol"] = {
                    str(s): int(n) for s, n in by_sym
                }
                filled = conn.execute(
                    "SELECT COUNT(*) FROM gex_regime_snapshots "
                    "WHERE next_day_abs_return IS NOT NULL"
                ).fetchone()
                audit["gex"]["rows_with_next_day_abs_return"] = (
                    int(filled[0]) if filled else 0
                )
        except Exception as e:
            audit["gex"]["read_error"] = str(e)

    gex_clears = gex_n >= MIN_ROWS_GEX
    audit["enough_for_validity_oos"] = bool(gex_clears)

    if gex_clears:
        audit["reason"] = (
            f"GEX store has {gex_n} rows (>= {MIN_ROWS_GEX}). "
            "A reduced GEX+vol validity pass may proceed; full composite "
            "still lacks a news-severity archive — declare the reduced "
            "scope before any run."
        )
    else:
        audit["reason"] = (
            f"GEX snapshots={gex_n}, skew snapshots={skew_n} "
            f"(need GEX>={MIN_ROWS_GEX} filled). Market-state composite "
            "cannot be replayed historically from live get_market_state. "
            "Validity test deferred — do not fabricate an elevated-vs-calm "
            "panel on empty/live-only data."
        )
    return audit


def run_market_state_validity_real(
    *,
    out_path: Optional[str] = None,
) -> Dict[str, Any]:
    audit = audit_market_state_data_availability()
    report: Dict[str, Any] = {
        "success": True,
        "signal_name": "market_state_validity",
        "disclosure": DISCLOSURE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ordering_note": (
            "Validity design (stressed vs quiet → fwd vol/DD, not direction) "
            "and floor MIN_ROWS_GEX=60 locked before this audit; floor not "
            "lowered after inspecting counts."
        ),
        "trial_justification_when_ready": TRIAL_JUSTIFICATION_WHEN_READY,
        "availability": audit,
        "recommend_live": False,
        "validity_oos": None,
        "real_counts": {
            "gex_snapshot_rows": audit["gex"]["total_rows"],
            "skew_snapshot_rows": audit["skew"]["total_rows"],
        },
    }

    if not audit.get("enough_for_validity_oos"):
        report["status"] = "insufficient_data"
        report["deferred"] = True
        report["note"] = audit.get("reason")
    else:
        report["status"] = "data_ready_oos_not_yet_wired"
        report["deferred"] = True
        report["note"] = (
            "GEX floor cleared. Wire elevated-vs-calm vs forward vol/DD "
            "in a follow-up with an explicit reduced-scope declaration "
            "(GEX+vol only, or wait for news-severity archive) — do not "
            "rush a borderline sample."
        )

    path = (
        Path(out_path)
        if out_path
        else ROOT / "data" / "market_state_validity_real.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["wrote"] = str(path)
    return report


__all__ = [
    "MIN_ROWS_GEX",
    "PREFERRED_ROWS_GEX",
    "DISCLOSURE",
    "TRIAL_JUSTIFICATION_WHEN_READY",
    "audit_market_state_data_availability",
    "run_market_state_validity_real",
]
