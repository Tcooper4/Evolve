# -*- coding: utf-8 -*-
"""IV-skew signal edge — signal-edge program Phase 5.

---------------------------------------------------------------------------
DATA-AVAILABILITY FIRST (same discipline as GEX / sentiment)
---------------------------------------------------------------------------
``get_options_skew`` is live-only (±5% moneyness proxy on free chains —
not true 25Δ RR/BF). Free chains are not a multi-year IV-surface archive.
This module audits ``data/skew_snapshots.db`` first. If under the locked
floor, write an honest ``insufficient_data`` deferral — do not invent OOS.

---------------------------------------------------------------------------
PREDECLARED (locked for when data eventually clears)
---------------------------------------------------------------------------
Universe: SPY, QQQ

  T1  signal=skew_diff         horizon=5   purge=5
  T2  signal=skew_diff         horizon=21  purge=21
  T3  signal=butterfly_proxy   horizon=5   purge=5
      butterfly_proxy = (put_otm_iv + call_otm_iv)/2 − atm_iv

Literature anchor: steep put skew / elevated wings as a risk-premium /
crash-hedging state that can relate to subsequent equity returns over
weekly–monthly windows. Metric honesty: moneyness proxy, not 25Δ.

Floor: MIN_ROWS_PER_SYMBOL = 60 (match GEX); prefer 120+.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from trading.data.skew_snapshot_logger import _db_path, count_snapshots

ROOT = Path(__file__).resolve().parents[2]

MIN_ROWS_PER_SYMBOL = 60
PREFERRED_ROWS_PER_SYMBOL = 120
MIN_SYMBOLS_WITH_FLOOR = 1
BASKET = ("SPY", "QQQ")

DISCLOSURE = (
    "IV-skew research: requires forward-accumulated daily snapshots from "
    "get_options_skew (±5% moneyness put-call skew_diff / butterfly proxy — "
    "not true 25Δ RR/BF). Live chains are not a historical archive — "
    "null/deferred until the store clears the locked floor. "
    "recommend_live never auto-wires."
)

TRIAL_JUSTIFICATION_WHEN_READY = (
    "Three trials only when data clears: (1) skew_diff → forward_return "
    "horizon=5; (2) skew_diff → forward_return horizon=21; (3) "
    "butterfly_proxy → forward_return horizon=5. Universe SPY/QQQ. "
    "Metric is ±5% moneyness proxy (honest — free chains lack reliable "
    "delta). Floor MIN_ROWS_PER_SYMBOL=60 locked before DSR; not lowered "
    "after inspecting thin samples."
)


def audit_skew_snapshot_availability() -> Dict[str, Any]:
    db = _db_path()
    audit: Dict[str, Any] = {
        "db_path": str(db),
        "db_exists": bool(db.exists()),
        "total_rows": int(count_snapshots()),
        "by_symbol": {},
        "filled_return_5d_by_symbol": {},
        "filled_return_21d_by_symbol": {},
        "date_range": None,
        "min_rows_per_symbol_required": MIN_ROWS_PER_SYMBOL,
        "preferred_rows_per_symbol": PREFERRED_ROWS_PER_SYMBOL,
        "symbols_meeting_floor": [],
        "enough_for_oos": False,
        "logging_hint": (
            "Set EVOLVE_SKEW_SNAPSHOT_LOG=1 "
            "(symbols via EVOLVE_SKEW_SNAPSHOT_SYMBOLS=SPY,QQQ)."
        ),
        "predeclared_trials_when_ready": [
            {"signal": "skew_diff", "horizon": 5, "label": "skew_h5"},
            {"signal": "skew_diff", "horizon": 21, "label": "skew_h21"},
            {
                "signal": "butterfly_proxy",
                "horizon": 5,
                "label": "bfly_h5",
            },
        ],
        "trial_justification_when_ready": TRIAL_JUSTIFICATION_WHEN_READY,
        "metric_honesty": (
            "skew_diff / butterfly from ±5% moneyness — not 25Δ risk "
            "reversal or 25Δ butterfly."
        ),
    }

    if not db.exists() or audit["total_rows"] == 0:
        audit["reason"] = (
            "IV-skew snapshot DB missing or empty — zero forward "
            "skew observations. Live get_options_skew cannot be "
            "replayed historically. Edge test deferred."
        )
        return audit

    try:
        with sqlite3.connect(str(db)) as conn:
            by_sym = conn.execute(
                "SELECT symbol, COUNT(*) FROM skew_snapshots "
                "GROUP BY symbol ORDER BY symbol"
            ).fetchall()
            audit["by_symbol"] = {str(s): int(n) for s, n in by_sym}

            f5 = conn.execute(
                "SELECT symbol, COUNT(*) FROM skew_snapshots "
                "WHERE return_5d IS NOT NULL GROUP BY symbol"
            ).fetchall()
            audit["filled_return_5d_by_symbol"] = {
                str(s): int(n) for s, n in f5
            }

            f21 = conn.execute(
                "SELECT symbol, COUNT(*) FROM skew_snapshots "
                "WHERE return_21d IS NOT NULL GROUP BY symbol"
            ).fetchall()
            audit["filled_return_21d_by_symbol"] = {
                str(s): int(n) for s, n in f21
            }

            dr = conn.execute(
                "SELECT MIN(as_of_date), MAX(as_of_date) FROM skew_snapshots"
            ).fetchone()
            if dr and dr[0]:
                audit["date_range"] = {"start": dr[0], "end": dr[1]}
    except Exception as e:
        audit["reason"] = f"DB read failed: {e}"
        return audit

    meeting: List[str] = []
    for sym, n in audit["by_symbol"].items():
        n5 = int(audit["filled_return_5d_by_symbol"].get(sym, 0))
        if n >= MIN_ROWS_PER_SYMBOL and n5 >= max(30, MIN_ROWS_PER_SYMBOL // 2):
            meeting.append(sym)
    audit["symbols_meeting_floor"] = meeting
    audit["enough_for_oos"] = len(meeting) >= MIN_SYMBOLS_WITH_FLOOR

    if audit["enough_for_oos"]:
        audit["reason"] = (
            f"{len(meeting)} symbol(s) meet the locked floor — OOS may proceed."
        )
    else:
        audit["reason"] = (
            f"No symbol has >={MIN_ROWS_PER_SYMBOL} snapshots with adequate "
            f"return_5d fills (have {audit['by_symbol'] or 'none'}). "
            "Edge test deferred — do not fabricate OOS on thin/live-only data."
        )
    return audit


def run_skew_signal_oos_real(
    *,
    out_path: Optional[str] = None,
) -> Dict[str, Any]:
    audit = audit_skew_snapshot_availability()
    report: Dict[str, Any] = {
        "success": True,
        "signal_name": "iv_skew",
        "disclosure": DISCLOSURE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ordering_note": (
            "Trials {skew_h5, skew_h21, bfly_h5} and floor 60/symbol locked "
            "before any DSR; floor not lowered after this audit."
        ),
        "basket": list(BASKET),
        "trial_justification_when_ready": TRIAL_JUSTIFICATION_WHEN_READY,
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
            "Snapshot floor cleared. Wire run_signal_edge_oos on stored "
            "skew_diff / butterfly_proxy vs horizons {5, 21} in a follow-up "
            "once the panel is confirmed stable — do not rush a borderline sample."
        )

    path = (
        Path(out_path)
        if out_path
        else ROOT / "data" / "skew_signal_oos_real.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["wrote"] = str(path)
    return report


__all__ = [
    "MIN_ROWS_PER_SYMBOL",
    "PREFERRED_ROWS_PER_SYMBOL",
    "BASKET",
    "DISCLOSURE",
    "TRIAL_JUSTIFICATION_WHEN_READY",
    "audit_skew_snapshot_availability",
    "run_skew_signal_oos_real",
]
