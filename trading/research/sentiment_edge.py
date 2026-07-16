# -*- coding: utf-8 -*-
"""FinBERT/VADER headline-sentiment edge — signal-edge program Phase 3.

---------------------------------------------------------------------------
DATA-AVAILABILITY FIRST (same discipline as GEX / AI Score)
---------------------------------------------------------------------------
``get_news_headline_sentiment`` is live-only. Free news feeds are not a
multi-year archive. This module audits
``data/news_sentiment_snapshots.db`` first. If under the locked floor,
write an honest ``insufficient_data`` deferral — do not invent OOS.

---------------------------------------------------------------------------
PREDECLARED (locked for when data eventually clears)
---------------------------------------------------------------------------
Horizons {1, 3} only — literature and prior Evolve notes treat headline
sentiment as short-lived / modest; do not assume long windows.

  TrialSpec(horizon=1), TrialSpec(horizon=3)
  target = forward_return matching horizon
  purge = horizon
  universe = SPY, QQQ, IWM
  signal = daily snapshot sentiment_score from the forward store
           (FinBERT+VADER blend when FinBERT loads)

Floor: MIN_ROWS_PER_SYMBOL = 30 filled return_1d (or return_3d) pairs.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from trading.data.news_sentiment_snapshot_logger import (
    _db_path,
    count_snapshots,
)

ROOT = Path(__file__).resolve().parents[2]

MIN_ROWS_PER_SYMBOL = 30
MIN_SYMBOLS_WITH_FLOOR = 1
BASKET = ("SPY", "QQQ", "IWM")

DISCLOSURE = (
    "Headline-sentiment research: requires forward-accumulated daily "
    "snapshots from get_news_headline_sentiment (FinBERT+VADER when "
    "available). Live news is not a historical archive — null/deferred "
    "until the store clears the locked floor. recommend_live never "
    "auto-wires."
)

TRIAL_JUSTIFICATION_WHEN_READY = (
    "Two trials only when data clears: hold horizons {1, 3} trading days "
    "(short-lived sentiment predictive window; not a long-horizon grid). "
    "Signal = stored daily sentiment_score; purge matches horizon; "
    "universe SPY/QQQ/IWM. Floor 30 filled rows/symbol locked before DSR."
)


def audit_sentiment_snapshot_availability() -> Dict[str, Any]:
    db = _db_path()
    audit: Dict[str, Any] = {
        "db_path": str(db),
        "db_exists": bool(db.exists()),
        "total_rows": int(count_snapshots()),
        "by_symbol": {},
        "filled_return_1d_by_symbol": {},
        "filled_return_3d_by_symbol": {},
        "date_range": None,
        "min_rows_per_symbol_required": MIN_ROWS_PER_SYMBOL,
        "symbols_meeting_floor": [],
        "enough_for_oos": False,
        "logging_hint": (
            "Set EVOLVE_NEWS_SENTIMENT_SNAPSHOT_LOG=1 "
            "(symbols via EVOLVE_NEWS_SENTIMENT_SNAPSHOT_SYMBOLS=SPY,QQQ,IWM)."
        ),
        "predeclared_horizons_when_ready": [1, 3],
        "trial_justification_when_ready": TRIAL_JUSTIFICATION_WHEN_READY,
    }

    if not db.exists() or audit["total_rows"] == 0:
        audit["reason"] = (
            "News-sentiment snapshot DB missing or empty — zero forward "
            "headline-sentiment observations. Live get_news_headline_sentiment "
            "cannot be replayed historically. Edge test deferred."
        )
        return audit

    try:
        with sqlite3.connect(str(db)) as conn:
            by_sym = conn.execute(
                "SELECT symbol, COUNT(*) FROM news_sentiment_snapshots "
                "GROUP BY symbol ORDER BY symbol"
            ).fetchall()
            audit["by_symbol"] = {str(s): int(n) for s, n in by_sym}

            f1 = conn.execute(
                "SELECT symbol, COUNT(*) FROM news_sentiment_snapshots "
                "WHERE return_1d IS NOT NULL GROUP BY symbol"
            ).fetchall()
            audit["filled_return_1d_by_symbol"] = {
                str(s): int(n) for s, n in f1
            }

            f3 = conn.execute(
                "SELECT symbol, COUNT(*) FROM news_sentiment_snapshots "
                "WHERE return_3d IS NOT NULL GROUP BY symbol"
            ).fetchall()
            audit["filled_return_3d_by_symbol"] = {
                str(s): int(n) for s, n in f3
            }

            dr = conn.execute(
                "SELECT MIN(as_of_date), MAX(as_of_date) "
                "FROM news_sentiment_snapshots"
            ).fetchone()
            if dr and dr[0]:
                audit["date_range"] = {"start": dr[0], "end": dr[1]}
    except Exception as e:
        audit["reason"] = f"DB read failed: {e}"
        return audit

    meeting: List[str] = []
    for sym, n in audit["by_symbol"].items():
        n1 = int(audit["filled_return_1d_by_symbol"].get(sym, 0))
        if n >= MIN_ROWS_PER_SYMBOL and n1 >= max(15, MIN_ROWS_PER_SYMBOL // 2):
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
            f"return fills (have {audit['by_symbol'] or 'none'}). "
            "Edge test deferred — do not fabricate OOS on thin/live-only data."
        )
    return audit


def run_sentiment_oos_real(
    *,
    out_path: Optional[str] = None,
) -> Dict[str, Any]:
    audit = audit_sentiment_snapshot_availability()
    report: Dict[str, Any] = {
        "success": True,
        "signal_name": "news_headline_sentiment",
        "disclosure": DISCLOSURE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ordering_note": (
            "Horizons {1,3} and floor 30/symbol locked before any DSR; "
            "floor not lowered after this audit."
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
            "sentiment_score vs horizons {1,3} in a follow-up once the panel "
            "is confirmed stable — do not rush a borderline sample."
        )

    path = Path(out_path) if out_path else ROOT / "data" / "sentiment_oos_real.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["wrote"] = str(path)
    return report


__all__ = [
    "MIN_ROWS_PER_SYMBOL",
    "BASKET",
    "DISCLOSURE",
    "TRIAL_JUSTIFICATION_WHEN_READY",
    "audit_sentiment_snapshot_availability",
    "run_sentiment_oos_real",
]
