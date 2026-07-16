# -*- coding: utf-8 -*-
"""Opt-in daily IV-skew snapshot logger — dataset builder ONLY.

---------------------------------------------------------------------------
Purpose
---------------------------------------------------------------------------
``get_options_skew`` is live-only (yfinance delayed chains). Free chains
are not a multi-year IV-surface archive, so a historical put-call skew /
risk-reversal OOS panel cannot be fabricated honestly today.

This module accumulates *forward* daily snapshots (skew_diff, shape,
OTM/ATM IVs, butterfly proxy) so a future harness pass can test
medium-horizon (5d / 21d) predictive power with purge + DSR. Enabling
logging validates nothing today.

Honesty: moneyness is ±5% strike-distance proxy — not true 25Δ RR/BF
(free chains lack reliable deltas). Column names stay honest about that.

---------------------------------------------------------------------------
Enable
---------------------------------------------------------------------------
* ``EVOLVE_SKEW_SNAPSHOT_LOG=1``
* Symbols: ``EVOLVE_SKEW_SNAPSHOT_SYMBOLS=SPY,QQQ`` (default)

Store: ``data/skew_snapshots.db``
Timeline: meaningful OOS typically needs ~60+ filled trading-day rows
per symbol (match GEX floor discipline).
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()

PURPOSE_DISCLOSURE = (
    "IV-skew snapshot logger: forward dataset for a FUTURE "
    "put-call skew_diff / butterfly-proxy vs equity-return validation. "
    "This write validates nothing today. Live get_options_skew has no "
    "historical archive. Metric is ±5% moneyness proxy, not true 25Δ."
)


def skew_snapshot_logging_enabled() -> bool:
    raw = (os.getenv("EVOLVE_SKEW_SNAPSHOT_LOG", "0") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def snapshot_symbols() -> List[str]:
    raw = (os.getenv("EVOLVE_SKEW_SNAPSHOT_SYMBOLS") or "SPY,QQQ").strip()
    syms = [s.strip().upper() for s in raw.split(",") if s.strip()]
    return syms or ["SPY", "QQQ"]


def _db_path() -> Path:
    root = Path(__file__).resolve().parent.parent.parent
    p = root / "data" / "skew_snapshots.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path()), check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS skew_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            as_of_date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            skew_diff REAL,
            shape TEXT,
            put_otm_iv REAL,
            call_otm_iv REAL,
            atm_iv REAL,
            butterfly_proxy REAL,
            far_skew_diff REAL,
            expiry TEXT,
            spot REAL,
            event_interpretation TEXT,
            return_5d REAL,
            return_21d REAL,
            disclosure TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(symbol, as_of_date)
        )
        """
    )
    conn.commit()
    return conn


def count_snapshots(symbol: Optional[str] = None) -> int:
    with _lock:
        try:
            with _connect() as conn:
                if symbol:
                    cur = conn.execute(
                        "SELECT COUNT(*) FROM skew_snapshots WHERE symbol = ?",
                        ((symbol or "").strip().upper(),),
                    )
                else:
                    cur = conn.execute("SELECT COUNT(*) FROM skew_snapshots")
                row = cur.fetchone()
                return int(row[0]) if row else 0
        except Exception as e:
            logger.debug("skew_snapshot count failed: %s", e)
            return 0


def append_snapshot_row(
    *,
    as_of_date: date,
    symbol: str,
    skew_diff: Optional[float],
    shape: Optional[str],
    put_otm_iv: Optional[float],
    call_otm_iv: Optional[float],
    atm_iv: Optional[float],
    butterfly_proxy: Optional[float],
    far_skew_diff: Optional[float] = None,
    expiry: Optional[str] = None,
    spot: Optional[float] = None,
    event_interpretation: Optional[str] = None,
    return_5d: Optional[float] = None,
    return_21d: Optional[float] = None,
    disclosure: Optional[str] = None,
) -> bool:
    sym = (symbol or "").strip().upper()
    if not sym:
        return False
    created = datetime.now(timezone.utc).isoformat()
    with _lock:
        try:
            with _connect() as conn:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO skew_snapshots (
                        as_of_date, symbol, skew_diff, shape,
                        put_otm_iv, call_otm_iv, atm_iv, butterfly_proxy,
                        far_skew_diff, expiry, spot, event_interpretation,
                        return_5d, return_21d, disclosure, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        as_of_date.isoformat(),
                        sym,
                        float(skew_diff) if skew_diff is not None else None,
                        shape,
                        float(put_otm_iv) if put_otm_iv is not None else None,
                        float(call_otm_iv) if call_otm_iv is not None else None,
                        float(atm_iv) if atm_iv is not None else None,
                        float(butterfly_proxy)
                        if butterfly_proxy is not None
                        else None,
                        float(far_skew_diff)
                        if far_skew_diff is not None
                        else None,
                        expiry,
                        float(spot) if spot is not None else None,
                        event_interpretation,
                        float(return_5d) if return_5d is not None else None,
                        float(return_21d) if return_21d is not None else None,
                        disclosure or PURPOSE_DISCLOSURE,
                        created,
                    ),
                )
                conn.commit()
                return cur.rowcount > 0
        except Exception as e:
            logger.debug("skew_snapshot append failed: %s", e)
            return False


def _fwd_return(hist: Any, as_of: date, horizon: int) -> Optional[float]:
    if hist is None or getattr(hist, "empty", True):
        return None
    try:
        import pandas as pd

        df = hist.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        cmap = {str(c).lower(): c for c in df.columns}
        close_c = cmap.get("close")
        if not close_c:
            return None
        dates = [d.date() if hasattr(d, "date") else d for d in df.index]
        if as_of not in dates:
            return None
        i = dates.index(as_of)
        j = i + int(horizon)
        if j >= len(df):
            return None
        c0 = float(df.iloc[i][close_c])
        c1 = float(df.iloc[j][close_c])
        if c0 <= 0:
            return None
        return float(c1 / c0 - 1.0)
    except Exception as e:
        logger.debug("skew_snapshot fwd return failed: %s", e)
        return None


def backfill_pending_outcomes() -> int:
    """Fill return_5d / return_21d when enough price history exists."""
    n = 0
    try:
        from trading.data.price_cache import get_history

        with _lock:
            with _connect() as conn:
                rows = conn.execute(
                    """
                    SELECT as_of_date, symbol FROM skew_snapshots
                    WHERE return_5d IS NULL OR return_21d IS NULL
                    """
                ).fetchall()
        for as_of_s, sym in rows:
            try:
                as_of = date.fromisoformat(str(as_of_s))
                hist = get_history(str(sym), period="1y")
                r5 = _fwd_return(hist, as_of, 5)
                r21 = _fwd_return(hist, as_of, 21)
                if r5 is None and r21 is None:
                    continue
                with _lock:
                    with _connect() as conn:
                        conn.execute(
                            """
                            UPDATE skew_snapshots
                            SET return_5d = COALESCE(?, return_5d),
                                return_21d = COALESCE(?, return_21d)
                            WHERE symbol = ? AND as_of_date = ?
                            """,
                            (r5, r21, str(sym).upper(), as_of.isoformat()),
                        )
                        conn.commit()
                        n += 1
            except Exception as e:
                logger.debug("skew_snapshot backfill row failed: %s", e)
    except Exception as e:
        logger.debug("skew_snapshot backfill failed: %s", e)
    return n


def log_daily_skew_snapshots() -> Dict[str, Any]:
    """Score live IV skew for each symbol and append today's row."""
    out: Dict[str, Any] = {
        "logged": 0,
        "backfilled": 0,
        "total_rows": 0,
        "enabled": skew_snapshot_logging_enabled(),
        "symbols": snapshot_symbols(),
    }
    if not out["enabled"]:
        return out

    today = date.today()
    try:
        from trading.data.options_skew import get_options_skew
    except Exception as e:
        out["error"] = str(e)
        return out

    for sym in snapshot_symbols():
        try:
            skew = get_options_skew(sym)
            if not skew.get("success"):
                continue
            put_iv = skew.get("put_otm_iv")
            call_iv = skew.get("call_otm_iv")
            atm = skew.get("atm_iv")
            butterfly = None
            try:
                if put_iv is not None and call_iv is not None and atm is not None:
                    butterfly = (float(put_iv) + float(call_iv)) / 2.0 - float(atm)
            except Exception as e:
                logger.debug("butterfly_proxy %s: %s", sym, e)

            event = skew.get("event_context") or {}
            ok = append_snapshot_row(
                as_of_date=today,
                symbol=sym,
                skew_diff=skew.get("skew_diff"),
                shape=skew.get("shape"),
                put_otm_iv=put_iv,
                call_otm_iv=call_iv,
                atm_iv=atm,
                butterfly_proxy=butterfly,
                far_skew_diff=skew.get("far_skew_diff"),
                expiry=skew.get("expiry"),
                spot=skew.get("spot"),
                event_interpretation=event.get("interpretation"),
            )
            if ok:
                out["logged"] += 1
        except Exception as e:
            logger.debug("skew_snapshot log %s failed: %s", sym, e)

    out["backfilled"] = backfill_pending_outcomes()
    out["total_rows"] = count_snapshots()
    return out


__all__ = [
    "PURPOSE_DISCLOSURE",
    "skew_snapshot_logging_enabled",
    "snapshot_symbols",
    "count_snapshots",
    "append_snapshot_row",
    "backfill_pending_outcomes",
    "log_daily_skew_snapshots",
    "_db_path",
]
