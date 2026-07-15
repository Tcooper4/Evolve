# -*- coding: utf-8 -*-
"""Opt-in daily GEX regime snapshot logger — dataset builder ONLY.

---------------------------------------------------------------------------
Purpose (read this before enabling)
---------------------------------------------------------------------------
Free yfinance option chains do **not** supply historical dealer GEX — only
a live/delayed snapshot (see ``strategy_chart_overlay`` docstring and
``trading.data.options_flow``'s single-row TTL cache keyed by symbol).
This project's ``data/options_cache.db`` overwrites per symbol; it is not
a time series. Polygon reference contract lists (when keyed) also do not
yield a reconstructible historical GEX panel with OI/γ as-of each past day.

Therefore a true OOS backtest of the ``near_flip`` 0.5%-of-spot boundary
(``NEAR_FLIP_PCT`` in ``gamma_exposure``) is **not feasible today**.

This module deliberately accumulates *forward* daily snapshots so a
future validation pass can ask whether long_gamma / short_gamma / near_flip
separate subsequent realized-vol/range regimes. **Enabling this logger
validates nothing today.**

---------------------------------------------------------------------------
Enable / disable
---------------------------------------------------------------------------
* ``EVOLVE_GEX_SNAPSHOT_LOG=1`` (or true/yes/on) → logging allowed
* Default: **off**
* Symbols: ``EVOLVE_GEX_SNAPSHOT_SYMBOLS=SPY,QQQ`` (default ``SPY``)

Store: ``data/gex_regime_snapshots.db`` (append-only; unique on
symbol+as_of_date). Subsequent runs backfill ``next_day_abs_return`` /
``next_day_range_pct`` for prior rows when daily bars are available.

Timeline reality: meaningful OOS separation tests typically need **several
months** of trading-day snapshots (roughly 60–120+ observations per symbol,
ideally more for regime subsets). Do not treat a handful of rows as
evidence.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

_lock = threading.Lock()

PURPOSE_DISCLOSURE = (
    "GEX snapshot logger: building a forward dataset for a FUTURE "
    "near_flip / regime-vs-realized-vol validation pass (and, once enough "
    "rows exist, the options-structure mapping's regime→pick association). "
    "This write validates nothing today. Historical GEX reconstruction is "
    "not available from free yfinance chains or this project's options cache."
)

NEAR_FLIP_BOUNDARY_NOTE = (
    "near_flip uses a design-choice boundary of 0.5% of spot from the "
    "gamma-flip level — not an Evolve-validated empirical threshold."
)


def gex_snapshot_logging_enabled() -> bool:
    raw = (os.getenv("EVOLVE_GEX_SNAPSHOT_LOG", "0") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def snapshot_symbols() -> List[str]:
    raw = (os.getenv("EVOLVE_GEX_SNAPSHOT_SYMBOLS") or "SPY").strip()
    syms = [s.strip().upper() for s in raw.split(",") if s.strip()]
    return syms or ["SPY"]


def _db_path() -> Path:
    root = Path(__file__).resolve().parent.parent.parent
    p = root / "data" / "gex_regime_snapshots.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path()), check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS gex_regime_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            as_of_date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            spot REAL,
            net_gex REAL,
            gamma_flip REAL,
            regime_short TEXT,
            near_flip_dist_pct REAL,
            near_flip_boundary_pct REAL,
            structure_pick TEXT,
            next_day_abs_return REAL,
            next_day_range_pct REAL,
            disclosure TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(symbol, as_of_date)
        )
        """
    )
    # Older DBs created before structure_pick — add column if missing.
    try:
        cols = {
            r[1]
            for r in conn.execute("PRAGMA table_info(gex_regime_snapshots)").fetchall()
        }
        if "structure_pick" not in cols:
            conn.execute(
                "ALTER TABLE gex_regime_snapshots ADD COLUMN structure_pick TEXT"
            )
    except Exception as e:
        logger.debug("gex_snapshot schema migrate: %s", e)
    conn.commit()
    return conn


def near_flip_distance_pct(spot: Optional[float], flip: Optional[float]) -> Optional[float]:
    """|(spot − flip) / spot| — same geometry as ``_regime_label``."""
    try:
        s = float(spot) if spot is not None else None
        f = float(flip) if flip is not None else None
    except (TypeError, ValueError):
        return None
    if s is None or f is None or s <= 0:
        return None
    return abs(s - f) / s


def append_snapshot_row(
    *,
    as_of_date: date,
    symbol: str,
    spot: Optional[float],
    net_gex: Optional[float],
    gamma_flip: Optional[float],
    regime_short: Optional[str],
    near_flip_boundary_pct: float = 0.005,
    structure_pick: Optional[str] = None,
    next_day_abs_return: Optional[float] = None,
    next_day_range_pct: Optional[float] = None,
    disclosure: Optional[str] = None,
) -> bool:
    """Insert one calendar-day row (no-op if symbol+date already logged)."""
    sym = (symbol or "").strip().upper()
    if not sym:
        return False
    dist = near_flip_distance_pct(spot, gamma_flip)
    text = disclosure or f"{PURPOSE_DISCLOSURE} {NEAR_FLIP_BOUNDARY_NOTE}"
    created = datetime.now(timezone.utc).isoformat()
    with _lock:
        try:
            with _connect() as conn:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO gex_regime_snapshots (
                        as_of_date, symbol, spot, net_gex, gamma_flip,
                        regime_short, near_flip_dist_pct, near_flip_boundary_pct,
                        structure_pick,
                        next_day_abs_return, next_day_range_pct,
                        disclosure, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        as_of_date.isoformat(),
                        sym,
                        float(spot) if spot is not None else None,
                        float(net_gex) if net_gex is not None else None,
                        float(gamma_flip) if gamma_flip is not None else None,
                        str(regime_short or ""),
                        dist,
                        float(near_flip_boundary_pct),
                        str(structure_pick) if structure_pick else None,
                        next_day_abs_return,
                        next_day_range_pct,
                        text,
                        created,
                    ),
                )
                conn.commit()
                return cur.rowcount > 0
        except Exception as e:
            logger.debug("gex_snapshot append failed: %s", e)
            return False


def update_next_day_outcomes(
    symbol: str,
    as_of_date: date,
    *,
    next_day_abs_return: float,
    next_day_range_pct: Optional[float] = None,
) -> bool:
    """Backfill subsequent realized move for a previously logged day."""
    sym = (symbol or "").strip().upper()
    with _lock:
        try:
            with _connect() as conn:
                cur = conn.execute(
                    """
                    UPDATE gex_regime_snapshots
                    SET next_day_abs_return = ?,
                        next_day_range_pct = COALESCE(?, next_day_range_pct)
                    WHERE symbol = ? AND as_of_date = ?
                      AND next_day_abs_return IS NULL
                    """,
                    (
                        float(next_day_abs_return),
                        float(next_day_range_pct)
                        if next_day_range_pct is not None
                        else None,
                        sym,
                        as_of_date.isoformat(),
                    ),
                )
                conn.commit()
                return cur.rowcount > 0
        except Exception as e:
            logger.debug("gex_snapshot outcome update failed: %s", e)
            return False


def count_snapshots(symbol: Optional[str] = None) -> int:
    with _lock:
        try:
            with _connect() as conn:
                if symbol:
                    cur = conn.execute(
                        "SELECT COUNT(*) FROM gex_regime_snapshots WHERE symbol = ?",
                        ((symbol or "").strip().upper(),),
                    )
                else:
                    cur = conn.execute("SELECT COUNT(*) FROM gex_regime_snapshots")
                row = cur.fetchone()
                return int(row[0]) if row else 0
        except Exception as e:
            logger.debug("gex_snapshot count failed: %s", e)
            return 0


def _next_day_stats_from_hist(
    hist: Any,
    as_of: date,
) -> Optional[Dict[str, float]]:
    """Compute next completed session |return| and range from a daily OHLC frame."""
    if hist is None or getattr(hist, "empty", True):
        return None
    try:
        import pandas as pd

        df = hist.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        cmap = {str(c).lower(): c for c in df.columns}
        close_c = cmap.get("close")
        high_c = cmap.get("high")
        low_c = cmap.get("low")
        if not close_c:
            return None
        dates = [d.date() if hasattr(d, "date") else d for d in df.index]
        if as_of not in dates:
            return None
        i = dates.index(as_of)
        if i + 1 >= len(df):
            return None
        c0 = float(df.iloc[i][close_c])
        c1 = float(df.iloc[i + 1][close_c])
        if c0 <= 0:
            return None
        out: Dict[str, float] = {"next_day_abs_return": abs(c1 / c0 - 1.0)}
        if high_c and low_c and c1 > 0:
            hi = float(df.iloc[i + 1][high_c])
            lo = float(df.iloc[i + 1][low_c])
            out["next_day_range_pct"] = (hi - lo) / c1
        return out
    except Exception as e:
        logger.debug("next_day_stats failed: %s", e)
        return None


def backfill_pending_outcomes(
    symbol: str,
    hist: Any,
) -> int:
    """Fill next-day fields for logged rows that still lack them."""
    sym = (symbol or "").strip().upper()
    updated = 0
    with _lock:
        try:
            with _connect() as conn:
                rows = conn.execute(
                    """
                    SELECT as_of_date FROM gex_regime_snapshots
                    WHERE symbol = ? AND next_day_abs_return IS NULL
                    ORDER BY as_of_date
                    """,
                    (sym,),
                ).fetchall()
        except Exception as e:
            logger.debug("gex_snapshot pending list failed: %s", e)
            return 0
    for (d_str,) in rows:
        try:
            d = date.fromisoformat(str(d_str)[:10])
        except Exception:
            continue
        stats = _next_day_stats_from_hist(hist, d)
        if not stats:
            continue
        if update_next_day_outcomes(
            sym,
            d,
            next_day_abs_return=stats["next_day_abs_return"],
            next_day_range_pct=stats.get("next_day_range_pct"),
        ):
            updated += 1
    return updated


def log_daily_gex_snapshots(
    *,
    symbols: Optional[Sequence[str]] = None,
    as_of: Optional[date] = None,
    fetch_gex: bool = True,
) -> Dict[str, Any]:
    """
    One daily pass: compute current GEX (optional), append rows, backfill.

    No-op when ``EVOLVE_GEX_SNAPSHOT_LOG`` is off. Network fetch is skipped
    when ``fetch_gex=False`` (tests inject via ``append_snapshot_row``).
    """
    summary: Dict[str, Any] = {
        "enabled": gex_snapshot_logging_enabled(),
        "logged": 0,
        "backfilled": 0,
        "symbols": [],
        "disclosure": PURPOSE_DISCLOSURE,
        "timeline_note": (
            "Meaningful regime-vs-realized-vol validation typically needs "
            "several months of daily snapshots before any OOS claim."
        ),
    }
    if not summary["enabled"]:
        return summary

    day = as_of or date.today()
    syms = list(symbols) if symbols is not None else snapshot_symbols()
    summary["symbols"] = [s.strip().upper() for s in syms if s and str(s).strip()]

    for sym in summary["symbols"]:
        try:
            if fetch_gex:
                from trading.data.gamma_exposure import (
                    NEAR_FLIP_PCT,
                    get_gamma_exposure,
                )

                profile = get_gamma_exposure(sym)
                if profile.get("success"):
                    structure_name = None
                    try:
                        from trading.analysis.options_structure_overlay import (
                            pick_options_structure,
                        )

                        pick = pick_options_structure(
                            regime_short=str(profile.get("regime_short") or ""),
                            spot=profile.get("spot"),
                            gamma_flip=profile.get("gamma_flip"),
                        )
                        structure_name = pick.get("structure")
                    except Exception as e:
                        logger.debug("gex_snapshot structure pick skip: %s", e)

                    wrote = append_snapshot_row(
                        as_of_date=day,
                        symbol=sym,
                        spot=profile.get("spot"),
                        net_gex=profile.get("net_gex"),
                        gamma_flip=profile.get("gamma_flip"),
                        regime_short=profile.get("regime_short"),
                        near_flip_boundary_pct=NEAR_FLIP_PCT,
                        structure_pick=structure_name,
                        disclosure=(
                            f"{PURPOSE_DISCLOSURE} {NEAR_FLIP_BOUNDARY_NOTE} "
                            f"{profile.get('disclosure') or ''}"
                        ).strip(),
                    )
                    if wrote:
                        summary["logged"] += 1
                else:
                    logger.debug(
                        "gex_snapshot: skip %s — %s",
                        sym,
                        profile.get("error"),
                    )

            # Backfill prior days when history is available
            try:
                import yfinance as yf

                hist = yf.Ticker(sym).history(period="3mo", auto_adjust=True)
                summary["backfilled"] += backfill_pending_outcomes(sym, hist)
            except Exception as e:
                logger.debug("gex_snapshot hist backfill %s: %s", sym, e)
        except Exception as e:
            logger.warning("gex_snapshot: %s failed: %s", sym, e)

    summary["total_rows"] = count_snapshots()
    return summary


__all__ = [
    "PURPOSE_DISCLOSURE",
    "NEAR_FLIP_BOUNDARY_NOTE",
    "gex_snapshot_logging_enabled",
    "snapshot_symbols",
    "near_flip_distance_pct",
    "append_snapshot_row",
    "update_next_day_outcomes",
    "count_snapshots",
    "backfill_pending_outcomes",
    "log_daily_gex_snapshots",
]
