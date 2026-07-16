# -*- coding: utf-8 -*-
"""Opt-in daily news-headline sentiment snapshot logger — dataset builder ONLY.

---------------------------------------------------------------------------
Purpose
---------------------------------------------------------------------------
``get_news_headline_sentiment`` is live-only (yfinance / NewsAPI / RSS).
Free feeds are not a multi-year headline archive, so a historical FinBERT
OOS panel cannot be fabricated honestly today.

This module accumulates *forward* daily snapshots (score + engine +
headline count) so a future harness pass can test short-horizon (1–3d)
predictive power with purge + DSR. Enabling logging validates nothing today.

---------------------------------------------------------------------------
Enable
---------------------------------------------------------------------------
* ``EVOLVE_NEWS_SENTIMENT_SNAPSHOT_LOG=1``
* Symbols: ``EVOLVE_NEWS_SENTIMENT_SNAPSHOT_SYMBOLS=SPY,QQQ,IWM`` (default)

Store: ``data/news_sentiment_snapshots.db``
Timeline: meaningful OOS typically needs ~30+ filled trading-day rows per
symbol (same floor discipline as AI Score / GEX).
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
    "News-sentiment snapshot logger: forward dataset for a FUTURE "
    "FinBERT/VADER headline-sentiment vs short-horizon return validation. "
    "This write validates nothing today. Live get_news_headline_sentiment "
    "has no historical archive."
)


def news_sentiment_snapshot_logging_enabled() -> bool:
    raw = (os.getenv("EVOLVE_NEWS_SENTIMENT_SNAPSHOT_LOG", "0") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def snapshot_symbols() -> List[str]:
    raw = (
        os.getenv("EVOLVE_NEWS_SENTIMENT_SNAPSHOT_SYMBOLS") or "SPY,QQQ,IWM"
    ).strip()
    syms = [s.strip().upper() for s in raw.split(",") if s.strip()]
    return syms or ["SPY", "QQQ", "IWM"]


def _db_path() -> Path:
    root = Path(__file__).resolve().parent.parent.parent
    p = root / "data" / "news_sentiment_snapshots.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path()), check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS news_sentiment_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            as_of_date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            sentiment_score REAL,
            sentiment_label TEXT,
            n_headlines INTEGER,
            engine TEXT,
            return_1d REAL,
            return_3d REAL,
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
                        "SELECT COUNT(*) FROM news_sentiment_snapshots "
                        "WHERE symbol = ?",
                        ((symbol or "").strip().upper(),),
                    )
                else:
                    cur = conn.execute(
                        "SELECT COUNT(*) FROM news_sentiment_snapshots"
                    )
                row = cur.fetchone()
                return int(row[0]) if row else 0
        except Exception as e:
            logger.debug("news_sentiment count failed: %s", e)
            return 0


def append_snapshot_row(
    *,
    as_of_date: date,
    symbol: str,
    sentiment_score: Optional[float],
    sentiment_label: Optional[str],
    n_headlines: Optional[int],
    engine: Optional[str],
    return_1d: Optional[float] = None,
    return_3d: Optional[float] = None,
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
                    INSERT OR IGNORE INTO news_sentiment_snapshots (
                        as_of_date, symbol, sentiment_score, sentiment_label,
                        n_headlines, engine, return_1d, return_3d,
                        disclosure, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        as_of_date.isoformat(),
                        sym,
                        float(sentiment_score)
                        if sentiment_score is not None
                        else None,
                        sentiment_label,
                        int(n_headlines) if n_headlines is not None else None,
                        engine,
                        float(return_1d) if return_1d is not None else None,
                        float(return_3d) if return_3d is not None else None,
                        disclosure or PURPOSE_DISCLOSURE,
                        created,
                    ),
                )
                conn.commit()
                return cur.rowcount > 0
        except Exception as e:
            logger.debug("news_sentiment append failed: %s", e)
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
        logger.debug("news_sentiment fwd return failed: %s", e)
        return None


def backfill_pending_outcomes() -> int:
    """Fill return_1d / return_3d when enough price history exists."""
    n = 0
    try:
        from trading.data.price_cache import get_history

        with _lock:
            with _connect() as conn:
                rows = conn.execute(
                    """
                    SELECT as_of_date, symbol FROM news_sentiment_snapshots
                    WHERE return_1d IS NULL OR return_3d IS NULL
                    """
                ).fetchall()
        for as_of_s, sym in rows:
            try:
                as_of = date.fromisoformat(str(as_of_s))
                hist = get_history(str(sym), period="1y")
                r1 = _fwd_return(hist, as_of, 1)
                r3 = _fwd_return(hist, as_of, 3)
                if r1 is None and r3 is None:
                    continue
                with _lock:
                    with _connect() as conn:
                        conn.execute(
                            """
                            UPDATE news_sentiment_snapshots
                            SET return_1d = COALESCE(?, return_1d),
                                return_3d = COALESCE(?, return_3d)
                            WHERE symbol = ? AND as_of_date = ?
                            """,
                            (r1, r3, str(sym).upper(), as_of.isoformat()),
                        )
                        conn.commit()
                        n += 1
            except Exception as e:
                logger.debug("news_sentiment backfill row failed: %s", e)
    except Exception as e:
        logger.debug("news_sentiment backfill failed: %s", e)
    return n


def log_daily_news_sentiment_snapshots() -> Dict[str, Any]:
    """Score live headlines for each symbol and append today's row."""
    out: Dict[str, Any] = {
        "logged": 0,
        "backfilled": 0,
        "total_rows": 0,
        "enabled": news_sentiment_snapshot_logging_enabled(),
        "symbols": snapshot_symbols(),
    }
    if not out["enabled"]:
        return out

    today = date.today()
    try:
        from trading.data.social_sentiment import get_news_headline_sentiment
    except Exception as e:
        out["error"] = str(e)
        return out

    for sym in snapshot_symbols():
        try:
            sent = get_news_headline_sentiment(sym, max_items=15)
            if not sent.get("success"):
                continue
            ok = append_snapshot_row(
                as_of_date=today,
                symbol=sym,
                sentiment_score=sent.get("sentiment_score"),
                sentiment_label=sent.get("sentiment_label"),
                n_headlines=sent.get("mention_count"),
                engine=sent.get("engine"),
            )
            if ok:
                out["logged"] += 1
        except Exception as e:
            logger.debug("news_sentiment log %s failed: %s", sym, e)

    out["backfilled"] = backfill_pending_outcomes()
    out["total_rows"] = count_snapshots()
    return out


__all__ = [
    "PURPOSE_DISCLOSURE",
    "news_sentiment_snapshot_logging_enabled",
    "snapshot_symbols",
    "count_snapshots",
    "append_snapshot_row",
    "backfill_pending_outcomes",
    "log_daily_news_sentiment_snapshots",
    "_db_path",
]
