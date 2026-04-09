"""
Signal score history store.
Persists per-dimension AI scores to
SQLite for IC computation.
Schema:
  signal_scores(
    id INTEGER PRIMARY KEY,
    symbol TEXT,
    scored_at REAL,  -- unix timestamp
    technical REAL,
    momentum REAL,
    sentiment REAL,
    fundamental REAL,
    overall REAL,
    price_at_score REAL,
    return_7d REAL DEFAULT NULL,
    return_filled_at REAL DEFAULT NULL
  )
"""

import logging
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DB_PATH = os.path.join("data", "signal_scores.db")
_TABLE = "signal_scores"


def _get_conn() -> sqlite3.Connection:
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(
        _DB_PATH,
        check_same_thread=False,
        timeout=10,
    )
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_score_db() -> None:
    """Create table if not exists."""
    try:
        with _get_conn() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS
                {_TABLE} (
                    id INTEGER
                        PRIMARY KEY
                        AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    scored_at REAL
                        NOT NULL,
                    technical REAL,
                    momentum REAL,
                    sentiment REAL,
                    fundamental REAL,
                    overall REAL,
                    price_at_score REAL,
                    return_7d REAL
                        DEFAULT NULL,
                    return_filled_at REAL
                        DEFAULT NULL
                )
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS
                idx_symbol_scored
                ON {_TABLE}
                (symbol, scored_at)
                """
            )
    except Exception as e:
        logger.warning("Score DB init failed: %s", e)


def record_score(
    symbol: str,
    technical: float,
    momentum: float,
    sentiment: float,
    fundamental: float,
    overall: float,
    price_at_score: float,
) -> None:
    """
    Persist one AI score observation.
    Called after every compute_ai_score.
    Deduplicates: skips if same symbol
    was scored in last 4 hours.
    """
    sym = str(symbol or "").strip().upper()
    if not sym:
        return
    try:
        _now = time.time()
        with _get_conn() as conn:
            _row = conn.execute(
                f"SELECT scored_at FROM {_TABLE} WHERE symbol=?"
                " ORDER BY scored_at DESC LIMIT 1",
                (sym,),
            ).fetchone()
            if _row and (_now - _row[0]) < 14400:
                return

            conn.execute(
                f"INSERT INTO {_TABLE} "
                "(symbol, scored_at, "
                "technical, momentum, "
                "sentiment, fundamental,"
                " overall, "
                "price_at_score) "
                "VALUES "
                "(?,?,?,?,?,?,?,?)",
                (
                    sym,
                    _now,
                    technical,
                    momentum,
                    sentiment,
                    fundamental,
                    overall,
                    price_at_score,
                ),
            )
    except Exception as e:
        logger.debug("record_score failed for %s: %s", sym, e)


def fill_forward_returns() -> int:
    """
    For rows scored ≥7 days ago with
    NULL return_7d, fetch actual price
    and compute realized return.
    Returns count of rows filled.
    """
    try:
        import datetime

        import pandas as pd

        from trading.data.price_cache import get_history

        _now = time.time()
        _cutoff = _now - 7 * 86400

        with _get_conn() as conn:
            _rows = conn.execute(
                f"SELECT id, symbol, "
                "scored_at, "
                "price_at_score FROM "
                f"{_TABLE} WHERE "
                "return_7d IS NULL "
                "AND scored_at < ?",
                (_cutoff,),
            ).fetchall()

        filled = 0
        for row_id, sym, scored_at, price_at in _rows:
            try:
                _target = scored_at + 7 * 86400
                _hist = get_history(sym, period="30d")
                if _hist.empty:
                    continue
                if "Close" in _hist.columns:
                    _close = _hist["Close"]
                else:
                    _close = _hist["close"]
                _target_dt = datetime.datetime.fromtimestamp(_target)
                _ts = pd.Timestamp(_target_dt)
                _idx = int(_close.index.searchsorted(_ts))
                if _idx >= len(_close):
                    _idx = len(_close) - 1
                _price_7d = float(_close.iloc[_idx])
                if price_at and price_at > 0:
                    _ret = (_price_7d - price_at) / price_at
                    with _get_conn() as conn:
                        conn.execute(
                            f"UPDATE {_TABLE} SET return_7d=?, "
                            "return_filled_at=? WHERE id=?",
                            (_ret, _now, row_id),
                        )
                    filled += 1
            except Exception as _re:
                logger.debug("Fill return failed %s: %s", sym, _re)

        return filled
    except Exception as e:
        logger.warning("fill_forward_returns failed: %s", e)
        return 0


def get_dimension_scores_and_returns(
    symbol: str,
    min_rows: int = 30,
) -> Optional[Dict[str, Any]]:
    """
    Returns historical dimension scores
    paired with realized 7d returns.
    Returns None if insufficient data.
    """
    sym = str(symbol or "").strip().upper()
    if not sym:
        return None
    try:
        with _get_conn() as conn:
            _rows = conn.execute(
                f"SELECT technical, "
                "momentum, sentiment, "
                "fundamental, overall,"
                " return_7d FROM "
                f"{_TABLE} WHERE "
                "symbol=? AND "
                "return_7d IS NOT NULL"
                " ORDER BY scored_at",
                (sym,),
            ).fetchall()

        if len(_rows) < min_rows:
            return None

        _arr: List[Dict[str, Any]] = [
            {
                "technical": r[0],
                "momentum": r[1],
                "sentiment": r[2],
                "fundamental": r[3],
                "overall": r[4],
                "return_7d": r[5],
            }
            for r in _rows
        ]
        return {
            "symbol": sym,
            "n_obs": len(_arr),
            "data": _arr,
        }
    except Exception as e:
        logger.debug("get_dimension_scores failed %s: %s", sym, e)
        return None


def get_global_dimension_scores(
    min_rows: int = 100,
) -> Optional[Dict[str, Any]]:
    """
    Returns all symbols' dimension
    scores for global IC computation.
    Used when per-symbol data is sparse.
    """
    try:
        with _get_conn() as conn:
            _rows = conn.execute(
                f"SELECT technical, "
                "momentum, sentiment, "
                "fundamental, overall,"
                " return_7d FROM "
                f"{_TABLE} WHERE "
                "return_7d IS NOT NULL"
                " ORDER BY scored_at"
                " DESC LIMIT 1000"
            ).fetchall()

        if len(_rows) < min_rows:
            return None

        return {
            "n_obs": len(_rows),
            "data": [
                {
                    "technical": r[0],
                    "momentum": r[1],
                    "sentiment": r[2],
                    "fundamental": r[3],
                    "overall": r[4],
                    "return_7d": r[5],
                }
                for r in _rows
            ],
        }
    except Exception as e:
        logger.debug("get_global_dimension_scores failed: %s", e)
        return None
