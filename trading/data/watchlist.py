"""SQLite-backed watchlist and alert manager for Evolve.

Stores user-defined tickers and alert thresholds (price and RSI) in
`data/watchlist.db` and exposes a small API for managing entries and
checking for triggered alerts based on live market data.
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path("data") / "watchlist.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS watchlist (
                symbol TEXT PRIMARY KEY,
                added_at TEXT NOT NULL,
                alert_price_above REAL,
                alert_price_below REAL,
                alert_rsi_below REAL,
                alert_rsi_above REAL,
                note TEXT,
                last_triggered TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS watchlist_alerts_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                trigger_value REAL,
                current_value REAL,
                triggered_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


_init_db()


@dataclass
class WatchlistEntry:
    symbol: str
    added_at: str
    alert_price_above: Optional[float] = None
    alert_price_below: Optional[float] = None
    alert_rsi_below: Optional[float] = None
    alert_rsi_above: Optional[float] = None
    note: Optional[str] = None
    last_triggered: Optional[str] = None


class WatchlistManager:
    """Manage persistent watchlist entries and price/RSI alerts."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def add_ticker(
        self,
        symbol: str,
        alert_price_above: Optional[float] = None,
        alert_price_below: Optional[float] = None,
        alert_rsi_below: Optional[float] = None,
        alert_rsi_above: Optional[float] = None,
        note: Optional[str] = None,
    ) -> None:
        symbol = (symbol or "").strip().upper()
        if not symbol:
            return
        entry = WatchlistEntry(
            symbol=symbol,
            added_at=datetime.utcnow().isoformat(),
            alert_price_above=alert_price_above,
            alert_price_below=alert_price_below,
            alert_rsi_below=alert_rsi_below,
            alert_rsi_above=alert_rsi_above,
            note=note or "",
            last_triggered=None,
        )
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO watchlist (
                    symbol, added_at, alert_price_above, alert_price_below,
                    alert_rsi_below, alert_rsi_above, note, last_triggered
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    alert_price_above=excluded.alert_price_above,
                    alert_price_below=excluded.alert_price_below,
                    alert_rsi_below=excluded.alert_rsi_below,
                    alert_rsi_above=excluded.alert_rsi_above,
                    note=excluded.note
                """,
                (
                    entry.symbol,
                    entry.added_at,
                    entry.alert_price_above,
                    entry.alert_price_below,
                    entry.alert_rsi_below,
                    entry.alert_rsi_above,
                    entry.note,
                    entry.last_triggered,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def remove_ticker(self, symbol: str) -> None:
        symbol = (symbol or "").strip().upper()
        if not symbol:
            return
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol,))
            conn.commit()
        finally:
            conn.close()

    def update_alert(self, symbol: str, **kwargs: Any) -> None:
        symbol = (symbol or "").strip().upper()
        if not symbol or not kwargs:
            return
        allowed_keys = {
            "alert_price_above",
            "alert_price_below",
            "alert_rsi_below",
            "alert_rsi_above",
            "note",
            "last_triggered",
        }
        to_set = {k: v for k, v in kwargs.items() if k in allowed_keys}
        if not to_set:
            return
        sets = ", ".join(f"{k} = ?" for k in to_set.keys())
        values = list(to_set.values())
        values.append(symbol)
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(f"UPDATE watchlist SET {sets} WHERE symbol = ?", values)
            conn.commit()
        finally:
            conn.close()

    def get_all(self) -> List[Dict[str, Any]]:
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM watchlist ORDER BY symbol ASC")
            rows = cur.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def _log_alert(
        self,
        symbol: str,
        alert_type: str,
        trigger_value: Optional[float],
        current_value: Optional[float],
    ) -> None:
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO watchlist_alerts_log (
                    symbol, alert_type, trigger_value, current_value, triggered_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    alert_type,
                    trigger_value,
                    current_value,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def check_alerts(
        self, current: Dict[str, Dict[str, Optional[float]]]
    ) -> List[Dict[str, Any]]:
        """
        Check all watchlist entries against current prices/RSI.

        current: mapping symbol -> {"price": float, "rsi": float}
        Returns list of triggered alert dicts.
        """
        alerts: List[Dict[str, Any]] = []
        entries = self.get_all()
        if not entries:
            return alerts

        for row in entries:
            symbol = row["symbol"]
            now_info = current.get(symbol.upper()) or current.get(symbol)
            if not now_info:
                continue
            price = now_info.get("price")
            rsi = now_info.get("rsi")
            last_trig = row.get("last_triggered")
            already_triggered_today = False
            if last_trig:
                try:
                    dt = datetime.fromisoformat(str(last_trig))
                    already_triggered_today = dt.date() == datetime.utcnow().date()
                except Exception:
                    already_triggered_today = False

            trigger_types: List[str] = []
            # Price above
            above = row.get("alert_price_above")
            if price is not None and above is not None and price >= above:
                trigger_types.append("price_above")
            # Price below
            below = row.get("alert_price_below")
            if price is not None and below is not None and price <= below:
                trigger_types.append("price_below")
            # RSI below
            rsi_below = row.get("alert_rsi_below")
            if rsi is not None and rsi_below is not None and rsi <= rsi_below:
                trigger_types.append("rsi_below")
            # RSI above
            rsi_above = row.get("alert_rsi_above")
            if rsi is not None and rsi_above is not None and rsi >= rsi_above:
                trigger_types.append("rsi_above")

            if not trigger_types or already_triggered_today:
                continue

            for t in trigger_types:
                trigger_val: Optional[float]
                current_val: Optional[float]
                if t == "price_above":
                    trigger_val, current_val = above, price
                elif t == "price_below":
                    trigger_val, current_val = below, price
                elif t == "rsi_below":
                    trigger_val, current_val = rsi_below, rsi
                else:
                    trigger_val, current_val = rsi_above, rsi

                alert = {
                    "symbol": symbol,
                    "alert_type": t,
                    "trigger_value": trigger_val,
                    "current_value": current_val,
                    "triggered_at": datetime.utcnow().isoformat(),
                }
                alerts.append(alert)
                self._log_alert(
                    symbol=symbol,
                    alert_type=t,
                    trigger_value=trigger_val,
                    current_value=current_val,
                )

            # Update last_triggered timestamp when any alert fires
            self.update_alert(symbol, last_triggered=datetime.utcnow().isoformat())

        return alerts

    def get_alert_history(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return recent alert history, optionally filtered by symbol."""
        conn = self._conn()
        try:
            cur = conn.cursor()
            if symbol:
                cur.execute(
                    "SELECT * FROM watchlist_alerts_log WHERE symbol = ? ORDER BY triggered_at DESC",
                    (symbol.upper(),),
                )
            else:
                cur.execute(
                    "SELECT * FROM watchlist_alerts_log ORDER BY triggered_at DESC"
                )
            rows = cur.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

