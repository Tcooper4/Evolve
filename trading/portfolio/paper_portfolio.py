# -*- coding: utf-8 -*-
"""Paper portfolio: per-user positions with average-cost accounting.

The competitive-analysis gap this closes: every mainstream research
platform tracks the user's portfolio so the tool can answer "how am I
doing?" and "why is MY portfolio moving?". This is the PAPER version -
no broker, no real money, consistent with the platform's honest scope.

Accounting rules (average cost, long-only v1):
* BUY:  new_avg = (qty*avg + q*price) / (qty + q); quantity += q.
* SELL: realized_pnl += q * (price - avg); quantity -= q; average cost
  is UNCHANGED by sells (that's the average-cost method); position rows
  at zero quantity are removed but their realized P&L persists in the
  ledger.
* Overselling is rejected (no shorts in v1) - an honest error beats a
  silent negative position.

Storage: SQLite at data/paper_portfolio.db, keyed by user_id exactly
like the watchlist, so multi-user isolation is inherited, and personal
mode uses 'local'.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path("data/paper_portfolio.db")


def _resolve_user(user_id: Optional[str]) -> str:
    return user_id or os.getenv("EVOLVE_SESSION_ID") or "local"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS positions (
               user_id TEXT NOT NULL,
               symbol TEXT NOT NULL,
               quantity REAL NOT NULL,
               avg_cost REAL NOT NULL,
               opened_at TEXT NOT NULL,
               PRIMARY KEY (user_id, symbol)
           )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS trades (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               user_id TEXT NOT NULL,
               symbol TEXT NOT NULL,
               side TEXT NOT NULL,
               quantity REAL NOT NULL,
               price REAL NOT NULL,
               realized_pnl REAL NOT NULL DEFAULT 0,
               executed_at TEXT NOT NULL
           )"""
    )
    return conn


class PaperPortfolio:
    """Per-user paper portfolio. All methods resolve identity per call."""

    def __init__(self, user_id: Optional[str] = None):
        self.user_id = _resolve_user(user_id)

    # ------------------------------------------------------------- trades
    def record_trade(self, symbol: str, side: str, quantity: float,
                     price: float) -> Dict[str, Any]:
        symbol = (symbol or "").strip().upper()
        side = (side or "").strip().lower()
        try:
            quantity = float(quantity)
            price = float(price)
        except (TypeError, ValueError):
            return {"success": False, "error": "quantity and price must be numbers"}
        if not symbol:
            return {"success": False, "error": "symbol required"}
        if side not in ("buy", "sell"):
            return {"success": False, "error": "side must be 'buy' or 'sell'"}
        if quantity <= 0 or price <= 0:
            return {"success": False, "error": "quantity and price must be positive"}

        now = datetime.now(timezone.utc).isoformat()
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT quantity, avg_cost FROM positions"
                " WHERE user_id=? AND symbol=?",
                (self.user_id, symbol),
            ).fetchone()
            realized = 0.0
            if side == "buy":
                if row:
                    qty, avg = float(row[0]), float(row[1])
                    new_qty = qty + quantity
                    new_avg = (qty * avg + quantity * price) / new_qty
                    conn.execute(
                        "UPDATE positions SET quantity=?, avg_cost=?"
                        " WHERE user_id=? AND symbol=?",
                        (new_qty, new_avg, self.user_id, symbol),
                    )
                else:
                    conn.execute(
                        "INSERT INTO positions VALUES (?,?,?,?,?)",
                        (self.user_id, symbol, quantity, price, now),
                    )
            else:  # sell
                if not row or float(row[0]) < quantity - 1e-9:
                    held = float(row[0]) if row else 0.0
                    return {
                        "success": False,
                        "error": f"can't sell {quantity:g} {symbol}: "
                                 f"you hold {held:g} (no shorts in paper v1)",
                    }
                qty, avg = float(row[0]), float(row[1])
                realized = quantity * (price - avg)
                new_qty = qty - quantity
                if new_qty <= 1e-9:
                    conn.execute(
                        "DELETE FROM positions WHERE user_id=? AND symbol=?",
                        (self.user_id, symbol),
                    )
                else:
                    conn.execute(
                        "UPDATE positions SET quantity=?"
                        " WHERE user_id=? AND symbol=?",
                        (new_qty, self.user_id, symbol),
                    )
            conn.execute(
                "INSERT INTO trades (user_id, symbol, side, quantity, price,"
                " realized_pnl, executed_at) VALUES (?,?,?,?,?,?,?)",
                (self.user_id, symbol, side, quantity, price, realized, now),
            )
            conn.commit()
            out = {"success": True, "symbol": symbol, "side": side,
                   "quantity": quantity, "price": price}
            if side == "sell":
                out["realized_pnl"] = round(realized, 2)
            return out
        finally:
            conn.close()

    # ---------------------------------------------------------- positions
    def get_positions(self) -> List[Dict[str, Any]]:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT symbol, quantity, avg_cost, opened_at FROM positions"
                " WHERE user_id=? ORDER BY symbol",
                (self.user_id,),
            ).fetchall()
            return [
                {"symbol": r[0], "quantity": r[1], "avg_cost": r[2],
                 "opened_at": r[3]}
                for r in rows
            ]
        finally:
            conn.close()

    def get_realized_pnl(self) -> float:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT COALESCE(SUM(realized_pnl), 0) FROM trades"
                " WHERE user_id=?",
                (self.user_id,),
            ).fetchone()
            return float(row[0])
        finally:
            conn.close()

    def get_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT symbol, side, quantity, price, realized_pnl,"
                " executed_at FROM trades WHERE user_id=?"
                " ORDER BY id DESC LIMIT ?",
                (self.user_id, int(limit)),
            ).fetchall()
            return [
                {"symbol": r[0], "side": r[1], "quantity": r[2],
                 "price": r[3], "realized_pnl": r[4], "executed_at": r[5]}
                for r in rows
            ]
        finally:
            conn.close()

    # ------------------------------------------------------------ summary
    def get_summary(
        self,
        price_fn: Optional[Callable[[str], Optional[float]]] = None,
    ) -> Dict[str, Any]:
        """Positions with live P&L. ``price_fn`` maps symbol -> last price
        (injected for tests; defaults to yfinance, degrading to None so
        the summary still works offline with cost-basis values only)."""
        if price_fn is None:
            def price_fn(sym: str) -> Optional[float]:  # noqa: ANN001
                try:
                    import yfinance as yf

                    p = yf.Ticker(sym).fast_info.last_price
                    return float(p) if p else None
                except Exception:
                    return None

        positions = self.get_positions()
        total_cost = 0.0
        total_value = 0.0
        priced_all = True
        for p in positions:
            cost = p["quantity"] * p["avg_cost"]
            total_cost += cost
            last = price_fn(p["symbol"])
            if last is None:
                priced_all = False
                p.update({"last_price": None, "market_value": None,
                          "unrealized_pnl": None, "unrealized_pct": None})
                total_value += cost  # neutral: value at cost when unpriced
            else:
                value = p["quantity"] * last
                pnl = value - cost
                p.update({
                    "last_price": round(last, 4),
                    "market_value": round(value, 2),
                    "unrealized_pnl": round(pnl, 2),
                    "unrealized_pct": round(pnl / cost * 100, 2) if cost else 0.0,
                })
                total_value += value
        return {
            "success": True,
            "positions": positions,
            "total_cost_basis": round(total_cost, 2),
            "total_market_value": round(total_value, 2),
            "total_unrealized_pnl": round(total_value - total_cost, 2),
            "realized_pnl": round(self.get_realized_pnl(), 2),
            "all_prices_live": priced_all,
        }
