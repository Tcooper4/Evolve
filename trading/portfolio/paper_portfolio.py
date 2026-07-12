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

# Every account starts with this much paper cash. Matches the figure the
# cashbook UI has shown since it was first built, so existing users see
# no surprise change in their displayed balance on first migration.
DEFAULT_STARTING_CASH = 100_000.0


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
    conn.execute(
        """CREATE TABLE IF NOT EXISTS cash (
               user_id TEXT PRIMARY KEY,
               balance REAL NOT NULL
           )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS recommendations (
               id TEXT PRIMARY KEY,
               user_id TEXT NOT NULL,
               symbol TEXT NOT NULL,
               source TEXT NOT NULL DEFAULT 'analyze',
               score REAL,
               price_at_rec REAL,
               note TEXT,
               created_at TEXT NOT NULL,
               status TEXT NOT NULL DEFAULT 'open',
               acted_at TEXT,
               acted_price REAL,
               closed_at TEXT,
               closed_price REAL
           )"""
    )
    # Migrate older DBs that predate acted/closed lifecycle columns
    try:
        existing = {
            str(r[1])
            for r in conn.execute("PRAGMA table_info(recommendations)").fetchall()
        }
        for col, typedef in (
            ("status", "TEXT NOT NULL DEFAULT 'open'"),
            ("acted_at", "TEXT"),
            ("acted_price", "REAL"),
            ("closed_at", "TEXT"),
            ("closed_price", "REAL"),
        ):
            if col not in existing:
                conn.execute(
                    f"ALTER TABLE recommendations ADD COLUMN {col} {typedef}"
                )
        conn.commit()
    except Exception as e:
        logger.debug("recommendations schema migrate: %s", e)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS limit_orders (
               id TEXT PRIMARY KEY,
               user_id TEXT NOT NULL,
               symbol TEXT NOT NULL,
               side TEXT NOT NULL,
               quantity REAL NOT NULL,
               limit_price REAL NOT NULL,
               status TEXT NOT NULL DEFAULT 'open',
               created_at TEXT NOT NULL,
               filled_at TEXT,
               filled_price REAL
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
        new_qty = 0.0
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT quantity, avg_cost FROM positions"
                " WHERE user_id=? AND symbol=?",
                (self.user_id, symbol),
            ).fetchone()
            cash_row = conn.execute(
                "SELECT balance FROM cash WHERE user_id=?", (self.user_id,)
            ).fetchone()
            cash = float(cash_row[0]) if cash_row else DEFAULT_STARTING_CASH
            trade_value = quantity * price

            realized = 0.0
            if side == "buy":
                # CASH INTEGRATION (2026-07): buys used to be free - the
                # cash book and positions were two disconnected ledgers,
                # so buying $1,000 of AAPL left the displayed cash
                # balance untouched at the full $100,000. A buy must
                # actually cost cash, and an account can't spend money it
                # doesn't have (paper trading should still teach real
                # constraints).
                if trade_value > cash + 1e-6:
                    conn.close()
                    return {
                        "success": False,
                        "error": f"insufficient paper cash: have "
                                 f"${cash:,.2f}, need ${trade_value:,.2f}",
                    }
                cash -= trade_value
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
                    conn.close()
                    return {
                        "success": False,
                        "error": f"can't sell {quantity:g} {symbol}: "
                                 f"you hold {held:g} (no shorts in paper v1)",
                    }
                qty, avg = float(row[0]), float(row[1])
                realized = quantity * (price - avg)
                cash += trade_value
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
                "INSERT INTO cash (user_id, balance) VALUES (?,?)"
                " ON CONFLICT(user_id) DO UPDATE SET balance=excluded.balance",
                (self.user_id, round(cash, 2)),
            )
            conn.execute(
                "INSERT INTO trades (user_id, symbol, side, quantity, price,"
                " realized_pnl, executed_at) VALUES (?,?,?,?,?,?,?)",
                (self.user_id, symbol, side, quantity, price, realized, now),
            )
            # Tracker lifecycle: open → acted on buy; acted → closed on full exit
            rec_update = self._apply_recommendation_trade(
                conn, symbol, side, price, now,
                fully_exited=(side == "sell" and new_qty <= 1e-9)
                if side == "sell" else False,
            )
            conn.commit()
            out = {"success": True, "symbol": symbol, "side": side,
                   "quantity": quantity, "price": price,
                   "cash_balance": round(cash, 2)}
            if side == "sell":
                out["realized_pnl"] = round(realized, 2)
            if rec_update:
                out["recommendation"] = rec_update
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

    # ----------------------------------------------------------------- cash
    def get_cash(self) -> float:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT balance FROM cash WHERE user_id=?", (self.user_id,)
            ).fetchone()
            return float(row[0]) if row else DEFAULT_STARTING_CASH
        finally:
            conn.close()

    def adjust_cash(self, amount: float, note: str = "") -> Dict[str, Any]:
        """Manual deposit/withdrawal (not a trade). Negative amounts
        withdraw; balance is floored at 0 (no paper margin/debt)."""
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT balance FROM cash WHERE user_id=?", (self.user_id,)
            ).fetchone()
            cash = float(row[0]) if row else DEFAULT_STARTING_CASH
            cash = max(0.0, cash + float(amount))
            conn.execute(
                "INSERT INTO cash (user_id, balance) VALUES (?,?)"
                " ON CONFLICT(user_id) DO UPDATE SET balance=excluded.balance",
                (self.user_id, round(cash, 2)),
            )
            conn.commit()
            return {"success": True, "cash": round(cash, 2), "note": note}
        finally:
            conn.close()

    # ------------------------------------------------------------ limit orders
    def place_limit_order(self, symbol: str, side: str, quantity: float,
                          limit_price: float,
                          order_id: Optional[str] = None) -> Dict[str, Any]:
        import uuid

        symbol = (symbol or "").strip().upper()
        side = (side or "").strip().lower()
        try:
            quantity = float(quantity)
            limit_price = float(limit_price)
        except (TypeError, ValueError):
            return {"success": False, "error": "quantity and limit_price must be numbers"}
        if side not in ("buy", "sell"):
            return {"success": False, "error": "side must be 'buy' or 'sell'"}
        if quantity <= 0 or limit_price <= 0:
            return {"success": False, "error": "quantity and limit_price must be positive"}
        if side == "sell":
            held = next((p["quantity"] for p in self.get_positions()
                        if p["symbol"] == symbol), 0.0)
            if held < quantity - 1e-9:
                return {"success": False,
                       "error": f"can't place sell limit for {quantity:g} "
                                f"{symbol}: you hold {held:g}"}
        oid = order_id or str(uuid.uuid4())[:8]
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO limit_orders (id, user_id, symbol, side,"
                " quantity, limit_price, status, created_at)"
                " VALUES (?,?,?,?,?,?, 'open', ?)",
                (oid, self.user_id, symbol, side, quantity, limit_price,
                 datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            return {"success": True, "id": oid}
        finally:
            conn.close()

    def cancel_limit_order(self, order_id: str) -> Dict[str, Any]:
        conn = _connect()
        try:
            cur = conn.execute(
                "DELETE FROM limit_orders WHERE id=? AND user_id=?"
                " AND status='open'",
                (order_id, self.user_id),
            )
            conn.commit()
            return {"success": cur.rowcount > 0}
        finally:
            conn.close()

    def get_limit_orders(self, include_filled: bool = True) -> List[Dict[str, Any]]:
        conn = _connect()
        try:
            q = ("SELECT id, symbol, side, quantity, limit_price, status,"
                 " created_at, filled_at, filled_price FROM limit_orders"
                 " WHERE user_id=?")
            if not include_filled:
                q += " AND status='open'"
            rows = conn.execute(q + " ORDER BY created_at DESC",
                                (self.user_id,)).fetchall()
            return [
                {"id": r[0], "symbol": r[1], "side": r[2], "quantity": r[3],
                 "limit_price": r[4], "status": r[5], "created_at": r[6],
                 "filled_at": r[7], "filled_price": r[8]}
                for r in rows
            ]
        finally:
            conn.close()

    def check_limit_orders(
        self,
        price_fn: Optional[Callable[[str], Optional[float]]] = None,
    ) -> List[Dict[str, Any]]:
        """Check open limit orders against the current price and FILL any
        that cross - mirrors the existing alert checker pattern. Buys fill
        at or below the limit; sells fill at or above. A fill that can't
        execute (insufficient cash, position closed elsewhere) is left
        open rather than silently dropped, and reported back."""
        if price_fn is None:
            def price_fn(sym: str) -> Optional[float]:  # noqa: ANN001
                try:
                    import yfinance as yf

                    p = yf.Ticker(sym).fast_info.last_price
                    return float(p) if p else None
                except Exception:
                    return None

        filled: List[Dict[str, Any]] = []
        for order in self.get_limit_orders(include_filled=False):
            last = price_fn(order["symbol"])
            if last is None:
                continue
            crosses = (
                (order["side"] == "buy" and last <= order["limit_price"])
                or (order["side"] == "sell" and last >= order["limit_price"])
            )
            if not crosses:
                continue
            result = self.record_trade(order["symbol"], order["side"],
                                       order["quantity"], last)
            if not result.get("success"):
                continue  # left open (e.g. insufficient cash); reported next check
            conn = _connect()
            try:
                conn.execute(
                    "UPDATE limit_orders SET status='filled', filled_at=?,"
                    " filled_price=? WHERE id=? AND user_id=?",
                    (datetime.now(timezone.utc).isoformat(), last,
                     order["id"], self.user_id),
                )
                conn.commit()
            finally:
                conn.close()
            filled.append({**order, "status": "filled", "filled_price": last})
        return filled

    # ---------------------------------------------------- recommendations
    def _apply_recommendation_trade(
        self,
        conn: sqlite3.Connection,
        symbol: str,
        side: str,
        price: float,
        now: str,
        fully_exited: bool,
    ) -> Optional[Dict[str, Any]]:
        """Advance tracker status from paper fills (same DB connection)."""
        try:
            if side == "buy":
                cur = conn.execute(
                    "UPDATE recommendations SET status='acted',"
                    " acted_at=?, acted_price=?"
                    " WHERE user_id=? AND symbol=? AND status='open'",
                    (now, float(price), self.user_id, symbol),
                )
                if cur.rowcount > 0:
                    return {"symbol": symbol, "status": "acted",
                            "acted_price": float(price)}
            elif side == "sell" and fully_exited:
                cur = conn.execute(
                    "UPDATE recommendations SET status='closed',"
                    " closed_at=?, closed_price=?"
                    " WHERE user_id=? AND symbol=? AND status='acted'",
                    (now, float(price), self.user_id, symbol),
                )
                if cur.rowcount > 0:
                    return {"symbol": symbol, "status": "closed",
                            "closed_price": float(price)}
        except Exception as e:
            logger.debug("recommendation trade sync skipped: %s", e)
        return None

    def track_recommendation(self, symbol: str, source: str = "analyze",
                             score: Optional[float] = None,
                             price_at_rec: Optional[float] = None,
                             note: str = "") -> Dict[str, Any]:
        """Save an idea to watch WITHOUT buying - closes the learning loop
        ('how do the ideas I liked actually perform?'). Price at rec time
        is captured so performance-since can be computed honestly later."""
        import uuid

        symbol = (symbol or "").strip().upper()
        if not symbol:
            return {"success": False, "error": "symbol required"}
        rid = str(uuid.uuid4())[:8]
        conn = _connect()
        try:
            # one live open/acted rec per symbol: re-tracking replaces those;
            # keep closed history rows for the learning loop
            conn.execute(
                "DELETE FROM recommendations WHERE user_id=? AND symbol=?"
                " AND status IN ('open', 'acted')",
                (self.user_id, symbol),
            )
            conn.execute(
                "INSERT INTO recommendations (id, user_id, symbol, source,"
                " score, price_at_rec, note, created_at, status,"
                " acted_at, acted_price, closed_at, closed_price)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rid, self.user_id, symbol, source,
                 float(score) if score is not None else None,
                 float(price_at_rec) if price_at_rec is not None else None,
                 note or "", datetime.now(timezone.utc).isoformat(),
                 "open", None, None, None, None),
            )
            conn.commit()
            return {"success": True, "id": rid, "symbol": symbol, "status": "open"}
        finally:
            conn.close()

    def delete_recommendation(self, rec_id: str) -> Dict[str, Any]:
        conn = _connect()
        try:
            cur = conn.execute(
                "DELETE FROM recommendations WHERE id=? AND user_id=?",
                (rec_id, self.user_id),
            )
            conn.commit()
            return {"success": cur.rowcount > 0}
        finally:
            conn.close()

    def get_recommendations(
        self,
        price_fn: Optional[Callable[[str], Optional[float]]] = None,
    ) -> List[Dict[str, Any]]:
        """Tracked ideas with performance-since-tracked, computed from the
        captured rec-time price vs current price (None-safe offline)."""
        if price_fn is None:
            def price_fn(sym: str) -> Optional[float]:  # noqa: ANN001
                try:
                    import yfinance as yf

                    p = yf.Ticker(sym).fast_info.last_price
                    return float(p) if p else None
                except Exception:
                    return None

        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT id, symbol, source, score, price_at_rec, note,"
                " created_at, status, acted_at, acted_price,"
                " closed_at, closed_price"
                " FROM recommendations WHERE user_id=?"
                " ORDER BY CASE status"
                " WHEN 'open' THEN 0 WHEN 'acted' THEN 1 ELSE 2 END,"
                " created_at DESC",
                (self.user_id,),
            ).fetchall()
        finally:
            conn.close()
        out: List[Dict[str, Any]] = []
        for r in rows:
            status = (r[7] or "open") if len(r) > 7 else "open"
            rec: Dict[str, Any] = {
                "id": r[0], "symbol": r[1], "source": r[2], "score": r[3],
                "price_at_rec": r[4], "note": r[5], "created_at": r[6],
                "status": status,
                "acted_at": r[8] if len(r) > 8 else None,
                "acted_price": r[9] if len(r) > 9 else None,
                "closed_at": r[10] if len(r) > 10 else None,
                "closed_price": r[11] if len(r) > 11 else None,
                "last_price": None, "change_pct": None,
                "change_since_acted_pct": None,
            }
            # Closed ideas: mark-to-market vs exit; open/acted vs live last
            if status == "closed" and rec["closed_price"] is not None:
                last = float(rec["closed_price"])
                rec["last_price"] = round(last, 4)
            else:
                last_opt = price_fn(rec["symbol"])
                last = float(last_opt) if last_opt is not None else None
                if last is not None:
                    rec["last_price"] = round(last, 4)
            if last is not None and rec["price_at_rec"]:
                rec["change_pct"] = round(
                    (float(last) / float(rec["price_at_rec"]) - 1) * 100, 2
                )
            if (
                last is not None
                and rec.get("acted_price")
                and float(rec["acted_price"] or 0) > 0
            ):
                rec["change_since_acted_pct"] = round(
                    (float(last) / float(rec["acted_price"]) - 1) * 100, 2
                )
            out.append(rec)
        return out

    # -------------------------------------------------------- trade stats
    def get_trade_stats(self) -> Dict[str, Any]:
        """Win rate + average win/loss ratio from CLOSED (realized) paper
        trades - the honest inputs to Kelly sizing. Only sells realize
        P&L, so only sells count; needs a handful before the numbers
        mean anything."""
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT realized_pnl FROM trades WHERE user_id=?"
                " AND side='sell'",
                (self.user_id,),
            ).fetchall()
        finally:
            conn.close()
        pnls = [float(r[0]) for r in rows]
        wins = [p for p in pnls if p > 0]
        losses = [-p for p in pnls if p < 0]
        n = len(pnls)
        out: Dict[str, Any] = {"closed_trades": n}
        if n == 0:
            out.update({"win_rate": None, "avg_win_loss_ratio": None})
            return out
        out["win_rate"] = round(len(wins) / n, 4)
        if wins and losses:
            out["avg_win_loss_ratio"] = round(
                (sum(wins) / len(wins)) / (sum(losses) / len(losses)), 4
            )
        else:
            out["avg_win_loss_ratio"] = None  # one-sided history: not usable
        return out

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
        cash = self.get_cash()
        return {
            "success": True,
            "positions": positions,
            "cash": round(cash, 2),
            "total_cost_basis": round(total_cost, 2),
            "total_market_value": round(total_value, 2),
            "total_equity": round(cash + total_value, 2),
            "total_unrealized_pnl": round(total_value - total_cost, 2),
            "realized_pnl": round(self.get_realized_pnl(), 2),
            "all_prices_live": priced_all,
        }
