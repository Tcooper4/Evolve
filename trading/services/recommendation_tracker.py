# -*- coding: utf-8 -*-
"""Persist and evaluate tracked AI recommendations per user."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_REC_KEY = "evolve_recommendations"


class RecommendationTracker:
    """
    Saves AI Score recommendations with entry price and checks outcomes.
    """

    def save_recommendation(
        self,
        session_id: str,
        symbol: str,
        action: str,
        entry_price: float,
        target_price: float,
        stop_price: float,
        ai_score: float,
        timestamp: str,
    ) -> None:
        """Save to user_store preferences under recommendations list."""
        if not session_id:
            raise ValueError("session_id required")
        try:
            from config.user_store import load_user_preferences, save_user_preferences

            prefs = load_user_preferences(session_id) or {}
            recs = list(prefs.get(_REC_KEY, []))
            if not isinstance(recs, list):
                recs = []
            entry = {
                "symbol": str(symbol).strip().upper(),
                "action": str(action or "HOLD").upper(),
                "entry_price": float(entry_price),
                "target_price": float(target_price),
                "stop_price": float(stop_price),
                "ai_score": float(ai_score),
                "timestamp": timestamp,
                "closed": False,
                "outcome": None,
            }
            recs.append(entry)
            prefs[_REC_KEY] = recs
            save_user_preferences(session_id, prefs)
        except Exception as e:
            logger.warning("save_recommendation failed: %s", e)
            raise

    def clear_closed_recommendations(self, session_id: str) -> int:
        """Remove closed rows from stored recommendations; returns count removed."""
        if not session_id:
            return 0
        try:
            from config.user_store import load_user_preferences, save_user_preferences

            prefs = load_user_preferences(session_id) or {}
            recs = prefs.get(_REC_KEY, [])
            if not isinstance(recs, list):
                return 0
            before = len(recs)
            kept = [
                r for r in recs
                if isinstance(r, dict) and not r.get("closed")
            ]
            prefs[_REC_KEY] = kept
            save_user_preferences(session_id, prefs)
            return before - len(kept)
        except Exception as e:
            logger.warning("clear_closed_recommendations failed: %s", e)
            return 0

    def check_outcomes(self, session_id: str) -> List[Dict[str, Any]]:
        """
        For each open recommendation, fetch current price and P&L stats.
        Auto-closes on target, stop, or >30d hold; persists updates to user_store.
        """
        rows: List[Dict[str, Any]] = []
        if not session_id:
            return rows
        try:
            from config.user_store import load_user_preferences, save_user_preferences
            from trading.data.price_cache import get_history, get_quote

            prefs = load_user_preferences(session_id) or {}
            recs = list(prefs.get(_REC_KEY, []))
            if not isinstance(recs, list):
                return rows

            prefs_dirty = False

            for i, r in enumerate(recs):
                if not isinstance(r, dict) or r.get("closed"):
                    continue
                sym = str(r.get("symbol", "")).strip().upper()
                if not sym:
                    continue
                try:
                    ep = float(r.get("entry_price") or 0)
                    tp = float(r.get("target_price") or 0)
                    sp = float(r.get("stop_price") or 0)
                    ts = r.get("timestamp") or ""
                    days_held = None
                    try:
                        t0 = datetime.fromisoformat(
                            str(ts).replace("Z", "+00:00")
                        )
                        days_held = max(
                            0,
                            (datetime.now() - t0.replace(tzinfo=None)).days,
                        )
                    except Exception:
                        pass

                    q = get_quote(sym)
                    px = q.get("price")
                    if px is None:
                        try:
                            h = get_history(sym, period="5d")
                            if h is not None and not h.empty:
                                _cm = {c.lower(): c for c in h.columns}
                                cc = _cm.get("close", h.columns[0])
                                px = float(h[cc].iloc[-1])
                        except Exception as _he:
                            logger.debug("check_outcomes hist fallback %s: %s", sym, _he)

                    act = str(r.get("action", "BUY")).upper()
                    outcome = None
                    if px is not None and ep > 0:
                        try:
                            px = float(px)
                            if act == "BUY":
                                if sp > 0 and px <= sp:
                                    outcome = "STOP_HIT"
                                elif tp > 0 and px >= tp:
                                    outcome = "TARGET_HIT"
                            elif act == "SELL":
                                if sp > 0 and px >= sp:
                                    outcome = "STOP_HIT"
                                elif tp > 0 and px <= tp:
                                    outcome = "TARGET_HIT"
                        except Exception as _pe:
                            logger.debug("check_outcomes price rules %s: %s", sym, _pe)

                    if outcome is None and days_held is not None and days_held > 30:
                        outcome = "EXPIRED"

                    if outcome is not None:
                        try:
                            recs[i]["closed"] = True
                            recs[i]["outcome"] = outcome
                            recs[i]["closed_at"] = datetime.now().isoformat()
                            prefs_dirty = True
                        except Exception as _ce:
                            logger.debug("check_outcomes close row %s: %s", sym, _ce)
                        continue

                    if px is None or ep <= 0:
                        continue
                    px = float(px)
                    pnl_pct = (px - ep) / ep * 100.0
                    if act == "SELL":
                        pnl_pct = -pnl_pct
                    t_hit = (
                        (act == "BUY" and px >= tp)
                        or (act == "SELL" and px <= tp)
                    ) if tp else False
                    s_hit = (
                        (act == "BUY" and px <= sp)
                        or (act == "SELL" and px >= sp)
                    ) if sp else False
                    rows.append({
                        "index": i,
                        "symbol": sym,
                        "action": act,
                        "entry_price": ep,
                        "current_price": px,
                        "pnl_pct": round(pnl_pct, 2),
                        "target_hit": t_hit,
                        "stop_hit": s_hit,
                        "days_held": days_held,
                        "timestamp": ts,
                    })
                except Exception as e:
                    logger.debug("check_outcomes row skip %s: %s", sym, e)
                    continue

            if prefs_dirty:
                try:
                    prefs[_REC_KEY] = recs
                    save_user_preferences(session_id, prefs)
                except Exception as se:
                    logger.warning("check_outcomes save after auto-close failed: %s", se)
            return rows
        except Exception as e:
            logger.debug("check_outcomes failed: %s", e)
            return rows

    def get_performance_summary(self, session_id: str) -> Dict[str, Any]:
        """Aggregate win rate, returns, counts from open + resolved logic."""
        outcomes = self.check_outcomes(session_id)
        if not outcomes:
            return {
                "total_recommendations": 0,
                "win_rate": None,
                "avg_return_pct": None,
                "best_trade_pct": None,
                "worst_trade_pct": None,
            }
        pnls = [float(o["pnl_pct"]) for o in outcomes if o.get("pnl_pct") is not None]
        wins = sum(
            1
            for o in outcomes
            if o.get("target_hit") and not o.get("stop_hit")
        )
        decided = sum(
            1 for o in outcomes
            if o.get("target_hit") or o.get("stop_hit")
        )
        return {
            "total_recommendations": len(outcomes),
            "win_rate": (
                round(wins / max(decided, 1), 3)
                if decided
                else None
            ),
            "avg_return_pct": (
                round(sum(pnls) / len(pnls), 3) if pnls else None
            ),
            "best_trade_pct": max(pnls) if pnls else None,
            "worst_trade_pct": min(pnls) if pnls else None,
        }
