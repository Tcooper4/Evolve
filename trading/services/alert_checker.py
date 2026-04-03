# -*- coding: utf-8 -*-
"""Evaluate saved user alerts against live quotes and AI scores."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _last_close_from_hist(hist) -> Optional[float]:
    if hist is None or hist.empty:
        return None
    try:
        _cm = {c.lower(): c for c in hist.columns}
        cc = _cm.get("close", hist.columns[0])
        return float(hist[cc].iloc[-1])
    except Exception as e:
        logger.debug("alert_checker: last close from hist failed: %s", e)
        return None


def check_alerts_for_user(session_id: str) -> List[Dict[str, Any]]:
    """
    Load user alerts, check current prices and AI scores against conditions.
    Returns list of triggered alerts.
    """
    out: List[Dict[str, Any]] = []
    if not session_id:
        return out
    try:
        from config.user_store import load_user_preferences
        from trading.data.price_cache import get_history, get_quote

        prefs = load_user_preferences(session_id) or {}
        raw = prefs.get("evolve_alerts", [])
        if not isinstance(raw, list):
            return out

        for a in raw:
            if not isinstance(a, dict):
                continue
            sym = str(a.get("symbol", "")).strip().upper()
            cond = str(a.get("condition", "") or "")
            try:
                thr = float(a.get("threshold", 0) or 0)
            except Exception:
                thr = 0.0
            if not sym or not cond:
                continue

            try:
                q = get_quote(sym)
                price = q.get("price")
                if price is None:
                    price = _last_close_from_hist(
                        get_history(sym, period="5d")
                    )

                row: Dict[str, Any] = {
                    "symbol": sym,
                    "condition": cond,
                    "threshold": thr,
                    "alert_id": a.get("id"),
                }
                if price is not None:
                    row["current_price"] = float(price)

                fired = False
                if cond == "price_above" and price is not None:
                    fired = float(price) >= thr
                elif cond == "price_below" and price is not None:
                    fired = float(price) <= thr
                elif cond == "ai_score_above":
                    from trading.analysis.ai_score import compute_ai_score

                    sc = compute_ai_score(sym)
                    ov = float(sc.get("overall_score") or 0)
                    row["ai_score"] = ov
                    fired = ov >= thr
                elif cond == "pct_change":
                    h = get_history(sym, period="5d")
                    if h is not None and len(h) >= 2:
                        _cm = {c.lower(): c for c in h.columns}
                        cc = _cm.get("close", h.columns[0])
                        c0 = float(h[cc].iloc[-1])
                        c1 = float(h[cc].iloc[-2])
                        if c1:
                            chg_pct = (c0 / c1 - 1.0) * 100.0
                            row["pct_change_1d"] = round(chg_pct, 3)
                            fired = abs(chg_pct) >= thr

                if fired:
                    out.append(row)
            except Exception as e:
                logger.debug("alert_checker: skip %s: %s", sym, e)
                continue

        return out
    except Exception as e:
        logger.debug("check_alerts_for_user failed: %s", e)
        return out
