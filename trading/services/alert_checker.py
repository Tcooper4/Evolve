# -*- coding: utf-8 -*-
"""Evaluate saved user alerts against live quotes and AI scores.

One-shot semantics: when a condition fires, the alert record is persisted
with ``status="triggered"`` and is not re-evaluated until an explicit
re-arm. This protects both the background job loop and GET /api/alerts.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

STATUS_ACTIVE = "active"
STATUS_TRIGGERED = "triggered"

# Optional second factor (AND). Absent/empty = single-factor legacy behavior.
CONFIRM_VOLUME_GE = "volume_ge"  # session vol >= Nx 20d average
CONFIRM_RSI_LE = "rsi_le"
CONFIRM_RSI_GE = "rsi_ge"
CONFIRM_TYPES = frozenset({CONFIRM_VOLUME_GE, CONFIRM_RSI_LE, CONFIRM_RSI_GE})


def is_alert_armed(alert: Dict[str, Any]) -> bool:
    """True when the alert may still fire (missing status => armed / legacy)."""
    if not isinstance(alert, dict):
        return False
    status = str(alert.get("status") or STATUS_ACTIVE).strip().lower()
    return status != STATUS_TRIGGERED


def mark_alert_triggered(
    alert: Dict[str, Any],
    *,
    price: Optional[float] = None,
    at: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a copy of ``alert`` stamped as one-shot fired."""
    row = dict(alert)
    row["status"] = STATUS_TRIGGERED
    row["triggered_at"] = at or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    if price is not None:
        try:
            row["triggered_price"] = float(price)
        except Exception:
            pass
    if extra:
        for k, v in extra.items():
            if v is not None:
                row[k] = v
    return row


def rearm_alert_record(alert: Dict[str, Any]) -> Dict[str, Any]:
    """Clear fired state so the alert can evaluate again."""
    row = dict(alert)
    row["status"] = STATUS_ACTIVE
    row.pop("triggered_at", None)
    row.pop("triggered_price", None)
    row.pop("triggered_ai_score", None)
    row.pop("triggered_pct_change", None)
    return row


def parse_confirm_factor(
    alert: Dict[str, Any],
) -> Optional[Tuple[str, float]]:
    """Return ``(confirm_type, confirm_threshold)`` or None for single-factor."""
    if not isinstance(alert, dict):
        return None
    raw = alert.get("confirm") or alert.get("confirm_type") or ""
    ctype = str(raw).strip().lower()
    if not ctype or ctype in ("none", "off", "null"):
        return None
    if ctype not in CONFIRM_TYPES:
        return None
    try:
        thr = float(alert.get("confirm_threshold", 0) or 0)
    except Exception:
        return None
    return ctype, thr


def compound_should_fire(
    primary_ok: bool,
    confirm_type: Optional[str],
    confirm_ok: Optional[bool],
) -> bool:
    """AND semantics: both factors required when a confirm type is set.

    Truth table (confirm unset): fire iff primary_ok.
    Truth table (confirm set): fire iff primary_ok AND confirm_ok is True.
    Missing confirm metric (confirm_ok is None) does **not** fire.
    """
    if not primary_ok:
        return False
    if not confirm_type:
        return True
    return confirm_ok is True


def confirming_factor_met(
    confirm_type: str,
    confirm_threshold: float,
    *,
    volume_ratio: Optional[float] = None,
    rsi: Optional[float] = None,
) -> Tuple[Optional[bool], Dict[str, Any]]:
    """Evaluate the optional second factor from precomputed metrics."""
    extras: Dict[str, Any] = {"confirm": confirm_type, "confirm_threshold": confirm_threshold}
    ct = str(confirm_type or "").strip().lower()
    if ct == CONFIRM_VOLUME_GE:
        if volume_ratio is None:
            return None, extras
        extras["volume_ratio"] = float(volume_ratio)
        return float(volume_ratio) >= float(confirm_threshold), extras
    if ct == CONFIRM_RSI_LE:
        if rsi is None:
            return None, extras
        extras["rsi"] = float(rsi)
        return float(rsi) <= float(confirm_threshold), extras
    if ct == CONFIRM_RSI_GE:
        if rsi is None:
            return None, extras
        extras["rsi"] = float(rsi)
        return float(rsi) >= float(confirm_threshold), extras
    return None, extras


def fetch_volume_ratio(symbol: str, hist=None) -> Optional[float]:
    """Last-bar volume / 20d avg — same math as ``detect_significant_candles``."""
    try:
        from trading.analysis.volume_news_linker import detect_significant_candles
        from trading.data.price_cache import get_history

        h = hist
        if h is None:
            h = get_history(symbol, period="3mo")
        if h is None or getattr(h, "empty", True):
            return None
        tagged = detect_significant_candles(h)
        if tagged is None or tagged.empty or "volume_ratio" not in tagged.columns:
            return None
        val = tagged["volume_ratio"].iloc[-1]
        if val is None or (isinstance(val, float) and val != val):  # NaN
            return None
        return float(val)
    except Exception as e:
        logger.debug("alert_checker: volume_ratio failed for %s: %s", symbol, e)
        return None


def fetch_rsi(symbol: str, hist=None, period: int = 14) -> Optional[float]:
    """Wilder RSI via market_scanner (aligned with scanner / charting)."""
    try:
        import numpy as np

        from trading.analysis.market_scanner import _rsi
        from trading.data.price_cache import get_history

        h = hist
        if h is None:
            h = get_history(symbol, period="6mo")
        if h is None or getattr(h, "empty", True):
            return None
        _cm = {c.lower(): c for c in h.columns}
        cc = _cm.get("close")
        if cc is None:
            return None
        closes = np.asarray(h[cc].astype(float).values, dtype=float)
        return _rsi(closes, period=period)
    except Exception as e:
        logger.debug("alert_checker: rsi failed for %s: %s", symbol, e)
        return None


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


def _persist_alerts(session_id: str, prefs: Dict[str, Any], alerts: List[Dict]) -> None:
    from config.user_store import save_user_preferences

    next_prefs = dict(prefs or {})
    next_prefs["evolve_alerts"] = alerts
    save_user_preferences(session_id, next_prefs)


def check_alerts_for_user(session_id: str) -> List[Dict[str, Any]]:
    """
    Load user alerts, check current prices and AI scores against conditions.
    Returns list of alerts that fired *on this call*, after marking them
    triggered in prefs so the next call will not re-fire them.
    """
    out: List[Dict[str, Any]] = []
    if not session_id:
        return out
    try:
        from config.user_store import load_user_preferences
        from trading.data.price_cache import get_history, get_quote

        try:
            import pytz
            from trading.utils.time_utils import MarketHours

            _mh = MarketHours(timezone="America/New_York")
            _market_open = _mh.is_market_open(
                datetime.now(pytz.timezone("America/New_York"))
            )
        except Exception:
            _market_open = True

        prefs = load_user_preferences(session_id) or {}
        raw = prefs.get("evolve_alerts", [])
        if not isinstance(raw, list):
            return out

        updated_alerts: List[Any] = []
        dirty = False
        fired_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        for a in raw:
            if not isinstance(a, dict):
                updated_alerts.append(a)
                continue

            if not is_alert_armed(a):
                updated_alerts.append(a)
                continue

            sym = str(a.get("symbol", "")).strip().upper()
            cond = str(a.get("condition", "") or "")
            try:
                thr = float(a.get("threshold", 0) or 0)
            except Exception:
                thr = 0.0
            if not sym or not cond:
                updated_alerts.append(a)
                continue

            try:
                q = get_quote(sym)
                price = q.get("price") if isinstance(q, dict) else None
                if price is None:
                    price = _last_close_from_hist(
                        get_history(sym, period="5d")
                    )

                from trading.services.alert_push_policy import (
                    MODE_ACTION,
                    normalize_alert_mode,
                )

                row: Dict[str, Any] = {
                    "symbol": sym,
                    "condition": cond,
                    "threshold": thr,
                    "alert_id": a.get("id"),
                    # Legacy missing mode = action (preserve prior push behavior).
                    "mode": normalize_alert_mode(a.get("mode"), default=MODE_ACTION),
                }
                confirm_parsed = parse_confirm_factor(a)
                if confirm_parsed:
                    row["confirm"] = confirm_parsed[0]
                    row["confirm_threshold"] = confirm_parsed[1]
                if price is not None:
                    row["current_price"] = float(price)

                primary_ok = False
                stamp_extra: Dict[str, Any] = {}
                if cond == "price_above" and price is not None:
                    if not _market_open:
                        updated_alerts.append(a)
                        continue
                    primary_ok = float(price) >= thr
                elif cond == "price_below" and price is not None:
                    if not _market_open:
                        updated_alerts.append(a)
                        continue
                    primary_ok = float(price) <= thr
                elif cond in ("ai_score_above", "score_above"):
                    from trading.analysis.ai_score import compute_ai_score

                    sc = compute_ai_score(sym)
                    ov = float(sc.get("overall_score") or 0)
                    row["ai_score"] = ov
                    stamp_extra["triggered_ai_score"] = ov
                    primary_ok = ov >= thr
                elif cond in ("ai_score_below", "score_below"):
                    from trading.analysis.ai_score import compute_ai_score

                    sc = compute_ai_score(sym)
                    ov = float(sc.get("overall_score") or 0)
                    row["ai_score"] = ov
                    stamp_extra["triggered_ai_score"] = ov
                    primary_ok = ov <= thr
                elif cond == "pct_change":
                    if not _market_open:
                        updated_alerts.append(a)
                        continue
                    h = get_history(sym, period="5d")
                    if h is not None and len(h) >= 2:
                        _cm = {c.lower(): c for c in h.columns}
                        cc = _cm.get("close", h.columns[0])
                        c0 = float(h[cc].iloc[-1])
                        c1 = float(h[cc].iloc[-2])
                        if c1:
                            chg_pct = (c0 / c1 - 1.0) * 100.0
                            row["pct_change_1d"] = round(chg_pct, 3)
                            stamp_extra["triggered_pct_change"] = round(chg_pct, 3)
                            primary_ok = abs(chg_pct) >= thr

                # Optional confirming factor — only evaluate when primary is true
                # (avoids extra I/O) and never alters legacy single-factor alerts.
                confirm_ok: Optional[bool] = None
                confirm_type: Optional[str] = None
                if confirm_parsed and primary_ok:
                    confirm_type, confirm_thr = confirm_parsed
                    vol_r = None
                    rsi_v = None
                    if confirm_type == CONFIRM_VOLUME_GE:
                        vol_r = fetch_volume_ratio(sym)
                    elif confirm_type in (CONFIRM_RSI_LE, CONFIRM_RSI_GE):
                        rsi_v = fetch_rsi(sym)
                    confirm_ok, c_extras = confirming_factor_met(
                        confirm_type,
                        confirm_thr,
                        volume_ratio=vol_r,
                        rsi=rsi_v,
                    )
                    row.update({k: v for k, v in c_extras.items() if k in (
                        "volume_ratio", "rsi",
                    )})
                    stamp_extra.update(c_extras)

                fired = compound_should_fire(
                    primary_ok,
                    confirm_parsed[0] if confirm_parsed else None,
                    confirm_ok if confirm_parsed else None,
                )

                if fired:
                    stamped = mark_alert_triggered(
                        a, price=float(price) if price is not None else None,
                        at=fired_at, extra=stamp_extra,
                    )
                    dirty = True
                    updated_alerts.append(stamped)
                    row["status"] = STATUS_TRIGGERED
                    row["triggered_at"] = stamped["triggered_at"]
                    if "triggered_price" in stamped:
                        row["triggered_price"] = stamped["triggered_price"]
                    out.append(row)
                else:
                    updated_alerts.append(a)
            except Exception as e:
                logger.debug("alert_checker: skip %s: %s", sym, e)
                updated_alerts.append(a)
                continue

        if dirty:
            try:
                _persist_alerts(session_id, prefs, updated_alerts)
            except Exception as e:
                logger.warning(
                    "alert_checker: failed to persist triggered state for %s: %s",
                    session_id, e,
                )

        return out
    except Exception as e:
        logger.debug("check_alerts_for_user failed: %s", e)
        return out


def rearm_alert_for_user(
    session_id: str, alert_id: str
) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
    """
    Explicitly re-arm one alert. Returns (ok, alerts, error).
    """
    if not session_id or not alert_id:
        return False, [], "missing session or alert id"
    try:
        from config.user_store import load_user_preferences, save_user_preferences

        prefs = dict(load_user_preferences(session_id) or {})
        raw = list(prefs.get("evolve_alerts") or [])
        found = False
        next_alerts: List[Dict[str, Any]] = []
        for a in raw:
            if not isinstance(a, dict):
                continue
            if str(a.get("id")) == str(alert_id):
                next_alerts.append(rearm_alert_record(a))
                found = True
            else:
                next_alerts.append(a)
        if not found:
            return False, [
                a for a in raw if isinstance(a, dict)
            ], "alert not found"
        prefs["evolve_alerts"] = next_alerts
        save_user_preferences(session_id, prefs)
        return True, next_alerts, None
    except Exception as e:
        logger.debug("rearm_alert_for_user failed: %s", e)
        return False, [], str(e)
