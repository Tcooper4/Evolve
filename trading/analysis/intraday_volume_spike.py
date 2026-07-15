# -*- coding: utf-8 -*-
"""Provisional intraday volume-spike detection for live chart overlays.

Evaluates today's *still-forming* session against the prior 20 completed
days (today excluded from the baseline). Markers are provisional until
regular-session close.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from trading.analysis.volume_news_linker import (
    LINK_FALLBACK,
    LINK_SAME_DAY,
    meets_spike_thresholds,
)

logger = logging.getLogger(__name__)

PROVISIONAL_COLOR = "#F5A623"


def prior_20d_avg_volume(
    hist: pd.DataFrame,
    as_of: Optional[datetime] = None,
    *,
    baseline_method: Optional[str] = None,
) -> Optional[float]:
    """Robust baseline volume of up to 20 *completed* daily bars before ``as_of``.

    Default method matches ``detect_significant_candles`` (trimmed mean) so
    live provisional spikes use the same cluster-resistant baseline.
    """
    if hist is None or hist.empty:
        return None
    df = hist.copy()
    _cm = {str(c).lower(): c for c in df.columns}
    vol_col = _cm.get("volume")
    if vol_col is None:
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            logger.debug("prior_20d_avg_volume: bad index: %s", e)
            return None

    cut = (as_of or datetime.utcnow()).date()
    completed = df[df.index.date < cut]
    if completed.empty:
        return None
    window = completed.tail(20)
    if len(window) < 5:
        return None
    try:
        from trading.analysis.volume_baseline import (
            DEFAULT_BASELINE_METHOD,
            prior_completed_baseline,
        )

        method = baseline_method or DEFAULT_BASELINE_METHOD
        return prior_completed_baseline(
            window[vol_col], method=method  # type: ignore[arg-type]
        )
    except Exception as e:
        logger.debug("prior_20d_avg_volume: baseline failed: %s", e)
        return None


def evaluate_provisional_spike(
    hist: pd.DataFrame,
    *,
    live_price: float,
    live_volume: float,
    as_of: Optional[datetime] = None,
    prev_close: Optional[float] = None,
    day_open: Optional[float] = None,
    volume_threshold: float = 2.0,
    price_threshold: float = 0.02,
) -> Optional[Dict[str, Any]]:
    """
    Pure evaluator (no I/O). Returns a provisional event dict or None.

    Price change uses day_open when available, else prev_close.
    """
    try:
        px = float(live_price)
        vol = float(live_volume)
    except (TypeError, ValueError):
        return None
    if px <= 0 or vol < 0:
        return None

    avg = prior_20d_avg_volume(hist, as_of=as_of)
    if avg is None or avg <= 0:
        return None
    volume_ratio = vol / avg

    base = None
    if day_open is not None:
        try:
            base = float(day_open)
        except (TypeError, ValueError):
            base = None
    if (base is None or base <= 0) and prev_close is not None:
        try:
            base = float(prev_close)
        except (TypeError, ValueError):
            base = None
    if base is None or base <= 0:
        return None

    price_change_pct = (px / base) - 1.0
    if not meets_spike_thresholds(
        volume_ratio,
        price_change_pct,
        volume_threshold=volume_threshold,
        price_threshold=price_threshold,
    ):
        return None

    when = as_of or datetime.utcnow()
    date_str = when.strftime("%Y-%m-%d")
    candle_type = "bullish" if price_change_pct > 0 else (
        "bearish" if price_change_pct < 0 else "neutral"
    )
    return {
        "time": date_str,
        "provisional": True,
        "active": True,
        "volume_ratio": float(volume_ratio),
        "price_change_pct": float(price_change_pct),
        "candle_type": candle_type,
        "text": "LIVE",
        "color": PROVISIONAL_COLOR,
        "shape": "circle",
        "title": (
            f"Provisional live volume spike · "
            f"{volume_ratio:.1f}x prior-20d avg · "
            f"{price_change_pct * 100:+.1f}% (may change by close)"
        ),
    }


def attach_news_honesty(
    event: Dict[str, Any],
    headlines: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Copy event and stamp Phase-2 link_quality / date_confirmed from headlines."""
    out = dict(event)
    rows = list(headlines or [])
    out["headlines"] = rows
    if not rows:
        out["link_quality"] = LINK_SAME_DAY
        out["date_confirmed"] = False
        return out
    qualities = {str(h.get("link_quality") or "") for h in rows if isinstance(h, dict)}
    if LINK_SAME_DAY in qualities and any(
        isinstance(h, dict) and h.get("date_confirmed") for h in rows
    ):
        out["link_quality"] = LINK_SAME_DAY
        out["date_confirmed"] = True
    elif LINK_FALLBACK in qualities:
        out["link_quality"] = LINK_FALLBACK
        out["date_confirmed"] = False
        title = str(out.get("title") or "")
        if "may not be same-day" not in title.lower():
            out["title"] = f"{title} · may not be same-day headline".strip(" ·")
    else:
        out["link_quality"] = LINK_SAME_DAY
        out["date_confirmed"] = bool(rows)
    return out


def evaluate_live_volume_spike(
    symbol: str,
    live_price: Optional[float],
    live_volume: Optional[float] = None,
    prev_close: Optional[float] = None,
    day_open: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Fetch history + optional news and return a provisional WS payload."""
    if live_price is None:
        return None
    try:
        from trading.data.price_cache import get_history

        hist = get_history(symbol, period="3mo", interval="1d")
    except Exception as e:
        logger.debug("evaluate_live_volume_spike history %s: %s", symbol, e)
        return None

    vol = live_volume
    if vol is None:
        try:
            _cm = {str(c).lower(): c for c in hist.columns}
            vc = _cm.get("volume")
            if vc is not None and len(hist):
                # Last row may be today — still use only completed days for baseline;
                # for live volume prefer caller. Fallback: last bar volume.
                vol = float(hist[vc].iloc[-1])
        except Exception:
            vol = None
    if vol is None:
        return None

    if day_open is None and hist is not None and not hist.empty:
        try:
            _cm = {str(c).lower(): c for c in hist.columns}
            oc = _cm.get("open")
            if oc is not None:
                last_idx = hist.index[-1]
                if hasattr(last_idx, "date") and last_idx.date() == datetime.utcnow().date():
                    day_open = float(hist[oc].iloc[-1])
        except Exception:
            pass

    if prev_close is None and hist is not None and len(hist) >= 2:
        try:
            _cm = {str(c).lower(): c for c in hist.columns}
            cc = _cm.get("close")
            if cc is not None:
                # If last bar is today, prev close is iloc[-2]; else last close.
                last_idx = hist.index[-1]
                if hasattr(last_idx, "date") and last_idx.date() == datetime.utcnow().date():
                    prev_close = float(hist[cc].iloc[-2])
                else:
                    prev_close = float(hist[cc].iloc[-1])
        except Exception:
            pass

    core = evaluate_provisional_spike(
        hist,
        live_price=float(live_price),
        live_volume=float(vol),
        prev_close=prev_close,
        day_open=day_open,
    )
    if core is None:
        return {
            "type": "volume_spike",
            "provisional": True,
            "active": False,
            "symbol": (symbol or "").upper(),
        }

    headlines: List[Dict[str, Any]] = []
    try:
        from trading.analysis.volume_news_linker import get_news_for_date

        headlines = get_news_for_date(symbol, core["time"], n_articles=3) or []
    except Exception as e:
        logger.debug("live spike news %s: %s", symbol, e)

    event = attach_news_honesty(core, headlines)
    event["type"] = "volume_spike"
    event["symbol"] = (symbol or "").upper()
    return event
