# -*- coding: utf-8 -*-
"""Toggleable strategy signal overlay for Dashboard/Analyze charts.

Research / backtest guide only — never framed as live trade instructions.
Default-off: no strategy is auto-selected; clearing an OOS gate for a
default-on pick is out of scope until a strategy clears a self_tune-style
champion/challenger bar (same discipline as VIX options sizing).

Gamma annotation
----------------
``trading.data.gamma_exposure`` only yields a *current* delayed-chain
snapshot. Free yfinance chains do not supply historical dealer GEX
as-of each past signal date. Therefore:

* Panel-level ``gamma_context`` may show today's regime (display-only).
* Per-marker ``gamma_tag`` is set ONLY when the signal bar is the latest
  bar in the series (i.e. "today's" / last closed bar context), using the
  current snapshot — never invents "fired during X regime" for older bars.
* Markers are NEVER filtered, weighted, or re-ranked by GEX.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Hard off — OOS gate must clear before any default-on behavior ships
DEFAULT_STRATEGY_OVERLAY_ENABLED = False

DISCLOSURE = (
    "Backtest signals / research guide only — not trade instructions. "
    "Markers use delayed free market data; past performance is not a live "
    "recommendation. Gamma regime tags (when present) reuse the current "
    "delayed-chain GEX snapshot and do not filter which signals appear."
)

LABEL_BUY = "backtest signal / research guide — buy signal"
LABEL_SELL = "backtest signal / research guide — sell signal"


def signal_events_from_series(
    dates: Sequence[Any],
    signals: Sequence[float],
    closes: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """
    Extract edge events from a strategy ``signal`` series.

    Hand-verifiable: emit when the series *enters* a non-zero state or
    flips sign. Flat zeros in between are ignored. ``+1`` → buy, ``-1`` → sell.
    """
    sig = pd.to_numeric(pd.Series(list(signals)), errors="coerce").fillna(0.0)
    if len(sig) != len(dates):
        raise ValueError("dates and signals length mismatch")
    closes_s: Optional[pd.Series] = None
    if closes is not None:
        closes_s = pd.to_numeric(pd.Series(list(closes)), errors="coerce")
        if len(closes_s) != len(dates):
            raise ValueError("closes length mismatch")

    events: List[Dict[str, Any]] = []
    prev = 0.0
    for i, raw in enumerate(sig.tolist()):
        s = float(raw)
        # Only emit on transition into a new signed state
        if s == 0.0:
            prev = 0.0
            continue
        if s == prev:
            continue
        side = "buy" if s > 0 else "sell"
        d = dates[i]
        if hasattr(d, "strftime"):
            time_s = d.strftime("%Y-%m-%d")
        else:
            time_s = str(d)[:10]
        px = None
        if closes_s is not None and np.isfinite(closes_s.iloc[i]):
            px = float(closes_s.iloc[i])
        events.append({
            "time": time_s,
            "side": side,
            "signal": int(1 if s > 0 else -1),
            "price": px,
            "label": LABEL_BUY if side == "buy" else LABEL_SELL,
            "gamma_tag": None,
        })
        prev = s
    return events


def reference_levels_from_last_bar(
    signals_df: pd.DataFrame,
    strategy: str,
) -> Dict[str, Any]:
    """Optional reference levels from the last bar — not a live order ticket."""
    if signals_df is None or signals_df.empty:
        return {"levels": [], "note": "No signal frame for reference levels."}
    last = signals_df.iloc[-1]
    cmap = {str(c).lower(): c for c in signals_df.columns}
    levels: List[Dict[str, Any]] = []
    name = (strategy or "").lower()

    def _add(key: str, label: str, *, price_scale: bool = True) -> None:
        col = cmap.get(key)
        if col is None:
            return
        try:
            v = float(last[col])
        except Exception:
            return
        if not np.isfinite(v):
            return
        levels.append({
            "key": key,
            "label": label,
            "value": round(v, 4),
            "price_scale": price_scale,
        })

    if "bollinger" in name or "atr" in name:
        _add("upper_band", "Upper band (reference)")
        _add("middle_band", "Middle band (reference)")
        _add("lower_band", "Lower band (reference)")
    elif "sma" in name:
        _add("short_sma", "Short SMA (reference)")
        _add("long_sma", "Long SMA (reference)")
    elif "macd" in name:
        _add("macd_line", "MACD line (reference)", price_scale=False)
        _add("signal_line", "MACD signal line (reference)", price_scale=False)
    elif "rsi" in name:
        _add("rsi", "RSI (reference)", price_scale=False)
        levels.append({
            "key": "rsi_os", "label": "RSI oversold ref (30)",
            "value": 30.0, "price_scale": False,
        })
        levels.append({
            "key": "rsi_ob", "label": "RSI overbought ref (70)",
            "value": 70.0, "price_scale": False,
        })
    elif "cci" in name:
        _add("cci", "CCI (reference)", price_scale=False)

    return {
        "levels": levels,
        "note": (
            "Reference levels from the last closed bar — research context, "
            "not a live trade target or invalidation instruction."
        ),
    }


def price_overlay_series(
    signals_df: pd.DataFrame,
    strategy: str,
) -> List[Dict[str, Any]]:
    """
    Full-history price-scale guides (SMA / Bollinger / ATR bands) as dotted
    line overlays. RSI/MACD are excluded — wrong scale for the candle pane.
    """
    if signals_df is None or signals_df.empty:
        return []
    name = (strategy or "").lower()
    cmap = {str(c).lower(): c for c in signals_df.columns}

    specs: List[tuple] = []
    if "bollinger" in name or "atr" in name:
        specs = [
            ("upper_band", "Upper band", "#7eb6ff"),
            ("middle_band", "Middle band", "#8a96ab"),
            ("lower_band", "Lower band", "#7eb6ff"),
        ]
    elif "sma" in name:
        specs = [
            ("short_sma", "Short SMA", "#5eead4"),
            ("long_sma", "Long SMA", "#c4b5fd"),
        ]
    else:
        return []

    out: List[Dict[str, Any]] = []
    for key, label, color in specs:
        col = cmap.get(key)
        if col is None:
            continue
        points: List[Dict[str, Any]] = []
        for idx, val in signals_df[col].items():
            try:
                v = float(val)
            except Exception:
                continue
            if not np.isfinite(v):
                continue
            t = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
            points.append({"time": t, "value": round(v, 4)})
        if len(points) < 2:
            continue
        out.append({
            "id": key,
            "label": label,
            "color": color,
            "style": "dotted",
            "points": points,
        })
    return out


def _attach_current_gamma_to_latest(
    events: List[Dict[str, Any]],
    last_time: Optional[str],
    gamma_context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Display-only: tag only the latest-bar signal with current GEX snapshot."""
    if not events or not last_time or not gamma_context.get("available"):
        return events
    regime = gamma_context.get("regime_short")
    regime_long = gamma_context.get("regime")
    if not regime:
        return events
    out = []
    for ev in events:
        e = dict(ev)
        if e.get("time") == last_time[:10]:
            e["gamma_tag"] = (
                f"current GEX snapshot: {str(regime).replace('_', ' ')} "
                f"(display-only; does not filter signals)"
            )
            e["gamma_regime_short"] = regime
            e["gamma_regime"] = regime_long
        out.append(e)
    return out


def _current_gamma_context(symbol: str) -> Dict[str, Any]:
    try:
        from trading.data.gamma_exposure import DATA_DISCLOSURE, get_gamma_exposure

        gex = get_gamma_exposure(symbol)
        if not gex.get("success"):
            return {
                "available": False,
                "reason": gex.get("error") or "GEX unavailable",
                "disclosure": DATA_DISCLOSURE,
            }
        return {
            "available": True,
            "regime_short": gex.get("regime_short"),
            "regime": gex.get("regime"),
            "net_gex": gex.get("net_gex"),
            "gamma_flip": gex.get("gamma_flip"),
            "spot": gex.get("spot"),
            "disclosure": gex.get("disclosure") or DATA_DISCLOSURE,
            "historical_note": (
                "GEX is a current delayed-chain snapshot only — it is not "
                "as-of each historical signal date, and it does not change "
                "which backtest markers appear."
            ),
        }
    except Exception as e:
        logger.debug("gamma context for overlay failed: %s", e)
        return {
            "available": False,
            "reason": str(e),
            "disclosure": (
                "Computed on delayed/free option-chain data when available."
            ),
        }


def build_strategy_overlay(
    symbol: str,
    strategy: str,
    *,
    period: str = "6mo",
    params: Optional[Dict[str, Any]] = None,
    include_gamma_context: bool = True,
    max_markers: int = 80,
) -> Dict[str, Any]:
    """
    Run a registered strategy and return chart markers + reference levels.

    Always includes honesty disclosure. Never recommends default-on.
    """
    sym = (symbol or "").strip().upper()
    strat = (strategy or "").strip()
    out: Dict[str, Any] = {
        "success": False,
        "symbol": sym,
        "strategy": strat,
        "markers": [],
        "reference_levels": {"levels": [], "note": None},
        "overlay_series": [],
        "gamma_context": None,
        "disclosure": DISCLOSURE,
        "default_on": DEFAULT_STRATEGY_OVERLAY_ENABLED,
        "framing": "backtest_signal_research_guide",
        "error": None,
    }
    if not sym or not strat:
        out["error"] = "symbol and strategy required"
        return out

    try:
        import trading.strategies  # noqa: F401 — populate registry
        from trading.data.price_cache import get_history
        from trading.optimization.strategy_backtest_objective import run_strategy
        from trading.services.self_tune import get_adopted_params

        hist = get_history(
            sym, period=period if period not in ("1d", "5d", "1w") else "3mo"
        )
        if hist is None or getattr(hist, "empty", True):
            out["error"] = f"no history for {sym}"
            return out

        adopted = get_adopted_params(strat, sym)
        use_params = dict(params or adopted or {})
        signals_df = run_strategy(strat, hist, use_params or None)
        if signals_df is None or signals_df.empty or "signal" not in signals_df.columns:
            out["error"] = "strategy produced no signal column"
            return out

        cm = {str(c).lower(): c for c in signals_df.columns}
        close_col = cm.get("close")
        closes = signals_df[close_col] if close_col else None
        events = signal_events_from_series(
            list(signals_df.index),
            list(signals_df["signal"]),
            list(closes) if closes is not None else None,
        )

        gamma_context = (
            _current_gamma_context(sym)
            if include_gamma_context
            else {"available": False}
        )
        last_time = None
        if len(signals_df.index):
            d = signals_df.index[-1]
            last_time = (
                d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
            )
        events = _attach_current_gamma_to_latest(events, last_time, gamma_context)

        if len(events) > max_markers:
            events = events[-max_markers:]

        markers = []
        for ev in events:
            buy = ev["side"] == "buy"
            title_parts = [ev["label"], f"{strat}"]
            if ev.get("price") is not None:
                title_parts.append(f"@ {ev['price']:.2f}")
            if ev.get("gamma_tag"):
                title_parts.append(str(ev["gamma_tag"]))
            markers.append({
                "time": ev["time"],
                "position": "belowBar" if buy else "aboveBar",
                "color": "#2eea8b" if buy else "#ff5470",
                "shape": "arrowUp" if buy else "arrowDown",
                "text": "B" if buy else "S",
                "title": " · ".join(title_parts),
                "side": ev["side"],
                "label": ev["label"],
                "price": ev.get("price"),
                "gamma_tag": ev.get("gamma_tag"),
            })

        levels = reference_levels_from_last_bar(signals_df, strat)
        overlays = price_overlay_series(signals_df, strat)
        out.update({
            "success": True,
            "markers": markers,
            "n_markers": len(markers),
            "reference_levels": levels,
            "overlay_series": overlays,
            "gamma_context": gamma_context,
            "params_used": use_params,
            "period": period,
            "last_bar": last_time,
        })
        return out
    except Exception as e:
        logger.warning("build_strategy_overlay failed: %s", e)
        out["error"] = str(e)
        return out


__all__ = [
    "DEFAULT_STRATEGY_OVERLAY_ENABLED",
    "DISCLOSURE",
    "signal_events_from_series",
    "reference_levels_from_last_bar",
    "price_overlay_series",
    "build_strategy_overlay",
]
