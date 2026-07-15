# -*- coding: utf-8 -*-
"""Options-specific Kelly × VIX sizing overlay (conditional, never >1.0).

Distinct from ``conditional_vol_sizing`` (asset realized-vol tercile). This
module scales options risk using **VIX level / VIX percentile vs VIX's own
history** — the pattern from S&P 500 options sizing research that combines
Kelly with VIX-based dynamic cuts.

Discipline (same family as conditional vol overlay):
* Cut size only when VIX is in an *elevated* band (upper tercile of its
  trailing percentile, or absolute level ≥ ``VIX_ABS_ELEVATED``).
* Multiplier is never above 1.0 (no levering up in calm VIX).
* Not continuous inverse-VIX scaling.

Live wiring: ``LIVE_OPTIONS_VIX_SIZING_ENABLED`` stays False until an OOS
champion/challenger-style gate clears (see ``validate_options_vix_universe``).
Phase 3 ships the utility + validation; Phase 4 may wire surfaces only if
``recommend_live`` is true — a null / mixed result is an acceptable outcome.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Do not auto-wire into get_position_size until OOS broad-win clears
LIVE_OPTIONS_VIX_SIZING_ENABLED = False

ELEVATED_VIX_MULTIPLIER = 0.50
# Upper tercile of trailing VIX distribution → elevated
VIX_PERCENTILE_ELEVATED = 2.0 / 3.0
# Absolute floor also counts as elevated (classic "VIX > 25" style band)
VIX_ABS_ELEVATED = 25.0
# Trailing window for VIX percentile (trading days)
VIX_LOOKBACK = 252

MIN_DD_IMPROVEMENT = 0.05
MAX_SHARPE_COST = 0.15


def _max_drawdown(equity: np.ndarray) -> float:
    if equity is None or len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak == 0, 1.0, peak)
    return float(-np.min(dd)) if len(dd) else 0.0


def _sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 5:
        return 0.0
    sig = float(np.std(r, ddof=1))
    if sig <= 1e-12:
        return 0.0
    return float(np.mean(r) / sig * np.sqrt(periods_per_year))


def vix_trailing_percentile(
    vix_history: Union[pd.Series, np.ndarray, Sequence[float]],
    *,
    lookback: int = VIX_LOOKBACK,
) -> Dict[str, Any]:
    """Percentile of the latest VIX print vs its own trailing window."""
    s = pd.to_numeric(pd.Series(vix_history), errors="coerce").dropna()
    out: Dict[str, Any] = {
        "vix": None,
        "percentile": None,
        "lookback": int(lookback),
        "n": int(len(s)),
        "sufficient": False,
    }
    if len(s) < max(20, lookback // 10):
        return out
    window = s.iloc[-min(len(s), lookback) :]
    latest = float(window.iloc[-1])
    # Fraction of window ≤ latest
    pct = float((window <= latest).mean())
    out.update({
        "vix": latest,
        "percentile": pct,
        "sufficient": True,
    })
    return out


def options_vix_multiplier(
    vix: Optional[float] = None,
    percentile: Optional[float] = None,
    *,
    elevated_multiplier: float = ELEVATED_VIX_MULTIPLIER,
    percentile_threshold: float = VIX_PERCENTILE_ELEVATED,
    abs_elevated: float = VIX_ABS_ELEVATED,
) -> Dict[str, Any]:
    """
    Conditional options size multiplier from VIX level / percentile.

    Elevated if ``percentile >= threshold`` OR ``vix >= abs_elevated``.
    Otherwise ×1.0. Never returns multiplier > 1.0.
    """
    try:
        high_m = float(elevated_multiplier)
    except Exception:
        high_m = ELEVATED_VIX_MULTIPLIER
    high_m = max(0.05, min(1.0, high_m))

    out: Dict[str, Any] = {
        "multiplier": 1.0,
        "regime": "insufficient",
        "scaled_down": False,
        "vix": None if vix is None else float(vix),
        "vix_percentile": None if percentile is None else float(percentile),
        "reason": "Insufficient VIX context — no options size cut.",
    }

    has_pct = percentile is not None and np.isfinite(float(percentile))
    has_lvl = vix is not None and np.isfinite(float(vix))
    if not has_pct and not has_lvl:
        return out

    elevated = False
    why: List[str] = []
    if has_pct and float(percentile) >= float(percentile_threshold):
        elevated = True
        why.append(
            f"VIX percentile {float(percentile):.0%} ≥ "
            f"{float(percentile_threshold):.0%} (upper-tercile band)"
        )
    if has_lvl and float(vix) >= float(abs_elevated):
        elevated = True
        why.append(f"VIX {float(vix):.1f} ≥ {float(abs_elevated):.0f}")

    if elevated:
        out["multiplier"] = high_m
        out["regime"] = "elevated"
        out["scaled_down"] = True
        out["reason"] = (
            "Elevated VIX for options sizing — "
            + "; ".join(why)
            + f" — size cut to {high_m:.0%} of Kelly "
            "(conditional; never scales above 1.0)."
        )
    else:
        out["regime"] = "normal"
        out["scaled_down"] = False
        bits = []
        if has_pct:
            bits.append(f"percentile {float(percentile):.0%}")
        if has_lvl:
            bits.append(f"VIX {float(vix):.1f}")
        out["reason"] = (
            "VIX outside elevated band ("
            + ", ".join(bits)
            + ") — options overlay leaves Kelly unchanged (×1.0)."
        )
    # Hard cap
    out["multiplier"] = min(1.0, float(out["multiplier"]))
    return out


def options_vix_multiplier_from_history(
    vix_history: Union[pd.Series, np.ndarray, Sequence[float]],
    *,
    elevated_multiplier: float = ELEVATED_VIX_MULTIPLIER,
) -> Dict[str, Any]:
    """Compute multiplier from a VIX price series (latest bar)."""
    stats = vix_trailing_percentile(vix_history)
    if not stats.get("sufficient"):
        return {
            "multiplier": 1.0,
            "regime": "insufficient",
            "scaled_down": False,
            "vix": stats.get("vix"),
            "vix_percentile": stats.get("percentile"),
            "reason": "Insufficient VIX history — no options size cut.",
        }
    return options_vix_multiplier(
        stats.get("vix"),
        stats.get("percentile"),
        elevated_multiplier=elevated_multiplier,
    )


def fetch_vix_history(period: str = "2y") -> Optional[pd.Series]:
    """Load ^VIX close via price cache / yfinance."""
    try:
        from trading.data.price_cache import get_history

        hist = get_history("^VIX", period=period)
        if hist is None or getattr(hist, "empty", True):
            import yfinance as yf

            hist = yf.Ticker("^VIX").history(period=period)
        if hist is None or hist.empty:
            return None
        cm = {str(c).lower(): c for c in hist.columns}
        close_c = cm.get("close")
        if not close_c:
            return None
        return pd.to_numeric(hist[close_c], errors="coerce").dropna()
    except Exception as e:
        logger.warning("fetch_vix_history failed: %s", e)
        return None


def options_vix_multiplier_live(
    *,
    elevated_multiplier: float = ELEVATED_VIX_MULTIPLIER,
    period: str = "2y",
) -> Dict[str, Any]:
    """Live VIX pull → conditional options multiplier."""
    series = fetch_vix_history(period=period)
    if series is None or series.empty:
        return {
            "multiplier": 1.0,
            "regime": "insufficient",
            "scaled_down": False,
            "reason": "Could not load VIX — no options size cut.",
        }
    return options_vix_multiplier_from_history(
        series, elevated_multiplier=elevated_multiplier
    )


def apply_kelly_options_vix_overlay(
    kelly_result: Dict[str, Any],
    vix_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach options-VIX-adjusted Kelly fields; raw Kelly left intact."""
    if not isinstance(kelly_result, dict) or not kelly_result.get("success"):
        return kelly_result
    out = dict(kelly_result)
    mult = float(vix_info.get("multiplier") or 1.0)
    mult = max(0.0, min(1.0, mult))
    half = float(out.get("half_kelly_fraction") or 0.0)
    full = float(out.get("full_kelly_fraction") or 0.0)
    dollars = out.get("half_kelly_dollars")
    out["options_vix_regime"] = vix_info.get("regime")
    out["options_vix_multiplier"] = round(mult, 4)
    out["options_vix_scaled_down"] = bool(vix_info.get("scaled_down"))
    out["options_vix"] = vix_info.get("vix")
    out["options_vix_percentile"] = vix_info.get("vix_percentile")
    out["options_vix_reason"] = vix_info.get("reason")
    out["half_kelly_fraction_options_vix_adjusted"] = round(half * mult, 4)
    out["full_kelly_fraction_options_vix_adjusted"] = round(full * mult, 4)
    if dollars is not None:
        try:
            out["half_kelly_dollars_options_vix_adjusted"] = round(
                float(dollars) * mult, 2
            )
        except Exception:
            pass
    note = str(out.get("note") or "")
    vix_note = str(vix_info.get("reason") or "")
    out["note"] = (note + " " + vix_note).strip()
    return out


def simulate_options_vix_paths(
    close: pd.Series,
    vix: pd.Series,
    *,
    elevated_multiplier: float = ELEVATED_VIX_MULTIPLIER,
    min_history: int = 60,
) -> Dict[str, Any]:
    """
    Causal OOS: constant exposure vs VIX-conditional size on the same returns.

    At each t, VIX percentile uses only vix[:t+1] (no peek-ahead).
    Proxy for options risk capital on an underlying — not a full options
    backtest (paper book has no options fills to replay).
    """
    px = pd.to_numeric(close, errors="coerce")
    vx = pd.to_numeric(vix, errors="coerce")
    # Align on calendar dates (VIX vs equity indexes often differ in tz)
    if isinstance(px.index, pd.DatetimeIndex):
        px = px.copy()
        px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    if isinstance(vx.index, pd.DatetimeIndex):
        vx = vx.copy()
        vx.index = pd.to_datetime(vx.index).tz_localize(None).normalize()
    df = pd.concat([px.rename("px"), vx.rename("vix")], axis=1).dropna()
    if len(df) < min_history + 5:
        return {
            "success": False,
            "error": (
                f"insufficient aligned VIX/price history "
                f"(n={len(df)}, need ≥{min_history + 5})"
            ),
        }

    rets = df["px"].pct_change()
    base_rets: List[float] = []
    overlay_rets: List[float] = []
    mults: List[float] = []

    for t in range(min_history, len(df)):
        r = float(rets.iloc[t]) if np.isfinite(rets.iloc[t]) else 0.0
        info = options_vix_multiplier_from_history(
            df["vix"].iloc[: t + 1],
            elevated_multiplier=elevated_multiplier,
        )
        m = float(info["multiplier"])
        base_rets.append(r)
        overlay_rets.append(r * m)
        mults.append(m)

    base_arr = np.asarray(base_rets, dtype=float)
    over_arr = np.asarray(overlay_rets, dtype=float)
    eq_base = np.cumprod(1.0 + base_arr)
    eq_over = np.cumprod(1.0 + over_arr)

    sharpe_base = _sharpe(base_arr)
    sharpe_over = _sharpe(over_arr)
    dd_base = _max_drawdown(eq_base)
    dd_over = _max_drawdown(eq_over)
    dd_improvement = dd_base - dd_over
    sharpe_delta = sharpe_over - sharpe_base

    adopt = bool(
        dd_improvement > abs(MIN_DD_IMPROVEMENT)
        and sharpe_delta >= -abs(MAX_SHARPE_COST)
    )
    reason = (
        "Overlay clears DD improvement gate without unacceptable Sharpe cost"
        if adopt
        else (
            f"No adopt: dd_improvement={dd_improvement:.3f} "
            f"(need >{MIN_DD_IMPROVEMENT}), sharpe_delta={sharpe_delta:.3f} "
            f"(floor -{MAX_SHARPE_COST})"
        )
    )
    return {
        "success": True,
        "n_steps": len(base_rets),
        "pct_time_scaled_down": round(float(np.mean(np.asarray(mults) < 1.0)), 3),
        "baseline": {
            "sharpe": round(sharpe_base, 4),
            "max_drawdown": round(dd_base, 4),
        },
        "options_vix_overlay": {
            "sharpe": round(sharpe_over, 4),
            "max_drawdown": round(dd_over, 4),
        },
        "dd_improvement": round(dd_improvement, 4),
        "sharpe_delta": round(sharpe_delta, 4),
        "adopt_overlay": adopt,
        "reason": reason,
        "elevated_multiplier": elevated_multiplier,
        "proxy": "underlying_equity_returns_x_vix_conditional_size",
    }


def validate_options_vix_universe(
    symbols: Optional[Sequence[str]] = None,
    *,
    period: str = "2y",
    elevated_multiplier: float = ELEVATED_VIX_MULTIPLIER,
) -> Dict[str, Any]:
    """Per-symbol OOS vs VIX overlay; report helped/hurt honestly."""
    from trading.data.price_cache import get_history

    vix = fetch_vix_history(period=period)
    if vix is None or vix.empty:
        return {
            "success": False,
            "error": "VIX history unavailable",
            "recommend_live": False,
            "broad_win": False,
        }

    syms = list(symbols or ("SPY", "QQQ", "IWM", "EFA", "TLT"))
    rows: List[Dict[str, Any]] = []
    for sym in syms:
        try:
            hist = get_history(sym, period=period)
            if hist is None or hist.empty:
                rows.append({"symbol": sym, "success": False, "error": "no history"})
                continue
            cm = {str(c).lower(): c for c in hist.columns}
            close = hist[cm.get("close", hist.columns[0])]
            sim = simulate_options_vix_paths(
                close, vix, elevated_multiplier=elevated_multiplier
            )
            sim["symbol"] = sym
            rows.append(sim)
        except Exception as e:
            rows.append({"symbol": sym, "success": False, "error": str(e)})

    ok = [r for r in rows if r.get("success")]
    adopt_n = sum(1 for r in ok if r.get("adopt_overlay"))
    broad = bool(ok and adopt_n / len(ok) >= 0.67)
    return {
        "success": True,
        "n_symbols": len(ok),
        "n_adopt": adopt_n,
        "broad_win": broad,
        "recommend_live": False,  # Phase 3 never auto-wires
        "recommend_live_candidate": broad,
        "live_flag": LIVE_OPTIONS_VIX_SIZING_ENABLED,
        "note": (
            "Broad win on equity-proxy OOS — candidate for Phase 4 options "
            "sizing wire after review (live flag still off in Phase 3)."
            if broad
            else (
                "No broad win — keep options VIX overlay research-only "
                "(same discipline as model-routing; null/mixed is acceptable)."
            )
        ),
        "min_dd_improvement": MIN_DD_IMPROVEMENT,
        "max_sharpe_cost": MAX_SHARPE_COST,
        "proxy_caveat": (
            "OOS uses underlying equity returns × VIX-conditional size as a "
            "risk-capital proxy — not a multi-leg 0DTE options fill replay "
            "(paper book has no options trade history to validate against). "
            "Equity-style ~5 bps spread assumptions understate liquid ATM "
            "options costs by roughly 20–100×+; use "
            "trading.backtesting.options_cost_model (ATM floor several %% "
            "of option mid; observed bid/ask preferred). Full options-fill "
            "replay remains out of scope."
        ),
        "results": rows,
    }


__all__ = [
    "LIVE_OPTIONS_VIX_SIZING_ENABLED",
    "ELEVATED_VIX_MULTIPLIER",
    "VIX_PERCENTILE_ELEVATED",
    "VIX_ABS_ELEVATED",
    "vix_trailing_percentile",
    "options_vix_multiplier",
    "options_vix_multiplier_from_history",
    "options_vix_multiplier_live",
    "apply_kelly_options_vix_overlay",
    "simulate_options_vix_paths",
    "validate_options_vix_universe",
]
