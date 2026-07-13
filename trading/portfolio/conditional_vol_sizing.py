# -*- coding: utf-8 -*-
"""Conditional volatility targeting for position size (overlay on Kelly).

Literature: *continuous* trailing-vol scaling often raises drawdowns and
turnover. *Conditional* targeting — cut exposure only at volatility
extremes — is the evidence-backed pattern. This module never leverages
up in low vol; multiplier is 1.0 outside the extreme-high band.

Reuses ``get_series_features(...).volatility_regime`` tercile bands
(high = upper third of 21d realized vol vs its own history).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Cut size only in extreme-high vol; never scale above 1.0
HIGH_VOL_MULTIPLIER = 0.50
# Champion/challenger: require this relative DD improvement to prefer overlay
MIN_DD_IMPROVEMENT = 0.05  # 5pp of max-DD (e.g. 0.30 → 0.25 clears)
# Reject overlay if Sharpe falls by more than this absolute amount
MAX_SHARPE_COST = 0.15


def _to_close_frame(data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, pd.Series):
        return pd.DataFrame({"Close": data.astype(float)})
    arr = np.asarray(data, dtype=float).ravel()
    return pd.DataFrame({"Close": arr})


def conditional_vol_multiplier(
    data: Union[pd.DataFrame, pd.Series, np.ndarray],
    *,
    high_multiplier: float = HIGH_VOL_MULTIPLIER,
) -> Dict[str, Any]:
    """Return size multiplier from causal vol regime on ``data``.

    Outside extreme-high: multiplier = 1.0 (conditional, not continuous).
    Extreme-high: multiplier = ``high_multiplier`` (< 1). Never > 1.
    """
    try:
        high_m = float(high_multiplier)
    except Exception:
        high_m = HIGH_VOL_MULTIPLIER
    high_m = max(0.05, min(1.0, high_m))

    out: Dict[str, Any] = {
        "multiplier": 1.0,
        "regime": "insufficient",
        "reason": "Insufficient history for volatility regime — no size cut.",
        "scaled_down": False,
    }
    try:
        from trading.models.forecast_router import get_series_features

        df = _to_close_frame(data)
        feats = get_series_features(df)
        regime = str(feats.get("volatility_regime") or "insufficient")
        out["regime"] = regime
        out["features"] = {
            "data_length": feats.get("data_length"),
            "volatility_regime": regime,
        }
        if regime == "high":
            out["multiplier"] = high_m
            out["scaled_down"] = True
            out["reason"] = (
                f"Extreme-high volatility regime (upper tercile of 21d "
                f"realized vol) — size cut to {high_m:.0%} of Kelly "
                f"(conditional targeting; not continuous vol scaling)."
            )
        elif regime == "insufficient":
            out["reason"] = (
                "Insufficient history for volatility regime — no size cut."
            )
        else:
            out["reason"] = (
                f"Volatility regime '{regime}' — outside extreme-high band; "
                "conditional overlay leaves Kelly size unchanged (×1.0)."
            )
    except Exception as e:
        logger.warning("conditional_vol_multiplier failed: %s", e)
        out["reason"] = f"Vol overlay unavailable ({e}) — using Kelly only."
    return out


def conditional_vol_multiplier_for_symbol(
    symbol: str,
    *,
    period: str = "1y",
    high_multiplier: float = HIGH_VOL_MULTIPLIER,
) -> Dict[str, Any]:
    """Load history for ``symbol`` and compute the conditional multiplier."""
    sym = (symbol or "").strip().upper()
    base = {
        "multiplier": 1.0,
        "regime": "insufficient",
        "reason": "No symbol — no vol overlay.",
        "scaled_down": False,
        "symbol": sym or None,
    }
    if not sym:
        return base
    try:
        from trading.data.price_cache import get_history

        hist = get_history(sym, period=period)
        if hist is None or getattr(hist, "empty", True):
            base["reason"] = f"No price history for {sym} — no vol overlay."
            return base
        out = conditional_vol_multiplier(hist, high_multiplier=high_multiplier)
        out["symbol"] = sym
        return out
    except Exception as e:
        logger.warning("conditional_vol_multiplier_for_symbol(%s): %s", sym, e)
        base["reason"] = f"Vol overlay failed for {sym}: {e}"
        return base


def apply_kelly_vol_overlay(
    kelly_result: Dict[str, Any],
    vol_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach vol-adjusted Kelly fields; does not replace raw Kelly numbers."""
    if not isinstance(kelly_result, dict) or not kelly_result.get("success"):
        return kelly_result
    out = dict(kelly_result)
    mult = float(vol_info.get("multiplier") or 1.0)
    mult = max(0.0, min(1.0, mult))
    half = float(out.get("half_kelly_fraction") or 0.0)
    full = float(out.get("full_kelly_fraction") or 0.0)
    dollars = out.get("half_kelly_dollars")
    out["vol_regime"] = vol_info.get("regime")
    out["vol_multiplier"] = round(mult, 4)
    out["vol_scaled_down"] = bool(vol_info.get("scaled_down"))
    out["vol_adjustment_reason"] = vol_info.get("reason")
    out["half_kelly_fraction_vol_adjusted"] = round(half * mult, 4)
    out["full_kelly_fraction_vol_adjusted"] = round(full * mult, 4)
    if dollars is not None:
        try:
            out["half_kelly_dollars_vol_adjusted"] = round(float(dollars) * mult, 2)
        except Exception:
            pass
    note = str(out.get("note") or "")
    vol_note = str(vol_info.get("reason") or "")
    out["note"] = (note + " " + vol_note).strip()
    return out


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


def simulate_exposure_paths(
    close: pd.Series,
    *,
    high_multiplier: float = HIGH_VOL_MULTIPLIER,
    min_history: int = 60,
) -> Dict[str, Any]:
    """Causal OOS: buy-and-hold vs conditional-vol-scaled exposure on same returns.

    At each t, regime uses only close[:t+1]. Overlay never peeks ahead.
    """
    px = pd.to_numeric(close, errors="coerce").dropna()
    if len(px) < min_history + 5:
        return {"success": False, "error": "insufficient price history"}

    rets = px.pct_change()
    base_rets: List[float] = []
    overlay_rets: List[float] = []
    mults: List[float] = []

    for t in range(min_history, len(px)):
        r = float(rets.iloc[t]) if np.isfinite(rets.iloc[t]) else 0.0
        hist = px.iloc[: t + 1]
        info = conditional_vol_multiplier(hist, high_multiplier=high_multiplier)
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
    dd_improvement = dd_base - dd_over  # positive = overlay helped DD
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
        "conditional_overlay": {
            "sharpe": round(sharpe_over, 4),
            "max_drawdown": round(dd_over, 4),
        },
        "dd_improvement": round(dd_improvement, 4),
        "sharpe_delta": round(sharpe_delta, 4),
        "adopt_overlay": adopt,
        "reason": reason,
        "high_multiplier": high_multiplier,
    }


def validate_conditional_vol_universe(
    symbols: Optional[Sequence[str]] = None,
    *,
    period: str = "2y",
    high_multiplier: float = HIGH_VOL_MULTIPLIER,
) -> Dict[str, Any]:
    """Per-symbol OOS for conditional overlay; report mixed helped/hurt honestly."""
    from trading.data.price_cache import get_history

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
            sim = simulate_exposure_paths(
                close, high_multiplier=high_multiplier
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
        "recommend_live": broad,
        "note": (
            "Broad win — overlay may be considered for live Kelly path"
            if broad
            else (
                "No broad win — keep overlay informational / research only "
                "(same discipline as model-routing Phase 3)."
            )
        ),
        "min_dd_improvement": MIN_DD_IMPROVEMENT,
        "max_sharpe_cost": MAX_SHARPE_COST,
        "results": rows,
    }
