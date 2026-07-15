# -*- coding: utf-8 -*-
"""Options-aware transaction cost / bid-ask model.

Why this exists
---------------
Equity backtests in this repo default to ``DEFAULT_SPREAD = 0.0005``
(5 bps). Even the most liquid ATM listed options routinely carry
bid-ask spreads of **several percentage points of the option mid** —
roughly 20–100×+ the equity assumption. Backtests that ignore that
gap look profitable and then underperform live fills. This module does
not silently replace the equity default (Phase 4 gate: no default
change without an options-fill comparison we still lack); it supplies
an explicit options cost path.

Preferred data → model fall-back
--------------------------------
When ``bid`` / ``ask`` are present on a chain row (yfinance
``option_chain`` frames via ``options_flow.fetch_option_chain``), use
the **observed** half-spread as a fraction of mid. Otherwise use the
modeled estimate below, labeled ``source="modeled"``.

Modeled estimate (fraction of option mid, one-way half-spread)
-------------------------------------------------------------
* ``ATM_LIQUID_FLOOR = 0.025`` (2.5%) — conservative floor inside the
  "several percentage points even for liquid ATM" industry finding;
  round-trip ≈ 5% of premium before slippage.
* OTM widening: add ``OTM_PER_5PCT * (|K/S − 1| / 0.05)`` capped —
  further OTM quotes are thinner (same geometry as skew research).
* Near expiry / near close: multiply by ``NEAR_EXPIRY_MULT`` when
  DTE ≤ 1 or minutes-to-close ≤ 30 — 0DTE research (e.g. Volos NDX)
  finds spreads widest near expiration/close.

Defaults stay on the equity cost path unless a caller opts into this
model. Full multi-leg 0DTE fill replay remains out of scope; this
closes the magnitude gap for research overlays that currently cite
equity-proxy validation.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, time
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

# One-way half-spread as fraction of option mid (not equity notional).
ATM_LIQUID_FLOOR = 0.025  # 2.5% of premium
OTM_PER_5PCT = 0.015  # +1.5pp of mid per 5% moneyness away from spot
NEAR_EXPIRY_MULT = 1.50
NEAR_CLOSE_MINUTES = 30
MAX_HALF_SPREAD = 0.25  # 25% of mid — pathological quotes capped
MIN_HALF_SPREAD = ATM_LIQUID_FLOOR

DISCLOSURE = (
    f"Options costs use a moneyness/liquidity-aware half-spread "
    f"(ATM liquid floor {ATM_LIQUID_FLOOR:.1%} of option mid; wider OTM "
    f"and near expiry/close). Observed bid/ask preferred when present; "
    f"otherwise modeled. Equity-style 5 bps understates liquid ATM "
    f"options by roughly 20–100×+. Not a full multi-leg fill replay."
)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def observed_half_spread_fraction(
    bid: Any,
    ask: Any,
) -> Optional[float]:
    """(ask − bid) / (2 · mid) — one-way half-spread as fraction of mid."""
    b = _safe_float(bid)
    a = _safe_float(ask)
    if b is None or a is None or a <= 0 or b < 0 or a < b:
        return None
    mid = 0.5 * (a + b)
    if mid <= 0:
        return None
    half = (a - b) / (2.0 * mid)
    return float(max(0.0, min(half, MAX_HALF_SPREAD)))


def moneyness_abs(spot: float, strike: float) -> float:
    s = float(spot)
    k = float(strike)
    if s <= 0 or k <= 0:
        return 0.0
    return abs(k / s - 1.0)


def modeled_half_spread_fraction(
    *,
    spot: float,
    strike: float,
    dte: Optional[float] = None,
    minutes_to_close: Optional[float] = None,
) -> Dict[str, Any]:
    """Modeled one-way half-spread (fraction of option mid)."""
    m = moneyness_abs(spot, strike)
    otm_add = OTM_PER_5PCT * (m / 0.05)
    half = ATM_LIQUID_FLOOR + max(0.0, otm_add)

    near = False
    try:
        if dte is not None and float(dte) <= 1.0:
            near = True
    except Exception:
        pass
    try:
        if minutes_to_close is not None and float(minutes_to_close) <= NEAR_CLOSE_MINUTES:
            near = True
    except Exception:
        pass
    if near:
        half *= NEAR_EXPIRY_MULT

    half = float(min(MAX_HALF_SPREAD, max(MIN_HALF_SPREAD, half)))
    return {
        "half_spread_fraction": half,
        "round_trip_fraction": 2.0 * half,
        "moneyness_abs": m,
        "near_expiry_or_close": near,
        "source": "modeled",
        "disclosure": DISCLOSURE,
    }


def estimate_option_half_spread(
    *,
    spot: Optional[float] = None,
    strike: Optional[float] = None,
    bid: Any = None,
    ask: Any = None,
    dte: Optional[float] = None,
    minutes_to_close: Optional[float] = None,
    premium_mid: Optional[float] = None,
) -> Dict[str, Any]:
    """Observed bid/ask half-spread if available; else modeled estimate."""
    obs = observed_half_spread_fraction(bid, ask)
    if obs is not None:
        mid = None
        b = _safe_float(bid)
        a = _safe_float(ask)
        if b is not None and a is not None:
            mid = 0.5 * (a + b)
        elif premium_mid is not None:
            mid = _safe_float(premium_mid)
        return {
            "success": True,
            "half_spread_fraction": obs,
            "round_trip_fraction": 2.0 * obs,
            "half_spread_dollars": (
                float(obs * mid) if mid is not None and mid > 0 else None
            ),
            "source": "observed_bid_ask",
            "moneyness_abs": (
                moneyness_abs(float(spot), float(strike))
                if spot and strike else None
            ),
            "near_expiry_or_close": None,
            "disclosure": DISCLOSURE,
        }

    if spot is None or strike is None or float(spot) <= 0 or float(strike) <= 0:
        return {
            "success": False,
            "error": "need bid/ask or spot+strike for modeled estimate",
            "half_spread_fraction": ATM_LIQUID_FLOOR,
            "round_trip_fraction": 2.0 * ATM_LIQUID_FLOOR,
            "source": "fallback_atm_floor",
            "disclosure": DISCLOSURE,
        }

    modeled = modeled_half_spread_fraction(
        spot=float(spot),
        strike=float(strike),
        dte=dte,
        minutes_to_close=minutes_to_close,
    )
    mid = _safe_float(premium_mid)
    modeled["success"] = True
    modeled["half_spread_dollars"] = (
        float(modeled["half_spread_fraction"] * mid)
        if mid is not None and mid > 0 else None
    )
    return modeled


def minutes_to_us_equity_close(now: Optional[datetime] = None) -> float:
    """Minutes until 16:00 America/New_York (0 if past close / weekend-ish)."""
    try:
        import pytz

        tz = pytz.timezone("America/New_York")
        if now is None:
            n = datetime.now(tz)
        elif now.tzinfo is None:
            n = tz.localize(now)
        else:
            n = now.astimezone(tz)
        close = n.replace(hour=16, minute=0, second=0, microsecond=0)
        return max(0.0, (close - n).total_seconds() / 60.0)
    except Exception as e:
        logger.debug("minutes_to_close failed: %s", e)
        return 999.0


def equity_vs_options_spread_ratio(
    equity_spread: float = 0.0005,
    options_half_spread: float = ATM_LIQUID_FLOOR,
) -> float:
    """How many times wider the options half-spread is vs equity default."""
    eq = max(float(equity_spread), 1e-12)
    return float(options_half_spread) / eq


__all__ = [
    "ATM_LIQUID_FLOOR",
    "OTM_PER_5PCT",
    "NEAR_EXPIRY_MULT",
    "MAX_HALF_SPREAD",
    "DISCLOSURE",
    "observed_half_spread_fraction",
    "moneyness_abs",
    "modeled_half_spread_fraction",
    "estimate_option_half_spread",
    "minutes_to_us_equity_close",
    "equity_vs_options_spread_ratio",
]
