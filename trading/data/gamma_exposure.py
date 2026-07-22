# -*- coding: utf-8 -*-
"""Gamma exposure (GEX) from delayed free option chains.

Computes retail-tier net GEX, gamma flip level, and pin-candidate strikes
from yfinance chains via ``trading.data.options_flow.fetch_option_chain``.
Does not duplicate chain fetching.

---------------------------------------------------------------------------
Sign convention (verify against tests — common GEX bug source)
---------------------------------------------------------------------------
Retail / SpotGamma-style dealer convention used here:

* Per-contract magnitude::
      gex = gamma * open_interest * multiplier * spot^2 * 0.01
  (``0.01`` = 1% spot move; ``multiplier`` defaults to 100).

* **Calls contribute +gex**, **puts contribute −gex**.
  This encodes the usual assumption that the listed market is dominated by
  customers long calls / long puts vs dealers on the other side of those
  flows, so call OI is treated as dealer-short-call (+ to net GEX when
  gamma is high) and put OI as dealer-short-put (− to net GEX). Black–
  Scholes gamma itself is always ≥ 0; the put minus sign is the
  positioning convention, not a property of γ.

* **Net GEX > 0** → dealers interpreted as **net long gamma** → hedging
  flows lean **counter-trend** → dampened / pinning-prone tape.
* **Net GEX < 0** → dealers **net short gamma** → hedge **with** the
  trend → amplified moves / breakout risk.
* **Gamma flip** → spot level where the strike-aggregated GEX profile's
  cumulative sum (low→high strikes) crosses zero (linear interpolation).

---------------------------------------------------------------------------
Honesty
---------------------------------------------------------------------------
Every public return dict includes ``disclosure``: this is delayed/free
chain data (not real-time OPRA). Treat as directional context only —
not a precise live dealer-positioning readout.

The ``near_flip`` regime boundary (``NEAR_FLIP_PCT`` = 0.5% of spot from
the gamma-flip level) is a **design choice**, not an Evolve-validated
empirical threshold. Free yfinance chains do not supply historical dealer
GEX, and ``data/options_cache.db`` is a TTL overwrite cache — not a time
series — so a real OOS check of this boundary is not possible yet.
Opt-in forward logging: ``EVOLVE_GEX_SNAPSHOT_LOG=1``
(``trading.data.gex_snapshot_logger``); that builds a future dataset and
validates nothing today.
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CONTRACT_MULTIPLIER = 100
GEX_PCT_MOVE = 0.01  # 1% spot move scaling

# Design choice — not Evolve OOS-validated (see module Honesty section).
NEAR_FLIP_PCT = 0.005

DATA_DISCLOSURE = (
    "Computed on delayed/free option-chain data (yfinance), not real-time "
    "OPRA. Treat net GEX, flip level, and pin strikes as directional "
    "context only — not a precise live dealer-positioning readout. "
    "near_flip uses a 0.5%-of-spot design-choice boundary around the "
    "gamma-flip level — not an Evolve-validated empirical threshold "
    "(historical GEX is unavailable from free yfinance; enable "
    "EVOLVE_GEX_SNAPSHOT_LOG=1 to accumulate a future validation set)."
)

# Short beginner-facing blurb (UI); full honesty stays in DATA_DISCLOSURE
UI_SUMMARY = (
    "Delayed options data — not a live exchange feed. "
    "Shows where dealers may hedge. Context only, not a trade signal."
)

REGIME_LONG = (
    "dealers net long gamma: expect dampened, pinning-prone price action "
    "(dealers hedge counter-trend against directional flow)"
)
REGIME_SHORT = (
    "dealers net short gamma: expect amplified moves, breakout risk "
    "(dealers hedge with the trend)"
)
REGIME_NEAR_FLIP = (
    "near gamma flip: mixed / unstable dealer-hedging regime — "
    "pinning and breakout risk can flip quickly with spot"
)

# Short regime lines for UI cards
REGIME_LONG_SHORT = "Calm / pin-prone — dealers hedge against big moves"
REGIME_SHORT_SHORT = "Choppy / breakout risk — dealers hedge with the trend"
REGIME_NEAR_FLIP_SHORT = "Unstable — near the gamma flip; wait for clarity"

# High-school plain read (no jargon) — canonical UI / chat field: plain_language
PLAIN_LANGUAGE_LONG = (
    "Market-makers may be damping moves today — price often sticks near key levels "
    "instead of running away."
)
PLAIN_LANGUAGE_SHORT = (
    "Be careful — price may swing more than usual and trends can run further "
    "before reversing."
)
PLAIN_LANGUAGE_NEAR_FLIP = (
    "The setup is unstable — wait for clearer direction before acting on big moves."
)


def plain_language_for_regime_short(regime_short: str) -> str:
    """Plain-language read keyed by regime_short (long_gamma / short_gamma / near_flip)."""
    key = (regime_short or "near_flip").strip().lower()
    if key == "long_gamma":
        return PLAIN_LANGUAGE_LONG
    if key == "short_gamma":
        return PLAIN_LANGUAGE_SHORT
    return PLAIN_LANGUAGE_NEAR_FLIP


def black_scholes_gamma(
    spot: float,
    strike: float,
    time_years: float,
    iv: float,
    rate: float = 0.0,
    dividend: float = 0.0,
) -> float:
    """BS gamma (identical for calls and puts). Returns 0 if inputs invalid."""
    try:
        s, k, t, sig = float(spot), float(strike), float(time_years), float(iv)
        if s <= 0 or k <= 0 or sig <= 0 or t <= 0:
            return 0.0
        # Floor extremely short dated T so 0DTE still has finite gamma
        t = max(t, 1.0 / (365.0 * 24.0))
        from scipy.stats import norm

        d1 = (
            math.log(s / k) + (rate - dividend + 0.5 * sig * sig) * t
        ) / (sig * math.sqrt(t))
        return float(
            math.exp(-dividend * t)
            * norm.pdf(d1)
            / (s * sig * math.sqrt(t))
        )
    except Exception as e:
        logger.debug("bs gamma failed: %s", e)
        return 0.0


def gex_contribution(
    gamma: float,
    open_interest: float,
    spot: float,
    *,
    is_call: bool,
    multiplier: float = CONTRACT_MULTIPLIER,
) -> float:
    """Signed GEX for one strike row under the module sign convention."""
    mag = (
        float(gamma)
        * float(open_interest)
        * float(multiplier)
        * float(spot) ** 2
        * GEX_PCT_MOVE
    )
    return float(mag if is_call else -mag)


def _years_to_expiry(expiry: str, as_of: Optional[date] = None) -> float:
    as_of = as_of or date.today()
    try:
        exp_d = datetime.strptime(str(expiry)[:10], "%Y-%m-%d").date()
        days = (exp_d - as_of).days
        # Same-day / weekend: treat as fraction of a day, not zero
        return max(days, 0) / 365.0 + (1.0 / 365.0 if days <= 0 else 0.0)
    except Exception as e:
        logger.debug("expiry parse failed (%s): %s", expiry, e)
        return 1.0 / 365.0


def _colmap(df: pd.DataFrame) -> Dict[str, str]:
    return {str(c).lower(): c for c in df.columns}


def _series_gamma(
    df: pd.DataFrame,
    spot: float,
    time_years: float,
    explicit_gamma: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Prefer explicit gamma column / override; else BS from IV."""
    n = len(df)
    if n == 0:
        return np.asarray([], dtype=float)
    if explicit_gamma is not None:
        g = np.asarray(list(explicit_gamma), dtype=float)
        if g.size != n:
            raise ValueError("explicit_gamma length must match dataframe rows")
        return g

    cmap = _colmap(df)
    if "gamma" in cmap:
        g = pd.to_numeric(df[cmap["gamma"]], errors="coerce").fillna(0.0).to_numpy()
        if np.any(g > 0):
            return g.astype(float)

    strike_c = cmap.get("strike")
    iv_c = cmap.get("impliedvolatility") or cmap.get("implied_volatility")
    if not strike_c or not iv_c:
        return np.zeros(n, dtype=float)

    strikes = pd.to_numeric(df[strike_c], errors="coerce").fillna(0.0).to_numpy()
    ivs = pd.to_numeric(df[iv_c], errors="coerce").fillna(0.0).to_numpy()
    out = np.zeros(n, dtype=float)
    for i in range(n):
        out[i] = black_scholes_gamma(spot, float(strikes[i]), time_years, float(ivs[i]))
    return out


def _aggregate_by_strike(
    rows: List[Tuple[float, float]],
) -> Dict[float, float]:
    """Sum signed GEX by strike."""
    out: Dict[float, float] = {}
    for k, gex in rows:
        out[float(k)] = out.get(float(k), 0.0) + float(gex)
    return out


def gamma_flip_point(gex_by_strike: Dict[float, float]) -> Optional[float]:
    """
    Spot level where cumulative strike GEX (sorted low→high) crosses zero.

    Linear interpolation between the bracketing strikes. Returns None if
    there is no sign change (entire profile same sign).
    """
    if not gex_by_strike:
        return None
    strikes = sorted(gex_by_strike.keys())
    running = 0.0
    for i, k in enumerate(strikes):
        before = running
        running += float(gex_by_strike[k])
        if i == 0:
            continue
        # Zero crossing of the cumulative profile
        if (before < 0 <= running) or (before > 0 >= running):
            prev_k = float(strikes[i - 1])
            if running == before:
                return float(k)
            # Distance toward k proportional to |before| / (|before|+|running|)
            w = abs(before) / (abs(before) + abs(running))
            return float(prev_k + w * (float(k) - prev_k))
        if before == 0.0:
            return float(strikes[i - 1])
    return None


def _regime_label(net_gex: float, flip: Optional[float], spot: float) -> str:
    # Near flip if within NEAR_FLIP_PCT of flip level (design choice —
    # not Evolve-validated; see module Honesty / NEAR_FLIP_PCT).
    if flip is not None and spot > 0:
        if abs(spot - flip) / spot <= NEAR_FLIP_PCT:
            return REGIME_NEAR_FLIP
    if abs(net_gex) < 1e-9:
        return REGIME_NEAR_FLIP
    if net_gex > 0:
        return REGIME_LONG
    return REGIME_SHORT


def compute_gex_profile(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot: float,
    *,
    expiry: Optional[str] = None,
    as_of: Optional[date] = None,
    multiplier: float = CONTRACT_MULTIPLIER,
    call_gammas: Optional[Sequence[float]] = None,
    put_gammas: Optional[Sequence[float]] = None,
    top_pins: int = 5,
) -> Dict[str, Any]:
    """
    Pure GEX calculation from call/put frames (no network).

    Pass ``call_gammas`` / ``put_gammas`` for hand-verified tests; otherwise
    gamma is taken from a ``gamma`` column or Black–Scholes via IV.
    """
    disclosure = DATA_DISCLOSURE
    base: Dict[str, Any] = {
        "success": False,
        "net_gex": 0.0,
        "gamma_flip": None,
        "spot": float(spot) if spot else None,
        "expiry": expiry,
        "pin_candidates": [],
        "gex_by_strike": [],
        "regime": REGIME_NEAR_FLIP,
        "regime_short": "near_flip",
        "plain_language": PLAIN_LANGUAGE_NEAR_FLIP,
        "near_flip_pct": NEAR_FLIP_PCT,
        "near_flip_validated": False,
        "sign_convention": (
            "calls=+gex, puts=-gex; "
            "gex=gamma*OI*multiplier*spot^2*0.01"
        ),
        "disclosure": disclosure,
        "summary": UI_SUMMARY,
        "delayed_data": True,
        "error": None,
    }
    if spot is None or not (float(spot) > 0):
        base["error"] = "spot required"
        return base

    spot_f = float(spot)
    t_years = _years_to_expiry(expiry or "", as_of=as_of) if expiry else (7.0 / 365.0)

    rows: List[Tuple[float, float]] = []

    def _consume(df: pd.DataFrame, is_call: bool, gammas: Optional[Sequence[float]]) -> None:
        if df is None or df.empty:
            return
        cmap = _colmap(df)
        strike_c = cmap.get("strike")
        oi_c = cmap.get("openinterest") or cmap.get("open_interest")
        if not strike_c or not oi_c:
            return
        strikes = pd.to_numeric(df[strike_c], errors="coerce").fillna(0.0).to_numpy()
        oi = pd.to_numeric(df[oi_c], errors="coerce").fillna(0.0).to_numpy()
        gam = _series_gamma(df, spot_f, t_years, explicit_gamma=gammas)
        for i in range(len(df)):
            if strikes[i] <= 0 or oi[i] <= 0 or gam[i] <= 0:
                continue
            gex = gex_contribution(
                float(gam[i]), float(oi[i]), spot_f,
                is_call=is_call, multiplier=multiplier,
            )
            rows.append((float(strikes[i]), gex))

    try:
        _consume(calls, True, call_gammas)
        _consume(puts, False, put_gammas)
    except Exception as e:
        base["error"] = str(e)
        return base

    by_strike = _aggregate_by_strike(rows)
    net = float(sum(by_strike.values()))
    flip = gamma_flip_point(by_strike)

    ranked = sorted(by_strike.items(), key=lambda kv: abs(kv[1]), reverse=True)
    pins = [
        {
            "strike": float(k),
            "gex": float(v),
            "abs_gex": float(abs(v)),
        }
        for k, v in ranked[: max(1, int(top_pins))]
    ]

    regime = _regime_label(net, flip, spot_f)
    if "long gamma" in regime:
        short = "long_gamma"
        plain = REGIME_LONG_SHORT
    elif "short gamma" in regime:
        short = "short_gamma"
        plain = REGIME_SHORT_SHORT
    else:
        short = "near_flip"
        plain = REGIME_NEAR_FLIP_SHORT

    return {
        "success": True,
        "net_gex": net,
        "gamma_flip": flip,
        "spot": spot_f,
        "expiry": expiry,
        "pin_candidates": pins,
        "gex_by_strike": [
            {"strike": float(k), "gex": float(by_strike[k])}
            for k in sorted(by_strike.keys())
        ],
        "regime": regime,
        "regime_plain": plain,
        "plain_language": plain_language_for_regime_short(short),
        "regime_short": short,
        "near_flip_pct": NEAR_FLIP_PCT,
        "near_flip_validated": False,
        "near_flip_note": (
            "near_flip boundary is a design choice (NEAR_FLIP_PCT), not "
            "an Evolve-validated empirical threshold"
        ),
        "sign_convention": base["sign_convention"],
        "disclosure": disclosure,
        "summary": UI_SUMMARY,
        "delayed_data": True,
        "error": None,
    }


def get_gamma_exposure(
    symbol: str,
    expiry: Optional[str] = None,
    *,
    top_pins: int = 5,
) -> Dict[str, Any]:
    """
    Fetch the chain via ``options_flow.fetch_option_chain`` and compute GEX.

    Default expiry = nearest listed (often 0DTE / short-dated for indexes).
    """
    from trading.data.options_flow import fetch_option_chain

    sym = (symbol or "").strip().upper()
    out: Dict[str, Any] = {
        "success": False,
        "symbol": sym,
        "net_gex": 0.0,
        "gamma_flip": None,
        "spot": None,
        "expiry": None,
        "pin_candidates": [],
        "gex_by_strike": [],
        "regime": REGIME_NEAR_FLIP,
        "regime_short": "near_flip",
        "plain_language": PLAIN_LANGUAGE_NEAR_FLIP,
        "near_flip_pct": NEAR_FLIP_PCT,
        "near_flip_validated": False,
        "disclosure": DATA_DISCLOSURE,
        "delayed_data": True,
        "source": "yfinance",
        "error": None,
    }
    if not sym:
        out["error"] = "symbol required"
        return out

    try:
        chain = fetch_option_chain(sym, expiry=expiry)
        if not chain.get("success"):
            out["error"] = chain.get("error") or "chain fetch failed"
            out["expiries"] = chain.get("expiries") or []
            return out
        profile = compute_gex_profile(
            chain["calls"],
            chain["puts"],
            float(chain["spot"] or 0),
            expiry=str(chain.get("expiry") or ""),
            top_pins=top_pins,
        )
        profile["symbol"] = sym
        profile["expiries"] = chain.get("expiries") or []
        profile["source"] = "yfinance"
        profile["delayed_data"] = True
        profile["disclosure"] = DATA_DISCLOSURE
        if chain.get("spot") is None or not profile.get("success"):
            profile["success"] = False
            profile["error"] = profile.get("error") or "missing spot or empty chain"
        return profile
    except Exception as e:
        logger.warning("get_gamma_exposure failed for %s: %s", sym, e)
        out["error"] = str(e)
        return out


__all__ = [
    "CONTRACT_MULTIPLIER",
    "DATA_DISCLOSURE",
    "UI_SUMMARY",
    "NEAR_FLIP_PCT",
    "black_scholes_gamma",
    "gex_contribution",
    "gamma_flip_point",
    "compute_gex_profile",
    "get_gamma_exposure",
]
