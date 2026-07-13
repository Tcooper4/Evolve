# -*- coding: utf-8 -*-
"""IV skew from delayed free option chains, with same-day event context.

Vertical skew (OTM put IV vs OTM call IV at matched moneyness) from
``trading.data.options_flow.fetch_option_chain``. Near-term skew is
**not** treated as standalone sentiment: we check earnings + Evolve's
macro calendar for a same-day catalyst and frame the reading accordingly.

---------------------------------------------------------------------------
Shape labels
---------------------------------------------------------------------------
* **flat** — |put OTM IV − call OTM IV| below threshold
* **smile** — both wings elevated vs ATM IV
* **put_smirk** — put wing steeper (classic downside-protection demand)
* **call_smirk** — call wing steeper

Skew uses a strike-distance proxy (default ±5% from spot) because free
yfinance chains do not reliably supply delta.

---------------------------------------------------------------------------
Honesty
---------------------------------------------------------------------------
Every response includes the delayed-data disclosure. Event checks that
fail say so plainly — we never silently treat every skew as structural
sentiment.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DISCLOSURE = (
    "Computed on delayed/free option-chain data (yfinance), not real-time "
    "OPRA. Skew is directional context only — not a live dealer or "
    "flow-tape readout."
)

# ± moneyness for OTM put / call legs (strike-distance proxy for delta)
DEFAULT_OTM_PCT = 0.05
# |put_iv - call_iv| below this → flat (vol points, e.g. 0.015 = 1.5 pts)
FLAT_DIFF = 0.015
# Wing vs ATM must clear this to count as "elevated" for a smile
WING_ELEVATED = 0.020


def _colmap(df: pd.DataFrame) -> Dict[str, str]:
    return {str(c).lower(): c for c in df.columns}


def _iv_at_moneyness(
    df: pd.DataFrame,
    target_strike: float,
) -> Optional[Tuple[float, float]]:
    """Nearest listed strike's IV to ``target_strike``. Returns (strike, iv)."""
    if df is None or df.empty or target_strike <= 0:
        return None
    cmap = _colmap(df)
    strike_c = cmap.get("strike")
    iv_c = cmap.get("impliedvolatility") or cmap.get("implied_volatility")
    if not strike_c or not iv_c:
        return None
    strikes = pd.to_numeric(df[strike_c], errors="coerce")
    ivs = pd.to_numeric(df[iv_c], errors="coerce")
    mask = strikes.notna() & ivs.notna() & (ivs > 0) & (strikes > 0)
    if not mask.any():
        return None
    sub_k = strikes[mask]
    sub_v = ivs[mask]
    idx = (sub_k - target_strike).abs().idxmin()
    return float(sub_k.loc[idx]), float(sub_v.loc[idx])


def classify_skew_shape(
    put_otm_iv: float,
    call_otm_iv: float,
    atm_iv: float,
    *,
    flat_diff: float = FLAT_DIFF,
    wing_elevated: float = WING_ELEVATED,
) -> Dict[str, Any]:
    """
    Hand-checkable shape classifier from three IV levels.

    Returns shape label, skew_diff (put − call), and wing elevations.
    """
    put_iv = float(put_otm_iv)
    call_iv = float(call_otm_iv)
    atm = float(atm_iv)
    skew_diff = put_iv - call_iv
    put_wing = put_iv - atm
    call_wing = call_iv - atm

    if (
        put_wing >= wing_elevated
        and call_wing >= wing_elevated
        and abs(skew_diff) <= max(flat_diff * 2, wing_elevated)
    ):
        shape = "smile"
        detail = (
            f"both wings elevated vs ATM "
            f"(put +{put_wing:.1%}, call +{call_wing:.1%})"
        )
    elif abs(skew_diff) < flat_diff:
        shape = "flat"
        detail = (
            f"|put−call IV|={abs(skew_diff):.1%} below flat threshold "
            f"({flat_diff:.1%})"
        )
    elif skew_diff > 0:
        shape = "put_smirk"
        detail = (
            f"put OTM IV exceeds call OTM IV by {skew_diff:.1%} vol "
            f"(downside-protection demand / put skew)"
        )
    else:
        shape = "call_smirk"
        detail = (
            f"call OTM IV exceeds put OTM IV by {-skew_diff:.1%} vol "
            f"(upside call skew)"
        )

    return {
        "shape": shape,
        "skew_diff": skew_diff,
        "put_wing": put_wing,
        "call_wing": call_wing,
        "detail": detail,
    }


def compute_vertical_skew(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot: float,
    *,
    otm_pct: float = DEFAULT_OTM_PCT,
    flat_diff: float = FLAT_DIFF,
    wing_elevated: float = WING_ELEVATED,
) -> Dict[str, Any]:
    """Pure skew calc from call/put frames (no network)."""
    disclosure = DATA_DISCLOSURE
    base: Dict[str, Any] = {
        "success": False,
        "shape": None,
        "skew_diff": None,
        "put_otm_iv": None,
        "call_otm_iv": None,
        "atm_iv": None,
        "put_strike": None,
        "call_strike": None,
        "atm_strike": None,
        "otm_pct": float(otm_pct),
        "detail": None,
        "disclosure": disclosure,
        "delayed_data": True,
        "error": None,
    }
    if spot is None or not (float(spot) > 0):
        base["error"] = "spot required"
        return base
    spot_f = float(spot)
    put_target = spot_f * (1.0 - float(otm_pct))
    call_target = spot_f * (1.0 + float(otm_pct))

    put_hit = _iv_at_moneyness(puts, put_target)
    call_hit = _iv_at_moneyness(calls, call_target)
    # ATM from either side; prefer call ATM then put
    atm_call = _iv_at_moneyness(calls, spot_f)
    atm_put = _iv_at_moneyness(puts, spot_f)
    if atm_call and atm_put:
        atm_strike, atm_iv = (
            atm_call
            if abs(atm_call[0] - spot_f) <= abs(atm_put[0] - spot_f)
            else atm_put
        )
        # Average if both nearly ATM
        if abs(atm_call[0] - spot_f) < spot_f * 0.005 and abs(atm_put[0] - spot_f) < spot_f * 0.005:
            atm_iv = 0.5 * (atm_call[1] + atm_put[1])
            atm_strike = spot_f
    elif atm_call:
        atm_strike, atm_iv = atm_call
    elif atm_put:
        atm_strike, atm_iv = atm_put
    else:
        base["error"] = "no ATM IV available"
        return base

    if not put_hit or not call_hit:
        base["error"] = "missing OTM put or call IV at target moneyness"
        base["atm_iv"] = float(atm_iv)
        base["atm_strike"] = float(atm_strike)
        return base

    put_k, put_iv = put_hit
    call_k, call_iv = call_hit
    classified = classify_skew_shape(
        put_iv, call_iv, float(atm_iv),
        flat_diff=flat_diff, wing_elevated=wing_elevated,
    )
    return {
        "success": True,
        "shape": classified["shape"],
        "skew_diff": classified["skew_diff"],
        "put_otm_iv": float(put_iv),
        "call_otm_iv": float(call_iv),
        "atm_iv": float(atm_iv),
        "put_strike": float(put_k),
        "call_strike": float(call_k),
        "atm_strike": float(atm_strike),
        "put_wing": classified["put_wing"],
        "call_wing": classified["call_wing"],
        "otm_pct": float(otm_pct),
        "detail": classified["detail"],
        "disclosure": disclosure,
        "delayed_data": True,
        "error": None,
    }


def check_same_day_catalysts(
    symbol: str,
    *,
    as_of: Optional[date] = None,
    earnings_fn=None,
    macro_fn=None,
) -> Dict[str, Any]:
    """
    Check earnings + macro calendar for a same-day catalyst.

    Injectable callables support hand tests without network I/O.
    """
    as_of = as_of or date.today()
    catalysts: List[Dict[str, Any]] = []
    errors: List[str] = []
    earnings_checked = False
    macro_checked = False

    # --- earnings ---
    try:
        if earnings_fn is None:
            from trading.data.earnings_calendar import get_upcoming_earnings
            earnings_fn = get_upcoming_earnings
        earn = earnings_fn(symbol) or {}
        earnings_checked = "error" not in earn or earn.get("days_until") is not None
        # If the call returned an explicit error and no date, mark failed
        if earn.get("error") and earn.get("days_until") is None and not earn.get("next_earnings_date"):
            earnings_checked = False
            errors.append(f"earnings: {earn.get('error')}")
        else:
            earnings_checked = True
            days = earn.get("days_until")
            if days == 0:
                catalysts.append({
                    "type": "earnings",
                    "name": f"{symbol} earnings",
                    "date": earn.get("next_earnings_date"),
                    "days_until": 0,
                })
    except Exception as e:
        errors.append(f"earnings: {e}")
        earnings_checked = False

    # --- macro (FOMC / CPI / jobs style) ---
    try:
        if macro_fn is None:
            from trading.data.earnings_calendar import get_macro_calendar
            macro_fn = get_macro_calendar
        macro = macro_fn(days_ahead=1) or {}
        if macro.get("success") is False and not macro.get("events"):
            macro_checked = False
            if macro.get("error"):
                errors.append(f"macro: {macro.get('error')}")
        else:
            macro_checked = True
            for ev in macro.get("events") or []:
                try:
                    d_until = ev.get("days_until")
                    if d_until is None and ev.get("date"):
                        d = datetime.fromisoformat(str(ev["date"])[:10]).date()
                        d_until = (d - as_of).days
                    if d_until == 0:
                        catalysts.append({
                            "type": ev.get("type") or "macro",
                            "name": ev.get("name") or "macro event",
                            "date": ev.get("date"),
                            "days_until": 0,
                        })
                except Exception as e:
                    logger.debug("macro event parse: %s", e)
    except Exception as e:
        errors.append(f"macro: {e}")
        macro_checked = False

    if catalysts:
        interpretation = "event_driven"
        note = (
            "Same-day catalyst present — treat near-term skew as "
            "event-driven positioning, not a structural sentiment shift."
        )
    elif earnings_checked or macro_checked:
        interpretation = "no_same_day_catalyst_found"
        note = (
            "No same-day earnings/macro catalyst found in available "
            "calendars — skew may reflect broader positioning, but "
            "calendar coverage is not exhaustive."
        )
    else:
        interpretation = "could_not_check"
        note = (
            "Could not check for a same-day catalyst "
            "(earnings/macro calendar unavailable)."
        )

    return {
        "interpretation": interpretation,
        "same_day_catalyst": bool(catalysts),
        "catalysts": catalysts,
        "earnings_checked": earnings_checked,
        "macro_checked": macro_checked,
        "note": note,
        "errors": errors,
        "as_of": as_of.isoformat(),
    }


def _term_structure_note(
    near: Dict[str, Any],
    far: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Flag sharp near-term skew vs flatter longer-dated skew."""
    if not far or not near.get("success") or not far.get("success"):
        return None
    near_abs = abs(float(near.get("skew_diff") or 0))
    far_abs = abs(float(far.get("skew_diff") or 0))
    if near_abs >= FLAT_DIFF and far_abs < near_abs * 0.5:
        return (
            f"Near-term |skew| ({near_abs:.1%}) is sharply steeper than "
            f"longer-dated ({far_abs:.1%}) — pattern often associated with "
            f"a same-day catalyst rather than a lasting sentiment shift."
        )
    return None


def get_options_skew(
    symbol: str,
    expiry: Optional[str] = None,
    *,
    otm_pct: float = DEFAULT_OTM_PCT,
    as_of: Optional[date] = None,
    earnings_fn=None,
    macro_fn=None,
) -> Dict[str, Any]:
    """
    Fetch chain(s), classify vertical skew, attach event-context framing.
    """
    from trading.data.options_flow import fetch_option_chain

    sym = (symbol or "").strip().upper()
    out: Dict[str, Any] = {
        "success": False,
        "symbol": sym,
        "shape": None,
        "skew_diff": None,
        "expiry": None,
        "event_context": None,
        "framing": None,
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
            out["event_context"] = check_same_day_catalysts(
                sym, as_of=as_of, earnings_fn=earnings_fn, macro_fn=macro_fn,
            )
            return out

        skew = compute_vertical_skew(
            chain["calls"], chain["puts"], float(chain["spot"] or 0),
            otm_pct=otm_pct,
        )
        skew["symbol"] = sym
        skew["expiry"] = chain.get("expiry")
        skew["spot"] = chain.get("spot")
        skew["source"] = "yfinance"
        skew["expiries"] = chain.get("expiries") or []

        # Optional next expiry for term-structure of skew
        term_note = None
        expiries: Sequence[str] = chain.get("expiries") or []
        chosen = str(chain.get("expiry") or "")
        next_exps = [e for e in expiries if e > chosen]
        if next_exps:
            far_chain = fetch_option_chain(sym, expiry=next_exps[0])
            if far_chain.get("success") and far_chain.get("spot"):
                far_skew = compute_vertical_skew(
                    far_chain["calls"], far_chain["puts"],
                    float(far_chain["spot"]),
                    otm_pct=otm_pct,
                )
                skew["far_expiry"] = far_chain.get("expiry")
                skew["far_shape"] = far_skew.get("shape")
                skew["far_skew_diff"] = far_skew.get("skew_diff")
                term_note = _term_structure_note(skew, far_skew)

        event = check_same_day_catalysts(
            sym, as_of=as_of, earnings_fn=earnings_fn, macro_fn=macro_fn,
        )
        skew["event_context"] = event

        # Framing for callers / UI
        if event["interpretation"] == "event_driven":
            names = ", ".join(c["name"] for c in event["catalysts"])
            framing = (
                f"Event-driven skew reading ({names}). "
                f"{skew.get('detail') or ''} "
                f"{event['note']}"
            )
        elif event["interpretation"] == "could_not_check":
            framing = (
                f"{skew.get('detail') or 'Skew classified'}. "
                f"{event['note']}"
            )
        else:
            framing = (
                f"{skew.get('detail') or 'Skew classified'}. "
                f"{event['note']}"
            )
        if term_note:
            framing = f"{framing} {term_note}"
            if event["interpretation"] != "event_driven":
                framing = (
                    f"{framing} Term-structure steepness alone is not proof "
                    f"of a catalyst — calendar check: {event['interpretation']}."
                )

        skew["framing"] = framing.strip()
        skew["term_structure_note"] = term_note
        skew["disclosure"] = DATA_DISCLOSURE
        skew["delayed_data"] = True
        return skew
    except Exception as e:
        logger.warning("get_options_skew failed for %s: %s", sym, e)
        out["error"] = str(e)
        out["event_context"] = {
            "interpretation": "could_not_check",
            "note": (
                "Could not check for a same-day catalyst "
                "(skew pipeline error)."
            ),
            "same_day_catalyst": False,
            "catalysts": [],
        }
        return out


__all__ = [
    "DATA_DISCLOSURE",
    "DEFAULT_OTM_PCT",
    "FLAT_DIFF",
    "WING_ELEVATED",
    "classify_skew_shape",
    "compute_vertical_skew",
    "check_same_day_catalysts",
    "get_options_skew",
]
