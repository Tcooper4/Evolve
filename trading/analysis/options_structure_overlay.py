# -*- coding: utf-8 -*-
"""Options structure research guide for chart overlay (default-off).

Maps *current* delayed GEX + IV-skew context to a defined-risk structure
idea (iron condor, put/call credit, etc.). Not trade instructions and not
historical signal backfill — last-bar guide only, same honesty bar as the
strategy overlay / Kelly sample caveats.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_OPTIONS_OVERLAY_ENABLED = False

DISCLOSURE = (
    "Options structure research guide only — not trade instructions. "
    "Uses delayed/free option-chain GEX + IV skew (not OPRA). "
    "Suggested wing distances are rough percentage guides, not live "
    "fills or broker tickets. Defined-risk premium-selling is fat-tailed; "
    "size with sample-size caveats in mind."
)

STRUCTURE_IRON_CONDOR = "iron_condor"
STRUCTURE_PUT_CREDIT = "put_credit_spread"
STRUCTURE_CALL_CREDIT = "call_credit_spread"
STRUCTURE_WAIT = "wait_mixed"


def pick_options_structure(
    *,
    regime_short: Optional[str],
    skew_shape: Optional[str] = None,
    skew_diff: Optional[float] = None,
    spot: Optional[float] = None,
    gamma_flip: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Pure mapping — hand-verifiable truth table for chart guide.

    long_gamma → favor short-premium defined-risk (condor / credit)
    short_gamma → do not favor naked/short-premium; wait
    near_flip → wait (regime unstable)
    """
    regime = str(regime_short or "").strip().lower()
    shape = str(skew_shape or "").strip().lower()
    try:
        sd = float(skew_diff) if skew_diff is not None else None
    except Exception:
        sd = None

    spot_vs_flip = None
    try:
        if spot is not None and gamma_flip is not None and float(gamma_flip) != 0:
            spot_vs_flip = (float(spot) - float(gamma_flip)) / float(gamma_flip)
    except Exception:
        spot_vs_flip = None

    if regime in ("short_gamma",):
        return {
            "structure": STRUCTURE_WAIT,
            "label": "Wait / avoid short premium",
            "mark_text": "WAIT",
            "color": "#E89B6B",
            "rationale": (
                "Dealers net short gamma — amplified moves / breakout risk. "
                "Short iron condors and credit spreads are the wrong side of "
                "this tape; wait or use defined-risk debit ideas separately."
            ),
            "wing_pct_guide": None,
        }

    if regime in ("near_flip", "", "unknown"):
        return {
            "structure": STRUCTURE_WAIT,
            "label": "Wait — near gamma flip / mixed",
            "mark_text": "WAIT",
            "color": "#F0C75E",
            "rationale": (
                "Near gamma flip or unclear regime — pinning and breakout "
                "risk can flip quickly. No structure overlay recommendation."
            ),
            "wing_pct_guide": None,
        }

    # long_gamma (and anything we treat as dampened / pin-prone)
    put_heavy = shape in ("put_smirk", "put_skew", "downside_skew") or (
        sd is not None and sd > 0.02
    )
    call_heavy = shape in ("call_smirk", "call_skew", "upside_skew") or (
        sd is not None and sd < -0.02
    )
    above_flip = spot_vs_flip is not None and spot_vs_flip > 0.005
    below_flip = spot_vs_flip is not None and spot_vs_flip < -0.005

    if put_heavy and above_flip:
        return {
            "structure": STRUCTURE_PUT_CREDIT,
            "label": "Put credit spread (research)",
            "mark_text": "PCS",
            "color": "#7EB6FF",
            "rationale": (
                "Long-gamma / pin-prone tape with elevated put skew and spot "
                "above gamma flip — a defined-risk put credit fits a "
                "supported-bullish tape better than a wide naked short put. "
                "Iron condor is the alternate if you want both wings."
            ),
            "wing_pct_guide": 0.03,
            "alternate": STRUCTURE_IRON_CONDOR,
        }

    if call_heavy and below_flip:
        return {
            "structure": STRUCTURE_CALL_CREDIT,
            "label": "Call credit spread (research)",
            "mark_text": "CCS",
            "color": "#C084FC",
            "rationale": (
                "Long-gamma tape with call-side skew and spot below flip — "
                "a defined-risk call credit is the directional short-premium "
                "read. Prefer an iron condor if you want non-directional."
            ),
            "wing_pct_guide": 0.03,
            "alternate": STRUCTURE_IRON_CONDOR,
        }

    return {
        "structure": STRUCTURE_IRON_CONDOR,
        "label": "Iron condor (research)",
        "mark_text": "IC",
        "color": "#00FF88",
        "rationale": (
            "Dealers net long gamma — dampened / pinning-prone tape favors "
            "a defined-risk iron condor (short premium both wings with "
            "long hedges) over directional single-legged short premium."
        ),
        "wing_pct_guide": 0.04,
        "alternate": (
            STRUCTURE_PUT_CREDIT if above_flip else STRUCTURE_CALL_CREDIT
            if below_flip else None
        ),
    }


def _levels_for_pick(
    pick: Dict[str, Any],
    spot: Optional[float],
    gamma_flip: Optional[float],
) -> List[Dict[str, Any]]:
    levels: List[Dict[str, Any]] = []
    if spot is not None:
        levels.append({
            "key": "spot",
            "label": "Spot (delayed)",
            "value": round(float(spot), 2),
            "price_scale": True,
        })
    if gamma_flip is not None:
        levels.append({
            "key": "gamma_flip",
            "label": "Gamma flip",
            "value": round(float(gamma_flip), 2),
            "price_scale": True,
        })
    wing = pick.get("wing_pct_guide")
    if spot is not None and wing:
        s = float(spot)
        w = float(wing)
        levels.append({
            "key": "short_put_guide",
            "label": f"Short-put ~guide (−{w:.0%})",
            "value": round(s * (1.0 - w), 2),
            "price_scale": True,
        })
        levels.append({
            "key": "short_call_guide",
            "label": f"Short-call ~guide (+{w:.0%})",
            "value": round(s * (1.0 + w), 2),
            "price_scale": True,
        })
    return levels


def build_options_structure_overlay(symbol: str) -> Dict[str, Any]:
    """Fetch current GEX (+ optional skew) and return last-bar chart guide."""
    sym = (symbol or "").strip().upper()
    out: Dict[str, Any] = {
        "success": False,
        "symbol": sym,
        "markers": [],
        "overlay_series": [],
        "reference_levels": {"levels": [], "note": None},
        "pick": None,
        "gex": None,
        "skew": None,
        "disclosure": DISCLOSURE,
        "default_on": DEFAULT_OPTIONS_OVERLAY_ENABLED,
        "framing": "options_structure_research_guide",
        "error": None,
    }
    if not sym:
        out["error"] = "symbol required"
        return out

    try:
        from trading.data.gamma_exposure import DATA_DISCLOSURE, get_gamma_exposure

        gex = get_gamma_exposure(sym)
        out["gex"] = {
            "success": bool(gex.get("success")),
            "regime_short": gex.get("regime_short"),
            "regime": gex.get("regime"),
            "net_gex": gex.get("net_gex"),
            "gamma_flip": gex.get("gamma_flip"),
            "spot": gex.get("spot"),
            "disclosure": gex.get("disclosure") or DATA_DISCLOSURE,
        }
        if not gex.get("success"):
            out["error"] = gex.get("error") or "GEX unavailable"
            return out

        skew_meta: Dict[str, Any] = {"success": False}
        try:
            from trading.data.options_skew import get_options_skew

            skew = get_options_skew(sym)
            skew_meta = {
                "success": bool(skew.get("success")),
                "shape": skew.get("shape"),
                "skew_diff": skew.get("skew_diff"),
                "note": skew.get("shape_note") or skew.get("note"),
            }
        except Exception as e:
            logger.debug("options structure skew skip: %s", e)
            skew_meta = {"success": False, "error": str(e)}
        out["skew"] = skew_meta

        pick = pick_options_structure(
            regime_short=str(gex.get("regime_short") or ""),
            skew_shape=str(skew_meta.get("shape") or "") or None,
            skew_diff=skew_meta.get("skew_diff"),
            spot=gex.get("spot"),
            gamma_flip=gex.get("gamma_flip"),
        )
        out["pick"] = pick

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # Prefer last history bar date when available
        try:
            from trading.data.price_cache import get_history

            hist = get_history(sym, period="5d")
            if hist is not None and not hist.empty:
                d = hist.index[-1]
                today = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
        except Exception:
            pass

        title = (
            f"[{pick['mark_text']}] {pick['label']} — {pick['rationale']} "
            f"(research guide; delayed chain)"
        )
        out["markers"] = [{
            "time": today,
            "text": pick["mark_text"],
            "title": title,
            "color": pick["color"],
            "shape": "square",
            "position": "aboveBar",
            "structure": pick["structure"],
        }]

        levels = _levels_for_pick(pick, gex.get("spot"), gex.get("gamma_flip"))
        out["reference_levels"] = {
            "levels": levels,
            "note": (
                "Wing prices are approximate % guides from spot — not live "
                "option strikes or order tickets."
            ),
        }

        # Horizontal guides: gamma flip + short wings when available
        series: List[Dict[str, Any]] = []
        try:
            from trading.data.price_cache import get_history

            hist = get_history(sym, period="3mo")
            if hist is not None and not hist.empty:
                times = [
                    (d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10])
                    for d in hist.index
                ]
                for lv in levels:
                    if lv["key"] in ("gamma_flip", "short_put_guide", "short_call_guide"):
                        color = {
                            "gamma_flip": "#F5A623",
                            "short_put_guide": "#7EB6FF",
                            "short_call_guide": "#C084FC",
                        }.get(lv["key"], "#8899aa")
                        series.append({
                            "id": lv["key"],
                            "label": lv["label"],
                            "color": color,
                            "style": "dotted",
                            "points": [
                                {"time": t, "value": float(lv["value"])} for t in times
                            ],
                        })
        except Exception as e:
            logger.debug("options overlay series skip: %s", e)
        out["overlay_series"] = series
        out["success"] = True
        return out
    except Exception as e:
        logger.warning("build_options_structure_overlay failed: %s", e)
        out["error"] = str(e)
        return out
