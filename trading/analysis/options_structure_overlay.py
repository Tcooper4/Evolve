# -*- coding: utf-8 -*-
"""Options structure research guide for chart overlay (default-off).

Maps *current* delayed GEX + IV-skew context to a defined-risk structure
idea (iron condor, put/call credit, etc.). Not trade instructions and not
historical signal backfill — last-bar guide only, same honesty bar as the
strategy overlay / Kelly sample caveats.

---------------------------------------------------------------------------
Research basis vs Evolve validation (Phase 3 scoping)
---------------------------------------------------------------------------
The regime→structure table below is a **reasonable research-backed
default**, grounded in documented dealer-hedging / GEX market-structure
findings used elsewhere in this project (dealers net long gamma →
counter-trend hedging → dampened / pinning-prone tape favors defined-risk
short premium; dealers net short gamma → with-trend hedging → amplified
moves — short iron condors / credit spreads are the wrong side; near
gamma flip → unstable — wait):

  long_gamma  → favor iron condor / defined-risk credit (skew can tilt PCS/CCS)
  short_gamma → WAIT (do not favor short premium)
  near_flip   → WAIT

What is **not** claimed here:
* This mapping has **not** been OOS-validated against Evolve's own GEX
  calculations or paper-trade outcomes.
* The GEX ``near_flip`` 0.5% boundary feeding the map is itself a
  design choice (see ``gamma_exposure.NEAR_FLIP_PCT``), not Evolve-proven.
* Free yfinance chains do not yield historical dealer GEX, so a past
  backtest of this map is not feasible today. Forward snapshots via
  ``EVOLVE_GEX_SNAPSHOT_LOG=1`` (``gex_snapshot_logger``) can later support
  regime / structure / next-day realized-range checks — that logger does
  not validate anything by itself.

``STRUCTURE_MAPPING_VALIDATED = False`` until such evidence exists.
Do not flip that flag on vibes; treat a null/no-improvement OOS result
the same way as elsewhere in this codebase.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_OPTIONS_OVERLAY_ENABLED = False

# Explicit honesty flag — do not set True without Evolve OOS evidence.
STRUCTURE_MAPPING_VALIDATED = False

STRUCTURE_MAPPING_NOTE = (
    "Regime→structure map is a research-backed default (long_gamma → "
    "defined-risk short premium; short_gamma / near_flip → wait), not an "
    "Evolve OOS-validated rule for this codebase's GEX or trade book."
)

DISCLOSURE = (
    "Options structure research guide only — not trade instructions. "
    "Uses delayed/free option-chain GEX + IV skew (not OPRA). "
    "Suggested wing distances are rough percentage guides, not live "
    "fills or broker tickets. Defined-risk premium-selling is fat-tailed; "
    "size with sample-size caveats in mind. "
    f"{STRUCTURE_MAPPING_NOTE} "
    "GEX near_flip (0.5% of spot from gamma flip) is a design choice, "
    "not an Evolve-validated empirical boundary — free yfinance chains "
    "do not supply historical dealer GEX; EVOLVE_GEX_SNAPSHOT_LOG=1 "
    "builds a future validation dataset only."
)

STRUCTURE_IRON_CONDOR = "iron_condor"
STRUCTURE_PUT_CREDIT = "put_credit_spread"
STRUCTURE_CALL_CREDIT = "call_credit_spread"
STRUCTURE_WAIT = "wait_mixed"


def _with_mapping_meta(pick: Dict[str, Any]) -> Dict[str, Any]:
    """Attach validation scoping to every pick (does not alter structure)."""
    out = dict(pick)
    out["mapping_validated"] = STRUCTURE_MAPPING_VALIDATED
    out["mapping_note"] = STRUCTURE_MAPPING_NOTE
    out["mapping_basis"] = "research_default"
    return out


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

    Mapping logic is unchanged from the research-backed default; every
    return includes ``mapping_validated=False`` until Evolve OOS exists.
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
        return _with_mapping_meta({
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
        })

    if regime in ("near_flip", "", "unknown"):
        return _with_mapping_meta({
            "structure": STRUCTURE_WAIT,
            "label": "Wait — near gamma flip / mixed",
            "mark_text": "WAIT",
            "color": "#F0C75E",
            "rationale": (
                "Near gamma flip or unclear regime — pinning and breakout "
                "risk can flip quickly. No structure overlay recommendation."
            ),
            "wing_pct_guide": None,
        })

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
        return _with_mapping_meta({
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
        })

    if call_heavy and below_flip:
        return _with_mapping_meta({
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
        })

    return _with_mapping_meta({
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
    })


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
        "mapping_validated": STRUCTURE_MAPPING_VALIDATED,
        "mapping_note": STRUCTURE_MAPPING_NOTE,
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
            f"(research guide; mapping not Evolve-OOS-validated; delayed chain)"
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
