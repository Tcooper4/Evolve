# -*- coding: utf-8 -*-
"""Explicit, user-stated risk profile (never inferred from clicks/engagement).

Design boundary
---------------
``risk_tolerance`` and related fields are **stated preferences** the user sets
in Settings. Do not infer them from watchlist activity, clicks, or engagement
history. Implicit preference-learning that narrows *what* is recommended is
out of scope (echo-chamber failure mode). Engagement-based personalization,
if ever added, must stay limited to tone/framing/communication style only —
never silently filtering symbols, ideas, or structures.

Default: ``moderate`` — preserves prior Kelly guidance unless the user
explicitly chooses conservative (which forces quarter-Kelly as a *profile*
reason, distinguishable from small-sample cuts).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

RISK_CONSERVATIVE = "conservative"
RISK_MODERATE = "moderate"
RISK_AGGRESSIVE = "aggressive"
VALID_RISK_TOLERANCES = frozenset(
    {RISK_CONSERVATIVE, RISK_MODERATE, RISK_AGGRESSIVE}
)
DEFAULT_RISK_TOLERANCE = RISK_MODERATE

# Educational undefined-risk idea (never the default lead for conservative).
STRUCTURE_NAKED_SHORT_PUT = "naked_short_put"


def normalize_risk_tolerance(raw: Any, *, default: str = DEFAULT_RISK_TOLERANCE) -> str:
    """Map free text / legacy synonyms → conservative|moderate|aggressive."""
    s = str(raw or "").strip().lower()
    if s in VALID_RISK_TOLERANCES:
        return s
    # Legacy MemoryStore / agent vocab
    if s in ("low", "defensive", "cautious"):
        return RISK_CONSERVATIVE
    if s in ("high", "speculative"):
        return RISK_AGGRESSIVE
    if s in ("med", "medium", "normal", "balanced", ""):
        return default if default in VALID_RISK_TOLERANCES else RISK_MODERATE
    return default if default in VALID_RISK_TOLERANCES else RISK_MODERATE


def load_stated_risk_profile(session_id: Optional[str] = None) -> Dict[str, Any]:
    """Load stated prefs from user_store — never inferred from behavior."""
    prefs: Dict[str, Any] = {}
    try:
        from config.user_store import load_user_preferences

        sid = (session_id or "").strip()
        if not sid:
            import os

            sid = (os.getenv("EVOLVE_SESSION_ID") or "local").strip()
        prefs = dict(load_user_preferences(sid) or {})
    except Exception:
        prefs = {}

    rt = normalize_risk_tolerance(prefs.get("risk_tolerance"))
    allow_undef = prefs.get("allow_undefined_risk")
    if allow_undef is None:
        # Stated default: conservatives assume "no"; others leave unset (False
        # for display consistency unless user opts in).
        allow_undef = False if rt == RISK_CONSERVATIVE else False
    else:
        allow_undef = bool(allow_undef)

    dte_min = prefs.get("preferred_dte_min")
    dte_max = prefs.get("preferred_dte_max")
    try:
        dte_min_i = int(dte_min) if dte_min is not None else None
    except Exception:
        dte_min_i = None
    try:
        dte_max_i = int(dte_max) if dte_max is not None else None
    except Exception:
        dte_max_i = None

    return {
        "risk_tolerance": rt,
        "allow_undefined_risk": bool(allow_undef),
        "preferred_dte_min": dte_min_i,
        "preferred_dte_max": dte_max_i,
        "source": "stated_settings",
    }


def chat_framing_block(profile: Optional[Dict[str, Any]] = None) -> str:
    """Plain-language framing instructions for the chat system from stated prefs."""
    p = profile or {}
    rt = normalize_risk_tolerance(p.get("risk_tolerance"))
    lines = [
        "[Stated risk profile — Settings; not inferred from clicks/engagement]",
        f"risk_tolerance={rt}",
    ]
    if rt == RISK_CONSERVATIVE:
        lines.append(
            "Framing: lead harder with downside and max-loss scenarios; prefer "
            "defined-risk structures; do not present undefined-risk (naked) "
            "ideas as the default lead — flag them clearly if discussed."
        )
    elif rt == RISK_AGGRESSIVE:
        lines.append(
            "Framing: standard analytical detail is fine; still disclose "
            "tails, but you need not soft-pedal every suggestion."
        )
    else:
        lines.append(
            "Framing: balanced — state downside clearly, then upside; "
            "defined-risk is the usual default for options ideas."
        )
    if p.get("preferred_dte_min") is not None or p.get("preferred_dte_max") is not None:
        lines.append(
            f"preferred_dte_range={p.get('preferred_dte_min')}–{p.get('preferred_dte_max')}"
        )
    return "\n".join(lines)


def undefined_risk_note_item() -> Dict[str, Any]:
    """Always-available educational undefined-risk idea (not auto-lead)."""
    return {
        "structure": STRUCTURE_NAKED_SHORT_PUT,
        "risk_class": "undefined",
        "label": "Naked short put (undefined risk)",
        "note": (
            "Shown for visibility — undefined max loss vs defined-risk "
            "credits/condors. Not the research default lead."
        ),
        "deprioritized": False,
        "priority_rank": 50,
    }


def apply_risk_profile_to_structure_pick(
    pick: Dict[str, Any],
    *,
    risk_tolerance: Optional[str] = None,
    allow_undefined_risk: bool = False,
) -> Dict[str, Any]:
    """Deprioritize/flag undefined-risk notes without removing them."""
    out = dict(pick or {})
    rt = normalize_risk_tolerance(risk_tolerance)
    out["risk_tolerance_applied"] = rt
    out["risk_class"] = out.get("risk_class") or "defined"
    if out.get("structure") == STRUCTURE_NAKED_SHORT_PUT:
        out["risk_class"] = "undefined"

    noted: List[Dict[str, Any]] = []
    for item in list(out.get("also_noted") or []):
        if isinstance(item, dict):
            noted.append(dict(item))
    # Ensure undefined-risk remains visible on defined-risk leads
    if out.get("structure") not in (None, "wait_mixed", STRUCTURE_NAKED_SHORT_PUT):
        if not any(
            str(x.get("structure") or "") == STRUCTURE_NAKED_SHORT_PUT for x in noted
        ):
            noted.append(undefined_risk_note_item())

    for item in noted:
        rc = str(item.get("risk_class") or "")
        if rc == "undefined" or str(item.get("structure")) == STRUCTURE_NAKED_SHORT_PUT:
            item["risk_class"] = "undefined"
            if rt == RISK_CONSERVATIVE or not allow_undefined_risk:
                item["deprioritized"] = True
                item["priority_rank"] = 90
                item["profile_flag"] = (
                    "flagged_undefined_risk — stated conservative profile "
                    "(or undefined-risk not opted in); not the default lead."
                )
            else:
                item["deprioritized"] = False
                item["priority_rank"] = int(item.get("priority_rank") or 50)

    noted.sort(key=lambda x: int(x.get("priority_rank") or 50))
    out["also_noted"] = noted

    if rt == RISK_CONSERVATIVE:
        out["conservative_note"] = (
            "Stated conservative profile: defined-risk / wait stays the lead; "
            "undefined-risk ideas remain listed under also_noted but are "
            "flagged and deprioritized — not hidden."
        )
        # Never let undefined be the structure lead for conservative
        if out.get("risk_class") == "undefined":
            out["structure"] = "wait_mixed"
            out["label"] = "Wait — undefined risk deprioritized"
            out["mark_text"] = "WAIT"
            out["risk_class"] = "defined"
            out["rationale"] = (
                (out.get("rationale") or "")
                + " Conservative profile will not lead with undefined-risk."
            ).strip()

    return out
