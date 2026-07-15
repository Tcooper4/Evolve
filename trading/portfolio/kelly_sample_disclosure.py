# -*- coding: utf-8 -*-
"""Sample-size caveats for Kelly guides derived from paper trade history.

Why ~25 closed trades for a high win-rate book
----------------------------------------------
Defined-risk premium-selling (iron condors, short puts with wings, etc.)
often shows ~65–80% win rate with infrequent large losses when a wing
is breached. If a rare adverse outcome occurs ~5–15% of the time, a short
streak of wins can look excellent before that tail has shown up:

  P(no rare loss in n trials) ≈ (1 - r)^n
  r=0.10, n=10  → ~35% chance the sample still has zero rare losses
  r=0.10, n=25  → ~7%
  r=0.05, n=25  → ~28%; n=40 → ~13%

So under ~25 closed trades — especially with a high observed win rate —
treating Kelly at face value is statistically premature. We flag that
explicitly rather than inventing fake precision.

Premium-selling recommendation
------------------------------
When the caller marks ``defined_risk_premium_selling=True``, recommend
**quarter-Kelly** (half of the usual half-Kelly guide) as the displayed
actionable fraction. Reasoning: practitioner half-Kelly already cuts for
noisy edges; fat left tails + uncertain loss magnitude argue for one more
halving until the sample has likely included a wing/breach class event.
Half-Kelly fields remain for transparency; ``recommended_*`` is what to use.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Primary threshold: see module docstring (rare-event visibility at ~10%).
SMALL_SAMPLE_N = 25
# Soft band for premium-selling after the primary bar clears.
PREMIUM_SOFT_N = 40


def assess_kelly_sample(
    n_closed_trades: Optional[int],
    win_rate: float,
    *,
    defined_risk_premium_selling: bool = False,
) -> Dict[str, Any]:
    """Return sample-size flags + human caveat (never mutates Kelly math)."""
    try:
        p = float(win_rate)
        if p > 1.0:
            p = p / 100.0
    except Exception:
        p = 0.0

    if n_closed_trades is None:
        return {
            "sample_size_flag": "unknown",
            "n_closed_trades": None,
            "sample_size_caveat": (
                "Sample size not provided — treat this Kelly fraction as a "
                "ceiling estimate only until you attach closed-trade count."
            ),
            "recommend_quarter_kelly": bool(defined_risk_premium_selling),
        }

    try:
        n = int(n_closed_trades)
    except Exception:
        n = 0

    out: Dict[str, Any] = {
        "sample_size_flag": "adequate",
        "n_closed_trades": n,
        "sample_size_caveat": None,
        "recommend_quarter_kelly": False,
        "defined_risk_premium_selling": bool(defined_risk_premium_selling),
        "small_sample_threshold": SMALL_SAMPLE_N,
    }

    if n < SMALL_SAMPLE_N:
        out["sample_size_flag"] = "insufficient" if n < 10 else "provisional"
        wr_note = ""
        if p >= 0.65:
            wr_note = (
                f" A ~{p:.0%} win rate needs a larger sample before a rare "
                "large loss would usually appear."
            )
        out["sample_size_caveat"] = (
            f"Based on {n} closed trades (threshold {SMALL_SAMPLE_N}). "
            "This sample is too small to treat Kelly at face value;"
            f"{wr_note} treat sizing as provisional."
        ).replace("  ", " ").strip()
        # Small sample always argues for a more conservative guided size.
        out["recommend_quarter_kelly"] = True
        return out

    if defined_risk_premium_selling:
        out["recommend_quarter_kelly"] = True
        if n < PREMIUM_SOFT_N:
            out["sample_size_flag"] = "provisional"
            out["sample_size_caveat"] = (
                f"Based on {n} closed trades — past the base bar ({SMALL_SAMPLE_N}) "
                f"but under {PREMIUM_SOFT_N} for a defined-risk premium-selling "
                "book. Prefer quarter-Kelly until a wing/breach-class loss has "
                "had a fair chance to show up."
            )
        else:
            out["sample_size_caveat"] = (
                f"Based on {n} closed trades. Defined-risk premium-selling "
                "still favors quarter-Kelly as the guided size (fat left "
                "tail); half-Kelly remains informational."
            )
        return out

    # Adequate non-premium sample — light note only when n is modest
    if n < PREMIUM_SOFT_N:
        out["sample_size_caveat"] = (
            f"Based on {n} closed trades — usable, but still a short history; "
            "keep half-Kelly as a ceiling, not a target."
        )
    return out


def attach_kelly_recommendation(
    kelly_out: Dict[str, Any],
    assessment: Dict[str, Any],
    account_size: float,
) -> Dict[str, Any]:
    """Merge assessment into a get_position_size payload; set recommended_*."""
    out = dict(kelly_out)
    out.update({
        "sample_size_flag": assessment.get("sample_size_flag"),
        "n_closed_trades": assessment.get("n_closed_trades"),
        "sample_size_caveat": assessment.get("sample_size_caveat"),
        "defined_risk_premium_selling": bool(
            assessment.get("defined_risk_premium_selling")
        ),
        "small_sample_threshold": assessment.get(
            "small_sample_threshold", SMALL_SAMPLE_N
        ),
    })

    half = float(out.get("half_kelly_fraction") or 0.0)
    full = float(out.get("full_kelly_fraction") or 0.0)
    quarter = full / 4.0  # = half/2 when half = full/2

    use_quarter = bool(assessment.get("recommend_quarter_kelly"))
    if use_quarter:
        out["quarter_kelly_fraction"] = round(quarter, 4)
        out["quarter_kelly_dollars"] = round(quarter * float(account_size), 2)
        out["recommended_fraction"] = out["quarter_kelly_fraction"]
        out["recommended_dollars"] = out["quarter_kelly_dollars"]
        out["recommended_basis"] = "quarter_kelly"
    else:
        out["recommended_fraction"] = round(half, 4)
        out["recommended_dollars"] = round(half * float(account_size), 2)
        out["recommended_basis"] = "half_kelly"

    caveat = assessment.get("sample_size_caveat")
    base_note = str(out.get("note") or "")
    if caveat:
        out["note"] = f"{base_note} {caveat}".strip()
    return out
