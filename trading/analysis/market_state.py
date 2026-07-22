# -*- coding: utf-8 -*-
"""Market-state synthesis — situational awareness, not a forecast.

Combines three *existing* Evolve signals into one plain-language read:

1. Mechanical regime — ``gamma_exposure.regime_short``
2. Event severity — breaking headlines + FinBERT/VADER magnitude, weighted by
   ``news_aggregator.source_reputation_factor`` and corroboration across
   independent sources
3. Statistical regime — ``forecast_router.get_series_features`` →
   ``volatility_regime``

This module does **not** predict price direction. It answers: "should you
pay closer attention right now, and why?" Mandatory ``DISCLOSURE`` is
attached to every public response.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

DISCLOSURE = (
    "Situational awareness synthesized from existing GEX, news-severity, and "
    "realized-vol regime signals — not a price prediction and not a claim "
    "about next-bar direction."
)

# Event severity thresholds (documented, hand-checkable).
# Severity = clip(|sentiment| * reputation * corroboration_mult, 0, 1).
# Corroboration: 1 source → ×1.0; 2 → ×1.25; 3+ → ×1.50
# (independent publishers mentioning the tape; not same-title dupes).
SEVERITY_MODERATE = 0.22
SEVERITY_HIGH = 0.45

# Composite levels for UI / push priority.
LEVEL_CALM = "calm"
LEVEL_WATCHFUL = "watchful"
LEVEL_ELEVATED = "elevated"
LEVEL_CRITICAL = "critical"
PUSH_PRIORITY_LEVELS = frozenset({LEVEL_ELEVATED, LEVEL_CRITICAL})


def _plain_language_market_state(
    level: str,
    stress: List[str],
    calm: List[str],
) -> str:
    """Plain composite read — no direction forecast."""
    if level == LEVEL_CRITICAL:
        return (
            "Several serious stress signals at once — pay close attention and "
            "be extra careful with new risk."
        )
    if level == LEVEL_ELEVATED:
        return (
            "Multiple stress signals are active — expect choppier conditions and "
            "size down if you are unsure."
        )
    if level == LEVEL_WATCHFUL:
        return "One thing looks off — worth watching, but not an all-clear alarm."
    if stress:
        return "Mostly calm, with one minor stress signal — stay aware."
    return "Overall calm right now — no major stress signals in the mix."


def event_severity_score(
    sentiment_magnitude: float,
    reputation: float,
    corroboration_count: int,
) -> float:
    """Hand-verifiable severity in [0, 1].

    Justification (same bar as other Evolve thresholds):
    - Magnitude: abs(sentiment) from FinBERT/VADER in [0, 1] — intensity only,
      not used for long/short direction in the composite.
    - Reputation: ``source_reputation_factor`` (Reuters/Bloomberg > Seeking Alpha).
    - Corroboration: independent sources agreeing raises severity; a lone
      low-rep blurb stays low even at the same abs(sentiment).

    Example (exact): |s|=0.40, Seeking Alpha (~0.17), n=1 → ~0.068
    vs |s|=0.40, Reuters (~0.60), n=3 → ~0.360 — ordering must hold in tests.
    """
    mag = min(1.0, max(0.0, abs(float(sentiment_magnitude))))
    rep = min(1.0, max(0.05, float(reputation)))
    n = max(1, int(corroboration_count))
    corr = 1.0 + 0.25 * min(n - 1, 2)
    return round(min(1.0, mag * rep * corr), 6)


def score_headlines_severity(
    headlines: Sequence[Dict[str, Any]],
    sentiment_scores: Sequence[float],
) -> Dict[str, Any]:
    """Aggregate breaking-news severity from parallel headline/sentiment lists.

    Uses the *max* per-headline severity (a single extreme corroborated blast
    should dominate a page of mild items). Corroboration count = number of
    distinct ``source`` values present in the batch (independent publishers).
    """
    from trading.data.news_aggregator import source_reputation_factor

    items = list(headlines or [])
    scores = list(sentiment_scores or [])
    n = min(len(items), len(scores))
    if n == 0:
        return {
            "severity": 0.0,
            "band": "none",
            "n_headlines": 0,
            "corroboration_count": 0,
            "top": None,
        }

    sources = []
    for i in range(n):
        src = str(items[i].get("source") or items[i].get("publisher") or "")
        sources.append(src)
    distinct = {s.strip().lower() for s in sources if s.strip()}
    corr_n = max(1, len(distinct))

    best = None
    best_sev = -1.0
    for i in range(n):
        rep = source_reputation_factor(sources[i])
        # Per-item corroboration still uses batch distinct count so one mild
        # wire alone under-scores vs the same |s| with multi-wire coverage.
        sev = event_severity_score(float(scores[i]), rep, corr_n)
        if sev > best_sev:
            best_sev = sev
            best = {
                "title": items[i].get("title"),
                "source": sources[i],
                "sentiment": float(scores[i]),
                "reputation": rep,
                "severity": sev,
            }

    band = "none"
    if best_sev >= SEVERITY_HIGH:
        band = "high"
    elif best_sev >= SEVERITY_MODERATE:
        band = "moderate"

    return {
        "severity": round(float(best_sev), 6),
        "band": band,
        "n_headlines": n,
        "corroboration_count": corr_n,
        "top": best,
    }


def compose_market_state(
    *,
    gex_regime: str,
    event_severity: float,
    volatility_regime: str,
) -> Dict[str, Any]:
    """Deterministic composite from three known inputs (no network).

    Label always names the driving factors — never a bare opaque score.
    """
    gex = (gex_regime or "unknown").strip().lower()
    vol = (volatility_regime or "insufficient").strip().lower()
    sev = float(event_severity or 0.0)

    stress: List[str] = []
    calm: List[str] = []

    if gex == "short_gamma":
        stress.append("dealers short gamma")
    elif gex == "near_flip":
        stress.append("near gamma flip")
    elif gex == "long_gamma":
        calm.append("dealers long gamma")
    else:
        calm.append("GEX regime unavailable")

    if sev >= SEVERITY_HIGH:
        stress.append("high-severity breaking news")
    elif sev >= SEVERITY_MODERATE:
        stress.append("moderate news severity")
    else:
        calm.append("no major catalysts")

    if vol == "high":
        stress.append("elevated realized vol")
    elif vol == "low":
        calm.append("low realized vol")
    elif vol == "medium":
        calm.append("normal realized vol")
    else:
        calm.append("vol regime insufficient")

    n_stress = len(stress)
    gex_hot = gex in ("short_gamma", "near_flip")
    if n_stress >= 3 or (sev >= SEVERITY_HIGH and gex_hot and vol == "high"):
        level = LEVEL_CRITICAL
    elif n_stress >= 2 or (sev >= SEVERITY_HIGH and gex_hot):
        level = LEVEL_ELEVATED
    elif n_stress == 1:
        level = LEVEL_WATCHFUL
    else:
        level = LEVEL_CALM

    if stress:
        label = f"{level}: " + " + ".join(stress)
    else:
        label = f"{level}: " + ", ".join(calm)

    return {
        "level": level,
        "label": label,
        "plain_language": _plain_language_market_state(level, stress, calm),
        "drivers_stress": stress,
        "drivers_calm": calm,
        "push_priority": level in PUSH_PRIORITY_LEVELS,
        "components": {
            "gex_regime": gex,
            "event_severity": round(sev, 6),
            "volatility_regime": vol,
        },
    }


def is_push_priority_state(state: Optional[Dict[str, Any]]) -> bool:
    """True when composite level should elevate watch-mode alert pushes."""
    if not isinstance(state, dict):
        return False
    if state.get("push_priority") is True:
        return True
    return str(state.get("level") or "") in PUSH_PRIORITY_LEVELS


def get_market_state(
    symbol: str = "SPY",
    *,
    max_headlines: int = 8,
    gex: Optional[Dict[str, Any]] = None,
    headlines: Optional[List[Dict[str, Any]]] = None,
    sentiment_scores: Optional[List[float]] = None,
    volatility_regime: Optional[str] = None,
) -> Dict[str, Any]:
    """Public entry — situational awareness composite for ``symbol``.

    Optional kwargs inject known values for tests / offline callers.
    """
    sym = (symbol or "SPY").strip().upper() or "SPY"
    out: Dict[str, Any] = {
        "success": False,
        "symbol": sym,
        "disclosure": DISCLOSURE,
        "predicts_direction": False,
        "framing": "situational_awareness",
        "error": None,
    }

    try:
        # --- 1. Mechanical GEX ---
        gex_regime = "unknown"
        gex_payload: Dict[str, Any] = {}
        if gex is not None:
            gex_payload = dict(gex)
            gex_regime = str(gex_payload.get("regime_short") or "unknown")
        else:
            try:
                from trading.data.gamma_exposure import get_gamma_exposure

                gex_payload = get_gamma_exposure(sym) or {}
                if gex_payload.get("success"):
                    gex_regime = str(gex_payload.get("regime_short") or "unknown")
                else:
                    gex_regime = "unknown"
            except Exception as e:
                logger.debug("market_state GEX skip: %s", e)
                gex_payload = {"success": False, "error": str(e)}

        # --- 2. Event severity (breaking + FinBERT/VADER) ---
        if headlines is not None and sentiment_scores is not None:
            sev_block = score_headlines_severity(headlines, sentiment_scores)
            hl_items = list(headlines)
        else:
            hl_items = []
            try:
                from trading.data.twitter_headlines import get_breaking_headlines

                hl_items = get_breaking_headlines(max_items=max_headlines) or []
            except Exception as e:
                logger.debug("market_state headlines skip: %s", e)

            texts = [
                str(h.get("title") or h.get("summary") or "") for h in hl_items
            ]
            scores: List[float] = []
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

                from trading.data.social_sentiment import (
                    blend_finbert_vader,
                    score_texts_finbert,
                )

                analyzer = SentimentIntensityAnalyzer()
                vader = [
                    float(analyzer.polarity_scores(t or "").get("compound", 0.0) or 0.0)
                    for t in texts
                ]
                fin = score_texts_finbert(texts)
                scores, _engine = blend_finbert_vader(fin, vader)
            except Exception as e:
                logger.debug("market_state sentiment skip: %s", e)
                scores = [0.0] * len(texts)

            sev_block = score_headlines_severity(hl_items, scores)

        # --- 3. Statistical vol regime ---
        vol_reg = volatility_regime
        feat: Dict[str, Any] = {}
        if vol_reg is None:
            try:
                from trading.data.price_cache import get_history
                from trading.models.forecast_router import get_series_features

                hist = get_history(sym, period="1y", interval="1d")
                feat = get_series_features(hist) if hist is not None else {}
                vol_reg = str(feat.get("volatility_regime") or "insufficient")
            except Exception as e:
                logger.debug("market_state vol features skip: %s", e)
                vol_reg = "insufficient"

        composite = compose_market_state(
            gex_regime=gex_regime,
            event_severity=float(sev_block.get("severity") or 0.0),
            volatility_regime=str(vol_reg or "insufficient"),
        )

        out.update({
            "success": True,
            "level": composite["level"],
            "label": composite["label"],
            "plain_language": composite["plain_language"],
            "drivers_stress": composite["drivers_stress"],
            "drivers_calm": composite["drivers_calm"],
            "push_priority": composite["push_priority"],
            "components": composite["components"],
            "event": sev_block,
            "gex": {
                "regime_short": gex_regime,
                "success": bool(gex_payload.get("success")),
                "disclosure": gex_payload.get("disclosure"),
            },
            "series_features": {
                "volatility_regime": vol_reg,
                "data_length": feat.get("data_length"),
            },
        })
        return out
    except Exception as e:
        logger.warning("get_market_state failed: %s", e)
        out["error"] = str(e)
        return out


__all__ = [
    "DISCLOSURE",
    "SEVERITY_HIGH",
    "SEVERITY_MODERATE",
    "PUSH_PRIORITY_LEVELS",
    "event_severity_score",
    "score_headlines_severity",
    "compose_market_state",
    "is_push_priority_state",
    "get_market_state",
]
