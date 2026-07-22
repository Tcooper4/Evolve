# -*- coding: utf-8 -*-
"""Shared plain-language helpers and jargon checks for beginner-facing fields."""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence

# Used by tests — keep in sync across plain_language field assertions.
BANNED_JARGON_TERMS: Sequence[str] = (
    "dealer",
    "gamma",
    "delta",
    "sharpe",
    "kelly",
    "dsr",
    "oos",
    "skew",
    "hedge",
    "regime",
)

_MAX_PLAIN_CHARS = 320


def find_banned_jargon(text: str, terms: Iterable[str] = BANNED_JARGON_TERMS) -> List[str]:
    """Return banned terms found in text (case-insensitive word boundaries)."""
    if not text:
        return []
    found: List[str] = []
    for term in terms:
        if re.search(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE):
            found.append(term)
    return found


def is_plain_language_length_ok(text: str, max_chars: int = _MAX_PLAIN_CHARS) -> bool:
    """Plain-language fields should be a sentence or two, not a paragraph."""
    if not text or not str(text).strip():
        return False
    return len(str(text).strip()) <= max_chars


def sentiment_label_plain_language(label: str, score: float | None = None) -> str:
    """Plain read on BULLISH / BEARISH / NEUTRAL sentiment labels."""
    lab = (label or "NEUTRAL").strip().upper()
    if lab == "BULLISH":
        return "People talking about this stock online sound mostly positive."
    if lab == "BEARISH":
        return "People talking about this stock online sound mostly negative."
    if score is not None and abs(float(score)) >= 0.08:
        if float(score) > 0:
            return "Online chatter is slightly positive but not strong either way."
        return "Online chatter is slightly negative but not strong either way."
    return "Online chatter is mixed — no clear positive or negative tone."
