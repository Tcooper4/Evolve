# -*- coding: utf-8 -*-
"""Hedged one-line context for headlines — context, not a call.

Used by React Dashboard/Analyze. Soft-fails when no LLM key is present.
Prompt rules deliberately avoid buy/sell language and certainty.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# One-line headline gloss on the Dashboard overlay path — keep tight so a
# stuck Anthropic call cannot stall news cards (SDK default is multi-minute).
ANTHROPIC_TIMEOUT_S = 15.0

_HEDGE_PROMPT = (
    "You are a market research assistant helping a trader skim headlines.\n"
    "In ONE short sentence, what context might this headline provide?\n"
    "Hard rules:\n"
    "- Do NOT give buy, sell, hold, long, or short advice.\n"
    "- Do NOT sound certain or predictive ('will', 'guarantees', 'must').\n"
    "- Phrase as possible context only ('may reflect…', 'could matter if…').\n"
    "- No ticker recommendations. No price targets.\n"
    "- If the headline is unclear, say it is unclear context — do not invent.\n\n"
    "Headline: {title}"
)


def _complete(prompt: str, user_id: Optional[str] = None) -> str:
    """Prefer resolve_api_key-aware providers; empty string if unavailable."""
    try:
        from config.api_keys import resolve_api_key

        anthropic_key = (
            resolve_api_key("ANTHROPIC_API_KEY", session_id=user_id) or ""
        ).strip()
        openai_key = (
            resolve_api_key("OPENAI_API_KEY", session_id=user_id) or ""
        ).strip()
    except Exception:
        anthropic_key = ""
        openai_key = ""

    if anthropic_key:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=anthropic_key, timeout=ANTHROPIC_TIMEOUT_S)
            msg = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=80,
                messages=[{"role": "user", "content": prompt}],
            )
            parts = []
            for block in getattr(msg, "content", []) or []:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
            return " ".join(parts).strip()
        except Exception as e:
            logger.debug("news_context anthropic failed: %s", e)

    if openai_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=80,
                messages=[{"role": "user", "content": prompt}],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.debug("news_context openai failed: %s", e)

    # Last resort: shared active-LLM path (env / personal mode)
    try:
        from agents.llm.active_llm_calls import call_active_llm_simple

        return (call_active_llm_simple(prompt, max_tokens=80) or "").strip()
    except Exception as e:
        logger.debug("news_context active llm failed: %s", e)
        return ""


def _sanitize_why(text: str) -> str:
    """Strip leftover advice-y phrasing; keep empty if nothing usable."""
    why = (text or "").strip().strip('"').strip("'")
    if not why:
        return ""
    lower = why.lower()
    banned = (
        "buy ", "sell ", "short ", "long this", "you should",
        "must buy", "must sell", "guaranteed", "will rise", "will fall",
    )
    if any(b in lower for b in banned):
        return ""
    # Cap length for UI
    if len(why) > 220:
        why = why[:217].rstrip() + "…"
    return why


def explain_headlines(
    titles: List[str],
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return [{title, why, hedged: true}] — why may be empty offline/no-key."""
    out: List[Dict[str, Any]] = []
    for title in titles:
        t = (title or "").strip()
        if not t:
            continue
        why = ""
        try:
            raw = _complete(_HEDGE_PROMPT.format(title=t[:220]), user_id=user_id)
            why = _sanitize_why(raw)
        except Exception as e:
            logger.debug("explain_headlines skipped: %s", e)
            why = ""
        out.append({
            "title": t[:220],
            "why": why,
            "hedged": True,
            "note": (
                "Context only — not a trade recommendation."
                if why else
                "No LLM context (add an Anthropic/OpenAI key in Settings, or skip)."
            ),
        })
    return out
