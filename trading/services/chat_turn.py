# -*- coding: utf-8 -*-
"""Shared, frontend-agnostic chat turn: ONE tool-calling brain.

Both the Streamlit chat page and the React /api/chat endpoint call this,
so whichever frontend the user picks, chat behaves identically: memory
context + Agent Skills + the platform tool loop (scan, score, forecast,
news, risk, patterns, backtests, options sentiment), with a plain-LLM
fallback when tools fail. Extracted from pages/6_Chat.py's orchestration
so the logic can never drift between frontends.

Personalization boundary: stated Settings risk profile may shape framing
tone only. Do not infer recommendations or filter symbols from clicks /
engagement — see docs/PERSONALIZATION.md and trading.portfolio.risk_profile.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

def _registry_tool_names() -> List[str]:
    """Single source of truth: whatever the platform registry exposes,
    the chat can use - in BOTH frontends. A hand-maintained subset here
    is how the guided analyst ends up blind to capabilities that exist
    (found 2026-07: chat had 8 of 13+ tools; the beginner-advisor skill
    mandated detect_market_regime, which chat couldn't call)."""
    try:
        from agents.llm.agent import get_evolve_platform_tool_registry

        return [t["name"] for t in get_evolve_platform_tool_registry()]
    except Exception as e:  # noqa: BLE001
        logger.warning("tool registry unavailable, using fallback: %s", e)
        return ["scan_universe", "get_ai_score", "get_forecast", "get_news",
                "get_risk_metrics", "get_pattern_analysis", "run_backtest",
                "get_options_sentiment"]


STANDARD_TOOLS = _registry_tool_names()


def run_chat_turn(
    user_message: str,
    conversation_messages: Optional[List[Dict[str, str]]] = None,
    focus_symbol: Optional[str] = None,
    max_tokens: int = 2048,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one full chat turn. Returns
    {"success", "reply", "tool_captions", "error"}.

    Never raises: every failure degrades to the next-simplest path and
    ultimately to a clean error dict.

    ``session_id`` (e.g. ``user:alice``) loads the *stated* Settings risk
    profile for framing only — never inferred from clicks/engagement.
    """
    conv = conversation_messages or []

    # 1) Memory context (per-user via the ambient identity)
    context_block = ""
    try:
        from trading.memory.memory_store import get_memory_store
        from trading.services import chat_nl_service

        store = get_memory_store()
        try:
            memory_ctx = chat_nl_service.get_memory_context(store)
        except Exception:
            memory_ctx = ""
        try:
            context_block = chat_nl_service.build_context_block(
                memory_ctx, {}, intent=None, store=store
            )
        except Exception:
            context_block = memory_ctx or ""
    except Exception as e:  # noqa: BLE001
        logger.warning("chat_turn: context build failed: %s", e)

    # 1b) Stated risk-profile framing (Settings prefs — explicit only)
    try:
        from trading.portfolio.risk_profile import (
            chat_framing_block,
            load_stated_risk_profile,
        )

        profile = load_stated_risk_profile(session_id)
        risk_block = chat_framing_block(profile)
        if risk_block:
            context_block = (
                f"{context_block}\n\n{risk_block}".strip()
                if context_block
                else risk_block
            )
    except Exception as e:  # noqa: BLE001
        logger.debug("chat_turn: risk profile framing skipped: %s", e)

    # 2) Agent Skills (playbooks matched to the message; '' when none)
    skills_ctx = ""
    try:
        from trading.services.skill_loader import render_skills_context

        skills_ctx = render_skills_context(user_message) or ""
    except Exception as e:  # noqa: BLE001
        logger.warning("chat_turn: skill loading failed: %s", e)

    # 3) Tool loop, then plain-chat fallback
    from trading.services import chat_nl_service

    system_prompt = getattr(chat_nl_service, "EVOLVE_CHAT_SYSTEM_PROMPT", "")
    try:
        from agents.llm.tool_executor import execute_with_tools

        res = execute_with_tools(
            user_message=user_message,
            context_block=context_block,
            conversation_messages=conv,
            system_prompt=system_prompt,
            platform_context_suffix=skills_ctx,
            focus_symbol=focus_symbol,
            available_tools=STANDARD_TOOLS,
            max_tokens=max_tokens,
        )
        reply = (getattr(res, "text", "") or "").strip()
        # The executor emits literal placeholders when no LLM answered;
        # treating them as success would show users "No response." as if
        # it were an answer. Fall through to the next path instead.
        if reply.lower() in {"no response.", "no response",
                             "i didn't get a response. please try again."}:
            reply = ""
        if reply:
            return {
                "success": True,
                "reply": reply,
                "tool_captions": list(getattr(res, "tool_captions", None) or []),
                "error": None,
            }
        logger.warning("chat_turn: tool loop returned empty text; falling back")
    except Exception as e:  # noqa: BLE001
        logger.warning("chat_turn: tool loop failed, fallback: %s", e)

    try:
        from agents.llm.active_llm_calls import call_active_llm_chat

        reply = (call_active_llm_chat(
            system_prompt, context_block, conv, user_message,
            max_tokens=max_tokens,
        ) or "").strip()
        if reply.lower() in {"no response.", "no response"}:
            reply = ""
        if reply:
            return {"success": True, "reply": reply,
                    "tool_captions": [], "error": None}
    except Exception as e:  # noqa: BLE001
        logger.warning("chat_turn: plain chat fallback failed: %s", e)

    return {
        "success": False,
        "reply": None,
        "tool_captions": [],
        "error": "No LLM responded - add an API key in Settings.",
    }
