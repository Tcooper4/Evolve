# -*- coding: utf-8 -*-
"""
Platform tool execution loop for Evolve chat.

Uses Option B (router JSON + heuristics): a lightweight LLM pass asks for
`{"tool_calls":[...]}`; tools run with per-call timeouts; results are injected
into the main chat context. Works across providers without Anthropic tool_use.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DEFAULT_TOOL_TIMEOUT_S = 45.0
MAX_TOOLS_PER_TURN = 3

_ROUTER_STOPWORDS = {
    "AND", "THE", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER", "WAS",
    "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM", "HIS", "HOW", "ITS", "LET",
    "MAY", "NEW", "NOW", "OLD", "SEE", "TWO", "WHO", "WAY",     "TOP", "LOW", "BIG",
    "EPS", "CEO", "IPO", "ETF", "USA", "FED",
}


@dataclass
class ToolChatResult:
    """Final assistant text plus UI captions for tools that ran."""

    text: str
    tool_captions: List[str] = field(default_factory=list)


def _registry_map() -> Dict[str, Tuple[Callable[..., Any], Dict[str, Any]]]:
    from agents.llm.agent import get_evolve_platform_tool_registry

    out: Dict[str, Tuple[Callable[..., Any], Dict[str, Any]]] = {}
    for entry in get_evolve_platform_tool_registry():
        name = entry.get("name")
        fn = entry.get("function")
        if name and callable(fn):
            out[str(name)] = (fn, entry.get("parameters") or {})
    return out


def _filter_kwargs(fn: Callable[..., Any], raw: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(fn)
        allowed = set(sig.parameters.keys())
    except Exception:
        allowed = set(raw.keys())
    return {k: v for k, v in (raw or {}).items() if k in allowed}


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    t = (text or "").strip()
    if not t:
        return None
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t)
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        return None
    chunk = t[start : end + 1]
    try:
        obj = json.loads(chunk)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _extract_symbols(user_message: str) -> List[str]:
    found = re.findall(r"\b([A-Z]{2,5})\b", (user_message or "").upper())
    return [s for s in found if s not in _ROUTER_STOPWORDS]


def _heuristic_tool_calls(
    user_message: str,
    focus_symbol: Optional[str],
    *,
    allowed_names: set,
) -> List[Dict[str, Any]]:
    msg = (user_message or "").lower()
    syms = _extract_symbols(user_message or "")
    sym = (syms[0] if syms else (focus_symbol or "").strip().upper()) or None
    calls: List[Dict[str, Any]] = []

    scan_triggers = (
        "scan", "screener", "screen", "universe", "stock picks", "top picks",
        "ideas", "what should i look", "what to buy", "look at today",
        "run the scanner", "run scanner", "best setups", "opportunities",
    )
    if any(x in msg for x in scan_triggers) and "scan_universe" in allowed_names:
        calls.append({
            "name": "scan_universe",
            "arguments": {"universe": "default", "min_score": 6.0, "max_results": 12},
        })

    if sym:
        if any(x in msg for x in ("forecast", "target", "price target", "7 day", "7-day")):
            if "get_forecast" in allowed_names:
                calls.append({
                    "name": "get_forecast",
                    "arguments": {"symbol": sym, "horizon": 7},
                })
        if any(x in msg for x in ("news", "headline", "headlines", "article")):
            if "get_news" in allowed_names:
                calls.append({
                    "name": "get_news",
                    "arguments": {"symbol": sym, "max_items": 8},
                })
        if any(x in msg for x in ("risk", "sharpe", "drawdown", "var", "volatility")):
            if "get_risk_metrics" in allowed_names:
                calls.append({
                    "name": "get_risk_metrics",
                    "arguments": {"symbol": sym, "period": "1y"},
                })
        if any(x in msg for x in ("ai score", "score", "analyze", "analysis", "deep dive")):
            if "get_ai_score" in allowed_names:
                calls.append({
                    "name": "get_ai_score",
                    "arguments": {"symbol": sym},
                })

    seen = set()
    deduped: List[Dict[str, Any]] = []
    for c in calls:
        n = c.get("name")
        if not n or n in seen or n not in allowed_names:
            continue
        seen.add(n)
        deduped.append(c)
    return deduped[:MAX_TOOLS_PER_TURN]


def _build_router_prompt(
    user_message: str,
    context_excerpt: str,
    tool_specs: List[Dict[str, Any]],
) -> str:
    lines = [
        "You are a tool router for Evolve (trading app). Reply with ONLY valid JSON, no markdown fences.",
        'Schema: {"tool_calls":[{"name":"<tool_name>","arguments":{...}}]}',
        "Use an empty tool_calls array if no tools are needed.",
        f"Maximum {MAX_TOOLS_PER_TURN} tool calls.",
        "",
        "Allowed tools and parameters:",
    ]
    for t in tool_specs:
        lines.append(f"- {t['name']}: {t['schema_hint']}")
    lines.extend([
        "",
        "Rules: Prefer live tools when the user asks for scanner/ideas, a specific ticker analysis, forecast, news, or risk stats.",
        "For scan_universe: universe is one of default|large|sp50|core; min_score 1-10; max_results integer.",
        "For symbol tools: symbol is a US ticker like AAPL.",
        "",
        "---",
        f"User message: {user_message}",
        "",
        f"Context excerpt:\n{(context_excerpt or '')[:3500]}",
    ])
    return "\n".join(lines)


def _compact_tool_result(name: str, result: Dict[str, Any]) -> str:
    if not result.get("success"):
        err = result.get("error") or "failed"
        return f"{name}: error — {err}"
    if name == "scan_universe":
        rows = result.get("results") or []
        scanned = result.get("scanned")
        parts = []
        if scanned is not None:
            parts.append(f"scanned={scanned} tickers")
        for r in rows[:10]:
            sym = r.get("symbol", "?")
            sc = r.get("ai_score", "")
            parts.append(f"  - {sym}: AI score {sc}")
        return "scan_universe:\n" + ("\n".join(parts) if parts else "  (no rows)")
    if name == "get_ai_score":
        sc = result.get("score") or {}
        overall = sc.get("overall_score", "")
        grade = sc.get("grade", "")
        summ = (sc.get("summary") or "")[:400]
        return f"get_ai_score({result.get('symbol')}): overall={overall} grade={grade}\n{summ}"
    if name == "get_forecast":
        fc = (result.get("forecast") or {})
        return (
            f"get_forecast({result.get('symbol')}): "
            f"direction={fc.get('direction')} consensus_price={fc.get('consensus_price')} "
            f"conviction={fc.get('conviction')} horizon_days={fc.get('horizon', '')}"
        )
    if name == "get_news":
        items = result.get("items") or []
        lines = [f"get_news({result.get('symbol')}):"]
        for it in items[:6]:
            title = (it.get("title") or it.get("headline") or str(it))[:200]
            lines.append(f"  - {title}")
        return "\n".join(lines) if len(lines) > 1 else lines[0] + " (none)"
    if name == "get_risk_metrics":
        m = result.get("metrics") or {}
        keys = ("sharpe_ratio", "sharpe", "max_drawdown", "win_rate", "annualized_return")
        bits = [f"{k}={m.get(k)}" for k in keys if m.get(k) is not None]
        return f"get_risk_metrics({result.get('symbol')}): " + ", ".join(bits[:8])
    return f"{name}: {json.dumps(result, default=str)[:1500]}"


def _caption_for_tool(name: str, result: Dict[str, Any]) -> str:
    if not result.get("success"):
        return f"{name} failed · {str(result.get('error', ''))[:80]}"
    if name == "scan_universe":
        scanned = result.get("scanned")
        n = len(result.get("results") or [])
        if scanned is not None:
            return f"Ran market scanner · {scanned} stocks analyzed · {n} above threshold"
        return f"Ran market scanner · {n} picks returned"
    labels = {
        "get_ai_score": "Fetched AI score",
        "get_forecast": "Ran consensus forecast",
        "get_news": "Loaded recent headlines",
        "get_risk_metrics": "Computed risk metrics",
    }
    sym = result.get("symbol") or ""
    base = labels.get(name, f"Ran {name}")
    return f"{base} · {sym}".strip()


def _run_tool(
    name: str,
    fn: Callable[..., Any],
    arguments: Dict[str, Any],
    timeout_s: float,
) -> Dict[str, Any]:
    kwargs = _filter_kwargs(fn, arguments)
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(fn, **kwargs)
            return fut.result(timeout=timeout_s)
    except FuturesTimeout:
        logger.warning("Tool %s timed out after %.0fs", name, timeout_s)
        return {"success": False, "error": f"timeout after {timeout_s:.0f}s"}
    except Exception as e:
        logger.warning("Tool %s failed: %s", name, e)
        return {"success": False, "error": str(e)}


def execute_with_tools(
    user_message: str,
    *,
    context_block: str,
    conversation_messages: List[Dict[str, str]],
    system_prompt: str,
    available_tools: Optional[Sequence[str]] = None,
    platform_context_suffix: str = "",
    focus_symbol: Optional[str] = None,
    timeout_per_tool: float = DEFAULT_TOOL_TIMEOUT_S,
    max_tokens: int = 2048,
) -> ToolChatResult:
    """
    Optionally run platform tools, then synthesize a final reply via call_active_llm_chat.

    ``platform_context_suffix`` is appended to ``context_block`` for both the
    router excerpt and the final assistant call (e.g. briefing, regime, deep-dive snapshot).
    """
    from agents.llm.active_llm_calls import (
        call_active_llm_chat,
        call_active_llm_simple,
    )

    full_context = (context_block or "").strip()
    sfx = (platform_context_suffix or "").strip()
    if sfx:
        full_context = f"{full_context}\n\n{sfx}" if full_context else sfx

    reg = _registry_map()
    allowed = set(reg.keys())
    if available_tools is not None:
        allowed &= {str(x) for x in available_tools}
    if not allowed:
        try:
            txt = call_active_llm_chat(
                system_prompt,
                full_context,
                conversation_messages,
                user_message,
                max_tokens=max_tokens,
            )
            return ToolChatResult(text=(txt or "").strip() or "No response.", tool_captions=[])
        except Exception as e:
            logger.exception("execute_with_tools: chat failed: %s", e)
            return ToolChatResult(text=f"Something went wrong: {e}", tool_captions=[])

    tool_specs = []
    for name in sorted(allowed):
        fn, params = reg[name]
        props = (params.get("properties") or {}) if isinstance(params, dict) else {}
        keys = ", ".join(props.keys()) if props else "(see agent_tools)"
        tool_specs.append({"name": name, "schema_hint": keys})

    router_prompt = _build_router_prompt(
        user_message,
        full_context[:4000],
        tool_specs,
    )

    tool_calls: List[Dict[str, Any]] = []
    try:
        raw = call_active_llm_simple(router_prompt, max_tokens=512)
        if raw and "api key" in raw.lower() and "no " in raw.lower():
            tool_calls = []
        else:
            obj = _extract_json_object(raw or "")
            if obj and isinstance(obj.get("tool_calls"), list):
                for item in obj["tool_calls"][:MAX_TOOLS_PER_TURN]:
                    if not isinstance(item, dict):
                        continue
                    n = item.get("name")
                    if n not in allowed:
                        continue
                    tool_calls.append({
                        "name": str(n),
                        "arguments": item.get("arguments")
                        if isinstance(item.get("arguments"), dict)
                        else {},
                    })
    except Exception as e:
        logger.debug("Tool router LLM failed, using heuristics: %s", e)

    if not tool_calls:
        tool_calls = _heuristic_tool_calls(
            user_message, focus_symbol, allowed_names=allowed
        )

    captions: List[str] = []
    compact_parts: List[str] = []

    for call in tool_calls[:MAX_TOOLS_PER_TURN]:
        name = call.get("name")
        if name not in reg:
            continue
        fn, _ = reg[name]
        args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        result = _run_tool(name, fn, args, timeout_per_tool)
        compact_parts.append(_compact_tool_result(name, result))
        captions.append(_caption_for_tool(name, result))

    if compact_parts:
        tool_blob = "\n\n".join(compact_parts)
        augmented = (
            f"{full_context}\n\n## Live platform tool results (use these numbers; authoritative)\n{tool_blob}"
        )
    else:
        augmented = full_context

    try:
        reply = call_active_llm_chat(
            system_prompt,
            augmented,
            conversation_messages,
            user_message,
            max_tokens=max_tokens,
        )
        text = (reply or "").strip() or "No response."
    except Exception as e:
        logger.exception("execute_with_tools: final chat failed: %s", e)
        text = f"Something went wrong: {e}"

    return ToolChatResult(text=text, tool_captions=captions)
