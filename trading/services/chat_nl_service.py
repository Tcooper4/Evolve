"""
Natural-language chat service for Evolve.

NL_INTERFACE: Builds memory context, routes intent via EnhancedPromptRouter,
runs agent actions (PromptAgent), and synthesizes personalized responses with Claude.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# System prompt for the conversational AI (Evolve's personal quant advisor)
EVOLVE_CHAT_SYSTEM_PROMPT = """You are Evolve's AI financial advisor. You have access to the user's full trading history, backtest results, strategy performance, risk profile, current portfolio state, and any live forecast data provided in the context below.

Your role:
- Give direct, specific advice grounded in the user's actual data—never generic advice.
- When making recommendations (e.g., reduce position size, change strategy), always explain why based on their history and current market context.
- If context includes agent output (e.g., backtest metrics, risk report, or a 7-day price forecast), use it to answer and **always quote the specific numbers** (e.g., "The ARIMA model forecasts TSLA at $245.20 in 7 days") rather than generic text like "run the Forecasting page".
- If the user asks for an action (backtest, risk report, etc.), the context will include the result of that action; synthesize it into a clear, actionable answer.
- Be concise but complete. Use bullet points or short paragraphs when listing recommendations or data.
- If you do not have enough data to answer personally, say so and suggest what would help (e.g., run a backtest, check Risk page).
"""


@dataclass
class ChatTurnResult:
    """Result of one chat turn: agent output (if any) and final assistant message."""

    assistant_message: str
    intent: Optional[str] = None
    agent_response_summary: Optional[str] = None
    action_data: Optional[Dict[str, Any]] = None  # e.g. metrics, equity_curve for UI


def get_trading_context_summary(store: Any) -> str:
    """
    Query MemoryStore for last backtests, forecast, trades, and risk snapshot.
    Return a single plain-English paragraph for Chat situational awareness.
    """
    from trading.memory.memory_store import MemoryType

    parts = []
    try:
        # Last 3 backtests (namespace=backtests, category=results)
        recs = store.list(MemoryType.LONG_TERM, namespace="backtests", category="results", limit=3)
        for r in recs:
            v = r.value if hasattr(r, "value") else r.get("value", r)
            if not isinstance(v, dict):
                continue
            name = v.get("strategy_name", "?")
            sym = v.get("symbol", "?")
            ret = v.get("total_return")
            sharpe = v.get("sharpe_ratio")
            ts = v.get("timestamp", "")
            ret_s = f"{ret:.1%}" if isinstance(ret, (int, float)) else str(ret)
            sharpe_s = f"{sharpe:.1f}" if isinstance(sharpe, (int, float)) else str(sharpe)
            when = _relative_time(ts)
            parts.append(f"Last backtest: {name} on {sym} returned {ret_s}, Sharpe {sharpe_s} ({when}).")
            if v.get("anomalous"):
                parts.append("Note: last backtest result was anomalous — treat with caution.")
        if parts:
            parts.append("")
    except Exception as e:
        logger.debug("Trading context backtests: %s", e)

    try:
        recs = store.list(MemoryType.LONG_TERM, namespace="forecasts", category="results", limit=1)
        if recs:
            r = recs[0]
            v = r.value if hasattr(r, "value") else r.get("value", r)
            if isinstance(v, dict):
                sym = v.get("symbol", "?")
                model = v.get("model_name", "?")
                horizon = v.get("horizon", "?")
                first = v.get("forecast_first")
                last = v.get("forecast_last")
                when = _relative_time(v.get("timestamp", ""))
                pred = ""
                if first is not None and last is not None:
                    try:
                        pct = ((float(last) - float(first)) / float(first)) * 100 if float(first) != 0 else 0
                        pred = f" predicts {pct:+.1f}% over {horizon} days"
                    except (TypeError, ValueError):
                        pred = f" over {horizon} days"
                else:
                    pred = f" over {horizon} days"
                parts.append(f"Last forecast: {model} on {sym}{pred} ({when}).")
                if v.get("model_failed"):
                    parts.append("Note: last forecast may indicate model failure — treat with caution.")
                parts.append("")
    except Exception as e:
        logger.debug("Trading context forecasts: %s", e)

    try:
        recs = store.list(MemoryType.LONG_TERM, namespace="trades", category="orders", limit=3)
        for r in recs:
            v = r.value if hasattr(r, "value") else r.get("value", r)
            if not isinstance(v, dict):
                continue
            action = v.get("action", "?")
            sym = v.get("symbol", "?")
            qty = v.get("quantity", "?")
            price = v.get("price")
            mode = v.get("mode", "paper")
            when = _relative_time(v.get("timestamp", ""))
            price_s = f"${price:.2f}" if isinstance(price, (int, float)) else str(price)
            parts.append(f"Last trade: {action.capitalize()} {qty} {sym} at {price_s} ({mode}, {when}).")
        if recs:
            parts.append("")
    except Exception as e:
        logger.debug("Trading context trades: %s", e)

    try:
        recs = store.list(MemoryType.LONG_TERM, namespace="risk", category="snapshots", limit=1)
        if recs:
            r = recs[0]
            v = r.value if hasattr(r, "value") else r.get("value", r)
            if isinstance(v, dict):
                var = v.get("var_95")
                vol = v.get("volatility")
                dd = v.get("max_drawdown")
                sharpe = v.get("sharpe_ratio")
                var_s = f"{abs(var):.2%}" if isinstance(var, (int, float)) else str(var)
                vol_s = f"{vol:.2%}" if isinstance(vol, (int, float)) else str(vol)
                dd_s = f"{abs(dd):.2%}" if isinstance(dd, (int, float)) else str(dd)
                sharpe_s = f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else str(sharpe)
                parts.append(f"Risk: VaR95={var_s}, volatility={vol_s}, max drawdown={dd_s}, Sharpe={sharpe_s}.")
    except Exception as e:
        logger.debug("Trading context risk: %s", e)

    if not parts:
        return "No recent backtests, forecasts, trades, or risk snapshot in memory."
    return " ".join(p.strip() for p in parts if p and p.strip())


def _relative_time(ts: str) -> str:
    """Return a short relative time string (e.g. '3 days ago')."""
    if not ts:
        return "unknown"
    try:
        from datetime import datetime
        s = str(ts).replace("Z", "").replace("+00:00", "")[:26]
        dt = datetime.fromisoformat(s)
        now = datetime.utcnow()
        if getattr(dt, "tzinfo", None):
            dt = dt.replace(tzinfo=None)
        delta = now - dt
        sec = delta.total_seconds()
        if sec < 60:
            return "just now"
        if sec < 3600:
            return f"{int(sec / 60)} min ago"
        if sec < 86400:
            return f"{int(sec / 3600)} hours ago"
        if sec < 604800:
            return f"{int(sec / 86400)} days ago"
        return f"{int(sec / 604800)} weeks ago"
    except Exception:
        return "recently"


def get_memory_context(
    store: Any,
    *,
    limit_lt: int = 25,
    limit_pref: int = 20,
) -> str:
    """Build a string of relevant long-term and preference memory for Claude."""
    from trading.memory.memory_store import MemoryType

    parts = []
    try:
        lt = store.list(MemoryType.LONG_TERM, limit=limit_lt)
        for r in lt:
            v = r.value if hasattr(r, "value") else r.get("value", r)
            cat = getattr(r, "category", None) or (r.get("category") if isinstance(r, dict) else None)
            parts.append(f"[Long-term | {cat or 'general'}]: {_safe_str(v)}")
    except Exception as e:
        logger.warning(f"Error listing long-term memory: {e}")
    try:
        pref = store.list(MemoryType.PREFERENCE, limit=limit_pref)
        for r in pref:
            v = r.value if hasattr(r, "value") else r.get("value", r)
            key = getattr(r, "key", None) or (r.get("key") if isinstance(r, dict) else None)
            parts.append(f"[Preference | {key or 'general'}]: {_safe_str(v)}")
    except Exception as e:
        logger.warning(f"Error listing preference memory: {e}")
    if not parts:
        return "No prior trading history or preferences in memory."
    return "\n".join(parts[: limit_lt + limit_pref])


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v[:2000]
    try:
        return json.dumps(v, default=str)[:2000]
    except Exception:
        return str(v)[:2000]


def parse_intent(router: Any, prompt: str) -> Any:
    """Parse user prompt with EnhancedPromptRouter. Returns ParsedIntent or route result."""
    try:
        agents_dict = {
            "forecasting": True,
            "backtesting": True,
            "portfolio": True,
            "risk": True,
            "research": True,
            "compare_strategies": True,
            "general_agent": True,
        }
        result = router.route(prompt, agents_dict)
        return result
    except Exception as e:
        logger.warning(f"Router failed: {e}")
        return {"intent": "unknown", "args": {}, "routed_agent": "general_agent"}


def run_agent_action(prompt: str) -> Dict[str, Any]:
    """
    Run PromptAgent.process_prompt(prompt) and return a normalized dict
    (success, message, data, recommendations, next_actions).
    """
    try:
        try:
            from agents.llm.agent import PromptAgent
        except ImportError:
            from trading.llm.agent import PromptAgent
        agent = PromptAgent()
        resp = agent.process_prompt(prompt)
        if hasattr(resp, "success"):
            return {
                "success": resp.success,
                "message": getattr(resp, "message", "") or "",
                "data": getattr(resp, "data", None),
                "recommendations": getattr(resp, "recommendations", None) or [],
                "next_actions": getattr(resp, "next_actions", None) or [],
            }
        return {
            "success": resp.get("success", False),
            "message": resp.get("message", ""),
            "data": resp.get("data"),
            "recommendations": resp.get("recommendations") or [],
            "next_actions": resp.get("next_actions") or [],
        }
    except Exception as e:
        logger.exception(f"Agent action failed: {e}")
        return {
            "success": False,
            "message": str(e),
            "data": None,
            "recommendations": [],
            "next_actions": [],
        }


def _agent_response_to_dict(agent_response: Any) -> Dict[str, Any]:
    """Normalize AgentResponse or dict to dict so .get() can be used safely."""
    if isinstance(agent_response, dict):
        return agent_response
    return {
        "success": getattr(agent_response, "success", False),
        "message": getattr(agent_response, "message", ""),
        "data": getattr(agent_response, "data", None),
        "recommendations": getattr(agent_response, "recommendations", None) or [],
        "next_actions": getattr(agent_response, "next_actions", None) or [],
    }


def build_context_block(memory_context: str, agent_response: Dict[str, Any], intent: Optional[str] = None, store: Any = None) -> str:
    """Build the context string to send to Claude (optional trading context + memory + last agent output)."""
    agent_response = _agent_response_to_dict(agent_response)
    blocks = []
    if store:
        try:
            trading = get_trading_context_summary(store)
            blocks.append(f"Current trading context:\n{trading}")
            blocks.append("")
        except Exception as e:
            logger.debug("Trading context for Chat failed: %s", e)
    blocks.append(f"Relevant memory and preferences:\n{memory_context}")
    blocks.append("\nLatest agent output (use this to answer the user):")
    blocks.append(f"Success: {agent_response.get('success', False)}")
    blocks.append(f"Message: {agent_response.get('message', '')}")
    if agent_response.get("data"):
        blocks.append(f"Data summary: {_safe_str(agent_response['data'])}")
    if agent_response.get("recommendations"):
        blocks.append(f"Recommendations: {agent_response['recommendations']}")
    if agent_response.get("next_actions"):
        blocks.append(f"Next actions: {agent_response['next_actions']}")
    if intent:
        blocks.append(f"(Detected intent: {intent})")
    return "\n".join(blocks)


def call_claude(
    system_prompt: str,
    context_block: str,
    conversation_messages: List[Dict[str, str]],
    user_message: str,
    *,
    max_tokens: int = 2048,
) -> str:
    """
    Generate the assistant reply using the app's active LLM (Admin setting).
    Routes through get_active_llm(); supports Claude, GPT-4, Gemini, Ollama, HuggingFace.
    On failure logs and returns a user-friendly error message (chat must not crash).
    """
    try:
        from agents.llm.active_llm_calls import call_active_llm_chat
        text = call_active_llm_chat(
            system_prompt,
            context_block,
            conversation_messages,
            user_message,
            max_tokens=max_tokens,
        )
        return text.strip() or "I didn't get a response. Please try again."
    except Exception as e:
        logger.exception("Active LLM call failed: %s", e)
        return f"I couldn't connect to the AI: {e}. Please check Admin → AI Model Settings or try again."


def summarize_conversation(messages: List[Dict[str, str]], max_turns: int = 50) -> str:
    """Produce a short summary of the conversation for storing in short-term memory. Uses active LLM."""
    if not messages:
        return "No conversation to summarize."
    try:
        from agents.llm.active_llm_calls import call_active_llm_simple
        conv = "\n".join(
            f"{m.get('role', 'user')}: {(m.get('content') or '')[:500]}"
            for m in messages[-max_turns:]
        )
        prompt = f"Summarize this trading chat in 2–4 short sentences (topics, decisions, and any actions taken):\n\n{conv}"
        return (call_active_llm_simple(prompt, max_tokens=256) or "").strip() or "Chat summary unavailable."
    except Exception as e:
        logger.warning(f"Summarization failed: {e}")
        return f"Chat with {len(messages)} messages (summary failed: {e})."
