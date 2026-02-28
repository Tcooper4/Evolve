"""
Home briefing service: builds context from MemoryStore, market data, and risk,
then asks Claude to produce a plain-English briefing and 2–4 dynamic cards.
Used by pages/0_Home.py for the personalized morning briefing.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default symbols when none from memory/preferences
DEFAULT_SYMBOLS = ["SPY", "AAPL"]


def _get_memory_context(store: Any, limit_lt: int = 30, limit_pref: int = 20) -> str:
    """Build a string of recent long-term and preference memory for the briefing."""
    from trading.memory.memory_store import MemoryType

    parts = []
    try:
        lt = store.list(MemoryType.LONG_TERM, limit=limit_lt)
        for r in lt:
            v = r.value if hasattr(r, "value") else r.get("value", r)
            cat = getattr(r, "category", None) or (r.get("category") if isinstance(r, dict) else None)
            parts.append(f"[{cat or 'general'}]: {_safe_str(v)}")
    except Exception as e:
        logger.warning(f"Error listing long-term memory: {e}")
    try:
        pref = store.list(MemoryType.PREFERENCE, limit=limit_pref)
        for r in pref:
            v = r.value if hasattr(r, "value") else r.get("value", r)
            key = getattr(r, "key", None) or (r.get("key") if isinstance(r, dict) else None)
            parts.append(f"[preference | {key or 'general'}]: {_safe_str(v)}")
    except Exception as e:
        logger.warning(f"Error listing preference memory: {e}")
    if not parts:
        return "No trading history or preferences in memory yet. The user is new."
    return "\n".join(parts[: limit_lt + limit_pref])


def _safe_str(v: Any, max_len: int = 1500) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v[:max_len]
    try:
        return json.dumps(v, default=str)[:max_len]
    except Exception:
        return str(v)[:max_len]


def _get_key_symbols(store: Any) -> List[str]:
    """Derive key symbols from preferences/memory or return defaults."""
    from trading.memory.memory_store import MemoryType

    symbols = []
    try:
        pref = store.list(MemoryType.PREFERENCE, limit=50)
        for r in pref:
            key = (getattr(r, "key", None) or (r.get("key") if isinstance(r, dict) else None)) or ""
            if "symbol" in key.lower() or "ticker" in key.lower():
                v = r.value if hasattr(r, "value") else r.get("value", r)
                if isinstance(v, str) and len(v) <= 10:
                    symbols.append(v.upper())
    except Exception:
        pass
    try:
        lt = store.list(MemoryType.LONG_TERM, limit=50)
        for r in lt:
            v = r.value if hasattr(r, "value") else r.get("value", r)
            if isinstance(v, dict):
                for k in ("symbol", "ticker", "symbols"):
                    if k in v and v[k]:
                        s = v[k]
                        if isinstance(s, str):
                            symbols.append(s.upper())
                        elif isinstance(s, list):
                            symbols.extend([x.upper() for x in s if isinstance(x, str)][:5])
            elif isinstance(v, str) and len(v) <= 10 and v.isalpha():
                symbols.append(v.upper())
    except Exception:
        pass
    seen = set()
    out = []
    for s in symbols:
        if s not in seen and len(s) <= 6:
            seen.add(s)
            out.append(s)
    return out[:8] if out else DEFAULT_SYMBOLS


def _get_market_data(symbols: List[str]) -> Dict[str, Any]:
    """Fetch current market data for symbols via FallbackDataProvider."""
    result = {}
    try:
        from trading.data.providers import get_fallback_provider
        provider = get_fallback_provider()
        end = datetime.now()
        start = end - timedelta(days=30)
        for sym in symbols:
            try:
                df = provider.fetch(sym, "1d", start_date=start.strftime("%Y-%m-%d"), end_date=end.strftime("%Y-%m-%d"))
                if df is not None and not df.empty:
                    close = df["close"] if "close" in df.columns else (df["Close"] if "Close" in df.columns else df.iloc[:, -1])
                    result[sym] = {
                        "current": float(close.iloc[-1]) if len(close) else None,
                        "week_ago": float(close.iloc[-5]) if len(close) >= 5 else None,
                        "series": close.tail(14).tolist() if len(close) >= 2 else [],
                    }
            except Exception as e:
                logger.debug(f"Market data for {sym}: {e}")
    except Exception as e:
        logger.warning(f"Market data fetch failed: {e}")
    return result


def _get_risk_snapshot() -> str:
    """Optional risk snapshot from portfolio/risk layer."""
    try:
        from trading.portfolio.portfolio_manager import PortfolioManager
        pm = PortfolioManager()
        summary = pm.get_performance_summary()
        risk = getattr(pm, "state", None) and getattr(pm.state, "risk_metrics", None) or {}
        parts = [f"Summary: {_safe_str(summary, 800)}"]
        if risk:
            parts.append(f"Risk metrics: {_safe_str(risk, 500)}")
        return "\n".join(parts)
    except Exception as e:
        logger.debug(f"Risk snapshot not available: {e}")
        return "Risk/portfolio data not available."


def _get_recent_performance_metrics(store: Any) -> Dict[str, Any]:
    """Extract recent backtest/model metrics from MemoryStore for monitoring checks."""
    from trading.memory.memory_store import MemoryType

    out: Dict[str, Any] = {}
    try:
        records = store.list(MemoryType.LONG_TERM, limit=100, namespace=None, category=None)
        for r in records:
            v = r.value if hasattr(r, "value") else getattr(r, "value", None) or (r.get("value") if isinstance(r, dict) else None)
            if not isinstance(v, dict):
                continue
            sharpe = v.get("sharpe_ratio") or v.get("sharpe")
            drawdown = v.get("max_drawdown") or v.get("drawdown")
            win_rate = v.get("win_rate")
            if sharpe is not None or drawdown is not None or win_rate is not None:
                if "sharpe_ratio" not in out and sharpe is not None:
                    out["sharpe_ratio"] = float(sharpe)
                if "max_drawdown" not in out and drawdown is not None:
                    out["max_drawdown"] = float(drawdown)
                if "win_rate" not in out and win_rate is not None:
                    out["win_rate"] = float(win_rate)
                out["model_id"] = out.get("model_id") or v.get("model_id") or v.get("model")
                out["strategy_name"] = out.get("strategy_name") or v.get("strategy_name") or v.get("strategy")
                if len(out) >= 4:
                    break
    except Exception as e:
        logger.debug(f"Could not get recent performance metrics from memory: {e}")
    return out


def _get_recent_recommendations(store: Any, days: int = 7) -> List[str]:
    """Return plain-English recommendation strings from MemoryStore (monitoring, last N days)."""
    from trading.memory.memory_store import MemoryType

    since = (datetime.utcnow() - timedelta(days=days)).isoformat()
    lines: List[str] = []
    try:
        records = store.list(
            MemoryType.LONG_TERM,
            namespace="monitoring",
            category="recommendations",
            limit=50,
            newest_first=True,
        )
        for r in records:
            created = getattr(r, "created_at", None) or (r.get("created_at") if isinstance(r, dict) else None)
            if created and isinstance(created, str) and created < since:
                continue
            v = r.value if hasattr(r, "value") else getattr(r, "value", None) or (r.get("value") if isinstance(r, dict) else None)
            if isinstance(v, dict):
                title = v.get("title", "")
                text = v.get("text", "")
                if title or text:
                    lines.append(f"- {title}: {text}" if title else f"- {text}")
            elif isinstance(v, str) and v.strip():
                lines.append(f"- {v.strip()}")
    except Exception as e:
        logger.warning(f"Could not list recommendations: {e}")
    return lines


def generate_briefing(
    store: Any,
    *,
    force_empty_context: bool = False,
) -> Dict[str, Any]:
    """
    Gather memory, market data, and risk; call Claude to produce briefing text and 2–4 cards.
    Before Claude: run monitoring checks (model/strategy degradation) and fetch recent recommendations.
    Returns dict with keys: briefing_text, cards (list of {headline, detail, card_type, data}).
    """
    memory_context = "" if force_empty_context else _get_memory_context(store)
    symbols = [] if force_empty_context else _get_key_symbols(store)
    market_data = _get_market_data(symbols) if symbols else {}
    risk_snapshot = _get_risk_snapshot()

    # Run monitoring checks with whatever recent metrics we have (do not block briefing on failure)
    if not force_empty_context:
        metrics = _get_recent_performance_metrics(store)
        try:
            from trading.services.monitoring_tools import check_model_degradation, check_strategy_degradation
            check_model_degradation(
                model_id=metrics.get("model_id"),
                recent_sharpe=metrics.get("sharpe_ratio"),
                recent_drawdown=metrics.get("max_drawdown"),
            )
        except Exception as e:
            logger.warning(f"Monitoring check_model_degradation failed: {e}")
        try:
            from trading.services.monitoring_tools import check_strategy_degradation
            check_strategy_degradation(
                strategy_name=metrics.get("strategy_name"),
                recent_sharpe=metrics.get("sharpe_ratio"),
                recent_win_rate=metrics.get("win_rate"),
            )
        except Exception as e:
            logger.warning(f"Monitoring check_strategy_degradation failed: {e}")

    # Fetch recent system recommendations (last 7 days) for Claude context
    recommendation_lines = [] if force_empty_context else _get_recent_recommendations(store, days=7)
    recommendations_block = "\n".join(recommendation_lines) if recommendation_lines else "None yet."

    system_prompt = """You are Evolve's morning briefing assistant. Your job is to write a short, friendly briefing (2–4 paragraphs) in plain English for the user. No jargon: explain what matters today and why.

You will receive:
1) Memory: recent trades, backtest results, and user preferences (or "No trading history yet").
2) Market data: current and week-ago prices for key symbols (if any).
3) Risk snapshot: portfolio/risk summary if available.
4) System recommendations: any recent system recommendations (e.g. model/strategy degradation or data quality). Surface these in plain English in the briefing where relevant — e.g. "Your RSI strategy has underperformed its benchmark for 3 weeks. You may want to review it." Do not dump raw data; weave recommendations naturally into the narrative. If there are none, say nothing about them.

If memory is empty, give a brief market overview and suggest what to do first in Evolve (e.g. try a forecast, run a backtest, or ask a question in Chat). Keep it welcoming and simple.

Then output exactly 2–4 "cards" as a JSON array. Each card has:
- headline: one short, plain-English sentence (e.g. "Your NVDA position is up 8% this week", not "NVDA delta-adjusted return +8.2%").
- detail: 1–2 sentences of technical detail for an optional expand section.
- card_type: one of "price_chart", "risk", "signal", "news", "backtest".
- symbol (optional): for card_type "price_chart", include the ticker symbol so the UI can show a chart (e.g. "NVDA").

Only include cards that are relevant. If there's nothing urgent, you can suggest a card like "Run your first backtest" or "Check today's market snapshot". Output format: first write the briefing in plain text, then on a new line write "CARDS:" and then a valid JSON array of the cards."""

    context_parts = [
        "=== MEMORY (recent trades, backtests, preferences) ===",
        memory_context,
        "\n=== MARKET DATA (key symbols) ===",
        json.dumps(market_data, default=str) if market_data else "None",
        "\n=== RISK / PORTFOLIO SNAPSHOT ===",
        risk_snapshot,
        "\n=== System recommendations (surface in plain English if present) ===",
        recommendations_block,
    ]
    context_block = "\n".join(context_parts)

    try:
        from config.llm_config import get_llm_config, get_anthropic_client, CLAUDE_PRIMARY_MODEL
        client = get_anthropic_client()
        llm = get_llm_config()
        model = getattr(llm, "primary_model", CLAUDE_PRIMARY_MODEL)
    except Exception as e:
        logger.warning(f"Claude not available for briefing: {e}")
        return {
            "briefing_text": "We couldn't connect to the AI right now. Check that ANTHROPIC_API_KEY is set and try **Refresh briefing**.",
            "cards": [
                {"headline": "Set up your API key", "detail": "Add ANTHROPIC_API_KEY to your environment to get personalized briefings.", "card_type": "news"},
            ],
        }

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=0.3,
            system=system_prompt,
            messages=[{"role": "user", "content": f"Generate today's briefing and cards.\n\n{context_block}"}],
        )
        text = (resp.content[0].text if resp.content else "").strip()
    except Exception as e:
        logger.exception(f"Briefing API error: {e}")
        return {
            "briefing_text": f"Something went wrong generating your briefing: {e}. Try **Refresh briefing** in a moment.",
            "cards": [],
        }

    briefing_text = text
    cards = []

    if "CARDS:" in text:
        parts = text.split("CARDS:", 1)
        briefing_text = parts[0].strip()
        raw_json = parts[1].strip()
        # Strip markdown code block if present
        if raw_json.startswith("```"):
            lines = raw_json.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw_json = "\n".join(lines)
        try:
            cards = json.loads(raw_json)
            if not isinstance(cards, list):
                cards = []
            # Normalize card shape
            for c in cards:
                if not isinstance(c, dict):
                    continue
                c.setdefault("headline", "Update")
                c.setdefault("detail", "")
                c.setdefault("card_type", "news")
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse cards JSON: {e}")

    return {
        "briefing_text": briefing_text or "No briefing generated. Try **Refresh briefing**.",
        "cards": cards[:4],
        "market_data": market_data,
    }
