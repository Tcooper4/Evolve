"""
Home briefing service: builds context from MemoryStore, market data, and risk,
then asks the active LLM (Admin-selected) to produce a plain-English briefing and 2–4 dynamic cards.
Used by pages/0_Home.py for the personalized morning briefing.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default symbols when none from memory/preferences
DEFAULT_SYMBOLS = ["SPY"]


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
    """Derive key symbols from watchlist, portfolio, and memory; avoid hardcoded single names."""
    from trading.memory.memory_store import MemoryType

    symbols: List[str] = []

    # 1) Watchlist preferences (primary)
    try:
        pref = store.list(MemoryType.PREFERENCE, limit=100)
        for r in pref:
            key = (getattr(r, "key", None) or (r.get("key") if isinstance(r, dict) else None)) or ""
            v = r.value if hasattr(r, "value") else r.get("value", r)
            if key.lower().endswith("watchlist") and isinstance(v, (str, list)):
                if isinstance(v, str):
                    candidates = [s.strip().upper() for s in v.split(",") if s.strip()]
                else:
                    candidates = [str(s).upper() for s in v if isinstance(s, str)]
                symbols.extend(candidates[:20])
            elif "symbol" in key.lower() or "ticker" in key.lower():
                if isinstance(v, str) and len(v) <= 10:
                    symbols.append(v.upper())
    except Exception:
        pass

    # 2) Portfolio positions
    try:
        from trading.portfolio.portfolio_manager import PortfolioManager
        pm = PortfolioManager()
        positions = getattr(pm, "state", None) and getattr(pm.state, "positions", None) or []
        for pos in positions or []:
            sym = (
                pos.get("symbol")
                if isinstance(pos, dict)
                else getattr(pos, "symbol", None)
            )
            if isinstance(sym, str) and sym:
                symbols.append(sym.upper())
    except Exception:
        pass

    # 3) Memory-derived symbols (fallback)
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
    out: List[str] = []
    for s in symbols:
        if s not in seen and len(s) <= 6:
            seen.add(s)
            out.append(s)
    # If still empty, use top movers from session_state if available
    if not out:
        try:
            import streamlit as st  # type: ignore
            movers_state = st.session_state.get("home_last_top_movers") or {}
            gainers = movers_state.get("gainers") or []
            losers = movers_state.get("losers") or []
            for m in list(gainers) + list(losers):
                sym = m.get("symbol") if isinstance(m, dict) else None
                if isinstance(sym, str) and sym:
                    if sym.upper() not in seen and len(sym) <= 6:
                        seen.add(sym.upper())
                        out.append(sym.upper())
        except Exception:
            pass

    # Always include SPY as a market overview card if not already present
    if "SPY" not in out:
        out.insert(0, "SPY")

    return out[:8] if out else DEFAULT_SYMBOLS


def _get_market_data(symbols: List[str]) -> Dict[str, Any]:
    """Fetch current market data for symbols via data provider (fallback when name is None)."""
    result = {}
    try:
        from trading.data.providers import get_data_provider
        provider = get_data_provider()  # returns fallback provider when name is None
        if provider is None:
            return result
        end = datetime.now()
        start = end - timedelta(days=30)
        for sym in symbols:
            try:
                df = provider.fetch(sym, "1d", start_date=start.strftime("%Y-%m-%d"), end_date=end.strftime("%Y-%m-%d"))
                if df is not None and not df.empty:
                    close = df["close"] if "close" in df.columns else (df["Close"] if "Close" in df.columns else df.iloc[:, -1])
                    tail_df = df.tail(14)
                    series = close.tail(14).tolist() if len(close) >= 2 else []
                    dates = []
                    if len(tail_df) and hasattr(tail_df.index, "strftime"):
                        try:
                            dates = [d.strftime("%b %d") for d in tail_df.index]
                        except Exception:
                            dates = [str(d)[:10] for d in tail_df.index]
                    elif len(tail_df):
                        dates = [str(i) for i in range(len(tail_df))]
                    result[sym] = {
                        "current": float(close.iloc[-1]) if len(close) else None,
                        "week_ago": float(close.iloc[-5]) if len(close) >= 5 else None,
                        "series": series,
                        "dates": dates if len(dates) == len(series) else [],
                    }
            except Exception as e:
                logger.debug(f"Market data for {sym}: {e}")
    except Exception as e:
        logger.warning(f"Market data fetch failed: {e}")
    return result


def _format_market_data_readable(market_data: Dict[str, Any]) -> str:
    """Format market data dict as readable plain English (no JSON)."""
    if not market_data:
        return "No market data available."
    lines = []
    for sym, data in market_data.items():
        if not isinstance(data, dict):
            continue
        cur = data.get("current")
        week = data.get("week_ago")
        if cur is not None:
            line = f"{sym}: ${float(cur):,.2f}"
            if week is not None:
                pct = ((float(cur) - float(week)) / float(week) * 100) if week else 0
                line += f" (vs week ago: {pct:+.1f}%)"
            lines.append(line)
    return "\n".join(lines) if lines else "No market data available."


def _format_risk_snapshot_readable(risk_snapshot: str, summary: Any) -> str:
    """Format risk/portfolio snapshot as readable text (no raw JSON)."""
    parts = []
    if isinstance(summary, dict):
        equity = summary.get("current_equity") or summary.get("equity")
        if equity is not None:
            parts.append(f"Equity: ${float(equity):,.2f}")
        open_pos = summary.get("open_positions", 0)
        parts.append(f"Open positions: {open_pos}")
        pnl = summary.get("total_pnl")
        if pnl is not None:
            parts.append(f"P&L: ${float(pnl):,.2f}")
        if not parts:
            for k, v in list(summary.items())[:5]:
                if v is not None and k not in ("strategy_performance", "last_updated"):
                    parts.append(f"{k.replace('_', ' ').title()}: {v}")
    if not parts and risk_snapshot and "not available" not in risk_snapshot.lower():
        return risk_snapshot[:500]
    return " | ".join(parts) if parts else "No portfolio data available."


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
    Gather memory, market data, and risk; call the active LLM (Admin-selected) to produce briefing text and 2–4 cards.
    Before LLM: run monitoring checks (model/strategy degradation) and fetch recent recommendations.
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

    def _make_fallback_briefing(portfolio_summary: Any = None) -> Dict[str, Any]:
        """Build fallback briefing with readable text (no raw JSON)."""
        risk_text = _format_risk_snapshot_readable(risk_snapshot, portfolio_summary or {})
        memory_text = (memory_context[:2000] + ("..." if len(memory_context) > 2000 else "")) if memory_context else "No trading history or preferences in memory yet."
        market_text = _format_market_data_readable(market_data)
        fallback_lines = [
            "**AI briefing unavailable** — configure an LLM provider in Admin (e.g. Claude, GPT-4, Gemini, or Ollama) for personalized briefings.",
            "",
            "**Memory (recent):**",
            memory_text,
            "",
            "**Market data:**",
            market_text,
            "",
            "**Risk snapshot:**",
            risk_text,
        ]
        return {
            "briefing_text": "\n".join(fallback_lines),
            "cards": [
                {"headline": "Configure AI briefing", "detail": "Go to Admin → AI Model Settings, choose a provider and set its API key, then Save. Use Refresh briefing to try again.", "card_type": "news"},
            ],
            "market_data": market_data,
        }

    # Use active LLM (Admin-selected); if none configured or call fails, return readable fallback
    try:
        from agents.llm.active_llm_calls import get_active_llm, call_active_llm_chat
        provider, _model, _opts = get_active_llm()
        if not provider:
            portfolio_summary = None
            try:
                from trading.portfolio.portfolio_manager import PortfolioManager
                portfolio_summary = PortfolioManager().get_performance_summary()
            except Exception:
                pass
            return _make_fallback_briefing(portfolio_summary)
    except Exception as e:
        logger.warning(f"Active LLM not available for briefing: {e}")
        portfolio_summary = None
        try:
            from trading.portfolio.portfolio_manager import PortfolioManager
            portfolio_summary = PortfolioManager().get_performance_summary()
        except Exception:
            pass
        return _make_fallback_briefing(portfolio_summary)

    user_message = "Generate today's briefing and cards."
    try:
        logger.debug("Calling LLM with provider: %s, model: %s", provider, _model)
        text = call_active_llm_chat(
            system_prompt=system_prompt,
            context_block=context_block,
            conversation_messages=[],
            user_message=user_message,
            max_tokens=2048,
        )
        text = (text or "").strip()
    except Exception as e:
        logger.exception(f"Briefing LLM call failed: {e}")
        portfolio_summary = None
        try:
            from trading.portfolio.portfolio_manager import PortfolioManager
            portfolio_summary = PortfolioManager().get_performance_summary()
        except Exception:
            pass
        return _make_fallback_briefing(portfolio_summary)

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
