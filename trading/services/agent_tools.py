# -*- coding: utf-8 -*-
"""
Agent tools — standalone callable functions for Chat and other live paths.

Each function does one job, takes clear inputs, and returns a plain dict.
Replaces direct use of agent classes for: research, regime detection,
strategy/model recommendation, and backtest critique.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def market_research(topic: str, max_results: int = 3) -> Dict[str, Any]:
    """
    Run market/research search (GitHub + arXiv) and return findings.
    Used when user asks for research, papers, or code ideas.
    """
    try:
        from trading.agents.research_agent import ResearchAgent

        agent = ResearchAgent()
        findings = agent.research(topic, max_results)
        return {"success": True, "findings": findings, "count": len(findings), "topic": topic}
    except Exception as e:
        logger.exception("market_research failed: %s", e)
        return {"success": False, "error": str(e), "findings": []}


def detect_market_regime(symbol: str = "SPY", period: str = "1y") -> Dict[str, Any]:
    """
    Detect current market regime (bull, bear, sideways, volatile) and return
    regime name, confidence, and recommended strategies.
    """
    try:
        from trading.agents.market_regime_agent import MarketRegimeAgent

        agent = MarketRegimeAgent()
        analysis = agent.analyze_regime(symbol)
        return {
            "success": True,
            "symbol": symbol,
            "regime": analysis.current_regime.value,
            "confidence": analysis.regime_confidence,
            "recommended_strategies": getattr(
                analysis, "recommended_strategies", []
            ) or [],
            "risk_level": getattr(analysis, "risk_level", "medium"),
        }
    except Exception as e:
        logger.exception("detect_market_regime failed: %s", e)
        return {
            "success": False,
            "error": str(e),
            "regime": "unknown",
            "confidence": 0.0,
            "recommended_strategies": [],
        }


def recommend_strategy(
    market_regime: Optional[str] = None,
    symbol: Optional[str] = None,
    volatility: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Given market conditions, recommend a strategy name and optional parameters.
    """
    regime_map = {
        "bull": ["MACD", "SMA Crossover", "trend_following"],
        "bear": ["RSI", "Bollinger Bands", "mean_reversion"],
        "sideways": ["RSI", "Bollinger Bands", "mean_reversion"],
        "volatile": ["Bollinger Bands", "MACD", "volatility"],
        "trending": ["MACD", "SMA Crossover"],
    }
    regime = (market_regime or "sideways").lower()
    strategies = regime_map.get(regime, ["RSI", "Bollinger Bands"])
    return {"success": True, "market_regime": regime, "recommended_strategies": strategies}


def recommend_model(
    horizon: str = "short_term",
    market_regime: Optional[str] = None,
    n_data_points: int = 500,
) -> Dict[str, Any]:
    """
    Given horizon and regime, recommend a model name (e.g. LSTM, Prophet, ARIMA).
    Used when user asks 'what model should I use for X'.
    """
    try:
        from trading.agents.model_selector_agent import (
            ForecastingHorizon,
            MarketRegime as ModelSelectorRegime,
            ModelSelectorAgent,
        )

        agent = ModelSelectorAgent()
        h = getattr(ForecastingHorizon, horizon.upper(), ForecastingHorizon.SHORT_TERM)
        if isinstance(h, str):
            h = ForecastingHorizon.SHORT_TERM
        regime_str = (market_regime or "sideways").upper().replace(" ", "_")
        r = getattr(ModelSelectorRegime, regime_str, ModelSelectorRegime.SIDEWAYS)
        if isinstance(r, str):
            r = ModelSelectorRegime.SIDEWAYS
        model_id, confidence = agent.select_model(h, r, n_data_points)
        return {"success": True, "model": model_id, "confidence": confidence}
    except Exception as e:
        logger.exception("recommend_model failed: %s", e)
        by_horizon = {
            "short_term": ["LSTM", "ARIMA"],
            "medium_term": ["XGBoost", "Prophet"],
            "long_term": ["Prophet", "Ensemble"],
        }
        return {
            "success": False,
            "error": str(e),
            "fallback_models": by_horizon.get(
                (horizon or "short_term").lower(), ["LSTM", "ARIMA", "Prophet"]
            ),
        }


def critique_backtest(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given backtest metrics (sharpe_ratio, max_drawdown, win_rate, etc.),
    return a short critique and list of suggestions.
    Used when user says 'critique my last backtest'.
    """
    if not metrics:
        return {"success": False, "error": "No metrics provided", "critique": "", "suggestions": []}
    try:
        from trading.agents.performance_critic_agent import PerformanceCriticAgent

        agent = PerformanceCriticAgent()
        agent._setup()
        recommendations = []
        sharpe = metrics.get("sharpe_ratio") or metrics.get("sharpe")
        drawdown = metrics.get("max_drawdown") or metrics.get("drawdown")
        win_rate = metrics.get("win_rate")
        if sharpe is not None and float(sharpe) < 0.5:
            recommendations.append("Consider improving risk-adjusted returns (Sharpe < 0.5).")
        if drawdown is not None and float(drawdown) < -0.15:
            recommendations.append("Max drawdown is steep; consider position sizing or stop-losses.")
        if win_rate is not None and float(win_rate) < 0.45:
            recommendations.append("Win rate is below 45%; review entry/exit rules.")
        if not recommendations:
            recommendations.append("Metrics look reasonable; consider walk-forward validation.")
        critique = " ".join(recommendations)
        return {
            "success": True,
            "critique": critique,
            "suggestions": recommendations,
            "metrics_reviewed": list(metrics.keys()),
        }
    except Exception as e:
        logger.exception("critique_backtest failed: %s", e)
        return {
            "success": False,
            "error": str(e),
            "critique": "Could not run full critique.",
            "suggestions": ["Check the Backtest page for detailed metrics."],
        }


def scan_universe(
    universe: str = "default",
    min_score: float = 6.0,
    max_results: int = 15,
) -> Dict[str, Any]:
    """
    Screen tickers with optional AI score floor. Use for scanners, ideas, or
    “what looks good” questions.
    """
    try:
        from trading.analysis.market_scanner import _get_universe, scan_market

        u = (universe or "default").lower()
        uni = list(_get_universe(u))
        if u in ("sp50", "large", "mega"):
            uni = uni[:50]
        elif u in ("sp30", "core"):
            uni = uni[:30]
        # Wider cap so min_score can filter client-side
        raw = scan_market(
            filters=[],
            universe=uni,
            max_results=min(200, max(len(uni), max_results * 8)),
        )
        if raw.get("error"):
            return {"success": False, "error": raw["error"], "results": []}
        rows = raw.get("results") or []
        filtered = [
            r
            for r in rows
            if float(r.get("ai_score", 0) or 0) >= float(min_score)
        ][: int(max_results)]
        return {
            "success": True,
            "results": filtered,
            "scanned": raw.get("scanned"),
            "scan_time_s": raw.get("scan_time_s"),
        }
    except Exception as e:
        logger.exception("scan_universe failed: %s", e)
        return {"success": False, "error": str(e), "results": []}


def get_ai_score(symbol: str) -> Dict[str, Any]:
    """Compute Evolve AI Score for one ticker. Use for single-name quality checks."""
    try:
        import yfinance as yf
        from trading.analysis.ai_score import compute_ai_score

        sym = (symbol or "").strip().upper()
        if not sym:
            return {"success": False, "error": "symbol required"}
        hist = yf.Ticker(sym).history(period="6mo")
        if hist.empty:
            return {"success": False, "error": f"No data for {sym}"}
        out = compute_ai_score(sym, hist)
        return {"success": True, "symbol": sym, "score": out}
    except Exception as e:
        logger.exception("get_ai_score failed: %s", e)
        return {"success": False, "error": str(e)}


def get_forecast(symbol: str, horizon: int = 7) -> Dict[str, Any]:
    """Consensus multi-model forecast. Use when user asks price targets or direction."""
    try:
        import yfinance as yf
        from trading.models.forecast_router import ForecastRouter

        sym = (symbol or "").strip().upper()
        if not sym:
            return {"success": False, "error": "symbol required"}
        hist = yf.Ticker(sym).history(period="2y")
        if hist.empty:
            return {"success": False, "error": f"No data for {sym}"}
        router = ForecastRouter()
        fc = router.get_consensus_forecast(
            data=hist, horizon=int(horizon), symbol=sym
        )
        if fc.get("error"):
            return {"success": False, "error": fc["error"], "forecast": fc}
        return {"success": True, "symbol": sym, "forecast": fc}
    except Exception as e:
        logger.exception("get_forecast failed: %s", e)
        return {"success": False, "error": str(e)}


def get_news(symbol: str, max_items: int = 10) -> Dict[str, Any]:
    """Headlines and ranked articles for a symbol."""
    try:
        from trading.data.news_aggregator import get_news as _gn

        sym = (symbol or "").strip().upper()
        if not sym:
            return {"success": False, "error": "symbol required", "items": []}
        items = _gn(sym, max_items=int(max_items))
        return {"success": True, "symbol": sym, "items": items}
    except Exception as e:
        logger.exception("get_news failed: %s", e)
        return {"success": False, "error": str(e), "items": []}


def get_risk_metrics(symbol: str, period: str = "1y") -> Dict[str, Any]:
    """Performance and risk stats from daily returns (Sharpe, drawdown, etc.)."""
    try:
        import yfinance as yf
        from utils.risk_metrics import compute_performance_metrics

        sym = (symbol or "").strip().upper()
        if not sym:
            return {"success": False, "error": "symbol required"}
        hist = yf.Ticker(sym).history(period=period or "1y")
        if hist.empty:
            return {"success": False, "error": f"No data for {sym}"}
        _cm = {c.lower(): c for c in hist.columns}
        cc = _cm.get("close", hist.columns[0])
        rets = hist[cc].pct_change().dropna()
        if len(rets) < 30:
            return {"success": False, "error": "Insufficient return history"}
        pm = compute_performance_metrics(rets)
        return {"success": True, "symbol": sym, "metrics": pm.to_dict()}
    except Exception as e:
        logger.exception("get_risk_metrics failed: %s", e)
        return {"success": False, "error": str(e)}


def get_pattern_analysis(symbol: str) -> Dict[str, Any]:
    """Detect chart patterns and trend context for a symbol."""
    try:
        import yfinance as yf
        from trading.analysis.chart_pattern_detector import ChartPatternDetector

        sym = (symbol or "").strip().upper()
        if not sym:
            return {"success": False, "error": "symbol required"}
        hist = yf.Ticker(sym).history(period="6mo")
        if hist.empty:
            return {"success": False, "error": f"No data for {sym}"}
        det = ChartPatternDetector(sym, hist)
        out = det.detect_all()
        patterns = out.get("patterns") or []
        trend = out.get("trend") or {}
        lines = [
            f"{sym}: {len(patterns)} pattern(s); "
            f"trend {trend.get('direction', '?')}"
        ]
        for p in patterns[:6]:
            lines.append(
                f"  - {p.get('name')}: "
                f"{(p.get('description') or '')[:160]}"
            )
        return {
            "success": True,
            "symbol": sym,
            "summary": "\n".join(lines),
            "pattern_count": len(patterns),
        }
    except Exception as e:
        logger.exception("get_pattern_analysis failed: %s", e)
        return {"success": False, "error": str(e)}


def run_backtest(symbol: str, days: int = 90) -> Dict[str, Any]:
    """Quick historical backtest using an RSI-family strategy when available."""
    try:
        import numpy as np
        import pandas as pd
        import yfinance as yf
        from trading.backtesting.backtester import Backtester
        from trading.backtesting.trade_models import TradeType
        from trading.strategies.registry import get_strategy_registry

        sym = (symbol or "").strip().upper()
        if not sym:
            return {"success": False, "error": "symbol required"}
        d = max(30, int(days))
        raw = yf.Ticker(sym).history(period=f"{d}d", auto_adjust=True)
        if raw.empty:
            return {"success": False, "error": f"No data for {sym}"}
        _cm = {c.lower(): c for c in raw.columns}
        _cc = _cm.get("close", raw.columns[0])
        closes = raw[_cc].astype(float)
        price_df = pd.DataFrame({sym: closes}).dropna()
        reg = get_strategy_registry()
        names = reg.get_strategy_names()
        strat_name = next(
            (n for n in names if "rsi" in n.lower()),
            names[0] if names else None,
        )
        if not strat_name:
            return {"success": False, "error": "No strategies in registry"}
        strat_res = reg.execute_strategy(strat_name, raw)

        def _sig_series(strat_signals, price_index: pd.Index):
            if strat_signals is None or strat_signals.empty:
                return pd.Series(0.0, index=price_index)
            if "signal" in strat_signals.columns:
                col = strat_signals["signal"]
            else:
                num = strat_signals.select_dtypes(include=[np.number])
                col = (
                    num.iloc[:, 0]
                    if not num.empty
                    else pd.Series(0.0, index=strat_signals.index)
                )
            aligned = col.reindex(price_index).ffill().fillna(0.0)
            return aligned.astype(float)

        sig_series = _sig_series(strat_res.signals, price_df.index)
        bt_engine = Backtester(data=price_df, initial_cash=100_000.0)
        buy_thr, sell_thr = 0.01, -0.01
        for ts in price_df.index:
            price = float(price_df.loc[ts, sym])
            sig = float(sig_series.loc[ts])
            pos = float(bt_engine.positions.get(sym, 0) or 0)
            if sig > buy_thr and pos < 1e-9:
                spend = bt_engine.cash_account * 0.95
                qty = int(spend / price) if price > 0 else 0
                if qty > 0:
                    bt_engine.execute_trade(
                        ts,
                        sym,
                        float(qty),
                        price,
                        TradeType.BUY,
                        strat_name,
                        sig,
                    )
            elif sig < sell_thr and pos > 1e-9:
                bt_engine.execute_trade(
                    ts,
                    sym,
                    pos,
                    price,
                    TradeType.SELL,
                    strat_name,
                    sig,
                )
        results = bt_engine.run()
        m = results.get("metrics") or {}
        ec = results.get("equity_curve")
        sharpe_v = None
        mdd_v = None
        wr_v = None
        if ec is not None and not ec.empty and "equity_curve" in ec.columns:
            rets = ec["equity_curve"].pct_change().dropna()
            if len(rets) > 5:
                from utils.risk_metrics import compute_performance_metrics

                pm = compute_performance_metrics(rets)
                sharpe_v = pm.sharpe_ratio
                mdd_v = pm.max_drawdown
                wr_v = pm.win_rate
        tr = m.get("total_return")
        summary = (
            f"{sym} {strat_name} ({d}d): "
            f"return {float(tr or 0)*100:.2f}%"
        )
        if sharpe_v is not None:
            summary += f", Sharpe ~{sharpe_v:.2f}"
        if mdd_v is not None:
            summary += f", max DD {float(mdd_v)*100:.2f}%"
        if wr_v is not None:
            summary += f", win rate {float(wr_v)*100:.1f}%"
        return {
            "success": True,
            "symbol": sym,
            "strategy": strat_name,
            "total_return": float(tr or 0),
            "sharpe": sharpe_v,
            "max_drawdown": mdd_v,
            "win_rate": wr_v,
            "summary": summary,
        }
    except Exception as e:
        logger.exception("run_backtest failed: %s", e)
        return {"success": False, "error": str(e)}


def get_options_sentiment(symbol: str) -> Dict[str, Any]:
    """Options flow summary: put/call ratio, max pain, unusual activity."""
    try:
        from trading.data.options_flow import get_options_flow

        sym = (symbol or "").strip().upper()
        if not sym:
            return {"success": False, "error": "symbol required"}
        flow = get_options_flow(sym)
        if not flow.get("success"):
            return {
                "success": False,
                "symbol": sym,
                "error": flow.get("error") or "options unavailable",
            }
        uc = len(flow.get("unusual_calls") or [])
        up = len(flow.get("unusual_puts") or [])
        unusual = (uc + up) > 0
        pc = float(flow.get("put_call_ratio") or 0)
        mp = float(flow.get("max_pain") or 0)
        nf = flow.get("net_flow") or "NEUTRAL"
        summary = (
            f"{sym}: P/C vol ratio {pc:.2f}, max pain {mp:.2f}, "
            f"net_flow={nf}, unusual_volume={'yes' if unusual else 'no'}"
        )
        return {
            "success": True,
            "symbol": sym,
            "put_call_ratio": pc,
            "max_pain": mp,
            "net_flow": nf,
            "unusual_activity": unusual,
            "summary": summary,
        }
    except Exception as e:
        logger.exception("get_options_sentiment failed: %s", e)
        return {"success": False, "error": str(e)}


def get_evolve_platform_tool_registry():
    """
    Re-export for callers using ``from trading.services.agent_tools import …``.
    Canonical definition lives in ``agents.llm.agent``.
    """
    from agents.llm.agent import get_evolve_platform_tool_registry as _registry

    return _registry()
