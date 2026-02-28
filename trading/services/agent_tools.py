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
            "suggestions": ["Check Strategy Testing page for detailed metrics."],
        }
