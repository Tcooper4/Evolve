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
    When ``symbol`` is set, infer regime from recent returns/volatility.
    """
    reason = "default sideways map"
    regime = (market_regime or "").lower().strip()
    metrics: Dict[str, Any] = {}

    if symbol and not regime:
        try:
            import numpy as np

            from trading.data.price_cache import get_history

            hist = get_history(str(symbol).strip().upper(), period="6mo")
            if hist is not None and not hist.empty:
                cm = {str(c).lower(): c for c in hist.columns}
                close = hist[cm.get("close", hist.columns[0])].astype(float)
                rets = close.pct_change().dropna()
                if len(rets) >= 20:
                    ret_20 = float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) > 21 else float(rets.tail(20).sum())
                    vol = float(rets.tail(40).std() * (252 ** 0.5))
                    metrics = {
                        "ret_20d": round(ret_20 * 100, 2),
                        "vol_ann": round(vol * 100, 2),
                    }
                    if volatility is None:
                        volatility = vol
                    if vol >= 0.28:
                        regime = "volatile"
                        reason = f"elevated annualized vol (~{vol*100:.0f}%)"
                    elif ret_20 >= 0.04:
                        regime = "bull"
                        reason = f"20d trend up ({ret_20*100:+.1f}%)"
                    elif ret_20 <= -0.04:
                        regime = "bear"
                        reason = f"20d trend down ({ret_20*100:+.1f}%)"
                    else:
                        regime = "sideways"
                        reason = "no strong 20d trend"
        except Exception as e:
            logger.debug("recommend_strategy regime infer failed: %s", e)

    if not regime:
        regime = "sideways"

    # Map to registry-friendly strategy class names used by Backtest
    regime_map = {
        "bull": ["MACDStrategy", "SMAStrategy", "RSIStrategy"],
        "bear": ["RSIStrategy", "BollingerStrategy", "MACDStrategy"],
        "sideways": ["BollingerStrategy", "RSIStrategy", "SMAStrategy"],
        "volatile": ["BollingerStrategy", "RSIStrategy", "MACDStrategy"],
        "trending": ["MACDStrategy", "SMAStrategy", "RSIStrategy"],
    }
    strategies = regime_map.get(regime, ["RSIStrategy", "BollingerStrategy"])
    primary = strategies[0]
    return {
        "success": True,
        "symbol": (symbol or "").upper() or None,
        "market_regime": regime,
        "reason": reason,
        "recommended_strategies": strategies,
        "primary_strategy": primary,
        "metrics": metrics,
        "note": "Rule-based playbook for this tape — backtest before sizing.",
    }


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
        from trading.models.forecast_router import (
            ForecastRouter,
            get_router_singleton,
        )

        sym = (symbol or "").strip().upper()
        if not sym:
            return {"success": False, "error": "symbol required"}
        hist = yf.Ticker(sym).history(period="2y")
        if hist.empty:
            return {"success": False, "error": f"No data for {sym}"}
        router = get_router_singleton()
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
            # STRUCTURED DATA (2026-07): the detector already computes
            # start_date/end_date/type/confidence per pattern - it was
            # being discarded down to a text summary. Exposed here so
            # the UI can plot patterns as dated chart markers instead of
            # only reading about them.
            "patterns": [
                {
                    "name": p.get("name"),
                    "type": p.get("type"),
                    "confidence": p.get("confidence"),
                    "start_date": p.get("start_date"),
                    "end_date": p.get("end_date"),
                    "description": p.get("description"),
                }
                for p in patterns[:12]
            ],
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
        # SELF-TUNE FEEDBACK (2026-07): if a champion parameter set was
        # adopted for this strategy+symbol by the self-tuning loop, apply
        # it - otherwise the loop learns and nothing listens.
        try:
            from trading.services.self_tune import get_adopted_params

            _adopted = get_adopted_params(strat_name, sym) if strat_name else None
        except Exception:
            _adopted = None

        if not strat_name:
            return {"success": False, "error": "No strategies in registry"}
        strat_res = reg.execute_strategy(strat_name, raw, parameters=_adopted)

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


# ---------------------------------------------------------------------------
# Connection pass (2026-07): the guided analyst gets EVERYTHING.
# ---------------------------------------------------------------------------

def optimize_strategy_params(
    strategy: str,
    symbol: str = "SPY",
    method: str = "pso",
    metric: str = "sharpe_ratio",
    max_evaluations: int = 60,
    period: str = "2y",
) -> Dict[str, Any]:
    """Optimize a strategy's parameters on real history with out-of-sample
    validation (train on first 75%, report held-out metrics)."""
    try:
        import yfinance as yf

        import trading.strategies  # noqa: F401
        from trading.optimization.strategy_backtest_objective import (
            optimize_strategy_validated,
        )

        hist_period = (period or "2y").strip().lower()
        if hist_period not in {"6mo", "1y", "2y", "5y"}:
            hist_period = "2y"
        raw = yf.Ticker(symbol).history(period=hist_period, interval="1d")
        if raw is None or raw.empty:
            return {"success": False, "error": f"no price data for {symbol}"}
        if getattr(raw.index, "tz", None) is not None:
            raw = raw.copy()
            raw.index = raw.index.tz_convert(None)
        run = optimize_strategy_validated(
            strategy, raw, train_fraction=0.75, method=method, metric=metric,
            max_evaluations=max_evaluations,
        )
        best_params = getattr(run, "best_params", {}) or {}
        oos = getattr(run, "oos_best_metrics", {}) or {}
        saved = False
        warning = None
        oos_active = int((oos or {}).get("active_bars") or 0)
        oos_buys = int((oos or {}).get("buy_events") or 0)
        try:
            from trading.services.self_tune import (
                adopt_params,
                get_adopted_entry,
            )

            metric_key = getattr(run, "metric", metric)
            challenger_val = (oos or {}).get(metric_key)
            champion_entry = get_adopted_entry(strategy, symbol)
            champion_val = (
                (champion_entry or {}).get("oos_metrics", {}).get(metric_key)
                if champion_entry else None
            )

            if not best_params or oos_active <= 0 or oos_buys <= 0:
                # DISCIPLINE FIX (found reviewing this diff): a dead OOS
                # window used to CLEAR whatever champion was already
                # adopted - a failed run destroying previously-good,
                # separately-earned state. Now it changes nothing.
                warning = (
                    "Out-of-sample window had no long trades with these "
                    "params - not saved, existing parameters (if any) "
                    "kept. Backtest will use current defaults/champion."
                )
            elif champion_val is None or challenger_val is None or (
                challenger_val > champion_val + 0.05
            ):
                # No prior champion (bootstrap), no comparable metric, or
                # a genuine out-of-sample win - proceed.
                adopt_params(
                    strategy, symbol, best_params,
                    oos_metrics=oos if isinstance(oos, dict) else {},
                    source="backtest_optimize",
                )
                saved = True
            else:
                # DISCIPLINE FIX: this used to adopt UNCONDITIONALLY once
                # the OOS window merely had trades - a strictly worse
                # challenger (verified: OOS Sharpe -1.2) silently
                # overwrote a good champion (OOS Sharpe 2.5) with zero
                # comparison. That defeated the entire point of
                # champion/challenger discipline documented in
                # self_tune.py's own module docstring. Now: same
                # OOS-margin gate as the scheduled tuner.
                warning = (
                    f"This run's out-of-sample {metric_key} "
                    f"({challenger_val:.3f}) didn't beat the current "
                    f"champion ({champion_val:.3f}) by enough - keeping "
                    f"the existing parameters."
                )
        except Exception as e:
            logger.warning("strategy param adopt failed: %s", e)
        return {
            "success": True,
            "strategy": strategy,
            "symbol": symbol,
            "period": hist_period,
            "train_range": getattr(run, "train_range", None),
            "test_range": getattr(run, "test_range", None),
            "best_params": best_params,
            "train_metrics": getattr(run, "best_metrics", {}),
            "oos_metrics": oos,
            "deflated_sharpe": getattr(run, "deflated_sharpe", None),
            "saved": saved,
            "warning": warning,
        }
    except Exception as e:  # noqa: BLE001
        logger.exception("optimize_strategy_params failed: %s", e)
        return {"success": False, "error": str(e)}


def retune_strategies(symbols: str = "SPY",
                      method: str = "pso") -> Dict[str, Any]:
    """Run one champion/challenger self-tuning cycle: re-optimize each
    strategy on recent history and ADOPT new parameters only if they beat
    the current ones on held-out data. Fully journaled."""
    try:
        from trading.services.self_tune import tune_all

        syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        out = tune_all(symbols=syms or ["SPY"], method=method,
                       max_evaluations=40)
        return {
            "success": True,
            "cycles": out["cycles"],
            "adoptions": out["adoptions"],
            "details": [
                {k: r.get(k) for k in ("strategy", "symbol", "adopted",
                                       "champion_oos", "challenger_oos")}
                for r in out["results"]
            ],
        }
    except Exception as e:  # noqa: BLE001
        logger.exception("retune_strategies failed: %s", e)
        return {"success": False, "error": str(e)}


def get_portfolio_allocation(symbols: str,
                             period: str = "1y") -> Dict[str, Any]:
    """Risk-parity allocation across a comma-separated symbol list: each
    holding contributes EQUAL risk (verified engine). Answers 'how should
    I split my money between these?'"""
    try:
        import yfinance as yf

        from trading.optimization.portfolio_optimizer import PortfolioOptimizer

        syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        if len(syms) < 2:
            return {"success": False,
                    "error": "need at least two symbols to allocate"}
        closes = {}
        for s in syms:
            h = yf.Ticker(s).history(period=period, interval="1d")
            if h is not None and not h.empty:
                closes[s] = h["Close"]
        if len(closes) < 2:
            return {"success": False, "error": "not enough price data"}
        import pandas as pd

        rets = pd.DataFrame(closes).pct_change().dropna()
        r = PortfolioOptimizer().risk_parity_optimization(rets)
        if "error" in r:
            return {"success": False, "error": str(r["error"])}
        return {
            "success": True,
            "weights": r.get("weights"),
            "risk_contributions": r.get("risk_contributions"),
            "note": "equal-risk weights: steadier names get more dollars, "
                    "jumpier names get fewer, so no single holding "
                    "dominates your swings",
        }
    except Exception as e:  # noqa: BLE001
        logger.exception("get_portfolio_allocation failed: %s", e)
        return {"success": False, "error": str(e)}


def get_leaderboard(top_n: int = 10) -> Dict[str, Any]:
    """What's been performing best lately: the platform's recorded
    model/strategy/agent performance history, best first."""
    try:
        from trading.agents.agent_leaderboard import AgentLeaderboard

        lb = AgentLeaderboard()
        res = lb.get_leaderboard(top_n=top_n) if hasattr(lb, "get_leaderboard") else None
        rows = (res or {}).get("result") if isinstance(res, dict) else res
        return {"success": True, "leaderboard": rows or []}
    except Exception as e:  # noqa: BLE001
        logger.exception("get_leaderboard failed: %s", e)
        return {"success": False, "error": str(e)}


def get_watchlist() -> Dict[str, Any]:
    """The current user's watchlist (per-user; identity is ambient)."""
    try:
        import os

        from trading.data.watchlist import WatchlistManager

        uid = os.getenv("EVOLVE_SESSION_ID", "local")
        rows = WatchlistManager(user_id=uid).get_all()
        return {"success": True,
                "symbols": [r.get("symbol") for r in rows], "rows": rows}
    except Exception as e:  # noqa: BLE001
        logger.exception("get_watchlist failed: %s", e)
        return {"success": False, "error": str(e)}


def get_position_size(
    win_rate: float,
    avg_win_loss_ratio: float = 1.5,
    account_size: float = 10_000.0,
) -> Dict[str, Any]:
    """Kelly-criterion position sizing from a strategy's stats (the 'Kelly
    tool' the position-sizing skill references). win_rate in [0,1] (or
    0-100), avg_win_loss_ratio = average win / average loss. Returns full
    Kelly, HALF Kelly (the practitioner reference - edge estimates are
    noisy), and dollar amounts."""
    try:
        p = float(win_rate)
        if p > 1.0:  # tolerate percentages
            p = p / 100.0
        b = float(avg_win_loss_ratio)
        if not (0.0 < p < 1.0) or b <= 0:
            return {"success": False,
                    "error": "need 0<win_rate<1 and ratio>0"}
        kelly = p - (1.0 - p) / b  # f* = p - q/b
        kelly = max(0.0, kelly)
        half = kelly / 2.0
        return {
            "success": True,
            "full_kelly_fraction": round(kelly, 4),
            "half_kelly_fraction": round(half, 4),
            "half_kelly_dollars": round(half * float(account_size), 2),
            "note": "half Kelly is the reference point, full Kelly the "
                    "ceiling; zero means this edge doesn't justify a "
                    "position at all",
        }
    except Exception as e:  # noqa: BLE001
        logger.exception("get_position_size failed: %s", e)
        return {"success": False, "error": str(e)}


def get_portfolio() -> Dict[str, Any]:
    """The user's paper portfolio: positions with live unrealized P&L,
    realized P&L, cost basis, and total value. Use for 'how am I doing?'
    and 'why is my portfolio moving?' (combine with get_news on the
    holdings)."""
    try:
        from trading.portfolio.paper_portfolio import PaperPortfolio

        return PaperPortfolio().get_summary()
    except Exception as e:  # noqa: BLE001
        logger.exception("get_portfolio failed: %s", e)
        return {"success": False, "error": str(e)}


def record_paper_trade(symbol: str, side: str, quantity: float,
                       price: Optional[float] = None) -> Dict[str, Any]:
    """Record a PAPER trade (no real money, ever). side: buy|sell. If
    price is omitted, uses the live market price. Average-cost
    accounting; overselling is rejected (long-only v1)."""
    try:
        from trading.portfolio.paper_portfolio import PaperPortfolio

        if price is None:
            import yfinance as yf

            from trading.data.ticker_resolver import normalize_ticker

            sym = normalize_ticker(symbol)
            p = yf.Ticker(sym).fast_info.last_price
            if not p:
                return {"success": False,
                        "error": "no live price - pass an explicit price"}
            price = float(p)
        return PaperPortfolio().record_trade(symbol, side, quantity, price)
    except Exception as e:  # noqa: BLE001
        logger.exception("record_paper_trade failed: %s", e)
        return {"success": False, "error": str(e)}


def track_recommendation(symbol: str, score: Optional[float] = None,
                         note: str = "", source: str = "chat") -> Dict[str, Any]:
    """Save an idea to the user's tracked list WITHOUT buying - so they can
    later see how ideas they liked actually performed. Captures the
    current price for honest performance-since measurement."""
    try:
        from trading.portfolio.paper_portfolio import PaperPortfolio

        price = None
        try:
            import yfinance as yf

            p = yf.Ticker((symbol or "").strip().upper()).fast_info.last_price
            price = float(p) if p else None
        except Exception:
            price = None
        return PaperPortfolio().track_recommendation(
            symbol, source=source, score=score, price_at_rec=price, note=note,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("track_recommendation failed: %s", e)
        return {"success": False, "error": str(e)}


def get_recommendations() -> Dict[str, Any]:
    """The user's tracked ideas with performance since each was tracked."""
    try:
        from trading.portfolio.paper_portfolio import PaperPortfolio

        return {"success": True,
                "recommendations": PaperPortfolio().get_recommendations()}
    except Exception as e:  # noqa: BLE001
        logger.exception("get_recommendations failed: %s", e)
        return {"success": False, "error": str(e)}


def get_sec_filings(symbol: str) -> Dict[str, Any]:
    """Latest SEC filings (annual/quarterly/material-event) for a symbol
    with plain-language labels, plus the filing-tone signal that already
    feeds the AI Score."""
    try:
        from trading.data.sec_edgar import get_latest_filing, get_sec_signal

        sym = (symbol or "").strip().upper()
        if not sym:
            return {"success": False, "error": "symbol required"}
        labels = {
            "10-K": "Annual report",
            "10-Q": "Quarterly report",
            "8-K": "Material event",
        }
        filings = []
        for form, label in labels.items():
            try:
                f = get_latest_filing(sym, form_type=form)
            except Exception:
                f = None
            if f:
                filings.append({"form": form, "label": label,
                                "date": f.get("date"), "url": f.get("url")})
        signal = None
        try:
            signal = get_sec_signal(sym)
        except Exception:
            signal = None
        return {"success": True, "symbol": sym, "filings": filings,
                "signal": signal}
    except Exception as e:  # noqa: BLE001
        logger.exception("get_sec_filings failed: %s", e)
        return {"success": False, "error": str(e)}
