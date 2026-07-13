# -*- coding: utf-8 -*-
"""React page-parity routes — thin wrappers over existing trading/* engines."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BriefingRequest(BaseModel):
    universe: str = "sp100"
    min_ai_score: float = 6.0
    max_positions: int = 3


class PairsRequest(BaseModel):
    symbols: List[str] = Field(default_factory=list)
    universe: str = ""
    max_symbols: int = 200
    max_pairs: int = 15


class MonteCarloRequest(BaseModel):
    symbol: str = "SPY"
    n_simulations: int = 400
    horizon_days: int = 63
    initial_capital: float = 10000.0


class OptimizeRequest(BaseModel):
    strategy: str = "RSIStrategy"
    symbol: str = "SPY"
    max_evaluations: int = 24
    metric: str = "sharpe_ratio"
    period: str = "2y"


class TuneModelsRequest(BaseModel):
    symbol: str = "SPY"
    n_trials: int = 12
    models: List[str] = Field(
        default_factory=lambda: ["xgboost", "ridge", "catboost", "prophet", "garch"]
    )


class AlertUpsertRequest(BaseModel):
    symbol: str
    condition: str = "price_above"
    threshold: float = 0.0
    id: Optional[str] = None


class LimitOrderRequest(BaseModel):
    symbol: str
    side: str = "buy"
    quantity: float = 1.0
    limit_price: float
    id: Optional[str] = None


class CashAdjustRequest(BaseModel):
    amount: float
    note: str = ""


class PrefsRequest(BaseModel):
    scoring_style: Optional[str] = None
    briefing_universe: Optional[str] = None
    min_ai_score: Optional[float] = None
    opportunity_direction: Optional[str] = None
    chart_timezone: Optional[str] = None


class AllocateRequest(BaseModel):
    symbols: List[str] = Field(default_factory=list)
    period: str = "1y"


class TrackRecRequest(BaseModel):
    symbol: str
    source: str = "analyze"
    score: Optional[float] = None
    price_at_rec: Optional[float] = None
    note: str = ""


class NewsContextRequest(BaseModel):
    titles: List[str] = Field(default_factory=list)


class GnnRequest(BaseModel):
    symbols: List[str] = Field(default_factory=list)
    period: str = "1y"


def _json_safe(obj: Any) -> Any:
    """Convert numpy / nested objects into plain JSON types."""
    try:
        import math

        import numpy as np
    except Exception:
        np = None  # type: ignore
        math = None  # type: ignore

    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        if math is not None and not math.isfinite(obj):
            return None
        return obj
    if np is not None:
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if (math is not None and not math.isfinite(v)) else v
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return [_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(x) for x in obj]
    if hasattr(obj, "item"):
        try:
            return _json_safe(obj.item())
        except Exception:
            return str(obj)
    return str(obj)


def build_router(current_user: Callable[..., str]) -> APIRouter:
    """Build parity routes with the shared JWT dependency from main.py."""
    router = APIRouter(tags=["parity"])

    @router.get("/api/pulse")
    def market_pulse(user: str = Depends(current_user)) -> Dict[str, Any]:
        """Broad tape: indices, vol, rates proxy, commodities, mega-caps."""
        import yfinance as yf

        out: Dict[str, Any] = {"success": True, "items": []}
        mapping = [
            ("^GSPC", "S&P 500"),
            ("^DJI", "Dow"),
            ("^IXIC", "Nasdaq"),
            ("^RUT", "Russell 2K"),
            ("^VIX", "VIX"),
            ("GC=F", "Gold"),
            ("CL=F", "Crude"),
            ("BTC-USD", "Bitcoin"),
            ("^TNX", "US 10Y"),
            ("DX-Y.NYB", "US Dollar"),
            ("AAPL", "AAPL"),
            ("MSFT", "MSFT"),
            ("NVDA", "NVDA"),
            ("AMZN", "AMZN"),
            ("META", "META"),
            ("GOOGL", "GOOGL"),
            ("TSLA", "TSLA"),
            ("SPY", "SPY"),
            ("QQQ", "QQQ"),
            ("IWM", "IWM"),
        ]
        try:
            # Batch download last 2 closes for speed
            tickers = " ".join(s for s, _ in mapping)
            try:
                hist = yf.download(
                    tickers, period="5d", interval="1d",
                    group_by="ticker", auto_adjust=True, progress=False, threads=True,
                )
            except Exception:
                hist = None
            for sym, label in mapping:
                price = None
                chg = None
                try:
                    if hist is not None and not hist.empty:
                        if len(mapping) == 1 or (hasattr(hist.columns, "nlevels") and hist.columns.nlevels >= 2):
                            if sym in getattr(hist.columns, "levels", [[]])[0] or sym in hist.columns:
                                sub = hist[sym] if hasattr(hist.columns, "nlevels") and hist.columns.nlevels >= 2 else hist
                                close_col = "Close" if "Close" in sub.columns else (
                                    "close" if "close" in sub.columns else None
                                )
                                if close_col:
                                    series = sub[close_col].dropna()
                                    if len(series) >= 1:
                                        price = float(series.iloc[-1])
                                    if len(series) >= 2 and series.iloc[-2]:
                                        chg = float((series.iloc[-1] / series.iloc[-2] - 1) * 100)
                        else:
                            series = hist["Close"].dropna() if "Close" in hist.columns else hist.iloc[:, 0].dropna()
                            if len(series) >= 1:
                                price = float(series.iloc[-1])
                            if len(series) >= 2:
                                chg = float((series.iloc[-1] / series.iloc[-2] - 1) * 100)
                    if price is None:
                        info = yf.Ticker(sym).fast_info
                        price = getattr(info, "last_price", None)
                        prev = getattr(info, "previous_close", None)
                        if price is not None and prev:
                            chg = float((price - prev) / prev * 100)
                            price = float(price)
                except Exception as e:
                    logger.debug("pulse %s: %s", sym, e)
                out["items"].append({
                    "symbol": sym,
                    "label": label,
                    "price": price,
                    "change_pct": chg,
                })
        except Exception as e:
            return {"success": False, "error": str(e), "items": []}
        return out

    @router.get("/api/pulse/{symbol}")
    def pulse_one(symbol: str, user: str = Depends(current_user)) -> Dict[str, Any]:
        """Refresh a single tape symbol (used when an item rolls off-screen)."""
        import yfinance as yf

        sym = (symbol or "").strip()
        try:
            info = yf.Ticker(sym).fast_info
            price = getattr(info, "last_price", None)
            prev = getattr(info, "previous_close", None)
            chg = (
                ((price - prev) / prev * 100)
                if price is not None and prev else None
            )
            return {
                "success": True,
                "symbol": sym,
                "price": float(price) if price is not None else None,
                "change_pct": float(chg) if chg is not None else None,
            }
        except Exception as e:
            return {"success": False, "symbol": sym, "error": str(e),
                    "price": None, "change_pct": None}

    @router.get("/api/news/breaking")
    def news_breaking(
        max_items: int = 12,
        user: str = Depends(current_user),
    ) -> Dict[str, Any]:
        """Fast market-wide breaking headlines (Twitter/X when keyed, else RSS)."""
        try:
            from trading.data.twitter_headlines import get_breaking_headlines

            items = get_breaking_headlines(max_items=max_items) or []
            return {
                "success": True,
                "items": items,
                "source": "twitter" if items and items[0].get("source_type") == "twitter"
                else ("twitter_rss" if items else "none"),
            }
        except Exception as e:
            logger.warning("breaking news failed: %s", e)
            return {"success": False, "items": [], "error": str(e)}

    @router.get("/api/news/{symbol}")
    def news(symbol: str, max_items: int = 8,
             user: str = Depends(current_user)) -> Dict[str, Any]:
        from trading.services import agent_tools
        return agent_tools.get_news(symbol, max_items=max_items)

    @router.post("/api/news/context")
    def news_context(
        req: NewsContextRequest,
        user: str = Depends(current_user),
    ) -> Dict[str, Any]:
        """Hedged one-line context for headlines — never a trade call.

        Soft-fails to empty whys when no LLM key is configured.
        """
        try:
            from trading.services.news_context import explain_headlines

            titles = [str(t).strip() for t in (req.titles or []) if str(t).strip()]
            items = explain_headlines(titles[:8], user_id=f"user:{user}")
            return {"success": True, "items": items}
        except Exception as e:
            logger.warning("news context failed: %s", e)
            return {"success": False, "items": [], "error": str(e)}

    @router.get("/api/forecast/{symbol}")
    def forecast(
        symbol: str,
        horizon: int = 7,
        full: bool = False,
        user: str = Depends(current_user),
    ) -> Dict[str, Any]:
        """Fast consensus by default. Pass full=1 for the full stack.

        Live path applies registry eligibility (Phase 2). Feature-routing
        rules stay off — Phase 3 found no broad OOS winner.
        """
        try:
            from trading.data.price_cache import get_history
            from trading.models.forecast_router import get_router_singleton
            from trading.models.routing_validation import live_forecast_policy

            sym = (symbol or "").strip().upper()
            if not sym:
                return {"success": False, "error": "symbol required"}
            hist = get_history(sym, period="2y")
            if hist is None or getattr(hist, "empty", True):
                return {"success": False, "error": f"No data for {sym}"}
            router = get_router_singleton()
            # Skip flat-prone backends + ensemble (re-fits ARIMA) + TCN (~60s).
            requested = None if full else [
                "arima", "xgboost", "ridge", "catboost", "prophet", "garch",
            ]
            policy = live_forecast_policy(
                hist,
                requested_models=requested,
                n_assets=1,
                apply_feature_rules=False,
            )
            models = None if full else policy["models"]
            fc = router.get_consensus_forecast(
                data=hist,
                horizon=int(horizon),
                symbol=sym,
                models=models,
                model_configs={"arima": {"fast_mode": True}},
            )
            if isinstance(fc, dict):
                fc = {
                    **fc,
                    "routing": {
                        "models": policy["models"],
                        "excluded_ineligible": policy["excluded_ineligible"],
                        "feature_routing_applied": policy["feature_routing_applied"],
                        "active_rule": policy["active_rule"],
                        "always_on_rules": policy["always_on_rules"],
                        "justification": policy["justification"],
                        "features": {
                            k: policy["features"].get(k)
                            for k in (
                                "trend_strength",
                                "seasonality_strength",
                                "noise_entropy",
                                "volatility_regime",
                                "data_length",
                            )
                        },
                    },
                }
            if fc.get("error"):
                return {"success": False, "error": fc["error"], "forecast": fc}
            return {
                "success": True,
                "symbol": sym,
                "forecast": fc,
                "routing": fc.get("routing"),
                "mode": "full" if full else "fast",
            }
        except Exception as e:
            logger.warning("forecast failed: %s", e)
            return {"success": False, "error": str(e)}

    @router.get("/api/risk/{symbol}")
    def risk(symbol: str, period: str = "1y",
             user: str = Depends(current_user)) -> Dict[str, Any]:
        from trading.services import agent_tools
        return agent_tools.get_risk_metrics(symbol, period=period)

    @router.post("/api/briefing")
    def briefing(req: BriefingRequest,
                 user: str = Depends(current_user)) -> Dict[str, Any]:
        try:
            from agents.briefing.morning_briefing import MorningBriefing

            eng = MorningBriefing(
                universe=req.universe or "sp100",
                min_ai_score=req.min_ai_score,
                max_positions=req.max_positions,
                prefs={"include_forecasts": False},
            )
            report = eng.generate()
            return {
                "success": not bool(report.get("error")),
                "error": report.get("error"),
                "timestamp": report.get("timestamp"),
                "market_regime": report.get("market_regime") or {},
                "top_opportunities": (report.get("top_opportunities") or [])[:5],
                "short_opportunities": (report.get("short_opportunities") or [])[:5],
                "markdown": (report.get("markdown") or "")[:4000],
                "risk_summary": report.get("risk_summary") or {},
            }
        except Exception as e:
            logger.warning("briefing failed: %s", e)
            return {"success": False, "error": str(e)}

    @router.post("/api/pairs")
    def pairs(req: PairsRequest,
              user: str = Depends(current_user)) -> Dict[str, Any]:
        try:
            import pandas as pd

            from trading.analysis.market_scanner import _get_universe
            from trading.data.price_cache import get_history
            from trading.strategies.pairs_trading_engine import PairsTradingEngine

            syms = [
                s.strip().upper()
                for s in (req.symbols or [])
                if s and str(s).strip()
            ]
            # Custom tickers win; otherwise load the selected universe in full
            universe_size = 0
            if len(syms) < 2:
                uni = (req.universe or "sp100").strip() or "sp100"
                syms = list(_get_universe(uni))
                universe_size = len(syms)
            # O(n²) cointegration — sample up to 250 names for large universes
            cap = max(2, min(int(req.max_symbols or 100), 250))
            truncated = len(syms) > cap
            if truncated:
                # Diversified sample: keep head (usually more liquid) + stride the rest
                head_n = min(80, cap // 2)
                rest = syms[head_n:]
                stride = max(1, len(rest) // max(1, cap - head_n))
                sampled = syms[:head_n] + rest[::stride]
                syms = sampled[:cap]
            if len(syms) < 2:
                return {
                    "success": False,
                    "error": "Need at least 2 symbols (pick a universe or enter tickers)",
                    "pairs": [],
                }
            price_data: Dict[str, Any] = {}
            for s in syms:
                try:
                    h = get_history(s, period="1y")
                    if h is None or h.empty:
                        continue
                    # Engine expects lowercase 'close' — normalize here
                    _cm = {str(c).lower(): c for c in h.columns}
                    _cc = _cm.get("close", h.columns[0])
                    price_data[s] = pd.DataFrame(
                        {"close": pd.to_numeric(h[_cc], errors="coerce")}
                    )
                except Exception:
                    continue
            if len(price_data) < 2:
                return {
                    "success": False,
                    "error": "Insufficient price history",
                    "pairs": [],
                }
            engine = PairsTradingEngine()
            found = engine.find_cointegrated_pairs(
                price_data, list(price_data.keys()),
            )
            rows = []
            for a, b, res in (found or [])[: max(1, req.max_pairs)]:
                rows.append({
                    "symbol1": a,
                    "symbol2": b,
                    "p_value": round(float(res.p_value), 4),
                    "hedge_ratio": round(float(res.hedge_ratio), 4),
                    "correlation": round(float(getattr(res, "correlation", 0) or 0), 4),
                    "is_cointegrated": bool(res.is_cointegrated),
                    "spread_mean": round(float(res.spread_mean), 4),
                    "spread_std": round(float(res.spread_std), 4),
                })
            note = None
            if truncated:
                note = (
                    f"Sampled {cap} of {universe_size or 'universe'} names "
                    f"(pair tests grow with n² — custom tickers focus the search)."
                )
            return {
                "success": True,
                "pairs": rows,
                "tested": len(price_data),
                "requested": len(syms),
                "universe_size": universe_size or len(syms),
                "truncated": truncated,
                "note": note,
            }
        except Exception as e:
            logger.warning("pairs failed: %s", e)
            return {"success": False, "error": str(e), "pairs": []}

    @router.get("/api/strategies")
    def strategies(user: str = Depends(current_user)) -> Dict[str, Any]:
        fallback = [
            "RSIStrategy", "MACDStrategy", "BollingerStrategy", "SMAStrategy",
        ]
        try:
            import trading.strategies  # noqa: F401
            from trading.strategies.registry import get_strategy_registry

            reg = get_strategy_registry()
            names: List[str] = []
            if hasattr(reg, "get_all_strategies"):
                names = sorted(reg.get_all_strategies().keys())
            elif hasattr(reg, "strategies"):
                names = sorted(reg.strategies.keys())
            return {"success": True, "strategies": names or fallback}
        except Exception as e:
            logger.warning("strategies list failed: %s", e)
            return {"success": True, "strategies": fallback, "error": str(e)}

    @router.get("/api/settings/prefs")
    def get_prefs(user: str = Depends(current_user)) -> Dict[str, Any]:
        from config.user_store import load_user_preferences
        prefs = load_user_preferences(f"user:{user}") or {}
        return {"success": True, "prefs": prefs}

    @router.post("/api/settings/prefs")
    def save_prefs(req: PrefsRequest,
                   user: str = Depends(current_user)) -> Dict[str, Any]:
        from config.user_store import load_user_preferences, save_user_preferences
        uid = f"user:{user}"
        prefs = dict(load_user_preferences(uid) or {})
        for k, v in req.model_dump().items():
            if v is not None:
                prefs[k] = v
        save_user_preferences(uid, prefs)
        return {"ok": True, "prefs": prefs}

    @router.get("/api/market-signals")
    def market_signals(user: str = Depends(current_user)) -> Dict[str, Any]:
        """Cached GPR + EPS revision breadth for Dashboard pulse.

        Prefers per-user prefs (Settings load). For GPR, also falls back to
        the on-disk MacroFactors cache so a prior load still shows without
        re-download. Never fabricates numbers.
        """
        from config.user_store import load_user_preferences

        prefs = load_user_preferences(f"user:{user}") or {}
        gpr = prefs.get("cached_gpr") if isinstance(prefs.get("cached_gpr"), dict) else None
        rb = (
            prefs.get("cached_revision_breadth")
            if isinstance(prefs.get("cached_revision_breadth"), dict)
            else None
        )
        if not gpr or gpr.get("current") is None:
            try:
                from trading.analysis.macro_factors import MacroFactors

                disk = MacroFactors()._get_gpr_index()
                if isinstance(disk, dict) and disk.get("current") is not None:
                    gpr = disk
            except Exception as e:
                logger.debug("market-signals GPR disk fallback: %s", e)
        return {
            "success": True,
            "gpr": gpr if (gpr and gpr.get("current") is not None) else None,
            "revision_breadth": rb if (rb and rb.get("success")) else None,
        }

    @router.post("/api/market-signals/gpr")
    def load_gpr(user: str = Depends(current_user)) -> Dict[str, Any]:
        """Download/refresh Caldara & Iacoviello GPR and save to user prefs."""
        from config.user_store import load_user_preferences, save_user_preferences

        try:
            from trading.analysis.macro_factors import MacroFactors

            gpr = MacroFactors()._get_gpr_index()
            if not isinstance(gpr, dict) or gpr.get("current") is None:
                return {
                    "success": False,
                    "error": gpr.get("error") if isinstance(gpr, dict)
                    else "GPR download failed",
                }
            uid = f"user:{user}"
            prefs = dict(load_user_preferences(uid) or {})
            prefs["cached_gpr"] = gpr
            save_user_preferences(uid, prefs)
            return {"success": True, "gpr": gpr}
        except Exception as e:
            logger.warning("load_gpr failed: %s", e)
            return {"success": False, "error": str(e)}

    @router.post("/api/market-signals/revision-breadth")
    def load_revision_breadth(
        sample_size: int = 150,
        user: str = Depends(current_user),
    ) -> Dict[str, Any]:
        """Sample S&P names for EPS revision breadth; may take minutes."""
        from config.user_store import load_user_preferences, save_user_preferences

        try:
            from trading.data.earnings_quality import get_revision_breadth

            n = max(20, min(int(sample_size or 150), 300))
            rb = get_revision_breadth(sample_size=n)
            if not isinstance(rb, dict) or not rb.get("success"):
                return {
                    "success": False,
                    "error": "Breadth compute returned no usable data",
                    "revision_breadth": rb if isinstance(rb, dict) else None,
                }
            uid = f"user:{user}"
            prefs = dict(load_user_preferences(uid) or {})
            prefs["cached_revision_breadth"] = rb
            save_user_preferences(uid, prefs)
            return {"success": True, "revision_breadth": rb}
        except Exception as e:
            logger.warning("revision-breadth failed: %s", e)
            return {"success": False, "error": str(e)}

    # ---- Progressive Labs (thin wrappers; keep UI uncluttered) ----

    @router.get("/api/chart-events/{symbol}")
    def chart_events(
        symbol: str,
        period: str = "6mo",
        user: str = Depends(current_user),
    ) -> Dict[str, Any]:
        """Significant volume/price candles with linked headlines (news overlay)."""
        try:
            import pandas as pd

            from trading.analysis.volume_news_linker import (
                build_chart_annotations,
                detect_significant_candles,
            )
            from trading.data.price_cache import get_history

            sym = (symbol or "").strip().upper()
            hist = get_history(sym, period=period if period not in ("1d", "5d", "1w") else "3mo")
            if hist is None or hist.empty:
                return {"success": False, "events": [], "error": "No history"}
            tagged = detect_significant_candles(hist)
            anns = build_chart_annotations(tagged, sym, max_annotations=12)
            events = []
            for a in anns or []:
                headlines = [
                    str(n.get("title") or "")[:100]
                    for n in (a.get("news") or [])
                    if n.get("title")
                ]
                title = headlines[0] if headlines else (
                    f"Vol {a.get('volume_ratio', 0):.1f}x · "
                    f"{float(a.get('price_change_pct') or 0) * 100:+.1f}%"
                )
                events.append({
                    "time": str(a.get("date") or "")[:10],
                    "price": a.get("price"),
                    "color": a.get("color"),
                    "text": "N",
                    "title": title,
                    "volume_ratio": a.get("volume_ratio"),
                    "price_change_pct": a.get("price_change_pct"),
                    "headlines": headlines[:3],
                })
            return {"success": True, "symbol": sym, "events": events}
        except Exception as e:
            logger.warning("chart-events failed: %s", e)
            return {"success": False, "events": [], "error": str(e)}

    @router.get("/api/causal/{symbol}")
    def causal(symbol: str, user: str = Depends(current_user)) -> Dict[str, Any]:
        try:
            from trading.analysis.econometric_diagnostics import EconometricDiagnostics
            from trading.data.price_cache import get_history

            sym = (symbol or "").strip().upper()
            hist = get_history(sym, period="2y")
            spy = get_history("SPY", period="2y")
            if hist is None or hist.empty:
                return {"success": False, "error": "No history"}
            diag = EconometricDiagnostics(sym, hist, benchmark_data=spy)
            out = diag.run_all()
            summary = out.get("summary") or {}
            if isinstance(summary, dict):
                flags = summary.get("flags") or summary.get("findings") or []
                recs = summary.get("recommendations") or []
                complexity = summary.get("overall_complexity") or summary.get("complexity")
            else:
                flags, recs, complexity = [], [], None
            lean = {
                "symbol": sym,
                "n_observations": out.get("n_observations"),
                "stationarity": out.get("stationarity"),
                "arch_effects": out.get("arch_effects"),
                "normality": out.get("normality"),
                "summary": summary,
                "flags": flags if isinstance(flags, list) else [],
                "recommendations": recs if isinstance(recs, list) else [],
                "complexity": complexity,
            }
            if hasattr(diag, "test_granger_causality"):
                try:
                    lean["granger"] = diag.test_granger_causality()
                except Exception:
                    pass
            return {"success": True, **_json_safe(lean)}
        except Exception as e:
            logger.warning("causal failed: %s", e)
            return {"success": False, "error": str(e)}

    @router.get("/api/patterns/{symbol}")
    def patterns(symbol: str, user: str = Depends(current_user)) -> Dict[str, Any]:
        from trading.services import agent_tools
        return agent_tools.get_pattern_analysis(symbol)

    @router.get("/api/playbook/{symbol}")
    def playbook(symbol: str, user: str = Depends(current_user)) -> Dict[str, Any]:
        """Recommended strategy + model for this symbol's current tape."""
        from trading.services import agent_tools

        sym = (symbol or "").strip().upper()
        strat = agent_tools.recommend_strategy(symbol=sym)
        model = agent_tools.recommend_model(horizon="short_term")
        return {
            "success": True,
            "symbol": sym,
            "strategy": strat,
            "model": model,
        }

    @router.get("/api/earnings/{symbol}")
    def earnings(symbol: str, user: str = Depends(current_user)) -> Dict[str, Any]:
        try:
            from trading.data.earnings_calendar import get_upcoming_earnings
            from trading.data.earnings_reaction import get_earnings_reactions

            sym = (symbol or "").strip().upper()
            upcoming = None
            try:
                upcoming = get_upcoming_earnings(sym)
            except Exception as e:
                logger.debug("upcoming earnings: %s", e)
            reactions = get_earnings_reactions(sym) or {}
            return _json_safe({
                "success": not bool(reactions.get("error") and not reactions.get("reactions")),
                "symbol": sym,
                "next_earnings": upcoming,
                "avg_move_1d": reactions.get("avg_move_1d"),
                "beat_rate": reactions.get("beat_rate"),
                "positive_reaction_rate": reactions.get("positive_reaction_rate"),
                "typical_range": reactions.get("typical_range"),
                "error": reactions.get("error"),
            })
        except Exception as e:
            logger.warning("earnings failed: %s", e)
            return {"success": False, "error": str(e)}

    @router.post("/api/tune-models")
    def tune_models(req: TuneModelsRequest,
                    user: str = Depends(current_user)) -> Dict[str, Any]:
        """Tune consensus forecast models; saves params under models/best_params/."""
        try:
            import json
            from pathlib import Path

            import numpy as np
            import pandas as pd
            from sklearn.metrics import mean_squared_error
            from sklearn.model_selection import TimeSeriesSplit

            from trading.data.price_cache import get_history

            sym = (req.symbol or "SPY").strip().upper()
            n_trials = max(5, min(int(req.n_trials or 12), 30))
            hist = get_history(sym, period="2y")
            if hist is None or hist.empty:
                return {"success": False, "error": f"No history for {sym}"}
            cm = {str(c).lower(): c for c in hist.columns}
            close = hist[cm.get("close", hist.columns[0])].astype(float)
            df = pd.DataFrame({"close": close})
            df["ret_1"] = df["close"].pct_change()
            df["ret_5"] = df["close"].pct_change(5)
            df["ret_10"] = df["close"].pct_change(10)
            df["vol_10"] = df["ret_1"].rolling(10).std()
            df["sma_10"] = df["close"].rolling(10).mean() / df["close"] - 1
            df["sma_20"] = df["close"].rolling(20).mean() / df["close"] - 1
            df["target"] = df["ret_1"].shift(-1)
            df = df.dropna()
            y = df["target"]
            X = df.drop(columns=["target", "close"])
            out_dir = Path("models/best_params")
            out_dir.mkdir(parents=True, exist_ok=True)
            results: Dict[str, Any] = {}
            skipped: List[str] = []
            default_models = ["xgboost", "ridge", "catboost", "prophet", "garch"]
            wanted = [m.lower() for m in (req.models or default_models)]
            tscv = TimeSeriesSplit(n_splits=3)

            def _save(name: str, payload: Dict[str, Any]) -> None:
                (out_dir / f"{name}_consensus_best.json").write_text(
                    json.dumps(payload, indent=2), encoding="utf-8",
                )
                results[name] = payload

            if "xgboost" in wanted:
                from trading.optimization.optuna_optimizer import HyperparameterOptimizer

                opt = HyperparameterOptimizer(
                    backend="optuna",
                    study_name=f"evolve_tune_{sym.lower()}",
                )
                result = opt.optimize_xgboost(X, y, n_trials=n_trials)
                _save("xgboost", {
                    "symbol": sym,
                    "model": "xgboost",
                    "best_params": result.get("best_params") or {},
                    "best_score": result.get("best_score"),
                    "n_trials": n_trials,
                    "note": "Next-day return RMSE via Optuna",
                })

            if "ridge" in wanted:
                from sklearn.linear_model import Ridge

                best_alpha, best_rmse = 1.0, 1e9
                for alpha in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
                    rmses = []
                    for tr, te in tscv.split(X):
                        m = Ridge(alpha=alpha, max_iter=2000)
                        m.fit(X.iloc[tr], y.iloc[tr])
                        pred = m.predict(X.iloc[te])
                        rmses.append(mean_squared_error(y.iloc[te], pred) ** 0.5)
                    rmse = float(np.mean(rmses))
                    if rmse < best_rmse:
                        best_rmse, best_alpha = rmse, alpha
                _save("ridge", {
                    "symbol": sym,
                    "model": "ridge",
                    "best_params": {"alpha": best_alpha},
                    "best_score": best_rmse,
                    "note": "Time-series CV alpha sweep",
                })

            if "catboost" in wanted:
                try:
                    from catboost import CatBoostRegressor

                    grid = [
                        {"depth": d, "learning_rate": lr, "iterations": it}
                        for d in (4, 6, 8)
                        for lr in (0.03, 0.08, 0.15)
                        for it in (200, 400)
                    ]
                    # Cap grid by n_trials budget
                    grid = grid[: max(6, min(n_trials, len(grid)))]
                    best_cb, best_cb_rmse = grid[0], 1e9
                    for params in grid:
                        rmses = []
                        for tr, te in tscv.split(X):
                            m = CatBoostRegressor(
                                **params, loss_function="RMSE",
                                verbose=False, random_seed=42,
                            )
                            m.fit(X.iloc[tr], y.iloc[tr])
                            pred = m.predict(X.iloc[te])
                            rmses.append(mean_squared_error(y.iloc[te], pred) ** 0.5)
                        rmse = float(np.mean(rmses))
                        if rmse < best_cb_rmse:
                            best_cb_rmse, best_cb = rmse, params
                    _save("catboost", {
                        "symbol": sym,
                        "model": "catboost",
                        "best_params": best_cb,
                        "best_score": best_cb_rmse,
                        "note": "Time-series CV grid over depth/lr/iterations",
                    })
                except Exception as e:
                    skipped.append(f"catboost ({e})")

            if "prophet" in wanted:
                # Cheap prior sweep — Prophet Stan fits are slower, keep tiny
                try:
                    from prophet import Prophet

                    series = close.dropna().tail(180)
                    pdf = pd.DataFrame({
                        "ds": pd.to_datetime(series.index).tz_localize(None)
                        if getattr(series.index, "tz", None) is not None
                        else pd.to_datetime(series.index),
                        "y": series.values.astype(float),
                    })
                    split = int(len(pdf) * 0.8)
                    train, test = pdf.iloc[:split], pdf.iloc[split:]
                    best_cps, best_prophet_rmse = 0.05, 1e9
                    for cps in (0.01, 0.05, 0.1, 0.25):
                        m = Prophet(
                            changepoint_prior_scale=cps,
                            daily_seasonality=False,
                            weekly_seasonality=True,
                            yearly_seasonality=False,
                        )
                        m.fit(train)
                        fc = m.predict(test[["ds"]])
                        rmse = float(np.sqrt(mean_squared_error(test["y"], fc["yhat"])))
                        if rmse < best_prophet_rmse:
                            best_prophet_rmse, best_cps = rmse, cps
                    _save("prophet", {
                        "symbol": sym,
                        "model": "prophet",
                        "best_params": {
                            "prophet_params": {"changepoint_prior_scale": best_cps},
                            "changepoint_prior_scale": best_cps,
                        },
                        "best_score": best_prophet_rmse,
                        "note": "Changepoint prior sweep on held-out price RMSE",
                    })
                except Exception as e:
                    skipped.append(f"prophet ({e})")

            if "garch" in wanted:
                try:
                    from arch import arch_model

                    rets = close.pct_change().dropna().tail(400) * 100.0
                    best_g, best_g_aic = {"p": 1, "q": 1}, 1e18
                    for p in (1, 2):
                        for q in (1, 2):
                            try:
                                am = arch_model(rets, vol="Garch", p=p, q=q, rescale=False)
                                fit = am.fit(disp="off")
                                aic = float(fit.aic)
                                if aic < best_g_aic:
                                    best_g_aic, best_g = aic, {"p": p, "q": q}
                            except Exception:
                                continue
                    _save("garch", {
                        "symbol": sym,
                        "model": "garch",
                        "best_params": best_g,
                        "best_score": best_g_aic,
                        "note": "AIC grid over GARCH(p,q)",
                    })
                except Exception as e:
                    skipped.append(f"garch ({e})")

            if "lstm" in wanted or "arima" in wanted:
                skipped.append(
                    "lstm/arima skipped in interactive tune "
                    "(LSTM is slow; ARIMA uses auto_arima at fit time)"
                )

            note = (
                "Best params saved under models/best_params/ and used by consensus."
            )
            if skipped:
                note += " Skipped: " + "; ".join(skipped[:4])

            return _json_safe({
                "success": True,
                "symbol": sym,
                "results": results,
                "skipped": skipped,
            })
        except Exception as e:
            logger.warning("tune-models failed: %s", e)
            return {"success": False, "error": str(e)}

    @router.post("/api/gnn")
    def gnn_lab(req: GnnRequest,
                user: str = Depends(current_user)) -> Dict[str, Any]:
        """Light GNN multi-asset probe (needs ≥3 symbols)."""
        try:
            from trading.data.price_cache import get_history
            from trading.models.advanced.gnn.gnn_model import GNNForecaster

            syms = [s.strip().upper() for s in (req.symbols or []) if s and str(s).strip()]
            if len(syms) < 3:
                return {
                    "success": False,
                    "error": (
                        f"GNN requires at least 3 symbols (got {len(syms)}). "
                        "It models cross-asset relationships — pass ≥3 tickers "
                        "or use a single-asset model."
                    ),
                }
            frames = {}
            for s in syms[:8]:
                h = get_history(s, period=req.period or "1y")
                if h is None or h.empty:
                    continue
                _cm = {str(c).lower(): c for c in h.columns}
                cc = _cm.get("close", h.columns[0])
                frames[s] = h[cc].astype(float)
            if len(frames) < 3:
                return {
                    "success": False,
                    "error": (
                        f"GNN requires price history for ≥3 symbols "
                        f"(got usable history for {len(frames)})."
                    ),
                }
            import pandas as pd
            df = pd.DataFrame(frames).dropna()
            if len(df) < 60:
                return {"success": False, "error": "Not enough aligned history"}
            model = GNNForecaster(
                num_assets=len(df.columns),
                hidden_size=32,
                num_layers=2,
            )
            target = list(df.columns)[0]
            model.fit(df, epochs=5)
            pred = model.forecast(df, horizon=5, target_asset=target)
            fc = pred.get("forecast") if isinstance(pred, dict) else pred
            if hasattr(fc, "tolist"):
                fc = fc.tolist()
            return _json_safe({
                "success": True,
                "target": target,
                "assets": list(df.columns),
                "forecast": list(fc)[:5] if fc is not None else [],
                "note": "Multi-asset GNN probe — relational structure across the basket.",
            })
        except Exception as e:
            logger.warning("gnn lab failed: %s", e)
            return {"success": False, "error": str(e)}

    @router.post("/api/monte-carlo")
    def monte_carlo(req: MonteCarloRequest,
                    user: str = Depends(current_user)) -> Dict[str, Any]:
        try:
            import numpy as np
            import pandas as pd

            from trading.data.price_cache import get_history

            sym = (req.symbol or "SPY").strip().upper()
            hist = get_history(sym, period="2y")
            if hist is None or hist.empty:
                return {"success": False, "error": f"No data for {sym}"}
            _cm = {str(c).lower(): c for c in hist.columns}
            cc = _cm.get("close", hist.columns[0])
            rets = pd.to_numeric(hist[cc], errors="coerce").pct_change().dropna()
            if len(rets) < 40:
                return {"success": False, "error": "Need more return history"}
            n_sims = max(50, min(int(req.n_simulations or 400), 1500))
            horizon = max(5, min(int(req.horizon_days or 63), 252))
            capital = float(req.initial_capital or 10000)
            rng = np.random.default_rng(42)
            arr = rets.values.astype(float)
            paths = []
            for _ in range(n_sims):
                draws = rng.choice(arr, size=horizon, replace=True)
                equity = capital * np.cumprod(1.0 + draws)
                paths.append(equity)
            mat = np.asarray(paths)
            finals = mat[:, -1]
            p5, p50, p95 = np.percentile(finals, [5, 50, 95])
            mean_path = mat.mean(axis=0)
            return {
                "success": True,
                "symbol": sym,
                "n_simulations": n_sims,
                "horizon_days": horizon,
                "initial_capital": capital,
                "final_p5": round(float(p5), 2),
                "final_p50": round(float(p50), 2),
                "final_p95": round(float(p95), 2),
                "mean_path": [round(float(x), 2) for x in mean_path[:: max(1, horizon // 24)]],
                "note": "Bootstrap of historical daily returns — not a price forecast.",
            }
        except Exception as e:
            logger.warning("monte-carlo failed: %s", e)
            return {"success": False, "error": str(e)}

    @router.get("/api/options/{symbol}")
    def options(symbol: str, user: str = Depends(current_user)) -> Dict[str, Any]:
        from trading.services import agent_tools
        return agent_tools.get_options_sentiment(symbol)

    @router.get("/api/options/gex/{symbol}")
    def options_gex(
        symbol: str,
        expiry: str = "",
        user: str = Depends(current_user),
    ) -> Dict[str, Any]:
        from trading.services import agent_tools
        return agent_tools.get_gamma_exposure(
            symbol, expiry=expiry.strip() or None
        )

    @router.get("/api/options/skew/{symbol}")
    def options_skew(
        symbol: str,
        expiry: str = "",
        user: str = Depends(current_user),
    ) -> Dict[str, Any]:
        from trading.services import agent_tools
        return agent_tools.get_options_skew(
            symbol, expiry=expiry.strip() or None
        )

    @router.get("/api/options/context/{symbol}")
    def options_context(
        symbol: str,
        expiry: str = "",
        user: str = Depends(current_user),
    ) -> Dict[str, Any]:
        """Sentiment + GEX + skew in one call for the Analyze Options tab."""
        from trading.services import agent_tools

        exp = expiry.strip() or None
        sentiment = agent_tools.get_options_sentiment(symbol)
        gex = agent_tools.get_gamma_exposure(symbol, expiry=exp)
        skew = agent_tools.get_options_skew(symbol, expiry=exp)
        ok = bool(
            sentiment.get("success") or gex.get("success") or skew.get("success")
        )
        return {
            "success": ok,
            "symbol": (symbol or "").strip().upper(),
            "sentiment": sentiment,
            "gex": gex,
            "skew": skew,
            "disclosure": (
                "Options analytics use delayed/free yfinance chains, not "
                "real-time OPRA. Treat GEX and skew as directional context."
            ),
            "error": None if ok else "options context unavailable",
        }

    @router.post("/api/optimize")
    def optimize(req: OptimizeRequest,
                 user: str = Depends(current_user)) -> Dict[str, Any]:
        from trading.services import agent_tools
        return agent_tools.optimize_strategy_params(
            strategy=req.strategy,
            symbol=req.symbol,
            method="pso",
            metric=req.metric or "sharpe_ratio",
            max_evaluations=max(8, min(int(req.max_evaluations or 24), 40)),
            period=req.period or "2y",
        )

    @router.get("/api/strategy-params/{strategy}/{symbol}")
    def get_strategy_params(strategy: str, symbol: str,
                            user: str = Depends(current_user)) -> Dict[str, Any]:
        from trading.data.ticker_resolver import normalize_ticker
        from trading.optimization.strategy_param_spaces import get_default_params
        from trading.services.self_tune import get_adopted_params

        sym = normalize_ticker(symbol)
        adopted = get_adopted_params(strategy, sym)
        return {
            "success": True,
            "strategy": strategy,
            "symbol": sym,
            "adopted": adopted,
            "defaults": get_default_params(strategy) or {},
            "using_saved": bool(adopted),
        }

    @router.delete("/api/strategy-params/{strategy}/{symbol}")
    def clear_strategy_params(strategy: str, symbol: str,
                              user: str = Depends(current_user)) -> Dict[str, Any]:
        from trading.data.ticker_resolver import normalize_ticker
        from trading.services.self_tune import clear_adopted_params

        sym = normalize_ticker(symbol)
        cleared = clear_adopted_params(strategy, sym)
        return {"success": True, "cleared": cleared, "strategy": strategy, "symbol": sym}

    @router.get("/api/ic/{symbol}")
    def signal_ic(symbol: str, user: str = Depends(current_user)) -> Dict[str, Any]:
        try:
            from trading.analysis.signal_ic import SignalICAnalyzer

            sym = (symbol or "").strip().upper()
            analyzer = SignalICAnalyzer()
            result = analyzer.compute_ic_for_symbol(sym, lookback_days=252)
            if result is None:
                return {"success": False, "error": "Insufficient history for IC"}
            # dataclass or object
            if hasattr(result, "__dict__"):
                payload = {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in vars(result).items()
                    if not k.startswith("_")
                }
            else:
                payload = {"result": str(result)}
            return {"success": True, "symbol": sym, **payload}
        except Exception as e:
            logger.warning("ic failed: %s", e)
            return {"success": False, "error": str(e)}

    @router.post("/api/allocate")
    def allocate(req: AllocateRequest,
                 user: str = Depends(current_user)) -> Dict[str, Any]:
        from trading.services import agent_tools
        syms = ",".join(s.strip().upper() for s in (req.symbols or []) if s.strip())
        if not syms:
            return {"success": False, "error": "symbols required"}
        return agent_tools.get_portfolio_allocation(syms, period=req.period or "1y")

    @router.get("/api/portfolio/risk")
    def portfolio_risk(user: str = Depends(current_user)) -> Dict[str, Any]:
        """Whole-account risk: the CURRENT holdings' weighted daily-return
        history run through the verified risk engine (VaR/CVaR/vol/
        drawdown), plus stress scenarios and Kelly sizing derived from
        the user's OWN closed paper trades - not generic numbers."""
        try:
            import pandas as pd

            from trading.data.price_cache import get_history
            from trading.portfolio.paper_portfolio import PaperPortfolio

            pp = PaperPortfolio(user_id=f"user:{user}")
            positions = pp.get_positions()
            out: Dict[str, Any] = {"success": True, "positions": len(positions)}

            # Kelly from the user's own realized trades
            stats = pp.get_trade_stats()
            out["trade_stats"] = stats
            kelly = None
            if stats.get("win_rate") is not None and stats.get(
                "avg_win_loss_ratio"
            ):
                from trading.services.agent_tools import get_position_size

                kelly = get_position_size(
                    stats["win_rate"], stats["avg_win_loss_ratio"],
                    account_size=pp.get_cash() + sum(
                        p["quantity"] * p["avg_cost"] for p in positions
                    ),
                    # Market vol proxy for the book; overlay is informational
                    # until validate_conditional_vol_universe shows a broad win.
                    symbol="SPY",
                    apply_vol_overlay=True,
                )
                # Options-VIX overlay: dual display only (live flag off)
                try:
                    from trading.portfolio.options_vix_sizing import (
                        LIVE_OPTIONS_VIX_SIZING_ENABLED,
                        apply_kelly_options_vix_overlay,
                        options_vix_multiplier_live,
                    )

                    vix_info = options_vix_multiplier_live()
                    kelly = apply_kelly_options_vix_overlay(kelly, vix_info)
                    kelly["options_vix_live_wired"] = bool(
                        LIVE_OPTIONS_VIX_SIZING_ENABLED
                    )
                except Exception as e:
                    logger.debug("options vix overlay skipped: %s", e)
            out["kelly"] = kelly
            out["kelly_note"] = (
                (kelly or {}).get("note")
                if isinstance(kelly, dict)
                else (
                    "Sizing guide only — derived from closed paper trades, "
                    "not live broker results."
                )
            )

            if not positions:
                out["portfolio_metrics"] = None
                out["stress"] = None
                return out

            # Weighted portfolio return series from current holdings
            frames = {}
            weights = {}
            total_cost = sum(p["quantity"] * p["avg_cost"] for p in positions)
            for p in positions:
                h = get_history(p["symbol"], period="1y")
                if h is None or h.empty:
                    continue
                cm = {str(c).lower(): c for c in h.columns}
                closes = h[cm.get("close", h.columns[0])].astype(float)
                frames[p["symbol"]] = closes.pct_change()
                weights[p["symbol"]] = (
                    p["quantity"] * p["avg_cost"] / total_cost
                    if total_cost else 0.0
                )
            if not frames:
                out["portfolio_metrics"] = None
                out["stress"] = None
                out["note"] = "no price history available offline"
                return out
            rets = pd.DataFrame(frames).dropna()
            port = (rets * pd.Series(weights)).sum(axis=1)

            from utils.risk_metrics import compute_performance_metrics
            out["portfolio_metrics"] = compute_performance_metrics(port).to_dict()

            from trading.risk.advanced_risk import AdvancedRiskAnalyzer
            stress_daily = AdvancedRiskAnalyzer().calculate_stress_test_metrics(port)
            equity = pp.get_cash() + sum(
                (p["quantity"] * p["avg_cost"]) for p in positions
            )
            # translate sigma-shock daily returns into plain dollars
            out["stress"] = {
                k: {
                    "daily_return_pct": round(float(v) * 100, 2),
                    "dollar_impact": round(float(v) * equity, 2),
                }
                for k, v in stress_daily.items()
            }
            out["stress_note"] = (
                "What a bad day looks like for THIS mix of holdings: a 1/2/3 "
                "standard-deviation down move, in dollars on your account."
            )
            return out
        except Exception as e:
            logger.warning("portfolio risk failed: %s", e)
            return {"success": False, "error": str(e)}

    @router.get("/api/recs")
    def list_recs(user: str = Depends(current_user)) -> Dict[str, Any]:
        from trading.portfolio.paper_portfolio import PaperPortfolio

        recs = PaperPortfolio(user_id=f"user:{user}").get_recommendations()
        return {"success": True, "recommendations": recs}

    @router.post("/api/recs")
    def track_rec(req: TrackRecRequest,
                  user: str = Depends(current_user)) -> Dict[str, Any]:
        from trading.portfolio.paper_portfolio import PaperPortfolio

        pp = PaperPortfolio(user_id=f"user:{user}")
        price = req.price_at_rec
        if price is None:
            try:
                import yfinance as yf

                p = yf.Ticker(req.symbol.strip().upper()).fast_info.last_price
                price = float(p) if p else None
            except Exception:
                price = None
        return pp.track_recommendation(
            req.symbol, source=req.source, score=req.score,
            price_at_rec=price, note=req.note,
        )

    @router.delete("/api/recs/{rec_id}")
    def delete_rec(rec_id: str,
                   user: str = Depends(current_user)) -> Dict[str, Any]:
        from trading.portfolio.paper_portfolio import PaperPortfolio

        return PaperPortfolio(user_id=f"user:{user}").delete_recommendation(rec_id)

    @router.get("/api/edgar/{symbol}")
    def edgar_filings(symbol: str,
                      user: str = Depends(current_user)) -> Dict[str, Any]:
        """Recent SEC filings, labeled in plain language. The EDGAR module
        already feeds the AI Score; this surfaces the documents themselves."""
        try:
            from trading.data.sec_edgar import get_latest_filing, get_sec_signal

            sym = (symbol or "").strip().upper()
            labels = {
                "10-K": "Annual report - the company's full yearly picture",
                "10-Q": "Quarterly report - the latest three months",
                "8-K": "Material event - something big enough to disclose now",
            }
            filings = []
            for form, label in labels.items():
                try:
                    f = get_latest_filing(sym, form_type=form)
                except Exception:
                    f = None
                if f:
                    filings.append({
                        "form": form,
                        "label": label,
                        "date": f.get("date"),
                        "url": f.get("url"),
                    })
            signal = None
            try:
                signal = get_sec_signal(sym)
            except Exception as e:
                logger.debug("sec signal skipped: %s", e)
            return {
                "success": True,
                "symbol": sym,
                "filings": filings,
                "signal": signal,
                "note": ("Filings straight from SEC EDGAR. The tone signal "
                         "below already feeds this symbol's AI Score."),
            }
        except Exception as e:
            logger.warning("edgar failed: %s", e)
            return {"success": False, "error": str(e)}

    @router.get("/api/cashbook")
    def cashbook(user: str = Depends(current_user)) -> Dict[str, Any]:
        # CASH INTEGRATION (2026-07): cashbook and the paper portfolio
        # used to be two disconnected ledgers - buying shares never
        # moved the displayed cash balance. Now backed by the SAME
        # per-user store as positions/trades, so cash + market value is
        # real account equity. Checking here also runs the limit-order
        # executor, mirroring how viewing alerts checks alerts.
        from trading.portfolio.paper_portfolio import PaperPortfolio

        pp = PaperPortfolio(user_id=f"user:{user}")
        newly_filled = pp.check_limit_orders()
        return {
            "success": True,
            "cash": pp.get_cash(),
            "limit_orders": pp.get_limit_orders(),
            "newly_filled": newly_filled,
        }

    @router.post("/api/cashbook/adjust")
    def cash_adjust(req: CashAdjustRequest,
                    user: str = Depends(current_user)) -> Dict[str, Any]:
        from trading.portfolio.paper_portfolio import PaperPortfolio

        return PaperPortfolio(user_id=f"user:{user}").adjust_cash(
            req.amount, note=req.note
        )

    @router.post("/api/cashbook/limit")
    def place_limit(req: LimitOrderRequest,
                    user: str = Depends(current_user)) -> Dict[str, Any]:
        from trading.portfolio.paper_portfolio import PaperPortfolio

        pp = PaperPortfolio(user_id=f"user:{user}")
        r = pp.place_limit_order(req.symbol, req.side, req.quantity,
                                 req.limit_price, order_id=req.id)
        if r.get("success"):
            r["limit_orders"] = pp.get_limit_orders()
        return r

    @router.delete("/api/cashbook/limit/{order_id}")
    def cancel_limit(order_id: str, user: str = Depends(current_user)) -> Dict[str, Any]:
        from trading.portfolio.paper_portfolio import PaperPortfolio

        pp = PaperPortfolio(user_id=f"user:{user}")
        r = pp.cancel_limit_order(order_id)
        r["limit_orders"] = pp.get_limit_orders()
        return r

    @router.get("/api/alerts")
    def list_alerts(user: str = Depends(current_user)) -> Dict[str, Any]:
        from config.user_store import load_user_preferences
        from trading.services.alert_checker import check_alerts_for_user

        uid = f"user:{user}"
        prefs = load_user_preferences(uid) or {}
        alerts = prefs.get("evolve_alerts") or []
        triggered = []
        try:
            triggered = check_alerts_for_user(uid) or []
        except Exception as e:
            logger.debug("alert check: %s", e)
        return {"success": True, "alerts": alerts, "triggered": triggered}

    @router.post("/api/alerts")
    def upsert_alert(req: AlertUpsertRequest,
                     user: str = Depends(current_user)) -> Dict[str, Any]:
        import uuid

        from config.user_store import load_user_preferences, save_user_preferences

        uid = f"user:{user}"
        prefs = dict(load_user_preferences(uid) or {})
        alerts = list(prefs.get("evolve_alerts") or [])
        aid = req.id or str(uuid.uuid4())[:8]
        row = {
            "id": aid,
            "symbol": req.symbol.strip().upper(),
            "condition": req.condition,
            "threshold": float(req.threshold),
        }
        alerts = [a for a in alerts if isinstance(a, dict) and a.get("id") != aid]
        alerts.append(row)
        prefs["evolve_alerts"] = alerts[-50:]
        save_user_preferences(uid, prefs)
        return {"success": True, "alerts": prefs["evolve_alerts"]}

    @router.delete("/api/alerts/{alert_id}")
    def delete_alert(alert_id: str, user: str = Depends(current_user)) -> Dict[str, Any]:
        from config.user_store import load_user_preferences, save_user_preferences

        uid = f"user:{user}"
        prefs = dict(load_user_preferences(uid) or {})
        alerts = [
            a for a in (prefs.get("evolve_alerts") or [])
            if isinstance(a, dict) and a.get("id") != alert_id
        ]
        prefs["evolve_alerts"] = alerts
        save_user_preferences(uid, prefs)
        return {"success": True, "alerts": alerts}

    return router
