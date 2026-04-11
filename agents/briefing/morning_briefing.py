"""
Morning Briefing Pipeline
==========================
INTEGRATION NOTES:
- Drop into: agents/briefing/morning_briefing.py
- Wire into: pages/6_Chat.py autonomous mode toggle
- Call pattern:
    from agents.briefing.morning_briefing import MorningBriefing
    briefing = MorningBriefing()
    report = briefing.generate()  # returns formatted markdown
    briefing.render_streamlit()   # renders full UI

Dependencies: existing platform modules only
"""

import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class MorningBriefing:
    """
    Autonomous morning briefing pipeline.

    Steps:
    1. Scan universe (SP100 by default) for top AI scores
    2. Run consensus forecast on top candidates
    3. Check short interest and earnings calendar
    4. Generate structured report with entry/target/stop
    5. Summarize market regime
    """

    # Cap universe for briefing speed; Scanner page uses full list.
    BRIEFING_UNIVERSE_CAP = 50

    # Fast consensus subset (Analyze page uses full 10-model stack).
    BRIEFING_MODELS = [
        "arima",
        "xgboost",
        "ridge",
        "catboost",
        "prophet",
    ]

    def __init__(
        self,
        universe: str = "sp100",
        min_ai_score: Optional[float] = None,
        max_positions: int = 3,
    ):
        self.universe = universe
        self.min_ai_score = float(min_ai_score or 5.5)
        self.max_positions = max_positions
        self._last_report: Optional[Dict[str, Any]] = None
        self._briefing_prefs: Dict[str, Any] = {}

    def _load_briefing_prefs(self) -> Dict[str, Any]:
        try:
            from config.user_store import load_user_preferences
            from utils.session_utils import get_stable_user_id

            return load_user_preferences(get_stable_user_id()) or {}
        except Exception:
            return {}

    def _opportunity_passes_direction_pref(self, opp: Dict[str, Any]) -> bool:
        prefs = getattr(self, "_briefing_prefs", {}) or {}
        od = prefs.get(
            "opportunity_direction",
            "Bullish only (BUY signals)",
        )
        fc = opp.get("forecast") or {}
        entry = float(opp.get("entry") or 0)
        target = float(opp.get("target") or fc.get("consensus_price") or 0)
        # No forecast / entry / target — use Quick Score as direction proxy
        if not fc and entry <= 0 and target <= 0:
            _qs = float(
                opp.get("quick_score")
                or opp.get("ai_score", 5.0)
                or 5.0,
            )
            if "Both directions" in od:
                return True
            if "Bearish only" in od:
                _sqs = float(opp.get("short_quick_score") or 0)
                return _sqs >= 5.5
            return _qs >= 5.5
        MIN_MOVE_PCT = 0.005
        if target > 0 and entry > 0:
            move_pct = abs(target - entry) / entry
            if move_pct < MIN_MOVE_PCT:
                return False
        exp_pct = float(fc.get("expected_move_pct", 0) or 0) / 100.0
        if entry > 0 and target > 0:
            forecast_return = (target - entry) / entry
        else:
            forecast_return = exp_pct

        sym = str(opp.get("symbol") or "?")

        if "Both directions" in od:
            return True
        if "Bearish only" in od:
            is_bear = forecast_return < -0.005 or (
                target > 0 and entry > 0 and target < entry * 0.998
            )
            if not is_bear:
                logger.info(
                    "Skipping %s: not bearish enough for bearish-only pref "
                    "(return=%.2f%%)",
                    sym,
                    forecast_return * 100.0,
                )
            return is_bear

        is_bull = forecast_return > 0.005 or (
            target > 0 and entry > 0 and target > entry * 1.002
        )
        if not is_bull:
            logger.info(
                "Skipping %s: bearish forecast (return=%.2f%%)",
                sym,
                forecast_return * 100.0,
            )
        return is_bull

    def generate(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        forecast_progress_callback: Optional[Callable[..., None]] = None,
    ) -> Dict[str, Any]:
        """
        Generate full morning briefing.
        Returns dict with markdown report and structured data.

        progress_callback: optional callable(done, total) during universe scan.
        forecast_progress_callback: optional callable(done, total, symbol)
            during per-ticker forecast phase (when include_forecasts is on).
        """
        self._briefing_prefs = self._load_briefing_prefs()
        _p = self._briefing_prefs
        if _p.get("min_ai_score") is not None:
            self.min_ai_score = float(_p["min_ai_score"])

        logger.info("Morning briefing: starting generation")
        report = {
            "timestamp": datetime.now().isoformat(),
            "market_regime": {},
            "top_opportunities": [],
            "short_opportunities": [],
            "watchlist_alerts": [],
            "risk_summary": {},
            "markdown": "",
            "error": None,
        }

        try:
            # Step 1: Market regime
            report["market_regime"] = self._get_market_regime()

            # Step 2: Scan universe
            candidates = self._scan_universe(
                progress_callback=progress_callback
            )
            pref_sectors = self._briefing_prefs.get("preferred_sectors") or []
            if pref_sectors and candidates:
                try:
                    import yfinance as yf

                    _syms = [
                        str(c.get("symbol"))
                        for c in candidates
                        if c.get("symbol")
                    ]
                    _tickers = yf.Tickers(" ".join(_syms))
                    _sector_map: Dict[str, str] = {}
                    for sym in _syms:
                        try:
                            _sector_map[sym] = (
                                _tickers.tickers[sym]
                                .info.get("sector", "")
                                or ""
                            )
                        except Exception:
                            _sector_map[sym] = ""

                    filtered_c: List[Dict[str, Any]] = []
                    for c in candidates:
                        sym = str(c.get("symbol", ""))
                        sec_l = (_sector_map.get(sym, "") or "").lower()
                        if not sec_l:
                            continue
                        if any(
                            str(s).strip().lower() in sec_l
                            for s in pref_sectors
                        ):
                            nc = dict(c)
                            nc["sector"] = _sector_map[sym]
                            filtered_c.append(nc)
                    if filtered_c:
                        candidates = filtered_c
                        logger.info(
                            "Sector filter applied: %d candidates in %s",
                            len(candidates),
                            pref_sectors,
                        )
                except Exception as _sf:
                    logger.debug("Sector filter skipped: %s", _sf)

            logger.info(
                "Morning briefing: found %d candidates above %.1f Quick Score",
                len(candidates),
                self.min_ai_score,
            )

            # Step 3: Deep analysis — try extra names until slots filled
            opportunities = []
            shared_router = None
            try:
                from trading.models.forecast_router import (
                    ForecastRouter,
                    get_router_singleton,
                )

                shared_router = get_router_singleton()
            except Exception as e:
                logger.warning("Morning briefing: ForecastRouter init failed: %s", e)

            _include_forecasts = bool(
                self._briefing_prefs.get("include_forecasts", False)
            )
            _max_try = min(len(candidates), max(self.max_positions * 6, 24))
            _pool = candidates[:_max_try]
            _n_pool = len(_pool)
            for i, candidate in enumerate(_pool):
                if len(opportunities) >= self.max_positions:
                    break
                sym = candidate.get("symbol") or "?"
                t0 = time.perf_counter()
                logger.info(
                    "Morning briefing: analyzing %s (%d/%d)",
                    sym,
                    i + 1,
                    _n_pool,
                )
                if (
                    _include_forecasts
                    and forecast_progress_callback is not None
                ):
                    try:
                        forecast_progress_callback(i, _n_pool, sym)
                    except TypeError:
                        try:
                            forecast_progress_callback(i, _n_pool)
                        except Exception:
                            pass
                    except Exception:
                        pass
                try:
                    opp = self._analyze_opportunity(
                        candidate,
                        router=shared_router,
                        briefing=True,
                        skip_forecast=not _include_forecasts,
                    )
                    if opp and self._opportunity_passes_direction_pref(opp):
                        opportunities.append(opp)
                except Exception as e:
                    logger.debug(
                        "Opportunity analysis failed for %s: %s",
                        sym,
                        e,
                    )
                elapsed = time.perf_counter() - t0
                logger.info(
                    "Morning briefing: %s complete (%.1fs)",
                    sym,
                    elapsed,
                )

            short_opportunities: List[Dict[str, Any]] = []
            _od = self._briefing_prefs.get(
                "opportunity_direction",
                "Bullish only (BUY signals)",
            )
            _include_shorts = (
                "Both directions" in _od
                or "Bearish only" in _od
            )
            if _include_shorts:
                try:
                    short_candidates = self._scan_shorts()
                    for sc in short_candidates[: self.max_positions * 2]:
                        if len(short_opportunities) >= self.max_positions:
                            break
                        _sym = sc.get("symbol")
                        if not _sym:
                            continue
                        _sq = float(
                            sc.get("short_quick_score", 5.0) or 5.0
                        )
                        short_opportunities.append(
                            {
                                "symbol": _sym,
                                "short_score": _sq,
                                "current_price": sc.get("price"),
                                "thesis": (
                                    f"Short Quick Score {_sq:.1f} — "
                                    "bearish technical setup"
                                ),
                                "direction": "SHORT",
                            }
                        )
                except Exception as _se:
                    logger.debug("Short scan failed: %s", _se)
            report["short_opportunities"] = short_opportunities

            if len(opportunities) >= 2 and _include_forecasts:
                try:
                    from trading.data.price_cache import get_history
                    from trading.optimization.portfolio_optimizer import (
                        get_portfolio_optimizer,
                    )

                    _opt = get_portfolio_optimizer()
                    _syms = [
                        str(o.get("symbol", "")).upper()
                        for o in opportunities
                    ]
                    _returns_dict: Dict[str, pd.Series] = {}
                    for sym in _syms:
                        try:
                            h = get_history(
                                sym, period="126d", interval="1d"
                            )
                            if h is not None and len(h) >= 30:
                                col = {
                                    c.lower(): c for c in h.columns
                                }.get("close", h.columns[0])
                                _returns_dict[sym] = (
                                    h[col].pct_change().dropna()
                                )
                        except Exception:
                            continue
                    if len(_returns_dict) >= 2:
                        _returns_df = pd.DataFrame(_returns_dict).dropna(
                            how="any"
                        )
                        if len(_returns_df) >= 30:
                            _result = _opt.mean_variance_optimization(
                                _returns_df, target_return=None
                            )
                            if (
                                isinstance(_result, dict)
                                and "weights" in _result
                                and "error" not in _result
                            ):
                                _weights = _result["weights"]
                                for opp in opportunities:
                                    sym = opp["symbol"]
                                    w = float(
                                        _weights.get(
                                            sym, 1.0 / len(_syms)
                                        )
                                    )
                                    opp["suggested_weight"] = round(w, 3)
                                    opp["weight_pct"] = f"{w * 100:.0f}%"
                                report["portfolio"] = {
                                    "method": "max_sharpe",
                                    "expected_sharpe": _result.get(
                                        "sharpe_ratio"
                                    ),
                                    "weights": _weights,
                                    "note": (
                                        "Mean-variance optimized weights "
                                        "(6-month history)"
                                    ),
                                }
                                logger.info(
                                    "Portfolio optimized: %s", _weights
                                )
                except Exception as e:
                    logger.debug("Portfolio opt failed: %s", e)

            report["top_opportunities"] = opportunities

            # Step 4: Watchlist alerts
            report["watchlist_alerts"] = self._get_watchlist_alerts()

            # Step 5: Generate markdown
            report["markdown"] = self._format_markdown(report)

            self._last_report = report
            logger.info("Morning briefing: complete")

        except Exception as e:
            logger.warning("Morning briefing generation failed: %s", e)
            report["error"] = str(e)
            report["markdown"] = (
                f"⚠️ Briefing generation encountered an error: {e}\n\n"
                "Please check the Scanner page for manual analysis."
            )

        return report

    def _get_market_regime(self) -> Dict[str, Any]:
        """Get current market regime from SPY/QQQ/VIX."""
        regime = {
            "spy_trend": "UNKNOWN",
            "vix_level": None,
            "regime": "NEUTRAL",
            "description": "",
        }
        try:
            import numpy as np
            from trading.data.price_cache import get_history, get_macro_history

            spy = get_history("SPY", period="3mo")
            vix_hist = get_macro_history("^VIX", period="5d")

            if not spy.empty:
                _col_map = {c.lower(): c for c in spy.columns}
                close_col = _col_map.get("close", spy.columns[0])
                closes = spy[close_col].values
                sma50 = np.mean(closes[-50:]) if len(closes) >= 50 else closes.mean()
                current = float(closes[-1])
                regime["spy_trend"] = "BULLISH" if current > sma50 else "BEARISH"
                regime["spy_50d_pct"] = round(
                    (current - sma50) / sma50 * 100, 2
                )

            if not vix_hist.empty:
                _col_map = {c.lower(): c for c in vix_hist.columns}
                close_col = _col_map.get("close", vix_hist.columns[0])
                vix_level = float(vix_hist[close_col].iloc[-1])
                regime["vix_level"] = round(vix_level, 2)

                if vix_level > 30:
                    regime["volatility"] = "HIGH"
                elif vix_level > 20:
                    regime["volatility"] = "ELEVATED"
                else:
                    regime["volatility"] = "NORMAL"

            # Determine overall regime (None = unknown VIX → neutral 20 for thresholds; keep real 0.0)
            is_bullish = regime["spy_trend"] == "BULLISH"
            vix_val = regime["vix_level"]
            if vix_val is None:
                vix_n = 20.0
            else:
                vix_n = float(vix_val)
            if is_bullish and vix_n < 20:
                regime["regime"] = "RISK_ON"
                regime["description"] = "Market is in risk-on mode — momentum strategies favored"
            elif not is_bullish and vix_n > 25:
                regime["regime"] = "RISK_OFF"
                regime["description"] = "Market is in risk-off mode — defensive positioning recommended"
            else:
                regime["regime"] = "NEUTRAL"
                regime["description"] = "Mixed signals — selective positioning recommended"

        except Exception as e:
            logger.debug("Market regime fetch failed: %s", e)
            regime["error"] = str(e)

        return regime

    def _scan_shorts(self) -> List[Dict[str, Any]]:
        """Scan for short candidates using Short Quick Score."""
        try:
            from trading.analysis.market_scanner import (
                _get_universe,
                scan_market,
            )

            u = (self.universe or "sp100").lower()
            uni = list(_get_universe(u))
            _p = getattr(self, "_briefing_prefs", {}) or {}
            pref_uni = str(_p.get("briefing_universe") or "")
            try:
                if "NASDAQ100" in pref_uni:
                    uni = list(_get_universe("nasdaq100"))
                elif "SP500" in pref_uni:
                    uni = list(_get_universe("sp500"))
                elif "SP100" in pref_uni:
                    uni = list(_get_universe("sp100"))
                elif "Top 25" in pref_uni:
                    uni = list(_get_universe("sp100"))[:25]
                if _p.get("watchlist_only"):
                    try:
                        from trading.data.watchlist import WatchlistManager

                        _wl = WatchlistManager().get_all() or []
                        _syms = [
                            str(r.get("symbol", "")).strip().upper()
                            for r in _wl
                            if r.get("symbol")
                        ]
                        if _syms:
                            uni = _syms
                    except Exception:
                        pass
            except Exception:
                pass

            _uni_cap_map = {
                "SP100": 50,
                "SP500": 150,
                "NASDAQ100": 100,
                "Top 25": 25,
            }
            _uni_cap = 50
            for _k, _v in _uni_cap_map.items():
                if _k in pref_uni:
                    _uni_cap = _v
                    break
            uni = uni[:_uni_cap]
            raw = scan_market(
                filters=["high_short_score"],
                universe=uni,
                max_results=self.max_positions * 4,
                min_quick_score=5.5,
            )
            rows = raw.get("results") or []
            rows.sort(
                key=lambda x: float(
                    x.get("short_quick_score", 0) or 0
                ),
                reverse=True,
            )
            return rows[: self.max_positions * 2]
        except Exception as e:
            logger.debug("Short scan: %s", e)
            return []

    def _scan_universe(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Run scanner on universe and return top candidates (symbol + ai_score)."""
        try:
            from trading.analysis.market_scanner import _get_universe, scan_market

            u = (self.universe or "default").lower()
            uni = list(_get_universe(u))
            _p = getattr(self, "_briefing_prefs", {}) or {}
            pref_uni = str(_p.get("briefing_universe") or "")
            try:
                if "NASDAQ100" in pref_uni:
                    uni = list(_get_universe("nasdaq100"))
                elif "SP500" in pref_uni:
                    uni = list(_get_universe("sp500"))
                elif "SP100" in pref_uni:
                    uni = list(_get_universe("sp100"))
                elif "Top 25" in pref_uni:
                    uni = list(_get_universe("sp100"))[:25]
                if _p.get("watchlist_only"):
                    try:
                        from trading.data.watchlist import WatchlistManager

                        _wl = WatchlistManager().get_all() or []
                        _syms = [
                            str(r.get("symbol", "")).strip().upper()
                            for r in _wl
                            if r.get("symbol")
                        ]
                        if _syms:
                            uni = _syms
                            logger.info(
                                "Morning briefing: using watchlist (%d tickers)",
                                len(uni),
                            )
                    except Exception as _wl_e:
                        logger.debug("Watchlist-only universe skipped: %s", _wl_e)
            except Exception:
                pass
            _legacy_top = 50
            if u in ("sp50", "large", "mega"):
                uni = uni[:_legacy_top]
            elif u in ("sp30", "core"):
                uni = uni[:30]
            _uni_cap_map = {
                "SP100": 50,
                "SP500": 150,
                "NASDAQ100": 100,
                "Top 25": 25,
            }
            _uni_cap = 50
            for _k, _v in _uni_cap_map.items():
                if _k in pref_uni:
                    _uni_cap = _v
                    break
            uni = uni[:_uni_cap]

            logger.info(
                "Morning briefing: scanning %d tickers...",
                len(uni),
            )
            # tz_localize(None) is naive-only; batch tz strip is in market_scanner.scan_market

            _cap = min(
                200,
                max(len(uni), self.max_positions * 16),
            )
            _min_q = max(
                4.5,
                float(self.min_ai_score) - 1.5,
            )
            _pre = scan_market(
                filters=["quick_technical"],
                universe=uni,
                max_results=_cap,
                min_quick_score=_min_q,
                progress_callback=progress_callback,
            )
            if _pre.get("error"):
                logger.warning("Universe scan error: %s", _pre.get("error"))
                return []
            raw = _pre
            rows = raw.get("results") or []
            candidates = []
            for r in rows:
                _qs = float(r.get("quick_score", 0) or 0)
                if _qs >= float(self.min_ai_score):
                    nc = dict(r)
                    nc["ai_score"] = _qs
                    candidates.append(nc)
            candidates.sort(
                key=lambda x: float(x.get("ai_score", 0) or 0),
                reverse=True,
            )
            return candidates[: self.max_positions * 2]

        except Exception as e:
            logger.warning("Universe scan failed: %s", e)
            return []

    def _analyze_opportunity(
        self,
        candidate: Dict[str, Any],
        router: Any = None,
        briefing: bool = True,
        skip_forecast: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Deep analysis on a single candidate.

        When ``briefing`` is True, skip Monte Carlo and strategy comparison
        (heavy paths run only when ``briefing`` is False — Analyze deep dive).
        When ``skip_forecast`` is True, skip consensus forecast (Quick Score
        briefing only — no entry/target/stop).
        """
        symbol = candidate.get("symbol")
        if not symbol:
            return None

        opp = {
            "symbol": symbol,
            "ai_score": candidate.get("ai_score"),
            "quick_score": candidate.get("quick_score"),
            "short_quick_score": candidate.get("short_quick_score"),
            "current_price": None,
            "forecast": {},
            "entry": None,
            "target": None,
            "stop": None,
            "conviction": "MEDIUM",
            "thesis": "",
            "risks": [],
            "catalysts": [],
        }

        try:
            from trading.data.price_cache import get_history

            hist = get_history(str(symbol), period="6mo")

            if hist.empty:
                return None

            _col_map = {c.lower(): c for c in hist.columns}
            close_col = _col_map.get("close", hist.columns[0])
            current_price = float(hist[close_col].iloc[-1])
            opp["current_price"] = round(current_price, 2)

            # Get consensus forecast (optional — off by default for Cloud speed)
            if not skip_forecast:
                try:
                    from trading.models.forecast_router import (
                        ForecastRouter,
                        get_router_singleton,
                    )

                    _router = router if router is not None else get_router_singleton()
                    forecast = _router.get_consensus_forecast(
                        data=hist,
                        horizon=7,
                        symbol=str(symbol),
                        models=self.BRIEFING_MODELS,
                        model_configs={"arima": {"fast_mode": True}},
                    )
                    if forecast and "error" not in forecast:
                        consensus_price = forecast.get("consensus_price")
                        if consensus_price:
                            pct_move = (
                                (consensus_price - current_price)
                                / current_price * 100
                            )
                            opp["forecast"] = {
                                "consensus_price": round(consensus_price, 2),
                                "expected_move_pct": round(pct_move, 2),
                                "direction": forecast.get("direction", "NEUTRAL"),
                                "conviction": forecast.get("conviction", "LOW"),
                                "models_used": forecast.get("models_used", []),
                            }

                            # Entry/target/stop
                            opp["entry"] = round(current_price, 2)
                            opp["target"] = round(consensus_price, 2)
                            stop_pct = 0.03  # 3% stop
                            if forecast.get("direction") == "BULLISH":
                                opp["stop"] = round(
                                    current_price * (1 - stop_pct), 2
                                )
                            else:
                                opp["stop"] = round(
                                    current_price * (1 + stop_pct), 2
                                )

                            opp["conviction"] = forecast.get(
                                "conviction", "MEDIUM"
                            )
                        for _w in forecast.get("walk_forward_warnings") or []:
                            opp.setdefault("risks", []).append(_w)
                        _wfc = forecast.get("walk_forward_confidence")
                        if _wfc:
                            opp.setdefault("catalysts", []).append(
                                f"Walk-forward confidence: {_wfc}"
                            )
                except Exception as e:
                    logger.debug("Forecast failed for %s: %s", symbol, e)

            if not briefing:
                try:
                    from trading.backtesting.monte_carlo import (
                        MonteCarloConfig,
                        MonteCarloSimulator,
                    )

                    import numpy as np

                    _cfg = MonteCarloConfig(n_simulations=100)
                    _mc = MonteCarloSimulator(_cfg)
                    _cm = {c.lower(): c for c in hist.columns}
                    _cc = _cm.get("close", hist.columns[0])
                    _rets = hist[_cc].astype(float).pct_change().dropna()
                    if len(_rets) >= 30:
                        _paths = _mc.simulate_portfolio_paths(
                            _rets, n_simulations=100
                        )
                        if _paths is not None and len(_paths) > 1:
                            _first = _paths.iloc[0].values.astype(float)
                            _last = _paths.iloc[-1].values.astype(float)
                            _term = (_last / np.maximum(_first, 1e-12)) - 1.0
                            _p10 = float(np.percentile(_term, 10))
                            _fmove = float(
                                (opp.get("forecast") or {}).get(
                                    "expected_move_pct", 0
                                )
                                or 0
                            )
                            if _p10 < -0.08:
                                opp["risk_note"] = (
                                    f"High downside risk: 10th pct = "
                                    f"{_p10:.1%} (vs ~{_fmove:+.1f}% "
                                    f"consensus move)"
                                )
                except Exception as _mce:
                    logger.debug("Monte Carlo skipped: %s", _mce)

                try:
                    from trading.strategies.strategy_comparison import (
                        get_strategy_comparison,
                    )

                    _cmp = get_strategy_comparison()
                    _best_name, _ = _cmp.get_best_strategy(
                        hist, metric="win_rate"
                    )
                    if _best_name:
                        _mat = _cmp.generate_comparison_matrix(hist)
                        _wr = None
                        if _mat is not None and not _mat.empty:
                            _row = _mat.loc[_mat["Strategy"] == _best_name]
                            if not _row.empty:
                                _wr = float(
                                    _row.iloc[0].get("win_rate") or 0.0
                                )
                        if _wr is not None:
                            opp["strategy_note"] = (
                                f"Best strategy: {_best_name} "
                                f"({_wr:.0%} win rate)"
                            )
                        else:
                            opp["strategy_note"] = (
                                f"Best strategy: {_best_name}"
                            )
                except Exception as _sce:
                    logger.debug("Strategy comparison skipped: %s", _sce)

            # Add AI score signals as thesis
            signals = candidate.get("signals", [])
            if signals:
                positive = [
                    s.get("description", "")
                    for s in signals
                    if s.get("impact") == "positive"
                ][:3]
                negative = [
                    s.get("description", "")
                    for s in signals
                    if s.get("impact") == "negative"
                ][:2]
                opp["thesis"] = "; ".join(positive) if positive else ""
                for _neg in negative:
                    if _neg:
                        opp.setdefault("risks", []).append(_neg)

            # Earnings catalyst
            try:
                from trading.data.earnings_calendar import get_upcoming_earnings
                earnings = get_upcoming_earnings(symbol)
                days_until = earnings.get("days_until")
                if days_until is not None and days_until <= 14:
                    opp["catalysts"].append(
                        f"⚠️ Earnings in {days_until} days"
                    )
                elif days_until is not None and days_until <= 30:
                    opp["catalysts"].append(
                        f"📅 Earnings in {days_until} days"
                    )
            except Exception:
                pass

        except Exception as e:
            logger.debug("Deep analysis failed for %s: %s", symbol, e)

        return opp

    def _get_watchlist_alerts(self) -> List[Dict[str, Any]]:
        """Check watchlist for notable moves or signals."""
        alerts = []
        try:
            from trading.data.watchlist import WatchlistManager

            _rows = WatchlistManager().get_all() or []
            watchlist = [
                str(r.get("symbol", "")).strip().upper()
                for r in _rows
                if r.get("symbol")
            ]
            if not watchlist:
                return []

            import numpy as np
            from trading.data.price_cache import get_history

            for symbol in watchlist[:10]:
                try:
                    hist = get_history(str(symbol), period="5d")
                    if hist.empty:
                        continue
                    _col_map = {c.lower(): c for c in hist.columns}
                    close_col = _col_map.get("close", hist.columns[0])
                    closes = hist[close_col].values
                    if len(closes) < 2:
                        continue

                    daily_change = float(
                        (closes[-1] / closes[-2] - 1) * 100
                    )

                    if abs(daily_change) > 3:
                        alerts.append({
                            "symbol": symbol,
                            "change_pct": round(daily_change, 2),
                            "alert_type": (
                                "SURGE" if daily_change > 0 else "DROP"
                            ),
                            "message": (
                                f"{symbol} moved {daily_change:+.1f}% today"
                            ),
                        })
                except Exception:
                    continue

        except Exception as e:
            logger.debug("Watchlist alerts failed: %s", e)

        return sorted(
            alerts,
            key=lambda x: abs(x["change_pct"]),
            reverse=True
        )[:5]

    def _format_markdown(self, report: Dict[str, Any]) -> str:
        """Format report as readable markdown."""
        lines = []
        now = datetime.now()
        lines.append(
            f"# 🌅 Morning Briefing — {now.strftime('%A, %B %d, %Y')}"
        )
        lines.append(
            f"*Generated at {now.strftime('%I:%M %p')}*\n"
        )

        # Market regime
        regime = report.get("market_regime", {})
        if regime:
            regime_emoji = {
                "RISK_ON": "🟢",
                "RISK_OFF": "🔴",
                "NEUTRAL": "🟡",
            }.get(regime.get("regime", "NEUTRAL"), "⚪")

            lines.append("## Market Regime")
            lines.append(
                f"{regime_emoji} **{regime.get('regime', 'NEUTRAL')}** "
                f"— {regime.get('description', '')}"
            )
            if regime.get("vix_level") is not None:
                lines.append(f"VIX: **{float(regime['vix_level']):.1f}**")
            if regime.get("spy_50d_pct"):
                lines.append(
                    f"SPY vs 50-day MA: **{regime['spy_50d_pct']:+.1f}%**"
                )
            lines.append("")

        # Top opportunities
        opportunities = report.get("top_opportunities", [])
        if opportunities:
            lines.append(f"## Top {len(opportunities)} Opportunities")
            for i, opp in enumerate(opportunities, 1):
                symbol = opp["symbol"]
                score = opp.get("ai_score", "N/A")
                _cp = opp.get("current_price")
                price_disp = f"${_cp:.2f}" if _cp is not None else "N/A"

                lines.append(
                    f"\n### {i}. {symbol} — "
                    f"Quick Score: {score} | {price_disp}"
                )

                forecast = opp.get("forecast", {})
                if forecast:
                    direction = forecast.get("direction", "")
                    dir_emoji = (
                        "🟢" if direction == "BULLISH"
                        else "🔴" if direction == "BEARISH"
                        else "🟡"
                    )
                    lines.append(
                        f"**Forecast:** {dir_emoji} {direction} "
                        f"| Target: ${forecast.get('consensus_price', 'N/A')} "
                        f"({forecast.get('expected_move_pct', 0):+.1f}%)"
                    )

                if opp.get("entry"):
                    lines.append(
                        f"**Trade:** Entry ${opp['entry']} | "
                        f"Target ${opp.get('target', 'N/A')} | "
                        f"Stop ${opp.get('stop', 'N/A')}"
                    )

                if opp.get("thesis"):
                    lines.append(f"**Thesis:** {opp['thesis']}")

                if opp.get("catalysts"):
                    lines.append(
                        f"**Catalysts:** {', '.join(opp['catalysts'])}"
                    )

                if opp.get("risks"):
                    lines.append(
                        f"**Risks:** {'; '.join(opp['risks'][:2])}"
                    )

                if opp.get("risk_note"):
                    lines.append(f"**Risk note:** {opp['risk_note']}")

                if opp.get("strategy_note"):
                    lines.append(f"**{opp['strategy_note']}**")

        else:
            lines.append(
                "## No High-Conviction Opportunities Today\n"
                f"No stocks cleared the {self.min_ai_score} Quick Score "
                "threshold. Consider lowering threshold or "
                "waiting for better setups."
            )

        short_opps = report.get("short_opportunities") or []
        if short_opps:
            _n_s = min(3, len(short_opps))
            lines.append(
                f"\n## Top {_n_s} Short Candidates"
            )
            for i, opp in enumerate(short_opps[:3], 1):
                sym = opp["symbol"]
                score = opp.get(
                    "short_score",
                    opp.get("short_quick_score", "N/A"),
                )
                _cp = opp.get("current_price")
                price_disp = (
                    f"${_cp:.2f}"
                    if _cp is not None
                    else "N/A"
                )
                lines.append(
                    f"\n### {i}. {sym} — Short Score: {score} | {price_disp}"
                )
                if opp.get("thesis"):
                    lines.append(f"**Thesis:** {opp['thesis']}")
            lines.append(
                "\n⚠️ *Short selling involves unlimited risk. "
                "Always use stop losses.*"
            )

        # Watchlist alerts
        alerts = report.get("watchlist_alerts", [])
        if alerts:
            lines.append("\n## Watchlist Alerts")
            for alert in alerts:
                emoji = "⬆️" if alert["change_pct"] > 0 else "⬇️"
                lines.append(
                    f"- {emoji} **{alert['symbol']}**: "
                    f"{alert['message']}"
                )

        lines.append(
            "\n---\n*Briefing uses **Quick Score** (technical estimate) for "
            "speed. For full 16-signal AI Score, open any ticker in Analyze. "
            "Always verify signals before trading.*"
        )

        return "\n".join(lines)

    def render_streamlit(self) -> None:
        """Render morning briefing in Streamlit."""
        try:
            import streamlit as st

            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader("🌅 Morning Briefing")
            with col2:
                if st.button(
                    "🔄 Regenerate",
                    key="briefing_regenerate"
                ):
                    st.session_state.pop("morning_briefing_cache", None)
                    st.session_state.pop("morning_briefing_ts", None)
                    st.rerun()

            # Cache for 30 minutes
            cache_key = "morning_briefing_cache"
            cache_ts_key = "morning_briefing_ts"

            import time
            now = time.time()
            cached = st.session_state.get(cache_key)
            cached_ts = st.session_state.get(cache_ts_key, 0)

            if cached and (now - cached_ts) < 1800:
                report = cached
            else:
                progress = st.progress(
                    0,
                    text=(
                        "Quick Score universe scan, then 5-model consensus "
                        "per top pick…"
                    ),
                )

                def _progress_cb(done: int, total: int) -> None:
                    if total <= 0:
                        return
                    frac = min(1.0, float(done) / float(total))
                    progress.progress(
                        frac,
                        text=f"Scanning universe… {done}/{total}",
                    )

                try:
                    report = self.generate(progress_callback=_progress_cb)
                finally:
                    progress.empty()
                st.session_state[cache_key] = report
                st.session_state[cache_ts_key] = now

            if report.get("error"):
                st.warning(f"Briefing error: {report['error']}")

            # Render markdown
            markdown = report.get("markdown", "")
            if markdown:
                st.markdown(markdown)

            # Structured opportunities table
            opps = report.get("top_opportunities", [])
            if opps:
                st.markdown("---")
                st.markdown("**Quick Reference Table**")
                rows = []
                for opp in opps:
                    forecast = opp.get("forecast", {})
                    rows.append({
                        "Symbol": opp["symbol"],
                        "Quick Score": opp.get("ai_score", "N/A"),
                        "Price": (
                            f"${float(opp['current_price']):.2f}"
                            if opp.get("current_price") is not None
                            else "N/A"
                        ),
                        "Direction": forecast.get("direction", "N/A"),
                        "Target": f"${opp.get('target', 0):.2f}"
                        if opp.get("target") else "N/A",
                        "Stop": f"${opp.get('stop', 0):.2f}"
                        if opp.get("stop") else "N/A",
                        "Expected Move": (
                            f"{forecast.get('expected_move_pct', 0):+.1f}%"
                            if forecast else "N/A"
                        ),
                        "Risk note": opp.get("risk_note") or "",
                        "Strategy": opp.get("strategy_note") or "",
                    })

                import pandas as pd
                df = pd.DataFrame(rows)
                st.dataframe(df, width='stretch')

        except Exception as e:
            try:
                import streamlit as st
                st.caption(f"Morning briefing unavailable: {e}")
            except Exception:
                pass
