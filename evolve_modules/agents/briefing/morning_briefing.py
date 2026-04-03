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
from datetime import datetime
from typing import Any, Dict, List, Optional

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

    def __init__(
        self,
        universe: str = "sp100",
        min_ai_score: float = 6.5,
        max_positions: int = 5,
    ):
        self.universe = universe
        self.min_ai_score = min_ai_score
        self.max_positions = max_positions
        self._last_report: Optional[Dict[str, Any]] = None

    def generate(self) -> Dict[str, Any]:
        """
        Generate full morning briefing.
        Returns dict with markdown report and structured data.
        """
        logger.info("Morning briefing: starting generation")
        report = {
            "timestamp": datetime.now().isoformat(),
            "market_regime": {},
            "top_opportunities": [],
            "watchlist_alerts": [],
            "risk_summary": {},
            "markdown": "",
            "error": None,
        }

        try:
            # Step 1: Market regime
            report["market_regime"] = self._get_market_regime()

            # Step 2: Scan universe
            candidates = self._scan_universe()
            logger.info(
                "Morning briefing: %d candidates above %.1f score",
                len(candidates), self.min_ai_score
            )

            # Step 3: Deep analysis on top candidates
            opportunities = []
            for candidate in candidates[:self.max_positions]:
                try:
                    opp = self._analyze_opportunity(candidate)
                    if opp:
                        opportunities.append(opp)
                except Exception as e:
                    logger.debug(
                        "Opportunity analysis failed for %s: %s",
                        candidate.get("symbol"), e
                    )

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
            import yfinance as yf
            import numpy as np

            spy = yf.Ticker("SPY").history(period="3mo")
            vix = yf.Ticker("^VIX").history(period="5d")

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

            if not vix.empty:
                _col_map = {c.lower(): c for c in vix.columns}
                close_col = _col_map.get("close", vix.columns[0])
                vix_level = float(vix[close_col].iloc[-1])
                regime["vix_level"] = round(vix_level, 2)

                if vix_level > 30:
                    regime["volatility"] = "HIGH"
                elif vix_level > 20:
                    regime["volatility"] = "ELEVATED"
                else:
                    regime["volatility"] = "NORMAL"

            # Determine overall regime
            is_bullish = regime["spy_trend"] == "BULLISH"
            vix = regime.get("vix_level", 20)
            if is_bullish and (vix or 20) < 20:
                regime["regime"] = "RISK_ON"
                regime["description"] = "Market is in risk-on mode — momentum strategies favored"
            elif not is_bullish and (vix or 20) > 25:
                regime["regime"] = "RISK_OFF"
                regime["description"] = "Market is in risk-off mode — defensive positioning recommended"
            else:
                regime["regime"] = "NEUTRAL"
                regime["description"] = "Mixed signals — selective positioning recommended"

        except Exception as e:
            logger.debug("Market regime fetch failed: %s", e)
            regime["error"] = str(e)

        return regime

    def _scan_universe(self) -> List[Dict[str, Any]]:
        """Run scanner on universe and return top candidates."""
        try:
            from trading.analysis.market_scanner import MarketScanner
            scanner = MarketScanner()
            results = scanner.scan_universe(
                universe=self.universe,
                min_score=self.min_ai_score,
            )
            if isinstance(results, dict):
                candidates = results.get("results", [])
            else:
                candidates = results or []

            # Sort by AI score
            candidates.sort(
                key=lambda x: float(x.get("ai_score", 0) or 0),
                reverse=True
            )
            return candidates[:self.max_positions * 2]

        except Exception as e:
            logger.warning("Universe scan failed: %s", e)
            return []

    def _analyze_opportunity(
        self, candidate: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Deep analysis on a single candidate."""
        symbol = candidate.get("symbol")
        if not symbol:
            return None

        opp = {
            "symbol": symbol,
            "ai_score": candidate.get("ai_score"),
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
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo")

            if hist.empty:
                return None

            _col_map = {c.lower(): c for c in hist.columns}
            close_col = _col_map.get("close", hist.columns[0])
            current_price = float(hist[close_col].iloc[-1])
            opp["current_price"] = round(current_price, 2)

            # Get consensus forecast
            try:
                from trading.models.forecast_router import ForecastRouter
                router = ForecastRouter()
                forecast = router.get_consensus_forecast(
                    data=hist, horizon=7
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

                        opp["conviction"] = forecast.get("conviction", "MEDIUM")
            except Exception as e:
                logger.debug("Forecast failed for %s: %s", symbol, e)

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
                opp["risks"] = negative

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
            from trading.data.watchlist import get_watchlist
            watchlist = get_watchlist()
            if not watchlist:
                return []

            import yfinance as yf
            import numpy as np

            for symbol in watchlist[:10]:
                try:
                    hist = yf.Ticker(symbol).history(period="5d")
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
            if regime.get("vix_level"):
                lines.append(f"VIX: **{regime['vix_level']:.1f}**")
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
                price = opp.get("current_price", "N/A")

                lines.append(
                    f"\n### {i}. {symbol} — "
                    f"AI Score: {score} | ${price}"
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

        else:
            lines.append(
                "## No High-Conviction Opportunities Today\n"
                f"No stocks cleared the {self.min_ai_score} AI Score "
                "threshold. Consider lowering threshold or "
                "waiting for better setups."
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
            f"\n---\n*Briefing covers {self.universe.upper()} universe. "
            f"Always verify signals before trading.*"
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
                with st.spinner("Generating briefing..."):
                    report = self.generate()
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
                        "AI Score": opp.get("ai_score", "N/A"),
                        "Price": f"${opp.get('current_price', 0):.2f}",
                        "Direction": forecast.get("direction", "N/A"),
                        "Target": f"${opp.get('target', 0):.2f}"
                        if opp.get("target") else "N/A",
                        "Stop": f"${opp.get('stop', 0):.2f}"
                        if opp.get("stop") else "N/A",
                        "Expected Move": (
                            f"{forecast.get('expected_move_pct', 0):+.1f}%"
                            if forecast else "N/A"
                        ),
                    })

                import pandas as pd
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)

        except Exception as e:
            try:
                import streamlit as st
                st.caption(f"Morning briefing unavailable: {e}")
            except Exception:
                pass
