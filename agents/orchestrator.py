"""
AgentOrchestrator: coordinates specialized agents for multi-agent chat mode.

Safety rules:
- Read-only: never triggers trades or writes to portfolio
- Transparent: always labels which agent provided each piece of data
- Graceful degradation: if an agent fails, returns partial results with clear labels
- No stale data: all agent calls fetch fresh data or are validated with timestamps
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Coordinates between specialized agents to answer complex queries.
    """

    # Configurable symbol blocklist (no orchestration for these)
    BLOCKLIST: Set[str] = set()

    def __init__(self, symbol: str, prompt: str, data_provider: Any):
        self.symbol = symbol
        self.prompt = prompt or ""
        self.data_provider = data_provider
        self.results: Dict[str, Any] = {}
        self.errors: Dict[str, str] = {}

    def run(self) -> Dict[str, Any]:
        """Run relevant agents based on prompt intent and return aggregated context."""
        start_time = time.time()

        # Safety: blocklisted symbols get no orchestration
        if self.symbol and self.symbol.upper() in self.BLOCKLIST:
            warning = (
                f"Orchestrator disabled for {self.symbol} (blocklisted symbol)."
            )
            logger.warning(warning)
            return {
                "context": f"[Orchestrator]: {warning}",
                "agents_used": [],
                "errors": {"orchestrator": "symbol_blocklisted"},
            }

        needs_forecast = any(
            w in self.prompt.lower()
            for w in [
                "forecast",
                "predict",
                "price target",
                "go up",
                "go down",
                "next week",
            ]
        )
        needs_sentiment = any(
            w in self.prompt.lower()
            for w in [
                "news",
                "sentiment",
                "opinion",
                "analyst",
                "why did",
                "why is",
            ]
        )
        needs_technicals = any(
            w in self.prompt.lower()
            for w in [
                "technical",
                "rsi",
                "macd",
                "trend",
                "support",
                "resistance",
                "moving average",
            ]
        )
        needs_strategy = any(
            w in self.prompt.lower()
            for w in [
                "strategy",
                "signal",
                "buy",
                "sell",
                "trade",
                "position",
            ]
        )

        context_parts: List[str] = []

        def _time_left() -> float:
            return 10.0 - (time.time() - start_time)

        # Always get live price (primary, fresh data source)
        if _time_left() > 0:
            logger.info(
                "Orchestrator: calling price agent for %s", self.symbol
            )
            context_parts.append(self._get_live_price())

        if _time_left() > 0 and needs_forecast:
            logger.info(
                "Orchestrator: calling forecast agent for %s", self.symbol
            )
            context_parts.append(self._get_forecast_context())

        if _time_left() > 0 and needs_sentiment:
            logger.info(
                "Orchestrator: calling news agent for %s", self.symbol
            )
            context_parts.append(self._get_news_context())

        if _time_left() > 0 and needs_technicals:
            logger.info(
                "Orchestrator: calling technical agent for %s", self.symbol
            )
            context_parts.append(self._get_technical_context())

        if _time_left() > 0 and needs_strategy:
            logger.info(
                "Orchestrator: calling strategy agent for %s", self.symbol
            )
            context_parts.append(self._get_strategy_context())

        elapsed = time.time() - start_time
        timeout_note = ""
        if elapsed >= 10.0:
            timeout_note = (
                "\n[Orchestrator]: Orchestration timed out after 10s, "
                "partial results shown."
            )

        return {
            "context": "\n\n".join(p for p in context_parts if p) + timeout_note,
            "agents_used": list(self.results.keys()),
            "errors": self.errors,
        }

    def _get_live_price(self) -> str:
        """
        Fetch last 14 days from data_provider and build a live price block.
        Read-only: no orders or state changes.
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=14)
            data = self.data_provider.get_historical_data(
                self.symbol, start_date, end_date, "1d"
            )
            if data is None or data.empty:
                self.errors["price"] = "No historical data available"
                return (
                    f"[Live market data — fetched now]\n"
                    f"No recent data available for {self.symbol}."
                )

            df = data.copy()
            if hasattr(df.columns, "levels"):
                df = (
                    df.droplevel(axis=1, level=0)
                    if df.columns.nlevels > 1
                    else df
                )
            df.columns = [str(c).lower().strip() for c in df.columns]

            close_col = (
                "close"
                if "close" in df.columns
                else next(
                    (c for c in df.columns if "close" in str(c).lower()), None
                )
            )
            if not close_col:
                self.errors["price"] = "No close column in data"
                return (
                    f"[Live market data — fetched now]\n"
                    f"Could not find a close price column for {self.symbol}."
                )

            closes = df[close_col].dropna()
            if len(closes) == 0:
                self.errors["price"] = "All close values are NaN"
                return (
                    f"[Live market data — fetched now]\n"
                    f"No valid close prices for {self.symbol} in recent data."
                )

            current_price = float(closes.iloc[-1])
            price_start = float(closes.iloc[0])
            pct_change = (
                (current_price - price_start) / price_start * 100
                if price_start
                else None
            )
            high_col = "high" if "high" in df.columns else close_col
            low_col = "low" if "low" in df.columns else close_col
            period_high = (
                float(df[high_col].max())
                if high_col in df.columns
                else current_price
            )
            period_low = (
                float(df[low_col].min())
                if low_col in df.columns
                else current_price
            )

            block = (
                f"[Live market data for {self.symbol} — fetched now]\n"
                f"Current price: ${current_price:.2f}\n"
                f"Price at start of period: ${price_start:.2f}\n"
            )
            if pct_change is not None:
                block += f"Period change (last ~2 weeks): {pct_change:+.2f}%\n"
            block += (
                f"Period high: ${period_high:.2f}, "
                f"Period low: ${period_low:.2f}"
            )

            self.results["price"] = {
                "current_price": current_price,
                "price_start": price_start,
                "period_high": period_high,
                "period_low": period_low,
            }
            return block
        except Exception as e:
            self.errors["price"] = str(e)
            return f"[Live market data — fetched now]\nPrice agent unavailable ({e})."

    def _get_forecast_context(self) -> str:
        """Read recent forecast from MemoryStore. Never triggers new model training."""
        try:
            from trading.memory import get_memory_store

            store = get_memory_store()
            forecast = store.get_preference(f"forecast_{self.symbol}")
            if not forecast:
                return (
                    f"[Forecast Agent]: No recent forecast for {self.symbol}. "
                    "Run the Forecasting page to generate one."
                )

            if not isinstance(forecast, dict):
                self.results["forecast"] = forecast
                return f"[Forecast Agent]\n{forecast}"

            ts_str = forecast.get("timestamp")
            minutes_str = "unknown"
            try:
                if ts_str:
                    s = (
                        str(ts_str)
                        .replace("Z", "")
                        .replace("+00:00", "")[:26]
                    )
                    ts_dt = datetime.fromisoformat(s)
                    now = datetime.utcnow()
                    if getattr(ts_dt, "tzinfo", None):
                        ts_dt = ts_dt.replace(tzinfo=None)
                    minutes = (now - ts_dt).total_seconds() / 60.0
                    minutes_str = f"{int(minutes)} min ago"
                    if minutes > 60:
                        return (
                            f"[Forecast Agent]: Last forecast for {self.symbol} "
                            f"is stale ({minutes_str}). Run the Forecasting page "
                            "to refresh model predictions."
                        )
            except Exception:
                # If timestamp is malformed, treat as stale
                return (
                    f"[Forecast Agent]: Last forecast for {self.symbol} has an "
                    "invalid timestamp. Run the Forecasting page to regenerate."
                )

            first = forecast.get("forecast_first")
            last = forecast.get("forecast_last")
            horizon = forecast.get("horizon", "?")
            direction = ""
            if isinstance(first, (int, float)) and isinstance(last, (int, float)):
                try:
                    pct = (
                        (float(last) - float(first)) / float(first) * 100
                        if float(first) != 0
                        else 0
                    )
                    direction = f"{pct:+.1f}% over {horizon} days"
                except Exception:
                    direction = f"over {horizon} days"

            model_name = forecast.get("model_name", "Unknown model")
            block = (
                f"[Forecast Agent — {minutes_str}]\n"
                f"Model: {model_name}, Horizon: {horizon} days\n"
            )
            if direction:
                block += f"Forecast move: {direction}\n"

            if forecast.get("model_failed"):
                block += (
                    "Note: model flagged this forecast as potentially unreliable; "
                    "treat with caution.\n"
                )

            self.results["forecast"] = forecast
            return block.strip()
        except Exception as e:
            self.errors["forecast"] = str(e)
            return f"[Forecast Agent]: Unavailable ({e})"

    def _get_news_context(self) -> str:
        """Fetch fresh news headlines."""
        try:
            from trading.data.news_fetcher import (
                fetch_recent_news,
                format_news_for_context,
            )

            headlines = fetch_recent_news(self.symbol, max_items=5)
            if not headlines:
                return (
                    f"[News Agent]: No recent headlines found for {self.symbol}."
                )
            ctx = format_news_for_context(headlines)
            self.results["news"] = {"count": len(headlines)}
            return f"[News Agent — fetched now]\n{ctx}"
        except Exception as e:
            self.errors["news"] = str(e)
            return f"[News Agent]: Unavailable ({e})"

    def _get_technical_context(self) -> str:
        """Compute basic technicals (RSI/SMA) from recent data."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)
            data = self.data_provider.get_historical_data(
                self.symbol, start_date, end_date, "1d"
            )
            if data is None or data.empty:
                return (
                    f"[Technical Agent]: No recent data for {self.symbol}."
                )

            df = data.copy()
            if hasattr(df.columns, "levels"):
                df = (
                    df.droplevel(axis=1, level=0)
                    if df.columns.nlevels > 1
                    else df
                )
            df.columns = [str(c).lower().strip() for c in df.columns]
            close_col = (
                "close"
                if "close" in df.columns
                else next(
                    (c for c in df.columns if "close" in str(c).lower()), None
                )
            )
            if not close_col:
                return (
                    f"[Technical Agent]: Could not locate a close column for "
                    f"{self.symbol}."
                )

            closes = df[close_col].astype("float64")
            if len(closes) < 20:
                return (
                    f"[Technical Agent]: Not enough data to compute technical "
                    f"indicators for {self.symbol}."
                )

            # Basic RSI (14) and simple MAs (20, 50)
            delta = closes.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            roll = 14
            avg_gain = gain.rolling(window=roll, min_periods=roll).mean()
            avg_loss = loss.rolling(window=roll, min_periods=roll).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            rsi_latest = float(rsi.dropna().iloc[-1]) if rsi.dropna().size else None

            sma20 = closes.rolling(window=20, min_periods=1).mean()
            sma50 = closes.rolling(window=50, min_periods=1).mean()
            sma20_latest = float(sma20.iloc[-1])
            sma50_latest = float(sma50.iloc[-1])
            price_latest = float(closes.iloc[-1])

            trend = "up" if sma20_latest > sma50_latest else "down"
            block = (
                f"[Technical Agent — computed now]\n"
                f"Price: ${price_latest:.2f}\n"
                f"RSI(14): {rsi_latest:.1f} " if rsi_latest is not None else ""
            )
            block += (
                f"\nSMA20: ${sma20_latest:.2f}, SMA50: ${sma50_latest:.2f} "
                f"(short-term trend: {trend})"
            )

            self.results["technicals"] = {
                "rsi_14": rsi_latest,
                "sma20": sma20_latest,
                "sma50": sma50_latest,
                "trend": trend,
            }
            return block.strip()
        except Exception as e:
            self.errors["technicals"] = str(e)
            return f"[Technical Agent]: Unavailable ({e})"

    def _get_strategy_context(self) -> str:
        """Read recent strategy signals from MemoryStore. Never places orders."""
        try:
            from trading.memory import get_memory_store

            store = get_memory_store()
            signals = store.get_preference(f"strategy_signals_{self.symbol}")
            if not signals:
                return (
                    f"[Strategy Agent]: No recent signals for {self.symbol}. "
                    "Run the Strategy page to generate signals."
                )

            self.results["strategy"] = signals
            return f"[Strategy Agent]\n{signals}"
        except Exception as e:
            self.errors["strategy"] = str(e)
            return f"[Strategy Agent]: Unavailable ({e})"

