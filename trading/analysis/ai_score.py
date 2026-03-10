"""
AI Score — composite signal strength rating (1-10) for any ticker.

Dimensions scored:
  Technical  (0-10): trend strength, momentum, volatility regime
  Sentiment  (0-10): short interest squeeze potential, insider flow
  Fundamental(0-10): earnings surprise, analyst estimates proximity
  Momentum   (0-10): price vs 20/50/200 SMA, RSI positioning

Each dimension is computed from already-available data pipelines.
No new external APIs required.
"""
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def compute_ai_score(symbol: str, hist: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Compute AI Score for a ticker.

    Args:
        symbol: ticker string
        hist: optional pre-fetched history DataFrame (Close, Volume, etc.)
              If None, fetches 6mo from yfinance.

    Returns dict:
        overall_score: float 1-10
        grade: str  "A" | "B" | "C" | "D" | "F"
        technical_score: float 0-10
        sentiment_score: float 0-10
        fundamental_score: float 0-10
        momentum_score: float 0-10
        signals: list of dicts {name, value, impact, description}
        summary: str — one-sentence plain-English verdict
        error: str | None
    """
    try:
        # --- Fetch data ---
        if hist is None or hist.empty:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo")
        if hist.empty or len(hist) < 20:
            return _error_score(symbol, "Insufficient price history")

        close = hist["Close"].values.astype(float)
        volume = hist["Volume"].values.astype(float) if "Volume" in hist.columns else None
        last_price = float(close[-1])

        signals = []

        # ── TECHNICAL SCORE (0-10) ──────────────────────────────────
        tech_points = 0.0

        # RSI (0-100 → score)
        rsi = _calc_rsi(close, 14)
        if rsi is not None:
            if 40 <= rsi <= 60:
                rsi_score = 5.0
            elif 30 <= rsi < 40 or 60 < rsi <= 70:
                rsi_score = 7.0
            elif rsi < 30:
                rsi_score = 9.0  # oversold = potential buy
            else:
                rsi_score = 3.0  # overbought
            tech_points += rsi_score
            signals.append(
                {
                    "name": "RSI",
                    "value": round(float(rsi), 1),
                    "impact": "positive" if rsi < 50 else "neutral" if rsi < 70 else "negative",
                    "description": f"RSI {rsi:.1f} — {'oversold' if rsi < 30 else 'neutral' if rsi < 70 else 'overbought'}",
                }
            )

        # Bollinger Band position
        if len(close) >= 20:
            sma20 = np.mean(close[-20:])
            std20 = np.std(close[-20:])
            bb_upper = sma20 + 2 * std20
            bb_lower = sma20 - 2 * std20
            bb_pct = (last_price - bb_lower) / (bb_upper - bb_lower + 1e-8)
            bb_score = 8.0 if bb_pct < 0.2 else 6.0 if bb_pct < 0.5 else 4.0 if bb_pct < 0.8 else 2.0
            tech_points += bb_score
            signals.append(
                {
                    "name": "Bollinger Position",
                    "value": round(float(bb_pct * 100), 1),
                    "impact": "positive" if bb_pct < 0.3 else "negative" if bb_pct > 0.8 else "neutral",
                    "description": f"Price at {bb_pct*100:.0f}% of Bollinger Band",
                }
            )
            tech_divisor = 2.0
        else:
            tech_divisor = 1.0

        technical_score = min(10.0, tech_points / tech_divisor)

        # ── MOMENTUM SCORE (0-10) ──────────────────────────────────
        mom_points = 0.0
        mom_signals = 0

        sma_periods = [(20, "SMA20"), (50, "SMA50"), (200, "SMA200")]
        for period, name in sma_periods:
            if len(close) >= period:
                sma = float(np.mean(close[-period:]))
                above = last_price > sma
                pct_diff = (last_price - sma) / sma * 100
                score = 7.0 if above else 3.0
                mom_points += score
                mom_signals += 1
                signals.append(
                    {
                        "name": f"Price vs {name}",
                        "value": round(pct_diff, 2),
                        "impact": "positive" if above else "negative",
                        "description": f"{'Above' if above else 'Below'} {name} by {abs(pct_diff):.1f}%",
                    }
                )

        # 20-day price momentum
        if len(close) >= 20:
            momentum_20d = (close[-1] / close[-20] - 1) * 100
            mom_score = 8.0 if momentum_20d > 5 else 6.0 if momentum_20d > 0 else 4.0 if momentum_20d > -5 else 2.0
            mom_points += mom_score
            mom_signals += 1
            signals.append(
                {
                    "name": "20d Momentum",
                    "value": round(float(momentum_20d), 2),
                    "impact": "positive" if momentum_20d > 0 else "negative",
                    "description": f"Price {momentum_20d:+.1f}% over 20 days",
                }
            )

        momentum_score = min(10.0, mom_points / max(mom_signals, 1))

        # ── SENTIMENT SCORE (0-10) ──────────────────────────────────
        sentiment_score = 5.0  # neutral default
        try:
            from trading.data.short_interest import get_short_interest

            si = get_short_interest(symbol)
            squeeze_score = si.get("short_squeeze_score", 0) or 0
            si_sentiment = min(10.0, 3.0 + squeeze_score * 0.07)
            short_pct = si.get("short_pct_float")
            if short_pct is None:
                short_pct = 0
            signals.append(
                {
                    "name": "Short Squeeze Score",
                    "value": round(float(squeeze_score), 1),
                    "impact": "positive" if squeeze_score > 50 else "neutral",
                    "description": f"Squeeze potential: {si.get('signal', 'UNKNOWN')} ({float(short_pct):.1f}% float short)",
                }
            )
            sentiment_score = si_sentiment
        except Exception:
            pass

        try:
            from trading.data.insider_flow import get_insider_flow

            insider = get_insider_flow(symbol)
            signal = insider.get("signal", "NO_ACTIVITY")
            insider_score = {
                "INSIDER_BUYING": 8.5,
                "MIXED": 5.5,
                "NO_ACTIVITY": 5.0,
                "INSIDER_SELLING": 2.5,
            }.get(signal, 5.0)
            sentiment_score = (sentiment_score + insider_score) / 2
            signals.append(
                {
                    "name": "Insider Flow",
                    "value": f"{insider.get('buy_count', 0)}B / {insider.get('sell_count', 0)}S",
                    "impact": "positive"
                    if signal == "INSIDER_BUYING"
                    else "negative"
                    if signal == "INSIDER_SELLING"
                    else "neutral",
                    "description": f"Insider activity (90d): {signal.replace('_', ' ').title()}",
                }
            )
        except Exception:
            pass

        # ── FUNDAMENTAL SCORE (0-10) ──────────────────────────────
        fundamental_score = 5.0
        try:
            from trading.data.earnings_calendar import get_upcoming_earnings

            earnings = get_upcoming_earnings(symbol)
            surprise = earnings.get("last_eps_surprise_pct")
            if surprise is not None:
                fund_score = 8.0 if surprise > 5 else 6.0 if surprise > 0 else 4.0 if surprise > -5 else 2.0
                fundamental_score = fund_score
                signals.append(
                    {
                        "name": "EPS Surprise",
                        "value": round(float(surprise), 1),
                        "impact": "positive" if surprise > 0 else "negative",
                        "description": f"Last earnings surprise: {surprise:+.1f}%",
                    }
                )
            days_until = earnings.get("days_until")
            if days_until is not None and 0 <= days_until <= 14:
                signals.append(
                    {
                        "name": "Earnings Risk",
                        "value": days_until,
                        "impact": "neutral",
                        "description": f"⚠️ Earnings in {days_until} days — elevated uncertainty",
                    }
                )
        except Exception:
            pass

        # ── COMPOSITE SCORE ─────────────────────────────────────────
        weights = {
            "technical": 0.30,
            "momentum": 0.35,
            "sentiment": 0.20,
            "fundamental": 0.15,
        }
        overall = (
            technical_score * weights["technical"]
            + momentum_score * weights["momentum"]
            + sentiment_score * weights["sentiment"]
            + fundamental_score * weights["fundamental"]
        )
        overall = round(min(10.0, max(1.0, overall)), 1)

        grade = (
            "A"
            if overall >= 8
            else "B"
            if overall >= 6.5
            else "C"
            if overall >= 5
            else "D"
            if overall >= 3.5
            else "F"
        )

        # Plain-English summary
        direction = "bullish" if overall >= 6.5 else "bearish" if overall < 4 else "neutral"
        summary = (
            f"{symbol} scores {overall}/10 ({grade}) — signals are predominantly "
            f"{direction} based on {len(signals)} technical and fundamental indicators."
        )

        return {
            "symbol": symbol,
            "overall_score": overall,
            "grade": grade,
            "technical_score": round(technical_score, 1),
            "momentum_score": round(momentum_score, 1),
            "sentiment_score": round(sentiment_score, 1),
            "fundamental_score": round(fundamental_score, 1),
            "signals": signals,
            "summary": summary,
            "last_price": last_price,
            "error": None,
        }

    except Exception as e:
        logger.error(f"AI Score failed for {symbol}: {e}")
        return _error_score(symbol, str(e))


def _calc_rsi(prices: np.ndarray, period: int = 14) -> Optional[float]:
    if len(prices) < period + 1:
        return None
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _error_score(symbol: str, error: str) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "overall_score": 5.0,
        "grade": "C",
        "technical_score": 5.0,
        "momentum_score": 5.0,
        "sentiment_score": 5.0,
        "fundamental_score": 5.0,
        "signals": [],
        "summary": f"Score unavailable: {error}",
        "last_price": None,
        "error": error,
    }
