"""
AI Score — composite signal strength rating (1-10) for any ticker.

Dimensions scored:
  Technical  (0-10): trend strength, momentum, volatility regime
  Sentiment  (0-10): short interest, insider flow, Reddit mention tone
  Fundamental(0-10): earnings surprise, analyst estimates proximity
  Momentum   (0-10): price vs 20/50/200 SMA, RSI positioning

Each dimension is computed from already-available data pipelines.
No new external APIs required.
"""
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from trading.data.price_cache import get_history as _pc_get_history
from trading.utils.safe_math import safe_rsi

logger = logging.getLogger(__name__)

_MACRO_FACTORS_INSTANCE = None
_ML_TRAINER_INSTANCE = None


def _get_macro_factors():
    global _MACRO_FACTORS_INSTANCE
    if _MACRO_FACTORS_INSTANCE is None:
        from trading.analysis.macro_factors import MacroFactors

        _MACRO_FACTORS_INSTANCE = MacroFactors()
    return _MACRO_FACTORS_INSTANCE


def _get_ml_trainer():
    global _ML_TRAINER_INSTANCE
    if _ML_TRAINER_INSTANCE is None:
        from trading.analysis.ml_score_trainer import MLScoreTrainer

        _ML_TRAINER_INSTANCE = MLScoreTrainer()
    return _ML_TRAINER_INSTANCE

SECTOR_PE = {
    "Technology": 28.0,
    "Healthcare": 22.0,
    "Financials": 14.0,
    "Consumer Discretionary": 25.0,
    "Consumer Staples": 20.0,
    "Industrials": 20.0,
    "Energy": 12.0,
    "Utilities": 17.0,
    "Materials": 18.0,
    "Real Estate": 35.0,
    "Communication Services": 20.0,
}

SECTOR_RISK_FLAGS = {
    "Utilities": [
        "⚠️ Rate case risk — regulatory approval "
        "required for earnings growth"
    ],
    "Healthcare": [
        "⚠️ FDA catalyst risk — binary event "
        "possible from regulatory decisions"
    ],
    "Financials": [
        "⚠️ Interest rate sensitivity — earnings "
        "exposed to Fed policy changes"
    ],
    "Energy": [
        "⚠️ Commodity price risk — earnings "
        "directly tied to oil/gas prices"
    ],
    "Real Estate": [
        "⚠️ Rate sensitivity — cap rate "
        "compression risk in rising rate environment"
    ],
    "Technology": [
        "⚠️ Antitrust/regulatory scrutiny "
        "elevated for large-cap names"
    ],
    "Communication Services": [
        "⚠️ Regulatory/antitrust risk — "
        "content moderation and competition scrutiny"
    ],
    "Consumer Discretionary": [
        "⚠️ Consumer spending sensitivity — "
        "exposed to rate and sentiment cycles"
    ],
}


def compute_ai_score(symbol: str, hist: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Compute AI Score for a ticker.

    Args:
        symbol: ticker string
        hist: optional pre-fetched history DataFrame (Close, Volume, etc.)
              If None, fetches 6mo via price_cache (shared TTL).

    Returns dict:
        overall_score: float 1-10
        grade: str  "A" | "B" | "C" | "D" | "F"
        technical_score: float 0-10
        sentiment_score: float 0-10
        fundamental_score: float 0-10
        momentum_score: float 0-10
        signals: list of dicts {name, value, impact, description}
        data_quality: per-dimension real vs unavailable
        summary: str — one-sentence plain-English verdict
        error: str | None
    """
    if hist is not None and not hist.empty:
        return _compute_ai_score_impl(symbol, hist)
    sym_key = str(symbol or "").strip().upper()
    if not sym_key:
        return _error_score(str(symbol or ""), "Invalid symbol")
    return _compute_ai_score_cached(sym_key)


@st.cache_data(ttl=300, show_spinner=False)
def _compute_ai_score_cached(symbol: str) -> Dict[str, Any]:
    try:
        h = _pc_get_history(symbol, period="6mo")
    except Exception as e:
        logger.warning("ai_score: price_cache get_history failed: %s", e)
        h = pd.DataFrame()
    return _compute_ai_score_impl(symbol, h)


def _compute_ai_score_impl(symbol: str, hist: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Internal AI score computation (used by compute_ai_score)."""
    try:
        _external_bundle: Optional[Dict[str, Any]] = None
        try:
            import asyncio

            from trading.data.external_signals import get_external_signals_manager

            _esm = get_external_signals_manager()
            _external_bundle = asyncio.run(
                _esm.get_all_signals(symbol, days_back=3)
            )
        except Exception:
            _external_bundle = None

        # --- Fetch data ---
        if hist is None or hist.empty:
            try:
                hist = _pc_get_history(symbol, period="6mo")
            except Exception as e:
                logger.warning("ai_score: price_cache get_history failed: %s", e)
                hist = pd.DataFrame()
        if hist.empty or len(hist) < 20:
            return _error_score(symbol, "Insufficient price history")

        _col_map = {c.lower(): c for c in hist.columns}
        _close_col = _col_map.get(
            "close",
            list(hist.select_dtypes(include="number").columns)[0],
        )
        _volume_col = _col_map.get("volume")
        close = hist[_close_col].values.astype(float)
        volume = (
            hist[_volume_col].values.astype(float)
            if _volume_col is not None
            else None
        )
        last_price = float(close[-1])

        signals = []
        data_quality: Dict[str, str] = {
            "sentiment": "unavailable",
            "options": "unavailable",
            "macro": "unavailable",
            "insider": "unavailable",
        }
        earnings_near = False
        earnings_days_until = None

        # ── TECHNICAL SCORE (0-10) ──────────────────────────────────
        tech_points = 0.0

        # RSI (0-100 → score) — Wilder smoothing via safe_rsi (same as XGBoost / platform)
        _rsi_series = safe_rsi(close, 14)
        _rsi_flat = np.asarray(_rsi_series, dtype=float).ravel()
        rsi = float(_rsi_flat[-1]) if _rsi_flat.size else None
        if rsi is not None and np.isfinite(rsi):
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

        try:
            from trading.analysis.chart_pattern_detector import (
                ChartPatternDetector,
            )

            _pat = ChartPatternDetector(symbol, hist).detect_all()
            _bull = {
                "Inverse Head and Shoulders",
                "Double Bottom",
                "Ascending Triangle",
                "Golden Cross",
            }
            _bear = {
                "Head and Shoulders",
                "Double Top",
                "Descending Triangle",
                "Death Cross",
            }
            _seen_bull = _seen_bear = False
            for _p in _pat.get("patterns") or []:
                _nm = str((_p or {}).get("name") or "")
                if _nm in _bull and not _seen_bull:
                    technical_score = min(10.0, technical_score + 0.3)
                    signals.append(
                        {
                            "name": f"Pattern: {_nm}",
                            "value": round(
                                float((_p or {}).get("confidence") or 0), 2
                            ),
                            "impact": "positive",
                            "description": (_p or {}).get(
                                "description", ""
                            )
                            or f"Bullish pattern: {_nm}",
                        }
                    )
                    _seen_bull = True
                elif _nm in _bear and not _seen_bear:
                    technical_score = max(0.0, technical_score - 0.3)
                    signals.append(
                        {
                            "name": f"Pattern: {_nm}",
                            "value": round(
                                float((_p or {}).get("confidence") or 0), 2
                            ),
                            "impact": "negative",
                            "description": (_p or {}).get(
                                "description", ""
                            )
                            or f"Bearish pattern: {_nm}",
                        }
                    )
                    _seen_bear = True
        except Exception as _pe:
            logger.debug("Chart pattern AI score hook skipped: %s", _pe)

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

        try:
            from utils.math_helpers import (
                calculate_momentum_score as _mh_momentum,
                calculate_regime_probability as _mh_regime,
            )

            close_s = pd.Series(close)
            _ms = _mh_momentum(close_s)
            if len(_ms) and not bool(pd.isna(_ms.iloc[-1])):
                _z = float(_ms.iloc[-1])
                _adj = 0.25 * float(np.tanh(_z))
                momentum_score = float(
                    min(10.0, max(0.0, momentum_score + _adj))
                )
            _rets = close_s.pct_change().dropna()
            if len(_rets) >= 30:
                _rp = _mh_regime(_rets, window=min(60, len(_rets)))
                _bull = _rp.get("bull")
                if _bull is not None and len(_bull) and not pd.isna(_bull.iloc[-1]):
                    _bp = float(_bull.iloc[-1])
                    momentum_score = float(
                        min(10.0, max(0.0, momentum_score + 0.3 * (_bp - 0.5)))
                    )
        except Exception as _e:
            logger.debug("math_helpers momentum/regime refinement skipped: %s", _e)

        try:
            from trading.data.options_flow import get_options_flow

            _of = get_options_flow(symbol)
            if _of.get("success"):
                data_quality["options"] = "real"
                _pcr = float(_of.get("put_call_ratio") or 0.0)
                _uc = len(_of.get("unusual_calls") or [])
                _up = len(_of.get("unusual_puts") or [])
                _unusual = _uc > 0 or _up > 0
                if _unusual and 0 < _pcr < 0.7:
                    momentum_score = min(10.0, momentum_score + 0.4)
                    signals.append(
                        {
                            "name": "Options flow",
                            "value": round(_pcr, 3),
                            "impact": "positive",
                            "description": "Options: unusual call activity",
                        }
                    )
                elif _pcr > 1.3:
                    momentum_score = max(0.0, momentum_score - 0.4)
                    signals.append(
                        {
                            "name": "Options flow",
                            "value": round(_pcr, 3),
                            "impact": "negative",
                            "description": "Options: unusual put activity",
                        }
                    )
        except Exception as _oe:
            logger.debug("Options flow AI score hook skipped: %s", _oe)

        # ── SENTIMENT SCORE (0-10) ──────────────────────────────────
        sentiment_score = 5.0  # neutral default
        try:
            from trading.data.short_interest import get_short_interest

            si = get_short_interest(symbol)
            squeeze_score = si.get("short_squeeze_score", 0) or 0
            short_pct = si.get("short_pct_float")
            if short_pct is None:
                short_pct = 0
            short_pct_f = float(short_pct)
            if short_pct_f >= 20:
                si_sentiment = 9.0
            elif short_pct_f >= 15:
                si_sentiment = 7.5
            elif short_pct_f >= 10:
                si_sentiment = 6.5
            elif short_pct_f >= 5:
                si_sentiment = 5.5
            else:
                si_sentiment = 4.0
            squeeze_tier = (
                "EXTREME"
                if short_pct_f >= 20
                else "HIGH"
                if short_pct_f >= 15
                else "ELEVATED"
                if short_pct_f >= 10
                else "MODERATE"
                if short_pct_f >= 5
                else "LOW"
            )

            # If float short is very elevated, also lift momentum
            if short_pct_f >= 25:
                momentum_score = min(10.0, momentum_score + 2.0)
            elif short_pct_f >= 15:
                momentum_score = min(10.0, momentum_score + 1.5)

            signals.append(
                {
                    "name": "Short Squeeze Score",
                    "value": round(float(squeeze_score), 1),
                    "impact": "positive" if squeeze_score > 50 else "neutral",
                    "description": f"Float short: {short_pct_f:.1f}% — Squeeze: {squeeze_tier}",
                }
            )
            sentiment_score = si_sentiment
        except Exception:
            pass

        if _external_bundle:
            try:
                _news = _external_bundle.get("news_sentiment") or []
                _n = len(_news) if isinstance(_news, list) else 0
                if _n > 0:
                    signals.append(
                        {
                            "name": "External signal bundle",
                            "value": float(_n),
                            "impact": "neutral",
                            "description": (
                                f"Unified external feed returned {_n} news/social "
                                "records (see data pipeline for details)."
                            ),
                        }
                    )
            except Exception as _e:
                logger.debug("external_signals bundle annotate skipped: %s", _e)

        try:
            from trading.data.insider_flow import get_insider_flow

            insider = get_insider_flow(symbol)
            if insider.get("error"):
                data_quality["insider"] = "unavailable"
            else:
                data_quality["insider"] = "real"
            signal = insider.get("signal", "NO_ACTIVITY")
            insider_score = {
                "INSIDER_BUYING": 8.5,
                "MIXED": 5.5,
                "NO_ACTIVITY": 5.0,
                "INSIDER_SELLING": 2.5,
            }.get(signal, 5.0)
            sentiment_score = (sentiment_score + insider_score) / 2
            _buy = insider.get("buy_count", 0) or 0
            _sell = insider.get("sell_count", 0) or 0
            _insider_val = (
                "No Activity"
                if (_buy == 0 and _sell == 0)
                else f"{_buy}B / {_sell}S"
            )
            signals.append(
                {
                    "name": "Insider Flow",
                    "value": _insider_val,
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

        try:
            from trading.data.social_sentiment import get_social_sentiment

            social = get_social_sentiment(symbol)
            if social and social.get("success") and social.get("source") == "reddit":
                data_quality["sentiment"] = "real"
            if social and social.get("source") == "unavailable":
                signals.append(
                    {
                        "name": "Social Sentiment",
                        "value": "N/A",
                        "impact": "neutral",
                        "description": (
                            "Social sentiment: unavailable "
                            f"({social.get('reason') or social.get('error') or 'fetch failed'})"
                        ),
                    }
                )
            elif (
                social
                and not social.get("error")
                and social.get("success")
                and int(social.get("mention_count") or 0) > 0
            ):
                sentiment_score_social = (
                    (float(social["sentiment_score"]) + 1) / 2 * 10
                )
                sentiment_score = (
                    sentiment_score * 0.7 + sentiment_score_social * 0.3
                )
                signals.append(
                    {
                        "name": "Social Sentiment",
                        "value": social["mention_count"],
                        "impact": (
                            "positive"
                            if social["sentiment_label"] == "BULLISH"
                            else "negative"
                            if social["sentiment_label"] == "BEARISH"
                            else "neutral"
                        ),
                        "description": (
                            f"Reddit: {social['sentiment_label']} "
                            f"({social['mention_count']} mentions today"
                            + (
                                " · trending"
                                if social.get("trending")
                                else ""
                            )
                            + ")"
                        ),
                    }
                )
        except Exception as e:
            logger.debug("Social sentiment skipped: %s", e)

        # ── FUNDAMENTAL SCORE (0-10) ──────────────────────────────
        fundamental_score = 5.0
        sector = ""  # initialise for use in risk flags
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
                earnings_near = True
                earnings_days_until = days_until

            # Valuation overlay vs sector-average P/E
            try:
                _ticker_obj = yf.Ticker(symbol)
                _info = _ticker_obj.info
                stock_pe = _info.get("trailingPE")
                sector = _info.get("sector", "")
                sector_pe = SECTOR_PE.get(sector, 20.0)
                if stock_pe and stock_pe > 0 and sector_pe > 0:
                    pe_premium = (stock_pe - sector_pe) / sector_pe * 100
                    if pe_premium > 30:
                        valuation_adj = -1.5
                        val_label = "Premium"
                        val_impact = "negative"
                    elif pe_premium > 10:
                        valuation_adj = -0.5
                        val_label = "Slight Premium"
                        val_impact = "negative"
                    elif pe_premium < -10:
                        valuation_adj = 1.0
                        val_label = "Discount"
                        val_impact = "positive"
                    else:
                        valuation_adj = 0.0
                        val_label = "Fair Value"
                        val_impact = "neutral"

                    fundamental_score = min(
                        10.0, max(0.0, fundamental_score + valuation_adj)
                    )
                    signals.append(
                        {
                            "name": "Valuation vs Sector",
                            "value": f"{pe_premium:+.0f}% vs sector",
                            "impact": val_impact,
                            "description": (
                                f"P/E {stock_pe:.1f}x vs {sector} avg "
                                f"{sector_pe:.1f}x — {val_label}"
                            ),
                        }
                    )
            except Exception:
                pass

            # Sector-specific risk flags
            try:
                _flags = SECTOR_RISK_FLAGS.get(sector, [])
                for _flag in _flags:
                    signals.append(
                        {
                            "name": "Sector Risk",
                            "value": sector or "Unknown",
                            "impact": "neutral",
                            "description": _flag,
                        }
                    )
            except Exception:
                pass

            # Macro factor adjustment
            try:
                _macro = _get_macro_factors()
                _macro_adj = _macro.get_ai_score_adjustment(
                    sector=sector
                )
                data_quality["macro"] = "real"
                _macro_score_adj = _macro_adj.get(
                    "score_adjustment", 0.0
                )
                fundamental_score = min(10.0, max(0.0,
                    fundamental_score + _macro_score_adj
                ))
                for _msig in _macro_adj.get("signals", []):
                    signals.append(_msig)
            except Exception:
                pass
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

        if earnings_near:
            overall = min(overall, 7.5)

        # ML Score blend (if model is trained) — after pre-earnings cap
        try:
            _ml_trainer = _get_ml_trainer()
            _ml_result = _ml_trainer.predict(symbol, hist)
            if (not _ml_result.get("fallback")
                    and _ml_result.get("ml_score") is not None):
                _ml_score = float(_ml_result["ml_score"])
                # Blend 40% ML, 60% rules-based
                overall = round(
                    overall * 0.6 + _ml_score * 0.4, 1
                )
                overall = float(np.clip(overall, 1.0, 10.0))
                signals.append({
                    "name": "ML Score",
                    "value": round(_ml_score, 1),
                    "impact": (
                        "positive" if _ml_score > 6.0
                        else "negative" if _ml_score < 4.0
                        else "neutral"
                    ),
                    "description": (
                        f"ML model: {_ml_result.get('predicted_7d_return', 0):+.1f}% "
                        f"7-day forecast "
                        f"({_ml_result.get('direction', 'NEUTRAL')})"
                    ),
                })
        except Exception:
            pass

        if earnings_near:
            overall = min(overall, 7.5)
            for sig in signals:
                if sig.get("name") == "Earnings Risk":
                    sig["description"] = (
                        f"⚠️ Earnings in {earnings_days_until}d "
                        f"— conviction capped at 7.5"
                    )

        grade = (
            "A"
            if overall >= 8.0
            else "B"
            if overall >= 6.0
            else "C"
            if overall >= 4.5
            else "D"
            if overall >= 3.0
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
            "data_quality": data_quality,
            "summary": summary,
            "last_price": last_price,
            "error": None,
        }

    except Exception as e:
        logger.error(f"AI Score failed for {symbol}: {e}")
        return _error_score(symbol, str(e))


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
        "data_quality": {
            "sentiment": "unavailable",
            "options": "unavailable",
            "macro": "unavailable",
            "insider": "unavailable",
        },
        "summary": f"Score unavailable: {error}",
        "last_price": None,
        "error": error,
    }
