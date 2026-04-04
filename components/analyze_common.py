# -*- coding: utf-8 -*-
"""Shared helpers for Analyze / Deep Dive (extracted from legacy Analyze page)."""
import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _extract_forecast_values(result):
    """Extract forecast array from result (dict/list/array). Handles all result formats."""
    if result is None:
        return None
    if isinstance(result, (list, np.ndarray)):
        arr = np.asarray(result)
        return arr if arr.size > 0 else None
    if isinstance(result, dict):
        for key in ["forecast", "predictions", "values", "forecast_values", "consensus_forecast"]:
            val = result.get(key)
            if val is not None:
                arr = np.asarray(val)
                if arr.size > 0:
                    return arr
    return None


def _news_sentiment_score(ticker: str) -> float:
    """Return a 0–10 news score with recency decay for AI sentiment."""
    try:
        from trading.data.price_cache import get_news as _get_news
        import time

        items = _get_news(ticker)
        if not items:
            return 5.0
        pos_kw = [
            "beat",
            "surge",
            "raises",
            "upgrade",
            "strong",
            "growth",
            "record",
            "above",
            "buy",
            "bullish",
            "jumps",
            "soars",
        ]
        neg_kw = [
            "miss",
            "falls",
            "cuts",
            "downgrade",
            "weak",
            "below",
            "layoffs",
            "loss",
            "investigation",
            "sell",
            "bearish",
            "drops",
            "plunges",
        ]
        total_score = 0.0
        total_weight = 0.0
        now = time.time()
        for item in items[:10]:
            content = item.get("content") or {}
            raw_title = (
                item.get("title")
                or item.get("headline")
                or content.get("title")
                or content.get("summary")
                or ""
            )
            title = raw_title.lower()
            if not title:
                continue
            pub = (
                item.get("providerPublishTime")
                or item.get("published")
                or content.get("pubDate")
                or now
            )
            try:
                from datetime import datetime

                if isinstance(pub, str):
                    dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                    pub_ts = dt.timestamp()
                elif hasattr(pub, "timestamp"):
                    pub_ts = float(pub.timestamp())
                else:
                    pub_ts = float(pub)
            except Exception:
                pub_ts = now
            age_hours = max(0.0, (now - pub_ts) / 3600.0)
            if age_hours < 1:
                decay = 1.0
            elif age_hours < 6:
                decay = 0.7
            elif age_hours < 24:
                decay = 0.4
            else:
                decay = 0.1
            raw = sum(1 for k in pos_kw if k in title) - sum(
                1 for k in neg_kw if k in title
            )
            total_score += raw * decay
            total_weight += decay
        if total_weight == 0:
            return 5.0
        normalized = total_score / total_weight
        return round(min(10.0, max(0.0, 5.0 + normalized * 2.0)), 1)
    except Exception:
        return 5.0


def _is_english(text: str) -> bool:
    if not text:
        return False
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return ascii_count / len(text) > 0.8


def _generate_recommendation(
    ticker,
    ai_score_result,
    forecast_result=None,
    trader_mode="Short-term",
):
    try:
        import streamlit as st
        from trading.data.price_cache import get_quote

        _sym = (ticker or "").strip().upper()

        # Entry is always current price from latest quote
        _quote = get_quote(ticker) if ticker else {}
        last_price = _quote.get("price") or _quote.get("regularMarketPrice") or 0.0
        try:
            last_price = float(last_price)
        except Exception:
            last_price = 0.0
        if last_price <= 0:
            # Fallback: no valid price, return neutral hold
            return {
                "action": "HOLD",
                "conviction": "LOW",
                "signal_score": 5.0,
                "fc_target": None,
                "fc_direction": "flat",
                "reasons": [("~", "Price unavailable — no trade recommendation")],
            }
        _entry = float(last_price)

        # Determine action from AI Score (not from consensus direction)
        _score = ai_score_result.get(
            "weighted_score", ai_score_result.get("overall_score", 5.0)
        )
        try:
            _score = float(_score)
        except Exception:
            _score = 5.0

        if _score >= 7.0:
            _action = "BUY"
            _conv = "HIGH"
        elif _score >= 6.0:
            _action = "BUY"
            _conv = "MEDIUM"
        elif _score <= 3.0:
            _action = "SELL"
            _conv = "HIGH"
        elif _score <= 4.0:
            _action = "SELL"
            _conv = "MEDIUM"
        else:
            _action = "HOLD"
            _conv = "LOW"

        # Consensus for direction / conviction: prefer explicit forecast_result from this
        # call (e.g. fresh router output for `ticker`). Session `current_forecast_result`
        # is only used when no argument was passed or it is unusable — otherwise a stale
        # non-empty session dict would override the correct ticker’s forecast.
        def _fc_symbol(fc):
            if not isinstance(fc, dict):
                return None
            v = fc.get("symbol") or fc.get("ticker") or fc.get("Symbol")
            return str(v).strip().upper() if v else None

        _passed = forecast_result
        _sess_raw = st.session_state.get("current_forecast_result")
        _sess = _sess_raw if isinstance(_sess_raw, dict) else {}

        if _passed is not None and isinstance(_passed, dict):
            if _passed.get("error"):
                _consensus = {}
            else:
                ps = _fc_symbol(_passed)
                if ps and _sym and ps != _sym:
                    _consensus = {}
                else:
                    _consensus = _passed
        elif _sess and not _sess.get("error"):
            ss = _fc_symbol(_sess)
            if ss and _sym and ss != _sym:
                _consensus = {}
            else:
                _consensus = _sess
        else:
            _consensus = {}
        _fc_arr = _consensus.get("forecast", [])
        _consensus_dir = _consensus.get("direction", "NEUTRAL")
        _consensus_conv = (
            _consensus.get("conviction")
            or _consensus.get("confidence_label", "LOW")
        )

        # Override action to HOLD if AI Score and consensus strongly conflict
        if (
            _action == "BUY"
            and _consensus_dir == "BEARISH"
            and str(_consensus_conv).upper() == "HIGH"
            and _score < 6.5
        ):
            _action = "HOLD"
            _conv = "LOW"
        elif (
            _action == "SELL"
            and _consensus_dir == "BULLISH"
            and str(_consensus_conv).upper() == "HIGH"
            and _score > 3.5
        ):
            _action = "HOLD"
            _conv = "LOW"

        # Set target and stop based on action direction
        # _vol from ai_score is annualized decimal (e.g. 0.2528 = 25.28%)
        # Convert to daily: ann_vol / sqrt(252)
        try:
            _ann_vol = max(float(ai_score_result.get("volatility", 0.15)), 0.05)
        except Exception:
            _ann_vol = 0.15
        _daily_vol = _ann_vol / np.sqrt(252)
        # 7-day expected move (1 std dev)
        _horizon_vol = _daily_vol * np.sqrt(7)
        # Target = 0.5 std dev move, cap 3%
        _target_pct = min(_horizon_vol * 0.5, 0.03)
        _target_pct = max(_target_pct, 0.005)
        # Stop = 0.75 std dev move, cap 2%
        _stop_pct = min(_horizon_vol * 0.75, 0.02)
        _stop_pct = max(_stop_pct, 0.005)

        if _action == "BUY":
            _target = _entry * (1 + _target_pct)
            _stop = _entry * (1 - _stop_pct)
        elif _action == "SELL":
            _target = _entry * (1 - _target_pct)
            _stop = _entry * (1 + _stop_pct)
        else:  # HOLD
            # Show a narrow range instead of directional target
            _target = _entry * 1.01
            _stop = _entry * 0.98
            _target_pct = 0.01
            _stop_pct = 0.02

        # R/R ratio and expected move
        _upside = abs(_target - _entry)
        _downside = abs(_entry - _stop)
        _rr = (_upside / _downside) if _downside > 0 else 1.0
        _pct_move = ((_target - _entry) / _entry * 100.0) if _entry else 0.0

        # Build reasoning bullets aligned with action/direction
        _reasons = []
        # AI Score strength
        if _score >= 6.0:
            _reasons.append(("+", f"AI Score bullish ({_score:.1f}/10)"))
        elif _score <= 4.0:
            _reasons.append(("-", f"AI Score bearish ({_score:.1f}/10)"))

        # Consensus direction
        if _consensus_dir == "BULLISH":
            _reasons.append(("+", "Forecast models project upside"))
        elif _consensus_dir == "BEARISH":
            _reasons.append(("-", "Forecast models project decline"))

        # Technical score
        tech = ai_score_result.get("technical_score", 5.0)
        try:
            tech = float(tech)
        except Exception:
            tech = 5.0
        if tech >= 7.0:
            _reasons.append(("+", "Technical trend is strong"))
        elif tech <= 4.0:
            _reasons.append(("-", "Technical trend is weak"))

        # Signal conflict warning
        if _action in ("BUY", "SELL"):
            if _action == "BUY" and _consensus_dir == "BEARISH":
                _reasons.append(
                    ("⚠", "AI Score bullish but models bearish — use caution")
                )
            elif _action == "SELL" and _consensus_dir == "BULLISH":
                _reasons.append(
                    ("⚠", "AI Score bearish but models bullish — use caution")
                )

        if not _reasons:
            _reasons.append(("~", "Signals are mixed — monitor closely"))

        # Limit to 3 reasons
        _reasons = _reasons[:3]

        # fc_target/fc_direction for backward compatibility (used by widget)
        if _action == "BUY":
            fc_direction = "up"
        elif _action == "SELL":
            fc_direction = "down"
        else:
            fc_direction = "flat"
        fc_target = _target

        return {
            "action": _action,
            "conviction": _conv,
            "signal_score": round(_score, 1),
            "fc_target": fc_target,
            "fc_direction": fc_direction,
            "reasons": _reasons,
            "entry": _entry,
            "target": _target,
            "stop": _stop,
            "target_pct": _target_pct,
            "stop_pct": _stop_pct,
            "risk_reward": _rr,
            "pct_move": _pct_move,
        }
    except Exception:
        return None
