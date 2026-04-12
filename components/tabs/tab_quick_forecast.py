# -*- coding: utf-8 -*-
"""Analyze page tab body (extracted; logic unchanged)."""
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from components.analyze_common import (
    _extract_forecast_values,
    _generate_recommendation,
    _is_english,
    _news_sentiment_score,
    sentiment_icon_for_label,
)
from trading.data.earnings_calendar import get_upcoming_earnings
from trading.data.insider_flow import get_insider_flow
from trading.data.price_cache import get_history, get_info, get_news, get_quote

try:
    from utils.dataframe_utils import normalize_for_display
except ImportError:
    def normalize_for_display(df):
        return df

logger = logging.getLogger(__name__)


def render(
    ticker: str,
    hist,
    period: str,
    period_label: str,
    trader_mode: str,
    _interval: str,
    _tf_label: str,
    *,
    score_mode: str = "Buy",
    scoring_style: str = "Balanced (default)",
    backend: dict,
) -> None:
    """Streamlit tab body (legacy Analyze)."""
    _ = backend  # passed by analyze_tabs_sections; forecast path uses price_cache
    try:
        QUICK_FORECAST_MODELS = [
            "ARIMA",
            "XGBoost",
            "Ridge",
            "CatBoost",
            "Prophet",
            "LSTM",
            "TCN",
            "Transformer",
            "GARCH",
            "Ensemble",
        ]
        st.header("Quick Forecast")
        st.markdown(
            "Generate forecasts via the consensus router (all major models); "
            "quick presets: "
            + ", ".join(QUICK_FORECAST_MODELS)
        )

        # Ticker from Analyze header — independent of the main chart range at top of page.
        symbol = str(
            st.session_state.get("analyze_ticker") or ticker or ""
        ).strip().upper()
        if not symbol:
            st.info(
                "Enter a ticker symbol at the top of the page to load data."
            )
            return

        _score_mode_lc = str(score_mode or "Buy").strip().lower()
        st.session_state["analyze_symbol"] = symbol

        _prev_fc_sym = str(
            st.session_state.get("analyze_forecast_data_symbol", ""),
        ).strip().upper()
        if _prev_fc_sym and _prev_fc_sym != symbol:
            st.session_state.pop("analyze_forecast_data", None)
            st.session_state.pop("analyze_forecast_data_symbol", None)
            st.session_state.pop("current_forecast", None)
            st.session_state.pop("current_forecast_result", None)

        if "quick_forecast_ai_lookback" not in st.session_state:
            st.session_state["quick_forecast_ai_lookback"] = "1M"

        st.caption(
            "The chart at the top of Analyze uses its own range. "
            "AI Score here uses the daily lookback you choose below (default **1 month**)."
        )
        _ai_lb_opts = ["1W", "1M", "3M", "6M", "1Y"]
        _ai_lb_default_i = (
            _ai_lb_opts.index(
                st.session_state["quick_forecast_ai_lookback"],
            )
            if st.session_state["quick_forecast_ai_lookback"] in _ai_lb_opts
            else 1
        )
        st.selectbox(
            "AI Score lookback (daily bars)",
            _ai_lb_opts,
            index=_ai_lb_default_i,
            key="quick_forecast_ai_lookback",
            help=(
                "Used for AI / Short scores only. Consensus models still train on "
                "~1 year of daily bars when you click Generate Forecast."
            ),
        )
        _ai_lb = str(
            st.session_state.get("quick_forecast_ai_lookback", "1M"),
        ).strip()
        _ai_period_map = {
            # compute_ai_score() requires len(hist) >= 20 (ai_score.py); 5d daily
            # never reaches that — use ~1mo of dailies for the "1W" preset label.
            "1W": "1mo",
            "1M": "1mo",
            "3M": "3mo",
            "6M": "6mo",
            "1Y": "1y",
        }
        _ai_yf_period = _ai_period_map.get(_ai_lb, "1mo")
        # Match compute_ai_score minimum (20 rows) so we do not pass the gate then
        # get an error dict with no UI (tab only renders the panel when error is None).
        _ai_min_rows = {
            "1W": 20,
            "1M": 20,
            "3M": 20,
            "6M": 20,
            "1Y": 20,
        }
        _min_ai_rows = int(_ai_min_rows.get(_ai_lb, 20))
        _hist_ai = None
        try:
            _hist_ai = get_history(
                symbol,
                period=_ai_yf_period,
                interval="1d",
            )
        except Exception as _e_ai_hist:
            logger.warning(
                "Quick forecast: AI lookback get_history failed: %s",
                _e_ai_hist,
            )
            _hist_ai = None

        if (
            _hist_ai is None
            or getattr(_hist_ai, "empty", True)
            or len(_hist_ai) < _min_ai_rows
        ):
            st.warning(
                f"Not enough daily data for {symbol} at this lookback. "
                "Try a longer window or a different ticker."
            )
            return

        if _ai_lb == "1W":
            st.caption(
                "Preset **1W** loads about one month of daily bars so the AI "
                "pipeline meets its 20-session minimum (true calendar-week dailies "
                "are too sparse)."
            )

        if _score_mode_lc == "short":
            st.warning(
                "Short Score mode active — forecasts still show price direction. "
                "For the short thesis, look for downward forecasts and high Short "
                "Score in the Technical tab.",
                icon="📉",
            )

        try:
            _e = get_upcoming_earnings(symbol)
            if _e.get("is_within_window"):
                _d, _dt = _e["days_until"], _e["next_earnings_date"]
                _s = (
                    f" | Last surprise: {_e['last_eps_surprise_pct']:+.1f}%"
                    if _e.get("last_eps_surprise_pct") is not None
                    else "  "
                )
                st.warning(
                    f"Earnings in {_d} day{'s' if _d != 1 else ''} ({_dt}){_s} — "
                    "forecasts may be less reliable near earnings."
                )
        except Exception as _e:
            logger.warning("Analyze: earnings proximity check failed: %s", _e)

        _raw_fc = st.session_state.get("analyze_forecast_data")
        _fc_sym = str(
            st.session_state.get("analyze_forecast_data_symbol", ""),
        ).strip().upper()
        data_fc = None
        if (
            _raw_fc is not None
            and not getattr(_raw_fc, "empty", True)
            and _fc_sym == symbol
        ):
            data_fc = _raw_fc

        data = (
            data_fc.copy()
            if data_fc is not None
            else _hist_ai.copy()
        )
        if "close" in data.columns and "Close" not in data.columns:
            data = data.rename(columns={"close": "Close"})
        if "Close" not in data.columns:
            data = data.copy()
            data["Close"] = data.iloc[:, 0]

        if data_fc is None:
            st.caption(
                "One year of daily bars for consensus models loads when you click "
                "**Generate Forecast**."
            )
        else:
            st.caption(
                "Preview below uses loaded training data (~1y daily). "
                "AI Score still uses the lookback selected above."
            )

        st.markdown("---")
        st.subheader("📊 Data Preview")

        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Data Points", len(data))
        with col2:
            st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
        with col3:
            change = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
            st.metric("Period Return", f"{change:.2f}%")
        with col4:
            volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
            st.metric("Annualized Volatility", f"{volatility:.2f}%")

        # AI Score panel with trader-mode display weights and news score
        try:
            from trading.analysis.ai_score import compute_ai_score
            _sym = st.session_state.get("analyze_symbol") or symbol
            if _sym:
                _hist = _hist_ai.copy()
                if "close" in _hist.columns and "Close" not in _hist.columns:
                    _hist = _hist.rename(columns={"close": "Close"})
                if "volume" in _hist.columns and "Volume" not in _hist.columns:
                    _hist = _hist.rename(columns={"volume": "Volume"})

                # Cache AI score per symbol for 5 minutes to avoid recompute on tab switch
                import time as _time

                _ai_score_key = (
                    f"ai_score_{_sym}_{trader_mode}_{scoring_style}_{_ai_lb}"
                )
                _ai_score_ts_key = (
                    f"ai_score_ts_{_sym}_{trader_mode}_{scoring_style}_{_ai_lb}"
                )
                _cached_score = st.session_state.get(_ai_score_key)
                _cached_ts = st.session_state.get(_ai_score_ts_key, 0.0)
                _score_age = _time.time() - _cached_ts

                if _cached_score is None or _score_age > 300:
                    with st.spinner("Computing AI Score..."):
                        score_result = compute_ai_score(
                            _sym,
                            _hist,
                            scoring_style=scoring_style,
                        )
                    if score_result.get("error"):
                        # Do not cache failures — avoids silent stale "unavailable"
                        # after fixing data or lookback.
                        st.session_state.pop(_ai_score_key, None)
                        st.session_state.pop(_ai_score_ts_key, None)
                        st.session_state.pop("ai_score_result", None)
                    else:
                        st.session_state[_ai_score_key] = score_result
                        st.session_state[_ai_score_ts_key] = _time.time()
                        st.session_state.pop(
                            f"short_score_{_sym}_{trader_mode}_{scoring_style}_{_ai_lb}",
                            None,
                        )
                        st.session_state.pop(
                            f"short_score_ts_{_sym}_{trader_mode}_{scoring_style}_{_ai_lb}",
                            None,
                        )
                    # Canonical key for cross-component access (only when usable)
                    if not score_result.get("error"):
                        st.session_state["ai_score_result"] = score_result
                else:
                    score_result = _cached_score

                short_result = None
                if (
                    _score_mode_lc == "short"
                    and score_result.get("error") is None
                ):
                    from trading.analysis.ai_score import compute_short_score

                    _sk = (
                        f"short_score_{_sym}_{trader_mode}_{scoring_style}_{_ai_lb}"
                    )
                    _stk = (
                        f"short_score_ts_{_sym}_{trader_mode}_{scoring_style}_{_ai_lb}"
                    )
                    _short_age = _time.time() - float(
                        st.session_state.get(_stk, 0.0) or 0.0,
                    )
                    if (
                        st.session_state.get(_sk) is None
                        or _short_age > 300
                    ):
                        short_result = compute_short_score(
                            _sym,
                            _hist,
                            score_result,
                            scoring_style=scoring_style,
                        )
                        if not short_result.get("error"):
                            st.session_state[_sk] = short_result
                            st.session_state[_stk] = _time.time()
                    else:
                        short_result = st.session_state.get(_sk)

                if score_result.get("error") is None:
                    if (
                        _score_mode_lc == "short"
                        and short_result is not None
                        and not short_result.get("error")
                    ):
                        score = float(
                            short_result.get("short_score", 0) or 0,
                        )
                        grade = str(short_result.get("grade") or "")
                        _summary_txt = str(
                            short_result.get("summary") or "",
                        )
                    elif _score_mode_lc == "short":
                        score = float(
                            score_result.get("overall_score", 0) or 0,
                        )
                        grade = str(score_result.get("grade") or "")
                        _summary_txt = str(
                            score_result.get("summary") or "",
                        )
                        st.caption(
                            "Short Score unavailable; showing buy-side AI metrics.",
                        )
                    else:
                        score = score_result["overall_score"]
                        grade = score_result["grade"]
                        _summary_txt = str(
                            score_result.get("summary") or "",
                        )

                    # Trader-mode-specific display weights (buy-side AI only)
                    if _score_mode_lc != "short":
                        if trader_mode == "Short-term":
                            display_weights = {
                                "technical": 0.45,
                                "momentum": 0.40,
                                "sentiment": 0.10,
                                "fundamental": 0.05,
                            }
                            mode_label = "Short-term weights"
                        else:
                            display_weights = {
                                "technical": 0.20,
                                "momentum": 0.20,
                                "sentiment": 0.15,
                                "fundamental": 0.45,
                            }
                            mode_label = "Long-term weights"

                        component_scores = {
                            "technical": score_result.get("technical_score", 0),
                            "momentum": score_result.get("momentum_score", 0),
                            "sentiment": score_result.get("sentiment_score", 0),
                            "fundamental": score_result.get("fundamental_score", 0),
                        }
                        weighted_score = sum(
                            component_scores[k] * display_weights[k]
                            for k in display_weights
                        )
                        weighted_score = round(
                            min(10.0, max(0.0, weighted_score)), 1,
                        )

                        # Show mode impact clearly (diff vs base score)
                        try:
                            base_score = float(
                                score_result.get("overall_score", 0) or 0,
                            )
                            diff = round(weighted_score - base_score, 1)
                            diff_str = (
                                f"+{diff}" if diff > 0
                                else str(diff) if diff < 0
                                else "="
                            )
                            diff_color = (
                                "#26a69a" if diff > 0
                                else "#ef5350" if diff < 0
                                else "#4a6080"
                            )
                            st.markdown(
                                f'<span style="font-size:11px;color:{diff_color}">'
                                f"{trader_mode} view: {diff_str} vs base score"
                                f"</span>",
                                unsafe_allow_html=True,
                            )
                        except Exception:
                            pass
                    else:
                        display_weights = {}
                        component_scores = {
                            "technical": score_result.get("technical_score", 0),
                            "momentum": score_result.get("momentum_score", 0),
                            "sentiment": score_result.get("sentiment_score", 0),
                            "fundamental": score_result.get("fundamental_score", 0),
                        }
                        weighted_score = 0.0
                        mode_label = (
                            f"{trader_mode} · Short Score mode"
                        )

                    # News sentiment score feeding into sentiment view
                    news_score = _news_sentiment_score(_sym)
                    if news_score >= 7.5:
                        ns_label, ns_color = "HOT", "#ff9800"
                    elif news_score >= 6.0:
                        ns_label, ns_color = "POS", "#26a69a"
                    elif news_score <= 3.0:
                        ns_label, ns_color = "NEG", "#ef5350"
                    else:
                        ns_label, ns_color = "NEU", "#4a6080"

                    from pathlib import Path as _Path_ml

                    _ml_model_p = (
                        _Path_ml(__file__).resolve().parents[2]
                        / ".cache"
                        / "ml_score"
                        / "ml_score_model.joblib"
                    )
                    if not _ml_model_p.is_file():
                        st.caption(
                            "⚠️ ML Score signal inactive — train the "
                            "model in Settings → AI & Signals."
                        )

                    _long_ov = float(
                        score_result.get("overall_score", 0) or 0,
                    )
                    if _score_mode_lc == "short":
                        st.markdown("### 📉 Short Score")
                    else:
                        st.markdown("### 🤖 AI Score")
                    top_cols = st.columns([2, 2, 2, 2])
                    with top_cols[0]:
                        st.metric(
                            "Short Score"
                            if _score_mode_lc == "short"
                            else "Model Score",
                            f"{score}/10",
                            delta=grade,
                            delta_color="normal" if score >= 5 else "inverse",
                        )
                    with top_cols[1]:
                        if _score_mode_lc == "short":
                            st.metric(
                                "Buy-side AI",
                                f"{_long_ov:.1f}/10",
                                help="Long/buy thesis (context)",
                            )
                        else:
                            st.metric(
                                "Weighted Score",
                                f"{weighted_score}/10",
                                help=mode_label,
                            )
                    with top_cols[2]:
                        st.markdown("**News Score**")
                        _ns_ic = sentiment_icon_for_label(ns_label)
                        st.markdown(
                            f"{_ns_ic} **{news_score:.1f}** · {ns_label}"
                        )
                    with top_cols[3]:
                        st.caption(mode_label)

                    # Component bars with weights (buy-side view)
                    if _score_mode_lc != "short":
                        bar_rows = [
                            ("Technical", "technical"),
                            ("Momentum", "momentum"),
                            ("Sentiment", "sentiment"),
                            ("Fundamental", "fundamental"),
                        ]
                        for label, key_name in bar_rows:
                            val = float(
                                component_scores.get(key_name, 0) or 0,
                            )
                            w = display_weights.get(key_name, 0)
                            pct = int(round(w * 100))
                            cols_row = st.columns([2, 5, 1])
                            with cols_row[0]:
                                st.markdown(f"**{label}**")
                            with cols_row[1]:
                                st.progress(
                                    min(1.0, max(0.0, val / 10.0)),
                                )
                            with cols_row[2]:
                                st.markdown(f"{val:.1f}  ({pct}%)")

                    st.caption(_summary_txt)
                    _drive_sigs = (
                        (short_result.get("signals") or [])
                        if (
                            _score_mode_lc == "short"
                            and short_result is not None
                            and not short_result.get("error")
                        )
                        else score_result.get("signals", [])
                    )
                    with st.expander("What drives this score?", expanded=False):
                        signals = _drive_sigs
                        if signals:
                            by_impact = sorted(
                                signals,
                                key=lambda s: (
                                    s.get("impact") == "positive",
                                    s.get("impact") == "negative",
                                ),
                                reverse=True,
                            )
                            for sig in by_impact[:3]:
                                st.caption(
                                    f"• {sig.get('name', '')}: {sig.get('value', '')} — {sig.get('description', '')}"
                                )
                        else:
                            st.caption("No signal breakdown available.")
                    with st.expander("📊 Signal Breakdown", expanded=False):
                        signals = _drive_sigs
                        if signals:
                            sig_df = pd.DataFrame(
                                signals
                            )[["name", "value", "impact", "description"]]
                            sig_df.columns = [
                                "Signal",
                                "Value",
                                "Impact",
                                "Description",
                            ]

                            def _color_impact(val):
                                colors = {
                                    "positive": (
                                        "background-color: #1a4a2a; color: #26a69a"
                                    ),
                                    "negative": (
                                        "background-color: #3a1a1a; color: #ef5350"
                                    ),
                                    "bearish": (
                                        "background-color: #3a1a1a; color: #ef5350"
                                    ),
                                    "risk": (
                                        "background-color: #3a2a0a; color: #ff9800"
                                    ),
                                    "neutral": (
                                        "background-color: #3a2a0a; color: #ff9800"
                                    ),
                                }
                                return colors.get(str(val).lower(), "")

                            try:
                                styler = sig_df.style.applymap(
                                    _color_impact, subset=["Impact"]
                                )
                                st.dataframe(styler, width='stretch')
                            except Exception:
                                st.dataframe(normalize_for_display(sig_df), width='stretch')
                    # Recommendation panel
                    try:
                        if (
                            _score_mode_lc == "short"
                            and short_result is not None
                            and not short_result.get("error")
                        ):
                            _rec = _generate_recommendation(
                                ticker,
                                score_result,
                                st.session_state.get(
                                    "current_forecast_result",
                                ),
                                trader_mode,
                                score_mode="Short",
                                short_score=float(
                                    short_result.get("short_score", 0)
                                    or 0,
                                ),
                            )
                        else:
                            _rec = _generate_recommendation(
                                ticker,
                                score_result,
                                st.session_state.get(
                                    "current_forecast_result",
                                ),
                                trader_mode,
                            )
                        if _rec:
                            _action = _rec["action"]
                            _conv = _rec["conviction"]
                            _score = _rec["signal_score"]
                            _color = (
                                "#26a69a"
                                if "BUY" in _action
                                else "#ef5350"
                                if "SELL" in _action or "SHORT" in _action
                                else "#ff9800"
                            )
                            _reasons_html = ""
                            for _s, _t2 in _rec["reasons"]:
                                # _s is one of "+", "-", "⚠", or "~"
                                _icon = _s if _s in ("+", "-", "⚠") else "~"
                                if _icon == "+":
                                    _rc = "#26a60a"
                                elif _icon == "-":
                                    _rc = "#ef5350"
                                elif _icon == "⚠":
                                    _rc = "#ffb74d"
                                else:
                                    _rc = "#8899aa"
                                _reasons_html += (
                                    f'<div style="display:flex;gap:10px;'
                                    f'margin-bottom:6px">'
                                    f'<span style="color:{_rc};'
                                    f'font-family:monospace;font-weight:bold;'
                                    f'min-width:14px">{_icon}</span>'
                                    f'<span style="color:#c8d4e0;'
                                    f'font-size:13px">{_t2}</span></div>'
                                )
                            _mh = ""
                            _info_row = ""
                            if _rec.get("fc_target") is not None:
                                _entry = float(_rec.get("entry", 0.0))
                                _target = float(_rec.get("target", _rec["fc_target"]))
                                _stop = float(_rec.get("stop", _entry))
                                if _entry and _entry > 0:
                                    _pct2 = ((_target / _entry) - 1.0) * 100.0
                                    _stop2 = _stop
                                    _rr2 = float(_rec.get("risk_reward", 1.0))
                                    _tc2 = "#26a69a" if _pct2 > 0 else "#ef5350"

                                    # Trade horizon estimate (7 business days by default)
                                    try:
                                        from datetime import datetime as _dtime
                                        from datetime import timedelta as _tdelta

                                        _today = _dtime.now()
                                        _bdays = 0
                                        _end_dt = _today
                                        _fc_horizon = 7
                                        while _bdays < _fc_horizon:
                                            _end_dt += _tdelta(days=1)
                                            if _end_dt.weekday() < 5:
                                                _bdays += 1
                                        _horizon_str = _end_dt.strftime("%b %d")
                                    except Exception:
                                        _horizon_str = "~7 days"
                                        _fc_horizon = 7

                                    _conv_explain = {
                                        "HIGH": "Multiple models agree",
                                        "MEDIUM": "Models show mixed signals",
                                        "LOW": "Weak or conflicting signals",
                                    }.get(str(_conv).upper(), "Signal strength unknown")

                                    _mh = (
                                        f'<div style="display:grid;'
                                        f'grid-template-columns:repeat(3,1fr);'
                                        f'gap:1px;background:#1e2d45;'
                                        f'border-radius:4px;overflow:hidden;'
                                        f'margin-top:10px">'
                                        f'<div style="background:#0f1525;'
                                        f'padding:10px 14px">'
                                        f'<div style="font-size:10px;'
                                        f'color:#4a6080;'
                                        f'letter-spacing:1px;'
                                        f'margin-bottom:3px">'
                                        f'ENTRY</div>'
                                        f'<div style="font-size:15px;'
                                        f'font-weight:bold;'
                                        f'color:#e0e6f0">'
                                        f'${_entry:.2f}</div></div>'
                                        f'<div style="background:#0f1525;'
                                        f'padding:10px 14px">'
                                        f'<div style="font-size:10px;'
                                        f'color:#4a6080;'
                                        f'letter-spacing:1px;'
                                        f'margin-bottom:3px">'
                                        f'{"TARGET ↑" if "BUY" in _action else "TARGET ↓" if "SELL" in _action or "SHORT" in _action else "RANGE"}</div>'
                                        f'<div style="font-size:15px;'
                                        f'font-weight:bold;'
                                        f'color:{_tc2}">'
                                        f'${_rec["fc_target"]:.2f}</div>'
                                        f'<div style="font-size:11px;'
                                        f'color:{_tc2}">'
                                        f'{_pct2:+.1f}%</div></div>'
                                        f'<div style="background:#0f1525;'
                                        f'padding:10px 14px">'
                                        f'<div style="font-size:10px;'
                                        f'color:#4a6080;'
                                        f'letter-spacing:1px;'
                                        f'margin-bottom:3px">'
                                        f'STOP</div>'
                                        f'<div style="font-size:15px;'
                                        f'font-weight:bold;'
                                        f'color:#ef5350">'
                                        f'${_stop2:.2f}</div>'
                                        f'<div style="font-size:11px;'
                                        f'color:#ef5350">'
                                        f'R/R {_rr2:.1f}:1</div>'
                                        f'</div></div>'
                                    )

                                    _info_row = (
                                        f'<div style="padding:8px 16px;'
                                        f'border-top:1px solid #1e2d45;'
                                        f'display:flex;justify-content:space-between;'
                                        f'flex-wrap:wrap;gap:8px">'
                                        f'<span style="font-size:11px;'
                                        f'color:#4a6080">⏱ Until: {_horizon_str}</span>'
                                        f'<span style="font-size:11px;'
                                        f'color:#4a6080">📊 Move: {_pct2:+.1f}%</span>'
                                        f'<span style="font-size:11px;'
                                        f'color:#4a6080">❌ Cut if: &lt;${_stop2:.2f}</span>'
                                        f'<span style="font-size:11px;'
                                        f'color:#4a6080" title="{_conv_explain}">💡 {_conv_explain}</span>'
                                        f'</div>'
                                    )
                            _rec_min_h = 300 if _mh else 160
                            st.html(
                                f'<div style="background:#0a0e1a;border:1px solid '
                                f'#1e2d45;border-radius:6px;overflow:hidden;'
                                f'font-family:Courier New,monospace;'
                                f'margin:4px 0;min-height:{_rec_min_h}px">'
                                f'<div style="padding:10px 16px;border-bottom:'
                                f'1px solid #1e2d45;display:flex;align-items:'
                                f'center;gap:14px">'
                                f'<span style="font-size:18px;font-weight:bold;'
                                f'color:{_color}">{_action}</span>'
                                f'<span style="font-size:11px;color:#4a6080;'
                                f'background:#0f1525;padding:2px 8px;border:1px '
                                f'solid #1e2d45;border-radius:3px">'
                                f'{_conv} conviction</span>'
                                f'<span style="font-size:11px;color:#4a6080;'
                                f'margin-left:auto">score {_score}/10</span>'
                                f'</div>'
                                f'<div style="padding:10px 16px">{_reasons_html}'
                                f'</div>{_mh}{_info_row}</div>',
                                width="stretch",
                            )
                    except Exception as _re:
                        st.caption(f"Recommendation unavailable: {_re}")
                else:
                    _err_msg = (
                        score_result.get("summary")
                        or score_result.get("error")
                        or "AI Score unavailable."
                    )
                    st.warning(_err_msg)
        except Exception as _e:
            st.caption(f"AI Score unavailable: {_e}")

        # Multi-timeframe chart
        with st.expander("📈 Multi-Timeframe Chart", expanded=False):
            try:
                from components.multi_timeframe_chart import render_multi_timeframe_chart
                _sym = st.session_state.get("analyze_symbol") or symbol
                render_multi_timeframe_chart(_sym, hist_daily=data)
            except Exception as _e:
                st.caption(f"Chart unavailable: {_e}")

        # Price chart - use advanced candlestick chart if OHLCV data available
        try:
            from utils.plotting_helper import create_candlestick_chart

            # Check if we have OHLCV data
            has_ohlcv = all(col in data.columns for col in ['open', 'high', 'low', 'close'])

            if has_ohlcv:
                fig = create_candlestick_chart(
                    data=data,
                    title=f"{st.session_state.analyze_symbol} Price History",
                    show_volume='volume' in data.columns,
                    show_ma=[20, 50, 200] if len(data) >= 200 else ([20, 50] if len(data) >= 50 else [20])
                )
            else:
                # Fallback to simple line chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='blue', width=2)
                ))
                fig.update_layout(
                    title=f"{st.session_state.analyze_symbol} Price History",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    hovermode='x unified',
                    height=400
                )

            st.plotly_chart(fig, width='stretch')
        except ImportError:
            # Fallback to basic chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            ))
            fig.update_layout(
                title=f"{st.session_state.analyze_symbol} Price History",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, width='stretch')

        # Data table (expandable)
        with st.expander("📋 View Full Data"):
            st.dataframe(normalize_for_display(data.tail(50)))

        # Model Selection & Forecasting
        st.markdown("---")
        st.subheader("🎯 Generate Forecast")

        # Forecast horizon slider (outside form so it updates immediately)
        forecast_horizon = st.slider(
            "Forecast Horizon (days)",
            min_value=1,
            max_value=30,
            value=st.session_state.analyze_forecast_horizon,
            key="forecast_horizon_slider",
            help="Number of days to forecast into the future"
        )
        # Update session state immediately
        st.session_state.analyze_forecast_horizon = forecast_horizon

        col1, col2 = st.columns([1, 2])

        with col1:
            forecast_button = st.button(
                "🚀 Generate Forecast",
                type="primary",
                help="Run all models and show consensus forecast",
            )

        with col2:
            if forecast_button:
                try:
                    _denorm_price = None  # safe default before any conditional that might skip assignment
                    with st.spinner("Running consensus forecast (all models)..."):
                        _tk_fc = str(ticker).strip().upper() or symbol
                        data = st.session_state.get("analyze_forecast_data")
                        _fc_ok = (
                            data is not None
                            and not getattr(data, "empty", True)
                            and str(
                                st.session_state.get(
                                    "analyze_forecast_data_symbol",
                                    "",
                                ),
                            ).strip().upper()
                            == _tk_fc
                            and len(data) >= 30
                        )
                        if not _fc_ok:
                            data = get_history(
                                _tk_fc,
                                period="1y",
                                interval="1d",
                            )
                        if (
                            data is None
                            or getattr(data, "empty", True)
                            or len(data) < 30
                        ):
                            raise RuntimeError(
                                f"Not enough daily history for {_tk_fc} to run "
                                "consensus (need at least 30 rows).",
                            )
                        data = data.copy()
                        # Normalize column case (yfinance uses Close, etc.)
                        if "close" in data.columns and "Close" not in data.columns:
                            data = data.rename(columns={"close": "Close"})
                        if "Close" not in data.columns:
                            data["Close"] = data.iloc[:, 0]
                        st.session_state["analyze_forecast_data"] = data
                        st.session_state["analyze_forecast_data_symbol"] = _tk_fc
                        st.session_state["analyze_symbol"] = _tk_fc
                        horizon = st.session_state.get("analyze_forecast_horizon", 7)
                        if not isinstance(data.index, pd.DatetimeIndex):
                            data.index = pd.to_datetime(data.index)
                        used_router = False
                        consensus = None
                        try:
                            from trading.models.forecast_router import (
                                ForecastRouter,
                                get_router_singleton,
                            )
                            import hashlib as _hashlib
                            import time as _time

                            _router = get_router_singleton()
                            _data_hash = _hashlib.md5(
                                str(data.index[-1]).encode()
                                + str(len(data)).encode()
                                + ticker.encode()
                            ).hexdigest()[:8]
                            _cons_key = f"consensus_{ticker}_{_data_hash}"
                            _cons_ts_key = f"consensus_ts_{ticker}_{_data_hash}"
                            _cached_cons = st.session_state.get(_cons_key)
                            _cached_cons_ts = st.session_state.get(_cons_ts_key, 0.0)
                            _cons_age = _time.time() - _cached_cons_ts

                            if _cached_cons is None or _cons_age > 600:
                                with st.spinner(
                                    "Running consensus forecast (all models)..."
                                ):
                                    consensus = _router.get_consensus_forecast(
                                        data,
                                        horizon=horizon,
                                        symbol=str(ticker).strip().upper()
                                        or None,
                                    )
                                st.session_state[_cons_key] = consensus
                                st.session_state[_cons_ts_key] = _time.time()
                            else:
                                consensus = _cached_cons
                                st.caption(
                                    "Using cached forecast (refreshes every 10 min)"
                                )
                            _used = consensus.get("models_used", []) if isinstance(consensus, dict) else []
                            _failed = consensus.get("models_failed", []) if isinstance(consensus, dict) else []
                            if _failed:
                                st.caption(
                                    f"Models available: {_used} | Unavailable: {_failed}"
                                )
                            if consensus and "error" not in consensus:
                                raw = consensus.get("consensus_forecast") or (consensus.get("consensus_price") and [consensus["consensus_price"]]) or []
                                forecast_values = np.asarray(raw, dtype="float64").ravel()
                                if forecast_values.size == 0 and consensus.get("price_targets"):
                                    pt = consensus["price_targets"]
                                    last_price = float(consensus.get("last_price") or 0)
                                    if pt and last_price:
                                        forecast_values = np.array([float(list(pt.values())[-1]) if pt else last_price])
                                    if forecast_values.size < horizon:
                                        forecast_values = np.resize(forecast_values, horizon)
                                if forecast_values.size < horizon:
                                    forecast_values = np.resize(forecast_values, horizon)
                                forecast_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=min(horizon, len(forecast_values)), freq="D")[:len(forecast_values)]
                                if hasattr(forecast_dates, "tz_localize"):
                                    try:
                                        forecast_dates = forecast_dates.tz_localize(None)
                                    except Exception:
                                        pass
                                st.session_state.current_forecast = pd.DataFrame({"forecast": forecast_values[:len(forecast_dates)]}, index=forecast_dates)
                                st.session_state.current_model = "Consensus"
                                st.session_state.current_forecast_result = {
                                    "forecast": np.asarray(consensus.get("consensus_forecast", [])).ravel().tolist(),
                                    "validation_mape": None,
                                    "in_sample_mape": None,
                                    "last_actual_price": consensus.get("last_price"),
                                    "confidence_label": consensus.get("conviction", "INSUFFICIENT"),
                                    "warnings": [],
                                    "lower_bound": consensus.get("lower_bound"),
                                    "upper_bound": consensus.get("upper_bound"),
                                }
                                st.session_state["forecast_debug"] = consensus.get("forecast_debug", {})
                                used_router = True
                                st.success("✅ Consensus forecast generated")
                                if st.session_state.get("forecast_debug"):
                                    with st.expander("🔍 Forecast debug (per-model std/mean/size)", expanded=False):
                                        st.json(st.session_state["forecast_debug"])
                        except Exception as _ce:
                            logger.debug("Consensus forecast failed: %s", _ce)
                            st.caption(f"Consensus error: {_ce}")
                            consensus = None
                        if not used_router:
                            _pt = consensus.get("price_targets") if isinstance(consensus, dict) else None
                            _failed = consensus.get("models_failed") if isinstance(consensus, dict) else None
                            if consensus and (_pt or _failed):
                                pt = _pt or {}
                                failed = _failed or []
                                n_ok = len(pt)
                                total = n_ok + len(failed)
                                st.caption(f"Partial consensus — {n_ok} of {total} models")
                            else:
                                st.error("Consensus forecast failed. Load data and try again, or use Tab 2 for a single-model forecast.")
                        # Quick Forecast now uses only the consensus router; single-model flows live in Tab 2.
                        # If consensus fails, we surface the above error and do not attempt any model initialization here.
                        if used_router:
                            # Store full forecast result for confidence intervals if available
                            forecast_result = st.session_state.get("current_forecast_result")
                            if forecast_result is not None:
                                # Postprocess forecast
                                try:
                                    from trading.forecasting.forecast_postprocessor import ForecastPostprocessor

                                    postprocessor = ForecastPostprocessor()

                                    # Extract forecast values (handles forecast, predictions, values, forecast_values, consensus_forecast)
                                    forecast_vals = _extract_forecast_values(forecast_result)
                                    if forecast_vals is not None:
                                        forecast_vals = forecast_vals.tolist() if hasattr(forecast_vals, 'tolist') else list(forecast_vals)
                                    else:
                                        forecast_vals = []

                                    # Postprocess forecast
                                    processed_forecast = postprocessor.process(
                                        forecast=forecast_vals,
                                        historical_data=data,
                                        apply_smoothing=True,
                                        remove_outliers=True,
                                        ensure_realistic_bounds=True
                                    )

                                    # Update forecast_result with processed version
                                    if isinstance(forecast_result, dict):
                                        forecast_result['forecast'] = processed_forecast['values']
                                        forecast_result['postprocessing_notes'] = processed_forecast.get('notes', [])
                                    else:
                                        forecast_result = processed_forecast['values']

                                    # Show what was done
                                    if processed_forecast.get('modifications'):
                                        with st.expander("⚙️ Forecast Postprocessing", expanded=False):
                                            st.write("**Modifications applied:**")
                                            for mod in processed_forecast['modifications']:
                                                st.write(f"• {mod}")

                                    # Update session state with processed forecast
                                    st.session_state.current_forecast_result = forecast_result
                                except ImportError as _e:
                                    logger.debug("Forecast postprocessor not available: %s", _e)
                                except Exception as e:
                                    logger.warning(f"Forecast postprocessing failed: {e}")

                            if 'model_log' in st.session_state and 'perf_logger' in st.session_state:
                                try:
                                    st.caption(
                                        "Training metrics available after "
                                        "running walk-forward validation "
                                        "on the Backtest page."
                                    )
                                except Exception as e:
                                    logger.warning(
                                        "Model performance logging failed: %s", e
                                    )

                            # Extract forecast values (robust for all result formats)
                            forecast_values = _extract_forecast_values(forecast_result)
                            if isinstance(forecast_result, dict):
                                forecast_dates = forecast_result.get('dates', pd.date_range(
                                    start=data.index[-1] + timedelta(days=1),
                                    periods=horizon,
                                    freq='D'
                                ))
                            else:
                                forecast_dates = pd.date_range(
                                    start=data.index[-1] + timedelta(days=1),
                                    periods=horizon,
                                    freq='D'
                                )
                            if forecast_values is not None:
                                forecast_values = np.asarray(forecast_values).ravel()

                            # Debug: Check what we got
                            if forecast_values is None or (hasattr(forecast_values, '__len__') and len(forecast_values) == 0):
                                st.warning(f"⚠️ Forecast returned empty values. Result type: {type(forecast_result)}")
                                if isinstance(forecast_result, dict):
                                    st.write("Forecast result keys:", list(forecast_result.keys()))
                                # Use last known price as fallback
                                last_price = data['Close'].iloc[-1] if 'Close' in data.columns else data.iloc[-1, 0]
                                forecast_values = np.full(horizon, float(last_price))

                            # Ensure forecast_values is array-like
                            if isinstance(forecast_values, (list, np.ndarray)):
                                forecast_values = np.array(forecast_values).flatten()
                                # Check for NaN/None values
                                if np.any(np.isnan(forecast_values)) or np.any(forecast_values == None):
                                    st.warning("⚠️ Forecast contains NaN/None values. Replacing with last known price.")
                                    last_price = float(data['Close'].iloc[-1] if 'Close' in data.columns else data.iloc[-1, 0])
                                    forecast_values = np.where(
                                        np.isnan(forecast_values) | (forecast_values == None),
                                        last_price,
                                        forecast_values
                                    )
                            else:
                                # Single value case
                                if forecast_values is None or (isinstance(forecast_values, float) and np.isnan(forecast_values)):
                                    last_price = float(data['Close'].iloc[-1] if 'Close' in data.columns else data.iloc[-1, 0])
                                    forecast_values = np.full(horizon, last_price)
                                else:
                                    forecast_values = np.array([float(forecast_values)] * horizon)
                            # Denormalize: model was trained on price/last_price
                            if _denorm_price and _denorm_price != 1.0:
                                forecast_values = np.asarray(forecast_values, dtype=float) * _denorm_price

                            # Create forecast DataFrame
                            forecast_df = pd.DataFrame({
                                'forecast': forecast_values
                            }, index=forecast_dates[:len(forecast_values)])

                            # Store in session state
                            st.session_state.current_forecast = forecast_df
                            selected_model = st.session_state.get("selected_model", "consensus")
                            st.session_state.current_model = selected_model
                            model = st.session_state.get("current_model_instance", None)
                            st.session_state.current_model_instance = model  # Store model instance for explainability

                            # Situational awareness: write to MemoryStore for Chat context (quality gate)
                            try:
                                from trading.memory import get_memory_store
                                from trading.memory.memory_store import MemoryType
                                _store = get_memory_store()
                                _vals = forecast_values if hasattr(forecast_values, "__len__") else [float(forecast_values)]
                                _first = _vals[0] if len(_vals) > 0 else None
                                _last = _vals[-1] if len(_vals) > 0 else None
                                try:
                                    _all_same = len(_vals) > 1 and (np.unique(np.asarray(_vals).flatten()).size <= 1)
                                except Exception:
                                    _all_same = False
                                _degenerate = _first is None or _last is None or _all_same
                                if _first is not None or _last is not None:
                                    _conf = None
                                    if isinstance(forecast_result, dict):
                                        if "lower_bound" in forecast_result and "upper_bound" in forecast_result:
                                            _conf = {"lower": forecast_result["lower_bound"], "upper": forecast_result["upper_bound"]}
                                    _store.add(
                                        MemoryType.LONG_TERM,
                                        namespace="forecasts",
                                        value={
                                            "symbol": st.session_state.get("analyze_symbol", ""),
                                            "model_name": selected_model,
                                            "horizon": horizon,
                                            "forecast_first": _first,
                                            "forecast_last": _last,
                                            "confidence": _conf,
                                            "timestamp": datetime.utcnow().isoformat(),
                                            **({"model_failed": True} if _degenerate else {}),
                                        },
                                        category="results",
                                    )
                            except Exception:
                                pass

                            st.success(f"✅ Forecast generated using {selected_model}")

                            # Natural Language Insights
                            st.markdown("---")
                            st.subheader("💬 Natural Language Insights")

                            if st.button("Generate Plain English Explanation", key="generate_nlg_insights"):
                                try:
                                    from nlp.natural_language_insights import NaturalLanguageInsights

                                    nlg = NaturalLanguageInsights()

                                    # Prepare forecast result for insights
                                    forecast_for_insights = {
                                        'forecast': forecast_df['forecast'].values.tolist() if hasattr(forecast_df['forecast'].values, 'tolist') else list(forecast_df['forecast'].values),
                                        'dates': forecast_df.index.tolist() if hasattr(forecast_df.index, 'tolist') else list(forecast_df.index),
                                        'model_type': selected_model,
                                        'symbol': st.session_state.analyze_symbol
                                    }

                                    # Add confidence intervals if available
                                    if isinstance(forecast_result, dict):
                                        if 'lower_bound' in forecast_result:
                                            forecast_for_insights['lower_bound'] = forecast_result['lower_bound']
                                        if 'upper_bound' in forecast_result:
                                            forecast_for_insights['upper_bound'] = forecast_result['upper_bound']

                                    # Generate explanation
                                    insights = nlg.generate_forecast_insights(
                                        forecast=forecast_for_insights,
                                        historical_data=data,
                                        model_type=selected_model,
                                        symbol=st.session_state.analyze_symbol
                                    )

                                    # Display insights
                                    st.info(insights['summary'])

                                    with st.expander("📊 Detailed Analysis", expanded=False):
                                        st.write("**Trend Analysis:**")
                                        st.write(insights['trend_analysis'])

                                        st.write("**Key Factors:**")
                                        for factor in insights['key_factors']:
                                            st.write(f"• {factor}")

                                        st.write("**Confidence Assessment:**")
                                        st.write(insights['confidence_explanation'])

                                        st.write("**Recommendations:**")
                                        for rec in insights['recommendations']:
                                            st.write(f"• {rec}")

                                except ImportError:
                                    st.error("Natural Language Insights not available")
                                except Exception as e:
                                    st.error(f"Error generating insights: {e}")
                                    import traceback
                                    if st.checkbox("Show technical details", key="nl_insights_trace"):
                                        st.code(traceback.format_exc())

                            # AI Commentary Service
                            if 'commentary_service' in st.session_state:
                                st.markdown("---")
                                st.subheader("💬 AI Commentary")

                                if st.button("Generate Commentary", key="generate_forecast_commentary"):
                                    commentary_service = st.session_state.commentary_service

                                    with st.spinner("Generating commentary..."):
                                        try:
                                            # Prepare forecast result for commentary
                                            forecast_for_commentary = {
                                                'forecast': forecast_df['forecast'].values.tolist() if hasattr(forecast_df['forecast'].values, 'tolist') else list(forecast_df['forecast'].values),
                                                'dates': forecast_df.index.tolist() if hasattr(forecast_df.index, 'tolist') else list(forecast_df.index),
                                                'model_type': selected_model,
                                                'symbol': st.session_state.analyze_symbol
                                            }

                                            # Add confidence intervals if available
                                            if isinstance(forecast_result, dict):
                                                if 'lower_bound' in forecast_result:
                                                    forecast_for_commentary['lower_bound'] = forecast_result['lower_bound']
                                                if 'upper_bound' in forecast_result:
                                                    forecast_for_commentary['upper_bound'] = forecast_result['upper_bound']

                                            commentary = commentary_service.generate_forecast_commentary(
                                                symbol=st.session_state.analyze_symbol,
                                                forecast_result=forecast_for_commentary,
                                                model_type=selected_model,
                                                historical_data=data
                                            )

                                            st.info(commentary.get('summary', 'Commentary generated'))

                                            with st.expander("📊 Detailed Analysis", expanded=False):
                                                if 'trend_analysis' in commentary:
                                                    st.write("**Trend Analysis:**")
                                                    st.write(commentary['trend_analysis'])

                                                if 'insights' in commentary and commentary['insights']:
                                                    st.write("**Key Insights:**")
                                                    for insight in commentary['insights']:
                                                        st.write(f"• {insight}")

                                                if 'warnings' in commentary and commentary['warnings']:
                                                    st.write("**Warnings:**")
                                                    for warning in commentary['warnings']:
                                                        st.warning(f"⚠️ {warning}")
                                        except Exception as e:
                                            st.error(f"Error generating commentary: {e}")
                                            import traceback
                                            st.code(traceback.format_exc())

                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.info("Try adjusting the date range or selecting a different model.")

            # Display consensus view (cached only; computation runs on Generate Forecast)
            try:
                hist_data_cons = st.session_state.get("analyze_forecast_data")
                if hist_data_cons is not None and len(hist_data_cons) >= 2:
                    horizon = st.session_state.get("analyze_forecast_horizon", 7)
                    import hashlib as _hashlib

                    try:
                        _data_hash = _hashlib.md5(
                            str(hist_data_cons.index[-1]).encode()
                            + str(len(hist_data_cons)).encode()
                            + ticker.encode()
                        ).hexdigest()[:8]
                        _cons_key = f"consensus_{ticker}_{_data_hash}"
                        consensus = st.session_state.get(_cons_key)
                        if consensus is None:
                            st.info(
                                "Run **Generate Forecast** "
                                "above to see model consensus.",
                                icon="🎯",
                            )
                        else:
                            st.caption(
                                "Using cached forecast (refreshes every 10 min)"
                            )
                    except Exception as _ce:
                        st.caption(f"Consensus error: {_ce}")
                        consensus = {"error": str(_ce)}
                    if consensus is not None:
                        _used = consensus.get("models_used", []) if isinstance(consensus, dict) else []
                        _failed = consensus.get("models_failed", []) if isinstance(consensus, dict) else []
                        if _failed:
                            st.caption(
                                f"Models available: {_used} | Unavailable: {_failed}"
                            )
                        if "error" not in consensus:
                            raw = consensus.get("consensus_forecast") or (
                                consensus.get("consensus_price")
                                and [consensus["consensus_price"]]
                            ) or []
                            if not raw and (
                                consensus.get("price_targets") or consensus.get("models_failed")
                            ):
                                pt = consensus.get("price_targets") or {}
                                failed = consensus.get("models_failed") or []
                                n_ok, total = len(pt), len(pt) + len(failed)
                                st.caption(f"Partial consensus — {n_ok} of {total} models")

                            with st.expander("🎯 Model Consensus", expanded=True):
                                direction = consensus.get("direction", "NEUTRAL")
                                conviction = consensus.get("conviction", "INSUFFICIENT")
                                last_price = float(consensus.get("last_price") or 0.0)
                                consensus_price = float(
                                    consensus.get("consensus_price")
                                    or consensus.get("consensus_forecast", [0.0])[-1]
                                )
                                price_targets = consensus.get("price_targets") or {}
                                models_failed = consensus.get("models_failed") or []

                                _ff_hint = ""
                                try:
                                    from trading.utils.forecast_formatter import (
                                        ForecastFormatter,
                                    )

                                    _ff = ForecastFormatter()
                                    _pv = pd.DataFrame(
                                        {"close": [last_price, consensus_price]}
                                    )
                                    _pv.index = pd.DatetimeIndex(
                                        pd.date_range(
                                            end=pd.Timestamp.utcnow(),
                                            periods=2,
                                            freq="D",
                                        )
                                    )
                                    _val = _ff.validate_forecast_format(_pv)
                                    if _val.get("warnings"):
                                        _ff_hint = " · ".join(
                                            _val["warnings"][:3]
                                        )
                                except Exception:
                                    pass
                                if _ff_hint:
                                    st.caption(f"Forecast format: {_ff_hint}")

                                # Row 1: direction, conviction, consensus price
                                col1, col2, col3 = st.columns(3)
                                direction_emoji = (
                                    "📈"
                                    if direction == "BULLISH"
                                    else "📉"
                                    if direction == "BEARISH"
                                    else "➡️"
                                )
                                with col1:
                                    col1.metric(
                                        "Consensus Direction",
                                        f"{direction_emoji} {direction}",
                                    )
                                with col2:
                                    col2.metric("Conviction", conviction)
                                with col3:
                                    delta_pct = (
                                        (consensus_price / last_price - 1.0) * 100.0
                                        if last_price
                                        else 0.0
                                    )
                                    col3.metric(
                                        f"Consensus Price (Day {horizon})",
                                        f"${consensus_price:.2f}",
                                        delta=f"{delta_pct:+.2f}%",
                                    )

                                # Signal synthesis: reconcile consensus, AI Score, and outlier exclusions
                                _consensus_dir = direction
                                _conv = conviction
                                # Try canonical key first, then dynamic key, then any ai_score key
                                _symbol = st.session_state.get("analyze_symbol", "")
                                _ai_result = (
                                    st.session_state.get("ai_score_result")
                                    or {}
                                )
                                _excluded = consensus.get("models_excluded", {}) or {}

                                try:
                                    _ai_val_f = float(
                                        _ai_result.get("weighted_score")
                                        or _ai_result.get("overall_score")
                                        or _ai_result.get("model_score")
                                        or 5.0)
                                except Exception:
                                    _ai_val_f = 5.0
                                if _ai_val_f == 5.0 and _ai_result:
                                    import logging as _lg
                                    _lg.getLogger(__name__).warning(
                                        "AI score result found but score keys missing: %s",
                                        list(_ai_result.keys()))

                                _signals = []
                                if _consensus_dir == "BULLISH":
                                    _signals.append("models lean bullish")
                                elif _consensus_dir == "BEARISH":
                                    _signals.append("models lean bearish")
                                else:
                                    _signals.append("models are mixed")

                                if _ai_val_f >= 7.0:
                                    _signals.append("AI Score is strong")
                                elif _ai_val_f >= 5.5:
                                    _signals.append("AI Score is neutral")
                                else:
                                    _signals.append("AI Score is weak")

                                _excl_str = ""
                                if _excluded:
                                    _excl_names = ", ".join(
                                        f"{k} ({v.get('deviation_std', 0)}σ)"
                                        for k, v in _excluded.items()
                                    )
                                    _excl_str = f" ⚠️ Outlier excluded: {_excl_names}."

                                _synthesis = (
                                    f"**Signal synthesis:** "
                                    f"{' · '.join(_signals)}."
                                    f"{_excl_str} "
                                    "Consensus and Monte Carlo measure different things — "
                                    "point forecast vs probability distribution. "
                                    "Use Monte Carlo for risk sizing and consensus for direction bias."
                                )
                                st.info(_synthesis)

                                # Row 2: per-model price targets table
                                if price_targets:
                                    targets_df = pd.DataFrame(
                                        [
                                            {
                                                "Model": k,
                                                "Day 7 Price": f"${v:.2f}",
                                                "vs Current": (
                                                    f"{((v/last_price)-1)*100:.1f}%"
                                                    if last_price
                                                    else "N/A"
                                                ),
                                            }
                                            for k, v in price_targets.items()
                                        ]
                                    )
                                    st.dataframe(
                                        normalize_for_display(targets_df),
                                        width='stretch',
                                        hide_index=True,
                                    )

                                # Row 3: failed models (optional)
                                if models_failed:
                                    with st.expander(
                                        f"⚠️ {len(models_failed)} model(s) excluded",
                                        expanded=False,
                                    ):
                                        for f in models_failed:
                                            st.caption(f"• {f}")
                        else:
                            st.warning("Consensus forecast temporarily unavailable.")
            except Exception:
                try:
                    if isinstance(consensus, dict) and consensus.get("error"):
                        st.warning("Consensus forecast temporarily unavailable.")
                except NameError:
                    st.warning("Consensus forecast temporarily unavailable.")

            # Display forecast using UI components
            if st.session_state.get('current_forecast') is not None:
                st.markdown(f"**Forecast using {st.session_state.current_model}**")

                try:
                    from trading.ui.forecast_components import render_forecast_results, render_confidence_metrics

                    hist_data = st.session_state.get("analyze_forecast_data")
                    forecast_df = st.session_state.current_forecast
                    forecast_result = st.session_state.get('current_forecast_result', {})

                    # Prepare forecast data for component
                    forecast_data = {
                        'dates': forecast_df.index.tolist() if hasattr(forecast_df.index, 'tolist') else list(forecast_df.index),
                        'forecast': forecast_df['forecast'].values.tolist() if hasattr(forecast_df['forecast'].values, 'tolist') else list(forecast_df['forecast'].values),
                        'model_name': st.session_state.current_model
                    }
                    if isinstance(forecast_result, dict):
                        if 'lower_bound' in forecast_result:
                            forecast_data['lower_bound'] = forecast_result['lower_bound']
                        if 'upper_bound' in forecast_result:
                            forecast_data['upper_bound'] = forecast_result['upper_bound']
                        if 'confidence' in forecast_result:
                            forecast_data['confidence'] = forecast_result['confidence']
                        if forecast_result.get('validation_mape') is not None:
                            forecast_data['validation_mape'] = forecast_result['validation_mape']
                        if forecast_result.get('confidence_label'):
                            forecast_data['confidence_label'] = forecast_result['confidence_label']
                        if forecast_result.get('last_actual_price') is not None:
                            forecast_data['last_actual_price'] = forecast_result['last_actual_price']
                    if forecast_data.get('last_actual_price') is None and hist_data is not None and len(hist_data) > 0:
                        close_col = 'close' if 'close' in hist_data.columns else (hist_data.columns[0] if len(hist_data.columns) > 0 else None)
                        if close_col:
                            forecast_data['last_actual_price'] = float(hist_data[close_col].iloc[-1])

                    # Show validation MAPE, confidence label, last actual vs first forecast
                    if forecast_data.get('validation_mape') is not None or forecast_data.get('last_actual_price') is not None:
                        with st.expander("📐 Accuracy & continuity", expanded=True):
                            mape_val = forecast_data.get('validation_mape') or forecast_data.get('in_sample_mape')
                            if mape_val is not None:
                                st.metric("Validation / in-sample MAPE", f"{mape_val:.2f}%")
                            if forecast_data.get('confidence_label'):
                                st.metric("Confidence (based on MAPE)", forecast_data['confidence_label'])
                            if mape_val is not None:
                                confidence_pct = max(0.0, min(100.0, 100.0 - float(mape_val)))
                                st.metric("Confidence score", f"{confidence_pct:.0f}%")
                            last_act = forecast_data.get('last_actual_price')
                            fvals = forecast_data.get('forecast', [])
                            first_fc = fvals[0] if fvals else None
                            if last_act is not None and first_fc is not None:
                                st.write("**Continuity:** Last actual price **${:.2f}** → First forecast **${:.2f}**".format(float(last_act), float(first_fc)))

                    # Render forecast results
                    render_forecast_results(
                        forecast=forecast_data,
                        historical_data=hist_data,
                        symbol=st.session_state.analyze_symbol,
                        show_chart=True,
                        show_table=True
                    )

                    # Model agreement bands (not calibrated CIs)
                    if isinstance(forecast_result, dict) and ('lower_bound' in forecast_result or 'confidence' in forecast_result):
                        render_confidence_metrics(forecast_data)
                    # News sentiment overlay below forecast results
                    try:
                        _ns = _news_sentiment_score(ticker)
                        if _ns >= 7.5:
                            _ns_label = "HOT"
                            _ns_text = "Breaking positive news may accelerate move"
                        elif _ns >= 6.0:
                            _ns_label = "POS"
                            _ns_text = "Positive news sentiment supports forecast"
                        elif _ns <= 3.0:
                            _ns_label = "NEG"
                            _ns_text = "Negative news may create headwinds"
                        else:
                            _ns_label = "NEU"
                            _ns_text = "News sentiment is neutral"
                        _news_ic = sentiment_icon_for_label(_ns_label)
                        st.markdown(
                            f"{_news_ic} **NEWS {_ns_label}** — {_ns_text} "
                            f"_(score {_ns:.1f}/10)_"
                        )
                    except Exception as _e:
                        logger.warning(
                            "Analyze: news sentiment overlay failed: %s", _e
                        )

                except ImportError:
                    # Fallback to original display code
                    from utils.plotting_helper import create_forecast_chart

                    hist_data = st.session_state.get("analyze_forecast_data")
                    forecast_df = st.session_state.current_forecast
                    forecast_result = st.session_state.get('current_forecast_result', {})

                    model_agreement_bands = None
                    if isinstance(forecast_result, dict) and 'lower_bound' in forecast_result and 'upper_bound' in forecast_result:
                        model_agreement_bands = {
                            'lower': forecast_result['lower_bound'],
                            'upper': forecast_result['upper_bound']
                        }

                    fig = create_forecast_chart(
                        historical=hist_data,
                        forecast=forecast_df['forecast'].values,
                        forecast_dates=forecast_df.index,
                        confidence_intervals=model_agreement_bands,
                        title=f"{st.session_state.analyze_symbol} - Historical & Forecast"
                    )
                    st.plotly_chart(fig, width='stretch')

                    # Forecast table (format dates as YYYY-MM-DD, no timezone)
                    st.markdown("**Forecast Values:**")
                    display_df = forecast_df.copy()
                    try:
                        _idx = pd.to_datetime(display_df.index)
                        if hasattr(_idx, "tz") and _idx.tz is not None:
                            _idx = _idx.tz_localize(None)
                        display_df.index = _idx.strftime("%Y-%m-%d")
                    except Exception:
                        pass
                    display_df["forecast"] = display_df["forecast"].apply(
                        lambda x: f"${x:.2f}"
                        if isinstance(x, (int, float)) and not pd.isna(x)
                        else "—"
                    )
                    display_df.rename(columns={"forecast": "Forecast"}, inplace=True)

                    # Add confidence bounds if available
                    _lb = None
                    _ub = None
                    _forecast_result = st.session_state.get("current_forecast_result")
                    if isinstance(_forecast_result, dict):
                        _lb = _forecast_result.get("lower_bound")
                        _ub = _forecast_result.get("upper_bound")
                    _n = len(display_df)
                    display_df["Lower Bound"] = (
                        [f"${v:.2f}" for v in _lb[:_n]]
                        if _lb and len(_lb) >= _n
                        else ["—"] * _n
                    )
                    display_df["Upper Bound"] = (
                        [f"${v:.2f}" for v in _ub[:_n]]
                        if _ub and len(_ub) >= _n
                        else ["—"] * _n
                    )
                    st.dataframe(normalize_for_display(display_df), width="stretch")

                # Download button
                csv = forecast_df.to_csv()
                st.download_button(
                    label="📥 Download Forecast CSV",
                    data=csv,
                    file_name=f"{st.session_state.analyze_symbol}_forecast.csv",
                    mime="text/csv"
                )

                # Add explainability section
                st.markdown("---")
                st.subheader("🔍 Model Explainability")

                with st.expander("View Feature Importance & Explanations", expanded=False):
                    try:
                        from trading.models.forecast_explainability import (
                            IntelligentForecastExplainability,
                        )
                        ForecastExplainability = IntelligentForecastExplainability

                        # Display cached explanation so it persists across reruns
                        if st.session_state.get('forecast_explanation') is not None:
                            cached = st.session_state.forecast_explanation
                            if cached.get('success') and cached.get('explanation'):
                                expl_obj = cached['explanation']
                                if hasattr(expl_obj, 'feature_importance') and expl_obj.feature_importance:
                                    st.write("**Feature Importance:**")
                                    importance_data = expl_obj.feature_importance
                                    if isinstance(importance_data, dict):
                                        importance_df = pd.DataFrame(
                                            list(importance_data.items()),
                                            columns=['Feature', 'Importance']
                                        ).sort_values('Importance', ascending=False)
                                        import plotly.express as px
                                        fig = px.bar(
                                            importance_df, x='Importance', y='Feature',
                                            orientation='h', title='Feature Importance (SHAP values)'
                                        )
                                        st.plotly_chart(fig)
                                if hasattr(expl_obj, 'explanation_text') and expl_obj.explanation_text:
                                    st.write("**Explanation:**")
                                    st.write(expl_obj.explanation_text)
                            if st.button("Clear explanation", key="clear_explanation_quick"):
                                st.session_state.forecast_explanation = None
                                st.rerun()

                        if st.button("Generate Explanation", key="generate_explanation"):
                            with st.spinner("Analyzing model predictions..."):
                                try:
                                    explainer = ForecastExplainability()
                                    model = st.session_state.get('current_model_instance')
                                    forecast_result = st.session_state.get('current_forecast_result', {})

                                    _fv = _extract_forecast_values(forecast_result)
                                    if _fv is not None and _fv.size > 0:
                                        forecast_value = float(np.asarray(_fv).flat[0])
                                    elif isinstance(forecast_result, (int, float)):
                                        forecast_value = float(forecast_result)
                                    else:
                                        forecast_value = float(data['Close'].iloc[-1])

                                    features = data.copy()
                                    target_history = features['Close'] if 'Close' in features.columns else (features['close'] if 'close' in features.columns else features.iloc[:, 0])

                                    explanation = None
                                    try:
                                        if hasattr(explainer, 'generate_forecast_explanation'):
                                            explanation = explainer.generate_forecast_explanation(
                                                model=model,
                                                X=features,
                                                forecast_value=forecast_value,
                                                forecast_horizon=st.session_state.analyze_forecast_horizon,
                                                actual_values=target_history,
                                            )
                                            st.session_state.forecast_explanation = {"success": True, "explanation": explanation}
                                        elif hasattr(explainer, 'explain_forecast'):
                                            explanation = explainer.explain_forecast(
                                                forecast_id=f"forecast_{st.session_state.analyze_symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                                symbol=st.session_state.analyze_symbol,
                                                forecast_value=forecast_value,
                                                model=model,
                                                features=features,
                                                target_history=target_history,
                                                horizon=st.session_state.analyze_forecast_horizon,
                                            )
                                            st.session_state.forecast_explanation = explanation
                                        else:
                                            st.caption("Model explainability unavailable for this model type.")
                                        if explanation is not None:
                                            st.success("✅ Explanation generated")
                                            st.rerun()
                                    except Exception:
                                        st.caption("Model explainability unavailable for this model type.")
                                except Exception as e:
                                    st.error(f"Error generating explanation: {e}")
                                    import traceback
                                    st.code(traceback.format_exc())

                    except ImportError as e:
                        st.caption(f"Feature unavailable: Forecast explainability requires SHAP (pip install shap). Details: {e}")
                    except Exception as e:
                        st.caption(f"Feature unavailable: {e}")

                # AI Commentary section
                st.markdown("---")
                st.subheader("🤖 AI Market Commentary")

                if st.button("Generate AI Commentary", key="generate_commentary"):
                    try:
                        from agents.llm.agent import get_prompt_agent
                        agent = get_prompt_agent()
                        model_name = st.session_state.get("current_model", "Unknown")
                        symbol = st.session_state.get("analyze_symbol", "Unknown")
                        last_price = float(data["Close"].iloc[-1]) if "Close" in data.columns else (float(data["close"].iloc[-1]) if "close" in data.columns else 0.0)
                        fcast = _extract_forecast_values(forecast_result)
                        fcast = fcast if fcast is not None and fcast.size > 0 else np.array([])
                        forecast_mean = float(np.mean(fcast)) if fcast.size > 0 else last_price
                        pct_change = ((forecast_mean - last_price) / last_price * 100) if last_price else 0.0
                        if agent:
                            with st.spinner("AI is analyzing the forecast..."):
                                commentary_prompt = (
                                    f"Provide 3 sentences of market commentary on the {model_name} forecast for {symbol}: "
                                    f"last price ${last_price:.2f}, 7-day forecast ${forecast_mean:.2f} ({pct_change:+.1f}%). Be specific and concise."
                                )
                                response = agent.process_prompt(commentary_prompt)
                                commentary = response.message if hasattr(response, "message") else str(response)
                                st.write(commentary)
                        else:
                            st.info("AI commentary requires a valid API key in .env")
                    except Exception as e:
                        st.info(f"AI commentary unavailable: {e}")
    except Exception as e:
        st.caption(f"Tab unavailable: {e}")
