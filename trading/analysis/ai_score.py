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
import copy
import logging
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from trading.data.price_cache import get_history as _pc_get_history
from trading.utils.credential_placeholders import is_placeholder_credential
from trading.utils.safe_math import safe_rsi

logger = logging.getLogger(__name__)

_MACRO_FACTORS_INSTANCE = None

# Order preserved for signal_completeness score denominator
SIGNAL_SOURCES = [
    "technical",
    "momentum",
    "options_flow",
    "chart_patterns",
    "social_sentiment",
    "macro_factors",
    "sector_rotation",
    "earnings_calendar",
    "earnings_quality",
    "insider_flow",
    "analyst_signals",
    "congressional_trading",
    "sec_edgar",
    "institutional_ownership",
    "dark_pool",
    "factor_model",
    "ml_score",
]

# Settings → Research → Score weighting style (must match pages/7_Settings.py)
SCORING_STYLE_WEIGHTS = {
    "Balanced (default)": {
        "technical": 0.30,
        "momentum": 0.35,
        "sentiment": 0.20,
        "fundamental": 0.15,
    },
    "Momentum-heavy": {
        "technical": 0.20,
        "momentum": 0.50,
        "sentiment": 0.15,
        "fundamental": 0.15,
    },
    "Technical-heavy": {
        "technical": 0.50,
        "momentum": 0.25,
        "sentiment": 0.15,
        "fundamental": 0.10,
    },
    "Fundamental-heavy": {
        "technical": 0.15,
        "momentum": 0.20,
        "sentiment": 0.20,
        "fundamental": 0.45,
    },
}


def _resolve_style_weights(
    scoring_style: Optional[str] = None,
) -> Dict[str, float]:
    """Static dimension weights for IC blend and composite overall_score."""
    _balanced = SCORING_STYLE_WEIGHTS["Balanced (default)"]
    _key = (scoring_style or "").strip()
    if _key and _key in SCORING_STYLE_WEIGHTS:
        return dict(SCORING_STYLE_WEIGHTS[_key])
    if _key:
        for _label, _w in SCORING_STYLE_WEIGHTS.items():
            if _label.split()[0] in _key or _key in _label:
                return dict(_w)
        return dict(_balanced)
    try:
        from config.user_store import load_user_preferences
        from utils.session_utils import get_stable_user_id

        _uid = None
        try:
            _uid = st.session_state.get("evolve_session_id")
        except Exception:
            pass
        if not _uid:
            _uid = get_stable_user_id()
        prefs = load_user_preferences(_uid) or {}
        sk = str(prefs.get("scoring_style", "Balanced (default)")).strip()
        return dict(SCORING_STYLE_WEIGHTS.get(sk, _balanced))
    except Exception:
        return dict(_balanced)


def _get_scoring_weights() -> Dict[str, float]:
    """Dimension weights from Settings → scoring_style (prefs path)."""
    return _resolve_style_weights(None)


def _compute_ic_weights(
    symbol: str,
    scoring_style: Optional[str] = None,
    regime_label: str = "",
    vix_level: float = 0.0,
) -> Dict[str, float]:
    """
    Computes IC-derived dimension weights
    for a specific symbol (or global
    pool if per-symbol data is sparse).

    Falls back to static weights when
    insufficient data exists.

    IC-to-weight conversion:
    1. Compute Spearman IC per dimension
    2. Clip ICs to [-1, 1]
    3. Shift to positive: ic + 1
    4. Softmax normalize to sum=1
    5. Blend 60% IC-derived +
       40% user style weights
    """
    try:
        _static = _resolve_style_weights(scoring_style)
    except Exception:
        _static = dict(SCORING_STYLE_WEIGHTS["Balanced (default)"])

    try:
        from scipy.stats import spearmanr

        import numpy as np
        import pandas as pd

        from trading.analysis.signal_score_store import (
            get_dimension_scores_and_returns,
            get_global_dimension_scores,
        )

        sym = str(symbol or "").strip().upper()
        _data = None
        if sym:
            _data = get_dimension_scores_and_returns(sym, min_rows=30)

        if _data is None:
            _global = get_global_dimension_scores(min_rows=100)
            if _global is None:
                _w = dict(_static)
                try:
                    if regime_label:
                        _w = _apply_regime_tilt(
                            _w,
                            regime_label,
                            vix_level,
                        )
                except Exception:
                    pass
                _blend_total = sum(_w.values())
                return {
                    d: round(_w[d] / _blend_total, 4)
                    for d in _w
                }
            _rows = _global["data"]
        else:
            _rows = _data["data"]

        _df = pd.DataFrame(_rows)
        _dims = [
            "technical",
            "momentum",
            "sentiment",
            "fundamental",
        ]
        _ics: Dict[str, float] = {}
        for dim in _dims:
            if dim not in _df.columns:
                _ics[dim] = 0.0
                continue
            _valid = _df[[dim, "return_7d"]].dropna()
            if len(_valid) < 10:
                _ics[dim] = 0.0
                continue
            _ic, _ = spearmanr(
                _valid[dim].values,
                _valid["return_7d"].values,
            )
            _ics[dim] = float(_ic if np.isfinite(_ic) else 0.0)

        _shifted = {d: max(0.01, _ics[d] + 1.0) for d in _dims}
        _total = sum(_shifted.values())
        _ic_weights = {d: _shifted[d] / _total for d in _dims}

        _blended = {
            d: (0.60 * _ic_weights[d] + 0.40 * _static[d])
            for d in _dims
        }

        try:
            if regime_label:
                _blended = _apply_regime_tilt(
                    _blended,
                    regime_label,
                    vix_level,
                )
        except Exception:
            pass
        _blend_total = sum(_blended.values())
        return {
            d: round(
                _blended[d] / _blend_total,
                4,
            )
            for d in _dims
        }

    except Exception as e:
        logger.debug("IC weight computation failed: %s", e)
        try:
            _tilted = _apply_regime_tilt(
                dict(_static),
                regime_label,
                vix_level,
            )
            return _tilted
        except Exception:
            return _static


def _apply_regime_tilt(
    weights: Dict[str, float],
    regime_label: str,
    vix_level: float,
) -> Dict[str, float]:
    """
    Applies regime-conditional tilt
    to base/IC-derived weights.

    Academic basis: in risk-off regimes,
    quality/fundamental factors
    outperform. In risk-on regimes,
    momentum factors outperform.
    (Asness, Moskowitz & Pedersen 2013;
    Fama & French 2015)

    Regime labels from MacroFactors:
      RISK_ON     → tilt toward momentum
      RISK_OFF    → tilt toward fundamental
      HIGH_VOL    → tilt toward fundamental
                    + reduce momentum
      LOW_VOL     → tilt toward momentum
                    + reduce fundamental
      NEUTRAL     → no tilt

    Each paired tilt (e.g. fundamental
    vs momentum) uses at most 0.08
    absolute weight shift in one step;
    combined adjustments on the same
    pair are merged so the cap is not
    exceeded cumulatively.
    """
    _w = copy.deepcopy(weights)
    _label = (regime_label or "").upper()

    # Maximum tilt per dimension
    _MAX_TILT = 0.08

    def _tilt(
        w: Dict[str, float],
        dim_up: str,
        dim_down: str,
        amount: float,
    ) -> None:
        _adj = min(amount, _MAX_TILT)
        _up_new = min(
            0.70,
            w.get(dim_up, 0) + _adj,
        )
        _dn_new = max(
            0.05,
            w.get(dim_down, 0) - _adj,
        )
        w[dim_up] = _up_new
        w[dim_down] = _dn_new

    if _label in ("RISK_ON", "LOW_VOL"):
        _tilt(_w, "momentum", "fundamental", 0.05)
        if vix_level > 0:
            _tilt_amt = min(
                0.03,
                max(0, (20 - vix_level) / 100),
            )
            _tilt(_w, "technical", "sentiment", _tilt_amt)

    elif _label in (
        "RISK_OFF",
        "HIGH_VOL",
        "HIGH",
        "STRESS",
    ):
        _extra = 0.0
        if vix_level > 25:
            _extra = min(
                0.04,
                (vix_level - 25) / 100,
            )
        _fm_amount = min(0.06 + _extra, _MAX_TILT)
        _tilt(_w, "fundamental", "momentum", _fm_amount)

    elif _label == "STAGFLATION":
        _tilt(_w, "fundamental", "momentum", 0.05)
        _tilt(_w, "technical", "sentiment", 0.03)

    _total = sum(_w.values())
    if _total > 0:
        _w = {
            d: round(v / _total, 4)
            for d, v in _w.items()
        }

    return _w


def _persist_ai_score_result(
    symbol: str, result: Dict[str, Any]
) -> None:
    try:
        from trading.analysis.signal_score_store import (
            init_score_db,
            record_score,
        )

        init_score_db()
        if result.get("error") is None:
            record_score(
                symbol=symbol,
                technical=float(result.get("technical_score", 5.0) or 5.0),
                momentum=float(result.get("momentum_score", 5.0) or 5.0),
                sentiment=float(result.get("sentiment_score", 5.0) or 5.0),
                fundamental=float(result.get("fundamental_score", 5.0) or 5.0),
                overall=float(result.get("overall_score", 5.0) or 5.0),
                price_at_score=float(result.get("last_price", 0.0) or 0.0),
            )
    except Exception:
        pass


def _has_external_api_keys() -> bool:
    """True only if at least one paid/external API key is configured."""
    import os

    keys = [
        "TWITTER_API_KEY",
        "FRED_API_KEY",
        "TRADIER_TOKEN",
        "NEWS_API_KEY",
        "REDDIT_CLIENT_ID",
        "REDDIT_CLIENT_SECRET",
    ]
    try:
        _ss = st.session_state
    except Exception:
        _ss = None

    for key in keys:
        env_v = (os.environ.get(key, "") or "").strip()
        sess_v = ""
        if _ss is not None:
            try:
                sess_v = str(
                    _ss.get(f"user_key_{key}", "") or ""
                ).strip()
            except Exception:
                sess_v = ""
        val = env_v or sess_v
        if not val:
            continue
        if is_placeholder_credential(val):
            continue
        return True
    return False
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

SECTOR_VALUATION_MEDIANS = {
    "Technology": {
        "pe": 28,
        "forward_pe": 24,
        "pb": 8,
        "ev_ebitda": 22,
    },
    "Healthcare": {
        "pe": 22,
        "forward_pe": 19,
        "pb": 4,
        "ev_ebitda": 16,
    },
    "Financials": {
        "pe": 14,
        "forward_pe": 12,
        "pb": 1.5,
        "ev_ebitda": 12,
    },
    "Consumer Discretionary": {
        "pe": 24,
        "forward_pe": 20,
        "pb": 5,
        "ev_ebitda": 18,
    },
    "Consumer Staples": {
        "pe": 20,
        "forward_pe": 18,
        "pb": 4,
        "ev_ebitda": 14,
    },
    "Industrials": {
        "pe": 20,
        "forward_pe": 17,
        "pb": 3.5,
        "ev_ebitda": 14,
    },
    "Energy": {
        "pe": 12,
        "forward_pe": 10,
        "pb": 2,
        "ev_ebitda": 8,
    },
    "Materials": {
        "pe": 16,
        "forward_pe": 14,
        "pb": 2.5,
        "ev_ebitda": 10,
    },
    "Real Estate": {
        "pe": 35,
        "forward_pe": 28,
        "pb": 2,
        "ev_ebitda": 20,
    },
    "Utilities": {
        "pe": 18,
        "forward_pe": 16,
        "pb": 1.8,
        "ev_ebitda": 12,
    },
    "Communication Services": {
        "pe": 20,
        "forward_pe": 17,
        "pb": 3,
        "ev_ebitda": 14,
    },
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


AI_SCORE_PARALLEL_TIMEOUT = 15


def _fetch_options_flow_safe(symbol: str) -> Dict[str, Any]:
    try:
        from trading.data.options_flow import get_options_flow

        return get_options_flow(symbol)
    except Exception as e:
        logger.debug("options_flow AI score fetch: %s", e)
        return {}


def _fetch_insider_flow_safe(symbol: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "flow": {},
        "insider_cluster": None,
    }
    try:
        from trading.data.insider_flow import get_insider_flow

        out["flow"] = get_insider_flow(symbol)
    except Exception as e:
        logger.debug("insider_flow AI score fetch: %s", e)
    try:
        from trading.data.insider_flow import get_insider_cluster_signal

        out["insider_cluster"] = get_insider_cluster_signal(symbol)
    except Exception as _ce:
        logger.debug("insider_cluster fetch: %s", _ce)
    return out


def _fetch_social_sentiment_safe(symbol: str) -> Optional[Dict[str, Any]]:
    try:
        from trading.data.social_sentiment import get_social_sentiment

        return get_social_sentiment(symbol)
    except Exception as e:
        logger.debug("social_sentiment AI score fetch: %s", e)
        return None


def _fetch_short_and_sec_safe(symbol: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "short": None,
        "sec": None,
        "dark_pool": None,
    }
    try:
        from trading.data.short_interest import get_short_interest

        out["short"] = get_short_interest(symbol)
    except Exception as e:
        logger.debug("short_interest AI score fetch: %s", e)
    try:
        from trading.data.sec_edgar import get_sec_signal

        out["sec"] = get_sec_signal(symbol)
    except Exception as e:
        logger.debug("sec_edgar AI score fetch: %s", e)
    try:
        from trading.data.dark_pool import get_dark_pool_activity

        out["dark_pool"] = get_dark_pool_activity(symbol)
    except Exception as e:
        logger.debug("dark_pool AI score fetch: %s", e)
    return out


def _fetch_analyst_safe(
    symbol: str, _hist: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    try:
        from trading.data.analyst_signals import get_analyst_signals

        return get_analyst_signals(symbol)
    except Exception as e:
        logger.debug("analyst_signals AI score fetch: %s", e)
        return {
            "signal": "NEUTRAL",
            "signal_strength": 5.0,
            "success": False,
        }


def _fetch_congressional_safe(
    symbol: str, _hist: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    try:
        from trading.data.congressional_trading import (
            get_congressional_trades,
        )

        return get_congressional_trades(symbol)
    except Exception as e:
        logger.debug("congressional AI score fetch: %s", e)
        return {
            "signal": "NEUTRAL",
            "signal_strength": 5.0,
            "success": False,
        }


def _fetch_institutional_safe(
    symbol: str, _hist: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    try:
        from trading.data.sec_edgar import get_institutional_ownership

        return get_institutional_ownership(symbol)
    except Exception as e:
        logger.debug("institutional AI score fetch: %s", e)
        return {
            "signal": "NEUTRAL",
            "signal_strength": 5.0,
            "success": False,
        }


def _fetch_earnings_quality_safe(
    symbol: str, _hist: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    try:
        from trading.data.earnings_quality import get_earnings_quality

        return get_earnings_quality(symbol)
    except Exception as e:
        logger.debug("earnings_quality AI score fetch: %s", e)
        return {
            "signal": "NEUTRAL",
            "composite_score": 5.0,
            "success": False,
        }


def _bundle_technical(
    symbol: str,
    hist: pd.DataFrame,
    close: np.ndarray,
    last_price: float,
) -> Dict[str, Any]:
    """RSI, Bollinger, chart patterns → technical_score and signal rows."""
    signals: List[Dict[str, Any]] = []
    signal_status: Dict[str, str] = {}
    tech_points = 0.0

    _rsi_series = safe_rsi(close, 14)
    _rsi_flat = np.asarray(_rsi_series, dtype=float).ravel()
    rsi = float(_rsi_flat[-1]) if _rsi_flat.size else None
    if rsi is not None and np.isfinite(rsi):
        if 40 <= rsi <= 60:
            rsi_score = 5.0
        elif 30 <= rsi < 40 or 60 < rsi <= 70:
            rsi_score = 7.0
        elif rsi < 30:
            rsi_score = 9.0
        else:
            rsi_score = 3.0
        tech_points += rsi_score
        signals.append(
            {
                "name": "RSI",
                "value": round(float(rsi), 1),
                "impact": "positive" if rsi < 50 else "neutral" if rsi < 70 else "negative",
                "description": f"RSI {rsi:.1f} — {'oversold' if rsi < 30 else 'neutral' if rsi < 70 else 'overbought'}",
            }
        )

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
    signal_status["technical"] = "real"

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
        signal_status["chart_patterns"] = "real"
    except Exception as _pe:
        logger.debug("Chart pattern AI score hook skipped: %s", _pe)

    return {
        "technical_score": technical_score,
        "signals": signals,
        "signal_status": signal_status,
    }


def _bundle_momentum_base(
    close: np.ndarray,
    last_price: float,
    hist: pd.DataFrame,
) -> Dict[str, Any]:
    """SMA / 20d momentum / regime refinement → momentum_score."""
    signals: List[Dict[str, Any]] = []
    signal_status: Dict[str, str] = {}
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
    signal_status["momentum"] = "real"

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

    # Overextension penalty: at/near 52w high after large recent run
    try:
        _cm = {c.lower(): c for c in hist.columns}
        _hi_col = _cm.get("high")
        _cl_col = _cm.get(
            "close",
            list(hist.select_dtypes(include="number").columns)[0],
        )
        if _hi_col is not None and len(hist) >= 32:
            _hi52 = float(
                hist[_hi_col].astype(float).rolling(252, min_periods=20).max().iloc[-1]
            )
            _last = float(hist[_cl_col].astype(float).iloc[-1])
            _clf = hist[_cl_col].astype(float)
            _run30 = float(
                (_clf.iloc[-1] - _clf.iloc[-31]) / _clf.iloc[-31]
            )
            _near_high = _last >= _hi52 * 0.95
            _big_run = _run30 >= 0.20
            if _near_high and _big_run:
                _penalty = min(1.5, round(_run30 * 3, 1))
                momentum_score = max(0.0, momentum_score - _penalty)
                signals.append(
                    {
                        "name": "Overextension",
                        "value": round(_penalty, 1),
                        "impact": "negative",
                        "description": (
                            "⚠️ Overextended: near 52w high after "
                            f"{_run30 * 100:.0f}% "
                            "30d run — elevated mean reversion risk"
                        ),
                    }
                )
    except Exception as _ox:
        logger.debug("Overextension momentum check skipped: %s", _ox)

    return {
        "momentum_score": momentum_score,
        "signals": signals,
        "signal_status": signal_status,
    }


def compute_ai_score(
    symbol: str,
    hist: Optional[pd.DataFrame] = None,
    scoring_style: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute AI Score for a ticker.

    Args:
        symbol: ticker string
        hist: optional pre-fetched history DataFrame (Close, Volume, etc.)
              If None, fetches 6mo via price_cache (shared TTL).
        scoring_style: optional Settings label (e.g. \"Momentum-heavy\").
              If None, loads from user preferences.

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
        result = _compute_ai_score_impl(
            symbol, hist, scoring_style=scoring_style,
        )
        _persist_ai_score_result(symbol, result)
        return result
    sym_key = str(symbol or "").strip().upper()
    if not sym_key:
        return _error_score(str(symbol or ""), "Invalid symbol")
    _style_key = (
        scoring_style
        if scoring_style is not None
        else "__prefs__"
    )
    result = _compute_ai_score_cached(sym_key, _style_key)
    _persist_ai_score_result(sym_key, result)
    return result


@st.cache_data(ttl=300, show_spinner=False)
def _compute_ai_score_cached(
    symbol: str,
    scoring_style_key: str = "__prefs__",
) -> Dict[str, Any]:
    try:
        h = _pc_get_history(symbol, period="6mo")
    except Exception as e:
        logger.warning("ai_score: price_cache get_history failed: %s", e)
        h = pd.DataFrame()
    _style = (
        None
        if scoring_style_key == "__prefs__"
        else scoring_style_key
    )
    return _compute_ai_score_impl(
        symbol, h, scoring_style=_style,
    )


def _compute_ai_score_impl(
    symbol: str,
    hist: Optional[pd.DataFrame] = None,
    scoring_style: Optional[str] = None,
) -> Dict[str, Any]:
    """Internal AI score computation (used by compute_ai_score)."""
    try:
        _regime_label = ""
        _vix_level = 0.0
        try:
            _mf = _get_macro_factors()
            _factors = _mf.get_factors()
            _ov = _factors.get("overall_regime") or {}
            _regime_label = str(_ov.get("label") or "")
            _vx = _factors.get("vix") or {}
            _vix_level = float(_vx.get("current") or 0)
        except Exception:
            pass

        try:
            weights = _compute_ic_weights(
                symbol,
                scoring_style=scoring_style,
                regime_label=_regime_label,
                vix_level=_vix_level,
            )
        except Exception as _weights_exc:
            # BUG FIX: previously hardcoded SCORING_STYLE_WEIGHTS["Balanced
            # (default)"] here regardless of what scoring_style was actually
            # requested - a user with e.g. "Technical-heavy" selected would
            # silently get Balanced weights instead whenever
            # _compute_ic_weights raised any exception, with no logging to
            # indicate this happened. _resolve_style_weights already
            # correctly falls back to Balanced only when scoring_style is
            # unset/unrecognized.
            logger.debug(
                "IC weight computation failed, falling back to static "
                "style weights for scoring_style=%r: %s",
                scoring_style, _weights_exc,
            )
            weights = _resolve_style_weights(scoring_style)
            try:
                if _regime_label:
                    weights = _apply_regime_tilt(
                        dict(weights),
                        _regime_label,
                        _vix_level,
                    )
            except Exception:
                pass
        _external_bundle: Optional[Dict[str, Any]] = None
        if _has_external_api_keys():
            try:
                import asyncio

                from trading.data.external_signals import (
                    get_external_signals_manager,
                )

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
        if _regime_label and _regime_label not in (
            "",
            "NEUTRAL",
            "UNKNOWN",
        ):
            _mv = (
                "momentum"
                if _regime_label in ("RISK_ON", "LOW_VOL")
                else "fundamental"
            )
            _env = (
                "risk-on"
                if _regime_label in ("RISK_ON", "LOW_VOL")
                else "risk-off"
            )
            signals.append(
                {
                    "name": "Market Regime",
                    "value": _regime_label.replace("_", " ").title(),
                    "impact": (
                        "positive"
                        if _regime_label in ("RISK_ON", "LOW_VOL")
                        else "negative"
                        if _regime_label
                        in (
                            "RISK_OFF",
                            "HIGH_VOL",
                            "HIGH",
                            "STRESS",
                            "STAGFLATION",
                        )
                        else "neutral"
                    ),
                    "description": (
                        f"Regime-conditional weights active: {_mv} "
                        f"signals weighted higher in current {_env} "
                        "environment"
                    ),
                }
            )
        data_quality: Dict[str, str] = {
            "sentiment": "unavailable",
            "options": "unavailable",
            "macro": "unavailable",
            "insider": "unavailable",
        }
        _signal_status: Dict[str, str] = {
            k: "unavailable" for k in SIGNAL_SOURCES
        }
        earnings_near = False
        earnings_days_until = None

        # ── Technical / momentum + I/O signals (parallel) ─────────
        _parallel_results: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=10) as ex:
            _futures = {
                ex.submit(
                    _bundle_technical, symbol, hist, close, last_price
                ): "technical",
                ex.submit(
                    _bundle_momentum_base, close, last_price, hist
                ): "momentum",
                ex.submit(_fetch_options_flow_safe, symbol): "options",
                ex.submit(_fetch_insider_flow_safe, symbol): "insider",
                ex.submit(_fetch_social_sentiment_safe, symbol): "social",
                ex.submit(_fetch_short_and_sec_safe, symbol): "short_sec",
                ex.submit(_fetch_analyst_safe, symbol, hist): "analyst",
                ex.submit(
                    _fetch_congressional_safe, symbol, hist
                ): "congressional",
                ex.submit(
                    _fetch_institutional_safe, symbol, hist
                ): "institutional",
                ex.submit(
                    _fetch_earnings_quality_safe, symbol, hist
                ): "eq",
            }
            _done, _not_done = wait(
                _futures.keys(),
                timeout=AI_SCORE_PARALLEL_TIMEOUT,
                return_when=ALL_COMPLETED,
            )
            for _nf in _not_done:
                _nf.cancel()
            for _df in _done:
                _lab = _futures[_df]
                try:
                    _parallel_results[_lab] = _df.result()
                except Exception as _pe:
                    logger.debug(
                        "AI score parallel task %s failed: %s", _lab, _pe
                    )
                    _parallel_results[_lab] = None

        _tb = _parallel_results.get("technical")
        if isinstance(_tb, dict) and _tb.get("technical_score") is not None:
            technical_score = float(_tb["technical_score"])
            signals.extend(_tb.get("signals") or [])
            for _k, _v in (_tb.get("signal_status") or {}).items():
                _signal_status[_k] = _v
        else:
            technical_score = 5.0
            _signal_status["technical"] = "unavailable"

        _mb = _parallel_results.get("momentum")
        if isinstance(_mb, dict) and _mb.get("momentum_score") is not None:
            momentum_score = float(_mb["momentum_score"])
            signals.extend(_mb.get("signals") or [])
            for _k, _v in (_mb.get("signal_status") or {}).items():
                _signal_status[_k] = _v
        else:
            momentum_score = 5.0
            _signal_status["momentum"] = "unavailable"

        _of = _parallel_results.get("options")
        if not isinstance(_of, dict):
            _of = {}

        try:
            if _of.get("success"):
                data_quality["options"] = "real"
                _signal_status["options_flow"] = "real"
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
            elif _of.get("error"):
                _signal_status["options_flow"] = "fallback"
            else:
                _signal_status["options_flow"] = "fallback"
        except Exception as _oe:
            logger.debug("Options flow AI score hook skipped: %s", _oe)

        # ── SENTIMENT SCORE (0-10) ──────────────────────────────────
        sentiment_score = 5.0  # neutral default
        _short_sec = _parallel_results.get("short_sec")
        si = None
        _sec = None
        if isinstance(_short_sec, dict):
            si = _short_sec.get("short")
            _sec = _short_sec.get("sec")

        try:
            if si is not None:
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

        try:
            _dp = {}
            if isinstance(_short_sec, dict):
                _dp = _short_sec.get("dark_pool") or {}
            if isinstance(_dp, dict) and _dp.get("success"):
                _dp_pct = float(_dp.get("dark_pool_pct", 0))
                _dp_sig = _dp.get("signal", "NEUTRAL")
                _signal_status["dark_pool"] = "real"
                if _dp_sig == "ACCUMULATION" and _dp_pct > 30:
                    sentiment_score = min(
                        10.0,
                        sentiment_score + 0.5,
                    )
                    signals.append(
                        {
                            "name": "Dark Pool",
                            "value": (
                                f"{_dp_pct:.0f}% OTC volume"
                            ),
                            "impact": "positive",
                            "description": (
                                f"High dark pool activity "
                                f"({_dp_pct:.0f}% of volume) — "
                                "institutional accumulation signal"
                            ),
                        }
                    )
                elif _dp_sig == "DISTRIBUTION" and _dp_pct < 10:
                    signals.append(
                        {
                            "name": "Dark Pool",
                            "value": (
                                f"{_dp_pct:.0f}% OTC volume"
                            ),
                            "impact": "neutral",
                            "description": (
                                f"Low dark pool activity "
                                f"({_dp_pct:.0f}% of volume) — "
                                "retail-driven price action"
                            ),
                        }
                    )
            else:
                _signal_status["dark_pool"] = "fallback"
        except Exception as _dpe:
            logger.debug(
                "dark pool AI score merge skipped: %s",
                _dpe,
            )
            _signal_status["dark_pool"] = "fallback"

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

        insider = _parallel_results.get("insider")
        if not isinstance(insider, dict):
            insider = {}
        _insider_flow = (
            insider.get("flow")
            if isinstance(insider.get("flow"), dict)
            else insider
        )
        if not isinstance(_insider_flow, dict):
            _insider_flow = {}
        try:
            if _insider_flow.get("error"):
                data_quality["insider"] = "unavailable"
                _signal_status["insider_flow"] = "unavailable"
            else:
                data_quality["insider"] = "real"
                _signal_status["insider_flow"] = "real"
            signal = _insider_flow.get("signal", "NO_ACTIVITY")
            insider_score = {
                "INSIDER_BUYING": 8.5,
                "MIXED": 5.5,
                "NO_ACTIVITY": 5.0,
                "INSIDER_SELLING": 2.5,
            }.get(signal, 5.0)
            sentiment_score = (sentiment_score + insider_score) / 2
            _buy = _insider_flow.get("buy_count", 0) or 0
            _sell = _insider_flow.get("sell_count", 0) or 0
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
            _cluster = insider.get("insider_cluster") or {}
            if _cluster.get("success"):
                _csig = _cluster.get("cluster_signal", "NEUTRAL")
                _cbuys = _cluster.get("cluster_buy_count", 0)
                _buyers = _cluster.get("recent_buyers", [])

                if _csig == "STRONG_BUY":
                    sentiment_score = min(10.0, sentiment_score + 1.2)
                    signals.append(
                        {
                            "name": "Insider Cluster",
                            "value": f"{_cbuys} insiders buying",
                            "impact": "positive",
                            "description": (
                                f"Strong cluster: {_cbuys} insiders "
                                f"bought in same 30d window — "
                                "historically predicts +4-6% abnormal return"
                                + (
                                    f" ({', '.join(_buyers[:2])})"
                                    if _buyers
                                    else ""
                                )
                            ),
                        }
                    )
                elif _csig == "BUY":
                    sentiment_score = min(10.0, sentiment_score + 0.6)
                    signals.append(
                        {
                            "name": "Insider Cluster",
                            "value": f"{_cbuys} insiders buying",
                            "impact": "positive",
                            "description": (
                                f"{_cbuys} insiders bought in same "
                                "30-day window"
                                + (
                                    f" ({', '.join(_buyers[:2])})"
                                    if _buyers
                                    else ""
                                )
                            ),
                        }
                    )
                elif _csig == "SELL":
                    _csells = _cluster.get("cluster_sell_count", 0)
                    sentiment_score = max(0.0, sentiment_score - 0.5)
                    signals.append(
                        {
                            "name": "Insider Cluster",
                            "value": f"{_csells} insiders selling",
                            "impact": "negative",
                            "description": (
                                f"{_csells} insiders sold in same "
                                "30-day window (note: sells are less "
                                "predictive than cluster buys)"
                            ),
                        }
                    )
        except Exception as _ice:
            logger.debug("insider cluster merge skipped: %s", _ice)

        social = _parallel_results.get("social")
        try:
            if social and social.get("success") and social.get("source") == "reddit":
                data_quality["sentiment"] = "real"
                _signal_status["social_sentiment"] = "real"
            elif social and social.get("source") == "unavailable":
                _signal_status["social_sentiment"] = "fallback"
            elif (
                social
                and social.get("success")
                and int(social.get("mention_count") or 0) > 0
            ):
                _signal_status["social_sentiment"] = "real"
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

        _analyst = _parallel_results.get("analyst")
        if not isinstance(_analyst, dict):
            _analyst = {}
        _an_sig = str(_analyst.get("signal", "NEUTRAL"))
        _an_str = float(_analyst.get("signal_strength", 5.0))
        _upside = _analyst.get("upside_pct")
        _n_analysts = int(_analyst.get("n_analysts", 0) or 0)

        if _analyst.get("success"):
            _signal_status["analyst_signals"] = "real"
            if _an_sig == "BUY" and _an_str >= 7.0:
                sentiment_score = min(10.0, sentiment_score + 0.8)
                _rec = str(_analyst.get("recommendation", "") or "").replace("_", " ").title()
                _desc = (
                    f"📈 Analyst consensus: {_rec} "
                    f"({_n_analysts} analysts"
                    + (
                        f", {_upside:+.1f}% to target"
                        if _upside is not None
                        else ""
                    )
                    + ")"
                )
                signals.append(
                    {
                        "name": "Analyst consensus",
                        "value": round(_an_str, 1),
                        "impact": "positive",
                        "description": _desc,
                    }
                )
            elif _an_sig == "SELL" and _an_str <= 3.0:
                sentiment_score = max(0.0, sentiment_score - 1.0)
                _rec = str(_analyst.get("recommendation", "") or "").replace("_", " ").title()
                _desc = (
                    f"📉 Analyst consensus: {_rec} "
                    f"({_n_analysts} analysts"
                    + (
                        f", {_upside:+.1f}% to target"
                        if _upside is not None
                        else ""
                    )
                    + ")"
                )
                signals.append(
                    {
                        "name": "Analyst consensus",
                        "value": round(_an_str, 1),
                        "impact": "negative",
                        "description": _desc,
                    }
                )
            if _upside is not None and _upside < -15:
                signals.append(
                    {
                        "name": "Analyst target",
                        "value": round(float(_upside), 1),
                        "impact": "neutral",
                        "description": (
                            f"⚠️ Trading {abs(float(_upside)):.0f}% "
                            "above analyst mean target"
                        ),
                    }
                )
        else:
            _signal_status["analyst_signals"] = "fallback"

        _cong = _parallel_results.get("congressional")
        if not isinstance(_cong, dict):
            _cong = {}
        if _cong.get("success") and _cong.get("total_trades", 0) > 0:
            _cong_sig = _cong.get("signal", "NEUTRAL")
            _cong_b = _cong.get("buys", 0)
            _cong_s = _cong.get("sells", 0)
            _signal_status["congressional_trading"] = "real"
            if _cong_sig == "BUY":
                sentiment_score = min(10.0, sentiment_score + 0.6)
                signals.append(
                    {
                        "name": "Congressional",
                        "value": "Buying",
                        "impact": "positive",
                        "description": (
                            f"{_cong_b} congressional"
                            f" buys vs {_cong_s} "
                            f"sells recently"
                        ),
                    }
                )
            elif _cong_sig == "SELL":
                sentiment_score = max(0.0, sentiment_score - 0.6)
                signals.append(
                    {
                        "name": "Congressional",
                        "value": "Selling",
                        "impact": "negative",
                        "description": (
                            f"{_cong_s} congressional"
                            f" sells vs {_cong_b} "
                            f"buys recently"
                        ),
                    }
                )
        else:
            _signal_status["congressional_trading"] = "fallback"

        try:
            if _sec is not None:
                _sec_sent = float(_sec.get("sec_sentiment", 0.0))
                _sec_source = str(_sec.get("sec_source", "unavailable"))
                if _sec_source in ("unavailable", "error"):
                    _signal_status["sec_edgar"] = "unavailable"
                elif _sec_source == "no_filing":
                    _signal_status["sec_edgar"] = "fallback"
                else:
                    _signal_status["sec_edgar"] = "real"

                if _sec_source not in ("unavailable", "error", "no_filing"):
                    _sec_score = 5.0 + _sec_sent * 4.0
                    _sec_score = max(0.0, min(10.0, _sec_score))
                    sentiment_score = sentiment_score * 0.70 + _sec_score * 0.30

                    _desc_parts = []
                    if _sec_sent > 0.1:
                        _desc_parts.append(
                            f"SEC filing: positive tone ({_sec.get('sec_label')})"
                        )
                    elif _sec_sent < -0.1:
                        _desc_parts.append(
                            f"SEC filing: cautious tone ({_sec.get('sec_label')})"
                        )
                    themes = _sec.get("sec_themes") or []
                    if themes:
                        _desc_parts.append(
                            "SEC themes: " + ", ".join(str(t) for t in themes[:2])
                        )
                    signals.append(
                        {
                            "name": "SEC filings",
                            "value": round(_sec_score, 1),
                            "impact": (
                                "positive"
                                if _sec_sent > 0.1
                                else "negative"
                                if _sec_sent < -0.1
                                else "neutral"
                            ),
                            "description": (
                                " · ".join(_desc_parts)
                                if _desc_parts
                                else f"SEC EDGAR ({_sec_source})"
                            ),
                        }
                    )
        except Exception:
            pass

        try:
            from trading.data.sec_edgar import (
                get_earnings_transcript_sentiment,
            )

            _trans = get_earnings_transcript_sentiment(symbol)
            if _trans.get("success"):
                _tadj = float(_trans.get("score_adj", 0.0))
                _tone = _trans.get("sentiment", "neutral")
                if abs(_tadj) > 0.2:
                    sentiment_score = max(
                        0.0,
                        min(
                            10.0,
                            sentiment_score + _tadj,
                        ),
                    )
                    _impact = (
                        "positive" if _tadj > 0 else "negative"
                    )
                    signals.append(
                        {
                            "name": "Transcript",
                            "value": _tone.title(),
                            "impact": _impact,
                            "description": (
                                f"Earnings call tone: {_tone}. "
                                + (
                                    "Themes: "
                                    + ", ".join(
                                        _trans.get("themes", [])[:3]
                                    )
                                    if _trans.get("themes")
                                    else ""
                                )
                            ),
                        }
                    )
        except Exception:
            pass

        _inst = _parallel_results.get("institutional")
        if not isinstance(_inst, dict):
            _inst = {}
        if _inst.get("success"):
            _inst_pct = float(_inst.get("institutional_pct", 0))
            _inst_sig = _inst.get("signal", "NEUTRAL")
            _signal_status["institutional_ownership"] = "real"
            if _inst_sig == "BUY" and _inst_pct > 0.50:
                sentiment_score = min(10.0, sentiment_score + 0.5)
                signals.append(
                    {
                        "name": "Institutional",
                        "value": (
                            f"{_inst_pct * 100:.0f}% institutional"
                        ),
                        "impact": "positive",
                        "description": (
                            "High institutional ownership — "
                            "smart money present"
                        ),
                    }
                )
            elif _inst_sig == "SELL" and _inst_pct < 0.20:
                signals.append(
                    {
                        "name": "Institutional",
                        "value": (
                            f"{_inst_pct * 100:.0f}% institutional"
                        ),
                        "impact": "neutral",
                        "description": (
                            "Low institutional coverage"
                        ),
                    }
                )
        else:
            _signal_status["institutional_ownership"] = "fallback"

        # ── FUNDAMENTAL SCORE (0-10) ──────────────────────────────
        fundamental_score = 5.0
        sector = ""  # initialise for use in risk flags
        try:
            from trading.data.earnings_calendar import get_upcoming_earnings

            earnings = get_upcoming_earnings(symbol)
            _signal_status["earnings_calendar"] = "real"
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

            _eq = _parallel_results.get("eq")
            if not isinstance(_eq, dict):
                _eq = {}
            if _eq.get("success"):
                _eq_sig = _eq.get("signal", "NEUTRAL")
                _beat_rate = float(_eq.get("beat_rate", 0.5))
                _accruals = _eq.get("accruals_signal", "NORMAL")
                _rev_sig = _eq.get("revision_signal", "NEUTRAL")
                _signal_status["earnings_quality"] = "real"

                if _eq_sig == "QUALITY":
                    fundamental_score = min(
                        10.0,
                        fundamental_score + 0.8,
                    )
                    signals.append(
                        {
                            "name": "Earnings Quality",
                            "value": "High Quality",
                            "impact": "positive",
                            "description": (
                                f"Beat rate: "
                                f"{_beat_rate * 100:.0f}%"
                                f" · EPS revisions: "
                                f"{_rev_sig} · "
                                f"Accruals: {_accruals}"
                            ),
                        }
                    )
                elif _eq_sig == "CONCERN":
                    fundamental_score = max(
                        0.0,
                        fundamental_score - 0.8,
                    )
                    signals.append(
                        {
                            "name": "Earnings Quality",
                            "value": "Concern",
                            "impact": "negative",
                            "description": (
                                f"Beat rate: "
                                f"{_beat_rate * 100:.0f}%"
                                f" · EPS revisions: "
                                f"{_rev_sig} · "
                                f"Accruals: {_accruals}"
                            ),
                        }
                    )

                if _accruals == "HIGH":
                    signals.append(
                        {
                            "name": "Accruals",
                            "value": "High",
                            "impact": "negative",
                            "description": (
                                "Earnings driven by "
                                "non-cash accruals — "
                                "Sloan anomaly risk "
                                "(earnings may revert)"
                            ),
                        }
                    )
            else:
                _signal_status["earnings_quality"] = "fallback"

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

            try:
                from trading.data.earnings_calendar import (
                    get_macro_calendar,
                )

                _mac = get_macro_calendar(days_ahead=7)
                for _ev in _mac.get("events", []):
                    if _ev.get("days_until", 99) <= 3:
                        signals.append(
                            {
                                "name": "Macro Event",
                                "value": _ev["name"],
                                "impact": "neutral",
                                "description": (
                                    f"In "
                                    f"{_ev['days_until']}"
                                    f" days — forecasts "
                                    f"may be less "
                                    f"reliable"
                                ),
                            }
                        )
            except Exception:
                pass

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

                try:
                    _fwd_pe = _info.get("forwardPE")
                    _pb = _info.get("priceToBook")
                    _ev_ebitda = _info.get("enterpriseToEbitda")
                    _med = SECTOR_VALUATION_MEDIANS.get(sector, {})
                    _overvalued_count = 0
                    _undervalued_count = 0

                    for _metric, _val, _key in [
                        ("Forward P/E", _fwd_pe, "forward_pe"),
                        ("P/B", _pb, "pb"),
                        ("EV/EBITDA", _ev_ebitda, "ev_ebitda"),
                    ]:
                        if _val and _key in _med:
                            _med_val = _med[_key]
                            if float(_val) > _med_val * 1.4:
                                _overvalued_count += 1
                            elif float(_val) < _med_val * 0.7:
                                _undervalued_count += 1

                    if _overvalued_count >= 2:
                        fundamental_score = max(
                            0.0,
                            fundamental_score - 0.8,
                        )
                        signals.append(
                            {
                                "name": "Valuation",
                                "value": "Overvalued",
                                "impact": "negative",
                                "description": (
                                    f"Trading at premium on {_overvalued_count}"
                                    f"/3 valuation metrics vs {sector} sector"
                                ),
                            }
                        )
                    elif _undervalued_count >= 2:
                        fundamental_score = min(
                            10.0,
                            fundamental_score + 0.8,
                        )
                        signals.append(
                            {
                                "name": "Valuation",
                                "value": "Undervalued",
                                "impact": "positive",
                                "description": (
                                    f"Attractive on {_undervalued_count}"
                                    f"/3 valuation metrics vs {sector} sector"
                                ),
                            }
                        )
                except Exception:
                    pass

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
                _signal_status["macro_factors"] = "real"
                _macro_score_adj = _macro_adj.get(
                    "score_adjustment", 0.0
                )
                fundamental_score = min(10.0, max(0.0,
                    fundamental_score + _macro_score_adj
                ))
                for _msig in _macro_adj.get("signals", []):
                    signals.append(_msig)
            except Exception:
                _signal_status["macro_factors"] = "unavailable"

            # Sector rotation context
            try:
                from trading.analysis.sector_rotation import (
                    get_sector_signal_for_ticker,
                )

                _sr = get_sector_signal_for_ticker(sector)
                if _sr and _sr.get("trend"):
                    _sr_score = float(_sr.get("composite_score", 0))
                    if _sr_score > 3:
                        fundamental_score = min(
                            10.0,
                            fundamental_score + 0.5,
                        )
                        _signal_status["sector_rotation"] = "real"
                        signals.append(
                            {
                                "name": "Sector Momentum",
                                "value": "Outperforming",
                                "impact": "positive",
                                "description": (
                                    f"{sector} sector outperforming SPY "
                                    f"by {_sr_score:.1f}% composite"
                                ),
                            }
                        )
                    elif _sr_score < -3:
                        fundamental_score = max(
                            0.0,
                            fundamental_score - 0.5,
                        )
                        _signal_status["sector_rotation"] = "real"
                        signals.append(
                            {
                                "name": "Sector Momentum",
                                "value": "Underperforming",
                                "impact": "negative",
                                "description": (
                                    f"{sector} sector underperforming SPY "
                                    f"by {abs(_sr_score):.1f}% composite"
                                ),
                            }
                        )
            except Exception:
                pass

            try:
                from trading.analysis.factor_model import FactorModel

                _fm = FactorModel()
                if hist is not None and len(hist) >= 20:
                    _col = {c.lower(): c for c in hist.columns}
                    _close = _col.get("close", hist.columns[0])
                    _prices = hist[_close].dropna()
                    _returns = _prices.pct_change().dropna()
                    if len(_returns) >= 10:
                        _exposures = _fm.compute_exposures(
                            symbol, _returns, ohlcv=hist
                        )
                        _signal_status["factor_model"] = "real"
                        _mom = float(_exposures.get("momentum", 0.0))
                        if abs(_mom) > 0.01:
                            _factor_adj = float(_mom * 2.0)
                            fundamental_score = float(
                                np.clip(
                                    fundamental_score + _factor_adj,
                                    0,
                                    10,
                                )
                            )
                            signals.append(
                                {
                                    "name": "Factor: Momentum",
                                    "value": round(_mom, 3),
                                    "impact": (
                                        "positive"
                                        if _mom > 0
                                        else "negative"
                                    ),
                                    "description": (
                                        f"Factor momentum exposure: {_mom:+.3f}"
                                    ),
                                }
                            )
                        _vol = float(_exposures.get("volatility", 0.0))
                        if _vol > 0:
                            _vol_score = max(0, 1 - _vol * 5)
                            fundamental_score = float(
                                np.clip(
                                    fundamental_score * 0.9
                                    + _vol_score * 10 * 0.1,
                                    0,
                                    10,
                                )
                            )
            except Exception:
                pass
        except Exception:
            pass

        # ── COMPOSITE SCORE ─────────────────────────────────────────
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
                _signal_status["ml_score"] = "real"
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
            else:
                _signal_status["ml_score"] = "fallback"
        except Exception:
            _signal_status["ml_score"] = "unavailable"

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

        _n_real = len(
            [v for v in _signal_status.values() if v == "real"]
        )
        signal_completeness = {
            "available": [
                k for k, v in _signal_status.items() if v == "real"
            ],
            "fallback": [
                k for k, v in _signal_status.items() if v == "fallback"
            ],
            "unavailable": [
                k for k, v in _signal_status.items() if v == "unavailable"
            ],
            "score": _n_real / float(len(SIGNAL_SOURCES)),
        }

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
            "signal_completeness": signal_completeness,
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
        "signal_completeness": {
            "available": [],
            "fallback": [],
            "unavailable": list(SIGNAL_SOURCES),
            "score": 0.0,
        },
        "summary": f"Score unavailable: {error}",
        "last_price": None,
        "error": error,
    }


def _neutral_short(symbol: str) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "short_score": 5.0,
        "grade": "C",
        "label": "Not a Short",
        "signals": [],
        "long_score": 5.0,
        "summary": "Short score unavailable",
        "error": "unavailable",
    }


def compute_short_score(
    symbol: str,
    hist: Optional[pd.DataFrame],
    ai_result: Optional[Dict[str, Any]] = None,
    scoring_style: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Short candidate score (0-10). Higher = stronger bearish thesis.
    Reweights existing AI Score signals (no extra data fetch when ai_result
    is passed).
    """
    try:
        if ai_result is None:
            ai_result = compute_ai_score(
                symbol, hist, scoring_style=scoring_style,
            )

        if ai_result.get("error"):
            return _neutral_short(symbol)

        signals = ai_result.get("signals") or []
        _sig_map = {
            str(s.get("name", "")).lower().strip(): s
            for s in signals
        }

        score = 5.0
        short_signals: List[Dict[str, Any]] = []

        tech = float(ai_result.get("technical_score", 5.0) or 5.0)
        score += (10.0 - tech) * 0.15 - 0.75

        mom = float(ai_result.get("momentum_score", 5.0) or 5.0)
        score += (10.0 - mom) * 0.20 - 1.0

        _rsi_sig = _sig_map.get("rsi") or {}
        try:
            _rsi_val = float(
                str(_rsi_sig.get("value", 50)).replace("%", "").strip()
            )
            if _rsi_val > 75:
                score += 1.5
                short_signals.append(
                    {
                        "name": "RSI",
                        "value": round(_rsi_val, 1),
                        "impact": "bearish",
                        "description": (
                            f"Overbought RSI {_rsi_val:.0f} — reversal risk"
                        ),
                    }
                )
            elif _rsi_val > 65:
                score += 0.5
            elif _rsi_val < 35:
                score -= 1.0
        except Exception:
            pass

        _ins_sig = (
            _sig_map.get("insider flow")
            or _sig_map.get("insider")
            or {}
        )
        _ins_impact = str(_ins_sig.get("impact", "")).lower()
        if _ins_impact == "negative":
            score += 1.0
            short_signals.append(
                {
                    "name": "Insider Flow",
                    "value": _ins_sig.get("value", "Selling"),
                    "impact": "bearish",
                    "description": "Insider selling detected",
                }
            )
        elif _ins_impact == "positive":
            score -= 0.5

        _eq_sig = _sig_map.get("earnings quality") or {}
        _eq_impact = str(_eq_sig.get("impact", "")).lower()
        if _eq_impact == "negative":
            score += 0.8
            short_signals.append(
                {
                    "name": "Earnings Quality",
                    "value": _eq_sig.get("value", "Concern"),
                    "impact": "bearish",
                    "description": (
                        "High accruals or deteriorating earnings quality"
                    ),
                }
            )

        _acc_sig = _sig_map.get("accruals") or {}
        if str(_acc_sig.get("impact", "")).lower() == "negative":
            score += 0.5
            short_signals.append(
                {
                    "name": "Accruals",
                    "value": _acc_sig.get("value", "High"),
                    "impact": "bearish",
                    "description": _acc_sig.get("description", ""),
                }
            )

        _si_sig = (
            _sig_map.get("short squeeze score")
            or _sig_map.get("short interest")
            or {}
        )
        try:
            _si_desc = str(_si_sig.get("description", "")).lower()
            _si_pct = 0.0
            if "float" in _si_desc and "%" in _si_desc:
                if "float:" in _si_desc:
                    _part = _si_desc.split("float:", 1)[1]
                elif "float short:" in _si_desc:
                    _part = _si_desc.split("float short:", 1)[1]
                else:
                    _part = ""
                if _part:
                    _si_pct = float(_part.split("%", 1)[0].strip())
            if _si_pct >= 20:
                score -= 1.5
                short_signals.append(
                    {
                        "name": "Short Interest",
                        "value": f"{_si_pct:.1f}%",
                        "impact": "risk",
                        "description": (
                            f"Already {_si_pct:.0f}% short — crowded, "
                            "squeeze risk"
                        ),
                    }
                )
            elif _si_pct >= 10:
                score -= 0.5
            elif 0 < _si_pct < 3:
                score += 0.3
        except Exception:
            pass

        fund = float(ai_result.get("fundamental_score", 5.0) or 5.0)
        if fund < 4.0:
            score += 0.8
            short_signals.append(
                {
                    "name": "Fundamentals",
                    "value": round(fund, 1),
                    "impact": "bearish",
                    "description": "Weak fundamental score",
                }
            )

        score = float(max(0.0, min(10.0, score)))

        if score >= 7.0:
            grade = "A"
            label = "Strong Short"
        elif score >= 6.0:
            grade = "B"
            label = "Moderate Short"
        elif score >= 5.0:
            grade = "C"
            label = "Weak Short"
        else:
            grade = "D"
            label = "Not a Short"

        return {
            "symbol": symbol,
            "short_score": round(score, 2),
            "grade": grade,
            "label": label,
            "signals": short_signals,
            "long_score": float(
                ai_result.get("overall_score", 5.0) or 5.0
            ),
            "summary": f"{label} — Short Score {score:.1f}/10",
            "error": None,
        }
    except Exception as e:
        logger.debug("Short score failed %s: %s", symbol, e)
        return _neutral_short(symbol)
