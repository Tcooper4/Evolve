"""
Macro Factors
==============
INTEGRATION NOTES:
- Drop into: trading/analysis/macro_factors.py
- Wire into: trading/analysis/ai_score.py fundamental section
- Call pattern:
    from trading.analysis.macro_factors import MacroFactors
    macro = MacroFactors()
    factors = macro.get_factors()
    score_adj = macro.get_ai_score_adjustment(sector)

Dependencies: yfinance (already installed), fredapi (optional)
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Cache duration in seconds
CACHE_TTL = 3600  # 1 hour


class MacroFactors:
    """
    Fetches and interprets macro economic factors.

    Factors tracked:
    - Fed Funds Rate (from yfinance ^IRX proxy or FRED)
    - Yield curve slope (10Y - 2Y spread)
    - Credit spreads (HYG/IEI ratio as proxy)
    - Dollar strength (DXY)
    - Market volatility regime (VIX)
    - Inflation regime (TIP/IEF ratio)
    """

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._cache_ts: float = 0

    def get_current_context(self) -> Dict[str, Any]:
        """
        Flat snapshot for dashboards (VIX, 10Y, DXY, regime label).
        """
        try:
            f = self.get_factors()
            if not f:
                return {}
            vix_b = f.get("vix") or {}
            yc = f.get("yield_curve") or {}
            dxy_b = f.get("dollar") or {}
            regime = f.get("overall_regime") or {}
            label = regime.get("label") or "UNKNOWN"
            desc = regime.get("description") or ""
            regime_label = f"Regime: {label}"
            if desc:
                regime_label = f"{regime_label} — {desc}"
            return {
                "vix": float(vix_b.get("current") or 0),
                "yield_10y": float(yc.get("rate_10y") or 0),
                "dxy": float(dxy_b.get("current") or 0),
                "regime_label": regime_label,
            }
        except Exception as e:
            logger.debug("get_current_context failed: %s", e)
            return {}

    def get_factors(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get all macro factors with caching.
        Returns dict with factor values and interpretations.
        """
        import time
        now = time.time()

        if (not force_refresh
                and self._cache
                and (now - self._cache_ts) < CACHE_TTL):
            return self._cache

        factors = {
            "timestamp": datetime.now().isoformat(),
            "yield_curve": self._get_yield_curve(),
            "vix": self._get_vix(),
            "credit_spreads": self._get_credit_spreads(),
            "dollar": self._get_dollar_strength(),
            "inflation": self._get_inflation_regime(),
            "overall_regime": {},
        }

        factors["overall_regime"] = self._interpret_regime(factors)

        self._cache = factors
        self._cache_ts = now
        return factors

    def get_ai_score_adjustment(
        self,
        sector: str = "",
    ) -> Dict[str, Any]:
        """
        Get AI Score adjustment based on macro factors.
        Returns score_adj (-2 to +2) and signals list.
        """
        factors = self.get_factors()
        regime = factors.get("overall_regime", {})
        signals = []
        score_adj = 0.0

        # Yield curve
        yc = factors.get("yield_curve", {})
        spread = yc.get("spread_10y2y")
        if spread is not None:
            if spread < -0.5:
                score_adj -= 1.0
                signals.append({
                    "name": "Yield Curve",
                    "value": f"{spread:+.2f}%",
                    "impact": "negative",
                    "description": (
                        f"⚠️ Inverted yield curve ({spread:+.2f}%) "
                        "— recession risk elevated"
                    ),
                })
            elif spread > 1.0:
                score_adj += 0.5
                signals.append({
                    "name": "Yield Curve",
                    "value": f"{spread:+.2f}%",
                    "impact": "positive",
                    "description": (
                        f"Healthy yield curve spread ({spread:+.2f}%)"
                    ),
                })

        # VIX
        vix = factors.get("vix", {})
        vix_level = vix.get("current")
        if vix_level is not None:
            if vix_level > 30:
                score_adj -= 1.0
                signals.append({
                    "name": "VIX",
                    "value": f"{vix_level:.1f}",
                    "impact": "negative",
                    "description": (
                        f"⚠️ High volatility (VIX {vix_level:.1f}) "
                        "— risk-off environment"
                    ),
                })
            elif vix_level < 15:
                score_adj += 0.5
                signals.append({
                    "name": "VIX",
                    "value": f"{vix_level:.1f}",
                    "impact": "positive",
                    "description": (
                        f"Low volatility (VIX {vix_level:.1f}) "
                        "— risk-on environment"
                    ),
                })

        # Sector-specific adjustments
        if sector:
            sector_adj = self._sector_macro_adjustment(sector, factors)
            score_adj += sector_adj.get("adjustment", 0)
            signals.extend(sector_adj.get("signals", []))

        return {
            "score_adjustment": round(
                float(np.clip(score_adj, -2.0, 2.0)), 2
            ),
            "signals": signals,
            "regime": regime.get("label", "NEUTRAL"),
        }

    def _get_yield_curve(self) -> Dict[str, Any]:
        """Get yield curve data (10Y-2Y spread)."""
        result = {}
        try:
            from trading.data.price_cache import get_macro_history
            # TNX = 10Y Treasury, IRX = 13-week T-bill
            tnx = get_macro_history("^TNX", period="5d")
            tyx = get_macro_history("^TYX", period="5d")  # 30Y
            fvx = get_macro_history("^FVX", period="5d")  # 5Y

            rates = {}
            for name, ticker_data in [
                ("10y", tnx), ("30y", tyx), ("5y", fvx)
            ]:
                if not ticker_data.empty:
                    _col_map = {
                        c.lower(): c for c in ticker_data.columns
                    }
                    close_col = _col_map.get("close", ticker_data.columns[0])
                    rates[name] = float(
                        ticker_data[close_col].iloc[-1]
                    )

            if "10y" in rates:
                result["rate_10y"] = round(rates["10y"], 3)
            if "5y" in rates:
                result["rate_5y"] = round(rates["5y"], 3)
                if "10y" in rates:
                    result["spread_10y5y"] = round(
                        rates["10y"] - rates["5y"], 3
                    )

            # Approximate 2Y from short-end
            irx = get_macro_history("^IRX", period="5d")
            if not irx.empty:
                _col_map = {c.lower(): c for c in irx.columns}
                close_col = _col_map.get("close", irx.columns[0])
                rate_2y = float(irx[close_col].iloc[-1])
                result["rate_2y"] = round(rate_2y, 3)
                if "10y" in rates:
                    spread = rates["10y"] - rate_2y
                    result["spread_10y2y"] = round(spread, 3)
                    result["inverted"] = spread < 0
                    result["interpretation"] = (
                        f"Yield curve {'inverted' if spread < 0 else 'normal'} "
                        f"({spread:+.2f}%)"
                    )

        except Exception as e:
            logger.debug("Yield curve fetch failed: %s", e)
            result["error"] = str(e)

        return result

    def _get_vix(self) -> Dict[str, Any]:
        """Get VIX level and regime."""
        result = {}
        try:
            from trading.data.price_cache import get_macro_history
            vix = get_macro_history("^VIX", period="1mo")
            if not vix.empty:
                _col_map = {c.lower(): c for c in vix.columns}
                close_col = _col_map.get("close", vix.columns[0])
                current = float(vix[close_col].iloc[-1])
                ma20 = float(vix[close_col].rolling(20).mean().iloc[-1])

                result = {
                    "current": round(current, 2),
                    "ma20": round(ma20, 2),
                    "vs_ma": round(current - ma20, 2),
                    "regime": (
                        "HIGH" if current > 30
                        else "ELEVATED" if current > 20
                        else "NORMAL" if current > 15
                        else "LOW"
                    ),
                    "trending_up": current > ma20,
                }
        except Exception as e:
            logger.debug("VIX fetch failed: %s", e)
            result["error"] = str(e)

        return result

    def _get_credit_spreads(self) -> Dict[str, Any]:
        """Get credit spread proxy using HYG/IEI."""
        result = {}
        try:
            from trading.data.price_cache import get_macro_history
            hyg = get_macro_history("HYG", period="1mo")
            iei = get_macro_history("IEI", period="1mo")

            if not hyg.empty and not iei.empty:
                _col_map_h = {c.lower(): c for c in hyg.columns}
                _col_map_i = {c.lower(): c for c in iei.columns}
                hyg_close = hyg[_col_map_h.get("close", hyg.columns[0])]
                iei_close = iei[_col_map_i.get("close", iei.columns[0])]

                # Ratio as credit spread proxy
                ratio = (hyg_close / iei_close).dropna()
                current = float(ratio.iloc[-1])
                ma20 = float(ratio.rolling(20).mean().iloc[-1])

                result = {
                    "hyg_iei_ratio": round(current, 4),
                    "ratio_vs_ma20": round(current - ma20, 4),
                    "credit_stress": current < ma20 * 0.98,
                    "interpretation": (
                        "Credit stress detected — risk-off signal"
                        if current < ma20 * 0.98
                        else "Credit markets stable"
                    ),
                }
        except Exception as e:
            logger.debug("Credit spreads fetch failed: %s", e)
            result["error"] = str(e)

        return result

    def _get_dollar_strength(self) -> Dict[str, Any]:
        """Get DXY dollar strength."""
        result = {}
        try:
            from trading.data.price_cache import get_macro_history
            dxy = get_macro_history("DX-Y.NYB", period="1mo")
            if dxy.empty:
                dxy = get_macro_history("UUP", period="1mo")

            if not dxy.empty:
                _col_map = {c.lower(): c for c in dxy.columns}
                close_col = _col_map.get("close", dxy.columns[0])
                closes = dxy[close_col].values
                current = float(closes[-1])
                ma20 = float(np.mean(closes[-20:])) if len(closes) >= 20 else current
                change_mo = float((closes[-1] / closes[0] - 1) * 100)

                result = {
                    "current": round(current, 2),
                    "vs_ma20_pct": round((current - ma20) / ma20 * 100, 2),
                    "monthly_change_pct": round(change_mo, 2),
                    "strengthening": current > ma20,
                    "interpretation": (
                        "Strong dollar — headwind for multinationals/commodities"
                        if current > ma20 * 1.02
                        else "Weak dollar — tailwind for emerging markets/commodities"
                        if current < ma20 * 0.98
                        else "Dollar neutral"
                    ),
                }
        except Exception as e:
            logger.debug("Dollar strength fetch failed: %s", e)
            result["error"] = str(e)

        return result

    def _get_inflation_regime(self) -> Dict[str, Any]:
        """Get inflation regime proxy using TIP/IEF."""
        result = {}
        try:
            from trading.data.price_cache import get_macro_history
            tip = get_macro_history("TIP", period="1mo")
            ief = get_macro_history("IEF", period="1mo")

            if not tip.empty and not ief.empty:
                _col_map_t = {c.lower(): c for c in tip.columns}
                _col_map_i = {c.lower(): c for c in ief.columns}
                tip_close = tip[_col_map_t.get("close", tip.columns[0])]
                ief_close = ief[_col_map_i.get("close", ief.columns[0])]

                ratio = (tip_close / ief_close).dropna()
                current = float(ratio.iloc[-1])
                ma20 = float(ratio.rolling(20).mean().iloc[-1])
                change = float(
                    (ratio.iloc[-1] / ratio.iloc[0] - 1) * 100
                )

                result = {
                    "tip_ief_ratio": round(current, 4),
                    "monthly_change_pct": round(change, 2),
                    "inflation_rising": current > ma20,
                    "interpretation": (
                        "Inflation expectations rising — "
                        "headwind for growth stocks"
                        if current > ma20
                        else "Inflation expectations falling — "
                        "supportive for growth stocks"
                    ),
                }
        except Exception as e:
            logger.debug("Inflation regime fetch failed: %s", e)
            result["error"] = str(e)

        return result

    def _interpret_regime(
        self, factors: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine overall macro regime."""
        bullish_factors = 0
        bearish_factors = 0

        vix = factors.get("vix", {})
        if vix.get("regime") == "LOW":
            bullish_factors += 1
        elif vix.get("regime") == "HIGH":
            bearish_factors += 2

        yc = factors.get("yield_curve", {})
        if yc.get("inverted"):
            bearish_factors += 1
        elif yc.get("spread_10y2y", 0) > 0.5:
            bullish_factors += 1

        credit = factors.get("credit_spreads", {})
        if credit.get("credit_stress"):
            bearish_factors += 1

        dollar = factors.get("dollar", {})
        if dollar.get("strengthening"):
            bearish_factors += 0.5  # mild headwind

        inflation = factors.get("inflation", {})
        if inflation.get("inflation_rising"):
            bearish_factors += 0.5

        net = bullish_factors - bearish_factors
        if net > 1:
            label = "RISK_ON"
            description = "Macro environment is supportive"
            score_bias = 0.5
        elif net < -1:
            label = "RISK_OFF"
            description = "Macro headwinds present — be selective"
            score_bias = -0.5
        else:
            label = "NEUTRAL"
            description = "Mixed macro signals"
            score_bias = 0.0

        return {
            "label": label,
            "description": description,
            "score_bias": score_bias,
            "bullish_count": bullish_factors,
            "bearish_count": bearish_factors,
        }

    def _sector_macro_adjustment(
        self,
        sector: str,
        factors: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Sector-specific macro adjustments."""
        signals = []
        adjustment = 0.0

        yc = factors.get("yield_curve", {})
        spread = yc.get("spread_10y2y", 0)
        vix = factors.get("vix", {}).get("current", 20)
        dollar = factors.get("dollar", {})
        inflation = factors.get("inflation", {})

        sector_lower = sector.lower()

        # Financials benefit from steeper yield curve
        if "financial" in sector_lower:
            if spread > 1.0:
                adjustment += 0.5
                signals.append({
                    "name": "Macro — Yield Curve",
                    "impact": "positive",
                    "description": "Steep yield curve supports bank margins",
                })
            elif spread < 0:
                adjustment -= 0.5
                signals.append({
                    "name": "Macro — Yield Curve",
                    "impact": "negative",
                    "description": "Inverted curve compresses bank margins",
                })

        # Utilities hurt by rising rates
        if "utilities" in sector_lower:
            rate_10y = yc.get("rate_10y", 3.5)
            if rate_10y > 4.5:
                adjustment -= 0.5
                signals.append({
                    "name": "Macro — Interest Rates",
                    "impact": "negative",
                    "description": (
                        f"High 10Y rate ({rate_10y:.1f}%) "
                        "pressures utility valuations"
                    ),
                })

        # Energy benefits from dollar weakness
        if "energy" in sector_lower:
            if not dollar.get("strengthening", True):
                adjustment += 0.3
                signals.append({
                    "name": "Macro — Dollar",
                    "impact": "positive",
                    "description": "Weak dollar supports commodity prices",
                })

        # Tech hurt by rising inflation/rates
        if "technology" in sector_lower:
            if inflation.get("inflation_rising"):
                adjustment -= 0.3
                signals.append({
                    "name": "Macro — Inflation",
                    "impact": "negative",
                    "description": "Rising inflation expectations "
                    "compress growth stock valuations",
                })

        return {"adjustment": adjustment, "signals": signals}
