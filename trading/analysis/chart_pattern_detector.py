"""
Chart Pattern Detector
=======================
INTEGRATION NOTES:
- Drop into: trading/analysis/chart_pattern_detector.py
- Wire into: pages/2_Analyze.py chart section
- Call pattern:
    from trading.analysis.chart_pattern_detector import ChartPatternDetector
    detector = ChartPatternDetector(symbol, hist_df)
    patterns = detector.detect_all()
    detector.render_streamlit()
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    """A detected chart pattern."""
    name: str
    pattern_type: str  # 'bullish', 'bearish', 'neutral'
    confidence: float  # 0-1
    start_idx: int
    end_idx: int
    start_date: Optional[str]
    end_date: Optional[str]
    description: str
    target_price: Optional[float] = None
    stop_price: Optional[float] = None


@dataclass
class SupportResistanceLevel:
    """A support or resistance level."""
    price: float
    level_type: str  # 'support' or 'resistance'
    strength: float  # 0-1, how many times tested
    touches: int
    last_touch_date: Optional[str]


class ChartPatternDetector:
    """
    Detects common chart patterns and support/resistance levels.

    Patterns detected:
    - Head and Shoulders (bearish reversal)
    - Inverse Head and Shoulders (bullish reversal)
    - Double Top (bearish reversal)
    - Double Bottom (bullish reversal)
    - Ascending/Descending Triangles
    - Support and Resistance levels
    - Higher Highs/Lower Lows (trend confirmation)
    - Golden/Death Cross (MA crossovers)
    """

    def __init__(self, symbol: str, data: pd.DataFrame):
        self.symbol = symbol
        self.data = data
        self._patterns: List[Pattern] = []
        self._sr_levels: List[SupportResistanceLevel] = []

        _col_map = {c.lower(): c for c in data.columns}
        self.close_col = _col_map.get("close", data.columns[0])
        self.high_col = _col_map.get("high", self.close_col)
        self.low_col = _col_map.get("low", self.close_col)
        self.volume_col = _col_map.get("volume")

        self.closes = data[self.close_col].values.astype(float)
        self.highs = data[self.high_col].values.astype(float)
        self.lows = data[self.low_col].values.astype(float)
        self.dates = data.index

    def detect_all(self) -> Dict[str, Any]:
        """Run all pattern detection and return results."""
        self._patterns = []
        self._sr_levels = []

        if len(self.closes) < 50:
            return {
                "patterns": [],
                "support_resistance": [],
                "trend": {"direction": "INSUFFICIENT_DATA"},
                "signals": [],
            }

        # Detect patterns
        self._detect_head_and_shoulders()
        self._detect_double_top_bottom()
        self._detect_triangles()
        self._detect_ma_crossovers()
        self._detect_support_resistance()
        trend = self._analyze_trend()

        # Generate trading signals
        signals = self._generate_signals(trend)

        return {
            "patterns": [self._pattern_to_dict(p) for p in self._patterns],
            "support_resistance": [
                self._sr_to_dict(sr) for sr in self._sr_levels
            ],
            "trend": trend,
            "signals": signals,
            "last_price": float(self.closes[-1]),
        }

    def _find_peaks(
        self, data: np.ndarray, window: int = 5
    ) -> np.ndarray:
        """Find local peaks (highs)."""
        peaks = []
        for i in range(window, len(data) - window):
            if data[i] == max(data[i - window:i + window + 1]):
                peaks.append(i)
        return np.array(peaks)

    def _find_troughs(
        self, data: np.ndarray, window: int = 5
    ) -> np.ndarray:
        """Find local troughs (lows)."""
        troughs = []
        for i in range(window, len(data) - window):
            if data[i] == min(data[i - window:i + window + 1]):
                troughs.append(i)
        return np.array(troughs)

    def _detect_head_and_shoulders(self) -> None:
        """Detect head and shoulders pattern."""
        try:
            peaks = self._find_peaks(self.highs, window=7)
            if len(peaks) < 3:
                return

            # Look for three peaks where middle is highest
            for i in range(len(peaks) - 2):
                left = peaks[i]
                head = peaks[i + 1]
                right = peaks[i + 2]

                left_h = self.highs[left]
                head_h = self.highs[head]
                right_h = self.highs[right]

                # H&S: head > left shoulder AND head > right shoulder
                # Shoulders roughly equal
                if (head_h > left_h * 1.02
                        and head_h > right_h * 1.02
                        and abs(left_h - right_h) / left_h < 0.05
                        and (right - left) > 20):

                    # Find neckline
                    neckline = min(
                        self.lows[left:head].min(),
                        self.lows[head:right].min()
                    )
                    target = neckline - (head_h - neckline)

                    confidence = min(0.9, 0.6 + (
                        1 - abs(left_h - right_h) / left_h
                    ) * 0.3)

                    self._patterns.append(Pattern(
                        name="Head and Shoulders",
                        pattern_type="bearish",
                        confidence=round(confidence, 2),
                        start_idx=left,
                        end_idx=right,
                        start_date=str(self.dates[left].date()) if hasattr(self.dates[left], 'date') else str(self.dates[left]),
                        end_date=str(self.dates[right].date()) if hasattr(self.dates[right], 'date') else str(self.dates[right]),
                        description=(
                            f"Bearish reversal pattern. "
                            f"Head at ${head_h:.2f}, "
                            f"neckline at ${neckline:.2f}"
                        ),
                        target_price=round(target, 2),
                        stop_price=round(head_h * 1.02, 2),
                    ))

            # Inverse H&S (bullish)
            troughs = self._find_troughs(self.lows, window=7)
            if len(troughs) >= 3:
                for i in range(len(troughs) - 2):
                    left = troughs[i]
                    head = troughs[i + 1]
                    right = troughs[i + 2]

                    left_l = self.lows[left]
                    head_l = self.lows[head]
                    right_l = self.lows[right]

                    if (head_l < left_l * 0.98
                            and head_l < right_l * 0.98
                            and abs(left_l - right_l) / left_l < 0.05
                            and (right - left) > 20):

                        neckline = max(
                            self.highs[left:head].max(),
                            self.highs[head:right].max()
                        )
                        target = neckline + (neckline - head_l)
                        confidence = min(0.9, 0.6 + (
                            1 - abs(left_l - right_l) / left_l
                        ) * 0.3)

                        self._patterns.append(Pattern(
                            name="Inverse Head and Shoulders",
                            pattern_type="bullish",
                            confidence=round(confidence, 2),
                            start_idx=left,
                            end_idx=right,
                            start_date=str(self.dates[left].date()) if hasattr(self.dates[left], 'date') else str(self.dates[left]),
                            end_date=str(self.dates[right].date()) if hasattr(self.dates[right], 'date') else str(self.dates[right]),
                            description=(
                                f"Bullish reversal pattern. "
                                f"Head at ${head_l:.2f}, "
                                f"neckline at ${neckline:.2f}"
                            ),
                            target_price=round(target, 2),
                            stop_price=round(head_l * 0.98, 2),
                        ))

        except Exception as e:
            logger.debug("H&S detection failed: %s", e)

    def _detect_double_top_bottom(self) -> None:
        """Detect double top and double bottom patterns."""
        try:
            peaks = self._find_peaks(self.highs, window=7)
            troughs = self._find_troughs(self.lows, window=7)

            # Double top
            for i in range(len(peaks) - 1):
                p1, p2 = peaks[i], peaks[i + 1]
                if p2 - p1 < 10:
                    continue

                h1, h2 = self.highs[p1], self.highs[p2]
                if abs(h1 - h2) / h1 < 0.03:  # within 3%
                    valley = self.lows[p1:p2].min()
                    target = valley - (max(h1, h2) - valley)

                    self._patterns.append(Pattern(
                        name="Double Top",
                        pattern_type="bearish",
                        confidence=round(
                            0.7 + (1 - abs(h1 - h2) / h1) * 0.2, 2
                        ),
                        start_idx=p1,
                        end_idx=p2,
                        start_date=str(self.dates[p1].date()) if hasattr(self.dates[p1], 'date') else str(self.dates[p1]),
                        end_date=str(self.dates[p2].date()) if hasattr(self.dates[p2], 'date') else str(self.dates[p2]),
                        description=(
                            f"Bearish reversal. Two tops near "
                            f"${max(h1,h2):.2f}"
                        ),
                        target_price=round(target, 2),
                        stop_price=round(max(h1, h2) * 1.02, 2),
                    ))

            # Double bottom
            for i in range(len(troughs) - 1):
                t1, t2 = troughs[i], troughs[i + 1]
                if t2 - t1 < 10:
                    continue

                l1, l2 = self.lows[t1], self.lows[t2]
                if abs(l1 - l2) / l1 < 0.03:
                    peak = self.highs[t1:t2].max()
                    target = peak + (peak - min(l1, l2))

                    self._patterns.append(Pattern(
                        name="Double Bottom",
                        pattern_type="bullish",
                        confidence=round(
                            0.7 + (1 - abs(l1 - l2) / l1) * 0.2, 2
                        ),
                        start_idx=t1,
                        end_idx=t2,
                        start_date=str(self.dates[t1].date()) if hasattr(self.dates[t1], 'date') else str(self.dates[t1]),
                        end_date=str(self.dates[t2].date()) if hasattr(self.dates[t2], 'date') else str(self.dates[t2]),
                        description=(
                            f"Bullish reversal. Two bottoms near "
                            f"${min(l1,l2):.2f}"
                        ),
                        target_price=round(target, 2),
                        stop_price=round(min(l1, l2) * 0.98, 2),
                    ))

        except Exception as e:
            logger.debug("Double top/bottom detection failed: %s", e)

    def _detect_triangles(self) -> None:
        """Detect ascending and descending triangles."""
        try:
            n = len(self.closes)
            if n < 40:
                return

            window = min(40, n // 2)
            recent_highs = self.highs[-window:]
            recent_lows = self.lows[-window:]
            x = np.arange(window)

            # Fit trend lines
            high_slope, high_intercept = np.polyfit(x, recent_highs, 1)
            low_slope, low_intercept = np.polyfit(x, recent_lows, 1)

            # Ascending triangle: flat resistance, rising support
            if (abs(high_slope) < 0.01 * recent_highs.mean()
                    and low_slope > 0.005 * recent_lows.mean()):
                resistance = recent_highs.mean()
                self._patterns.append(Pattern(
                    name="Ascending Triangle",
                    pattern_type="bullish",
                    confidence=0.65,
                    start_idx=n - window,
                    end_idx=n - 1,
                    start_date=str(self.dates[n - window].date()) if hasattr(self.dates[n - window], 'date') else None,
                    end_date=str(self.dates[-1].date()) if hasattr(self.dates[-1], 'date') else None,
                    description=(
                        f"Bullish continuation. "
                        f"Resistance at ${resistance:.2f}, rising support"
                    ),
                    target_price=round(
                        resistance + (resistance - recent_lows.mean()), 2
                    ),
                    stop_price=round(recent_lows[-1] * 0.98, 2),
                ))

            # Descending triangle: flat support, falling resistance
            elif (abs(low_slope) < 0.01 * recent_lows.mean()
                    and high_slope < -0.005 * recent_highs.mean()):
                support = recent_lows.mean()
                self._patterns.append(Pattern(
                    name="Descending Triangle",
                    pattern_type="bearish",
                    confidence=0.65,
                    start_idx=n - window,
                    end_idx=n - 1,
                    start_date=str(self.dates[n - window].date()) if hasattr(self.dates[n - window], 'date') else None,
                    end_date=str(self.dates[-1].date()) if hasattr(self.dates[-1], 'date') else None,
                    description=(
                        f"Bearish continuation. "
                        f"Support at ${support:.2f}, falling resistance"
                    ),
                    target_price=round(
                        support - (recent_highs.mean() - support), 2
                    ),
                    stop_price=round(recent_highs[-1] * 1.02, 2),
                ))

        except Exception as e:
            logger.debug("Triangle detection failed: %s", e)

    def _detect_ma_crossovers(self) -> None:
        """Detect golden cross and death cross."""
        try:
            if len(self.closes) < 200:
                return

            ma50 = pd.Series(self.closes).rolling(50).mean().values
            ma200 = pd.Series(self.closes).rolling(200).mean().values

            if np.isnan(ma50[-1]) or np.isnan(ma200[-1]):
                return

            # Check recent crossover (within last 10 days)
            for i in range(
                max(201, len(self.closes) - 10),
                len(self.closes)
            ):
                if np.isnan(ma50[i]) or np.isnan(ma200[i]):
                    continue
                if np.isnan(ma50[i-1]) or np.isnan(ma200[i-1]):
                    continue

                # Golden cross
                if (ma50[i] > ma200[i] and ma50[i-1] <= ma200[i-1]):
                    self._patterns.append(Pattern(
                        name="Golden Cross",
                        pattern_type="bullish",
                        confidence=0.75,
                        start_idx=i,
                        end_idx=i,
                        start_date=str(self.dates[i].date()) if hasattr(self.dates[i], 'date') else None,
                        end_date=str(self.dates[i].date()) if hasattr(self.dates[i], 'date') else None,
                        description=(
                            f"Bullish signal: 50MA crossed above 200MA. "
                            f"50MA=${ma50[i]:.2f}, 200MA=${ma200[i]:.2f}"
                        ),
                    ))

                # Death cross
                elif (ma50[i] < ma200[i] and ma50[i-1] >= ma200[i-1]):
                    self._patterns.append(Pattern(
                        name="Death Cross",
                        pattern_type="bearish",
                        confidence=0.75,
                        start_idx=i,
                        end_idx=i,
                        start_date=str(self.dates[i].date()) if hasattr(self.dates[i], 'date') else None,
                        end_date=str(self.dates[i].date()) if hasattr(self.dates[i], 'date') else None,
                        description=(
                            f"Bearish signal: 50MA crossed below 200MA. "
                            f"50MA=${ma50[i]:.2f}, 200MA=${ma200[i]:.2f}"
                        ),
                    ))

        except Exception as e:
            logger.debug("MA crossover detection failed: %s", e)

    def _detect_support_resistance(
        self, n_levels: int = 5, tolerance: float = 0.02
    ) -> None:
        """Detect key support and resistance levels."""
        try:
            peaks = self._find_peaks(self.highs, window=5)
            troughs = self._find_troughs(self.lows, window=5)

            # Cluster peaks and troughs
            all_levels = []
            for p in peaks:
                all_levels.append(("resistance", self.highs[p], p))
            for t in troughs:
                all_levels.append(("support", self.lows[t], t))

            # Group nearby levels
            clustered = {}
            for level_type, price, idx in all_levels:
                placed = False
                for cluster_price in clustered:
                    if abs(price - cluster_price) / cluster_price < tolerance:
                        clustered[cluster_price]["count"] += 1
                        clustered[cluster_price]["last_idx"] = max(
                            clustered[cluster_price]["last_idx"], idx
                        )
                        placed = True
                        break
                if not placed:
                    clustered[price] = {
                        "type": level_type,
                        "count": 1,
                        "last_idx": idx,
                    }

            # Sort by strength and take top levels
            sorted_levels = sorted(
                clustered.items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )[:n_levels * 2]

            current_price = self.closes[-1]
            for price, data in sorted_levels:
                # Determine if support or resistance relative to current price
                if price < current_price * 0.99:
                    level_type = "support"
                elif price > current_price * 1.01:
                    level_type = "resistance"
                else:
                    continue

                last_idx = data["last_idx"]
                self._sr_levels.append(SupportResistanceLevel(
                    price=round(float(price), 2),
                    level_type=level_type,
                    strength=round(min(1.0, data["count"] / 5), 2),
                    touches=data["count"],
                    last_touch_date=str(self.dates[last_idx].date()) if hasattr(self.dates[last_idx], 'date') else None,
                ))

            # Sort: supports descending, resistances ascending
            self._sr_levels.sort(
                key=lambda x: (
                    x.level_type,
                    -x.price if x.level_type == "support" else x.price
                )
            )

        except Exception as e:
            logger.debug("Support/resistance detection failed: %s", e)

    def _analyze_trend(self) -> Dict[str, Any]:
        """Analyze current trend using multiple timeframes."""
        try:
            n = len(self.closes)
            result = {}

            # Short term (20 days)
            if n >= 20:
                short_slope = np.polyfit(
                    range(20), self.closes[-20:], 1
                )[0]
                result["short_term"] = (
                    "BULLISH" if short_slope > 0 else "BEARISH"
                )
                result["short_slope"] = round(float(short_slope), 4)

            # Medium term (50 days)
            if n >= 50:
                med_slope = np.polyfit(
                    range(50), self.closes[-50:], 1
                )[0]
                result["medium_term"] = (
                    "BULLISH" if med_slope > 0 else "BEARISH"
                )

            # Long term (200 days)
            if n >= 200:
                long_slope = np.polyfit(
                    range(200), self.closes[-200:], 1
                )[0]
                result["long_term"] = (
                    "BULLISH" if long_slope > 0 else "BEARISH"
                )

            # Higher highs / lower lows
            recent_peaks = self._find_peaks(self.highs[-60:], window=5)
            recent_troughs = self._find_troughs(self.lows[-60:], window=5)

            if len(recent_peaks) >= 2:
                result["higher_highs"] = (
                    self.highs[-60:][recent_peaks[-1]]
                    > self.highs[-60:][recent_peaks[-2]]
                )
            if len(recent_troughs) >= 2:
                result["higher_lows"] = (
                    self.lows[-60:][recent_troughs[-1]]
                    > self.lows[-60:][recent_troughs[-2]]
                )

            # Overall direction
            bullish_signals = sum([
                result.get("short_term") == "BULLISH",
                result.get("medium_term") == "BULLISH",
                result.get("long_term") == "BULLISH",
                result.get("higher_highs", False),
                result.get("higher_lows", False),
            ])
            result["direction"] = (
                "BULLISH" if bullish_signals >= 3
                else "BEARISH" if bullish_signals <= 2
                else "NEUTRAL"
            )
            result["strength"] = round(bullish_signals / 5, 2)

            return result

        except Exception as e:
            logger.debug("Trend analysis failed: %s", e)
            return {"direction": "NEUTRAL", "error": str(e)}

    def _generate_signals(
        self, trend: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate trading signals from patterns and trend."""
        signals = []
        current_price = float(self.closes[-1])

        # Pattern-based signals
        for pattern in self._patterns:
            if pattern.confidence >= 0.6:
                signals.append({
                    "type": pattern.pattern_type.upper(),
                    "source": pattern.name,
                    "confidence": pattern.confidence,
                    "description": pattern.description,
                    "target": pattern.target_price,
                    "stop": pattern.stop_price,
                })

        # Support/resistance signals
        for level in self._sr_levels[:3]:
            pct_from_level = (
                (current_price - level.price) / level.price * 100
            )
            if (level.level_type == "support"
                    and abs(pct_from_level) < 3
                    and level.strength > 0.4):
                signals.append({
                    "type": "BULLISH",
                    "source": "Support Level",
                    "confidence": level.strength * 0.8,
                    "description": (
                        f"Price near strong support at "
                        f"${level.price:.2f} "
                        f"({level.touches} touches)"
                    ),
                    "target": None,
                    "stop": round(level.price * 0.97, 2),
                })
            elif (level.level_type == "resistance"
                    and abs(pct_from_level) < 3
                    and level.strength > 0.4):
                signals.append({
                    "type": "BEARISH",
                    "source": "Resistance Level",
                    "confidence": level.strength * 0.8,
                    "description": (
                        f"Price near strong resistance at "
                        f"${level.price:.2f} "
                        f"({level.touches} touches)"
                    ),
                    "target": None,
                    "stop": round(level.price * 1.03, 2),
                })

        return signals

    def _pattern_to_dict(self, p: Pattern) -> Dict[str, Any]:
        return {
            "name": p.name,
            "type": p.pattern_type,
            "confidence": p.confidence,
            "start_date": p.start_date,
            "end_date": p.end_date,
            "description": p.description,
            "target_price": p.target_price,
            "stop_price": p.stop_price,
        }

    def _sr_to_dict(self, sr: SupportResistanceLevel) -> Dict[str, Any]:
        return {
            "price": sr.price,
            "type": sr.level_type,
            "strength": sr.strength,
            "touches": sr.touches,
            "last_touch": sr.last_touch_date,
        }

    def render_streamlit(
        self, fig=None
    ) -> Optional[Any]:
        """
        Render pattern detection results in Streamlit.
        Optionally overlays on existing Plotly figure.
        Returns updated figure if provided.
        """
        try:
            import plotly.graph_objects as go
            import streamlit as st

            results = self.detect_all()

            if not results["patterns"] and not results["support_resistance"]:
                st.caption("No significant patterns detected")
                return fig

            # Show patterns
            patterns = results["patterns"]
            if patterns:
                st.markdown("**Detected Patterns**")
                for p in patterns:
                    color = (
                        "🟢" if p["type"] == "bullish"
                        else "🔴" if p["type"] == "bearish"
                        else "🟡"
                    )
                    with st.expander(
                        f"{color} {p['name']} "
                        f"(confidence: {p['confidence']:.0%})"
                    ):
                        st.write(p["description"])
                        if p.get("target_price"):
                            st.write(f"Target: ${p['target_price']:.2f}")
                        if p.get("stop_price"):
                            st.write(f"Stop: ${p['stop_price']:.2f}")

            # Show S/R levels
            sr_levels = results["support_resistance"]
            if sr_levels:
                st.markdown("**Support & Resistance**")
                current = results["last_price"]
                for level in sr_levels[:6]:
                    pct = (current - level["price"]) / level["price"] * 100
                    icon = "🟩" if level["type"] == "support" else "🟥"
                    st.write(
                        f"{icon} {level['type'].title()}: "
                        f"${level['price']:.2f} "
                        f"({pct:+.1f}% from current, "
                        f"{level['touches']} touches)"
                    )

            # Overlay on plotly figure
            if fig is not None:
                current_price = results["last_price"]

                # Add S/R lines
                for level in sr_levels[:6]:
                    color = (
                        "rgba(0,255,0,0.3)"
                        if level["type"] == "support"
                        else "rgba(255,0,0,0.3)"
                    )
                    fig.add_hline(
                        y=level["price"],
                        line_dash="dot",
                        line_color=color,
                        annotation_text=(
                            f"{level['type'].title()} "
                            f"${level['price']:.2f}"
                        ),
                        annotation_position="right",
                    )

                return fig

        except Exception as e:
            try:
                import streamlit as st
                st.caption(f"Pattern detection unavailable: {e}")
            except Exception:
                pass

        return fig
