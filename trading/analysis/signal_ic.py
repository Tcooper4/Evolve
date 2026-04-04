"""
Signal Information Coefficient Analysis
========================================
Measures whether the AI Score actually predicts
forward returns. IC > 0.05 is meaningful edge.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _historical_score(hist_slice: pd.DataFrame) -> float:
    """
    Compute a simplified score using ONLY the passed price/volume history.
    No external API calls. No caching. Pure technical + momentum signals.

    Returns score 0-10.
    """
    try:
        col_map = {c.lower(): c for c in hist_slice.columns}
        close_col = col_map.get("close", hist_slice.columns[0])
        vol_col = col_map.get("volume")

        prices = pd.to_numeric(hist_slice[close_col], errors="coerce").dropna()
        if len(prices) < 20:
            return 0.0

        score_components = []

        # --- MOMENTUM signals (weight 0.40) ---

        sma20 = prices.rolling(20).mean().iloc[-1]
        last = float(prices.iloc[-1])
        if sma20 > 0 and not np.isnan(sma20):
            sma_signal = (last / float(sma20) - 1) * 100
            sma_score = np.clip(5 + sma_signal * 2, 0, 10)
            score_components.append(("momentum_sma", float(sma_score), 0.15))

        if len(prices) >= 21:
            mom_20 = (prices.iloc[-1] / prices.iloc[-21] - 1) * 100
            mom_score = np.clip(5 + mom_20 * 0.5, 0, 10)
            score_components.append(("momentum_20d", float(mom_score), 0.15))

        if len(prices) >= 15:
            delta = prices.diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = (100 - 100 / (1 + rs)).iloc[-1]
            if not np.isnan(rsi):
                if 40 <= rsi <= 70:
                    rsi_score = 6 + (rsi - 40) / 10
                elif rsi < 40:
                    rsi_score = rsi / 40 * 6
                else:
                    rsi_score = max(0, 10 - (rsi - 70) / 3)
                score_components.append(("rsi", float(rsi_score), 0.10))

        # --- TECHNICAL signals (weight 0.35) ---

        if len(prices) >= 50:
            sma50 = prices.rolling(50).mean().iloc[-1]
            if sma50 > 0 and not np.isnan(sma50):
                t50_signal = (last / float(sma50) - 1) * 100
                t50_score = np.clip(5 + t50_signal * 1.5, 0, 10)
                score_components.append(("tech_sma50", float(t50_score), 0.10))

        if len(prices) >= 20:
            x = np.arange(20)
            y = prices.iloc[-20:].values.astype(float)
            if np.std(y) > 0:
                corr = np.corrcoef(x, y)[0, 1]
                if not np.isnan(corr):
                    trend_score = np.clip(5 + corr * 5, 0, 10)
                    score_components.append(("trend", float(trend_score), 0.10))

        if len(prices) >= 20:
            returns = prices.pct_change().dropna().iloc[-20:]
            if len(returns) > 0:
                vol = float(returns.std() * np.sqrt(252))
                vol_score = np.clip(8 - (vol - 0.10) / 0.10, 0, 10)
                score_components.append(("volatility", float(vol_score), 0.08))

        if vol_col is not None and len(hist_slice) >= 10:
            vols = pd.to_numeric(hist_slice[vol_col], errors="coerce").dropna()
            if len(vols) >= 10 and vols.iloc[-5:].mean() > 0:
                vol_ratio = (
                    vols.iloc[-5:].mean() / vols.iloc[-20:].mean()
                    if len(vols) >= 20
                    else 1.0
                )
                v_score = np.clip(5 + (vol_ratio - 1) * 5, 0, 10)
                score_components.append(("volume", float(v_score), 0.07))

        if not score_components:
            return 5.0

        total_weight = sum(w for _, _, w in score_components)
        weighted_sum = sum(s * w for _, s, w in score_components)

        if total_weight > 0:
            final = weighted_sum / total_weight
        else:
            final = 5.0

        return float(np.clip(final, 0, 10))

    except Exception:
        return 5.0


@dataclass
class ICResult:
    """Result of IC analysis for one symbol."""

    symbol: str
    n_signals: int
    ic_1d: float  # correlation at 1-day horizon
    ic_3d: float  # correlation at 3-day horizon
    ic_7d: float  # correlation at 7-day horizon (primary)
    ic_14d: float  # correlation at 14-day horizon
    win_rate: float  # % signals with positive return
    mean_return: float  # mean 7d forward return
    sharpe: float  # Sharpe of signal-based strategy
    hit_rate_above_6: float  # win rate when score > 6.0

    def has_edge(self) -> bool:
        return (
            self.ic_7d > 0.05
            and self.win_rate > 0.52
            and self.n_signals >= 10
        )

    @property
    def is_contrarian(self) -> bool:
        return self.ic_7d < -0.05

    def summary(self) -> str:
        edge_label = (
            "✅ EDGE"
            if self.has_edge()
            else "🔄 CONTRARIAN"
            if self.is_contrarian
            else "❌ NO EDGE"
        )
        return (
            f"{self.symbol}: IC(7d)={self.ic_7d:.3f}, "
            f"Win={self.win_rate:.1%}, "
            f"Sharpe={self.sharpe:.2f} — {edge_label}"
        )


@dataclass
class ICReport:
    """Aggregate IC report across multiple symbols."""

    results: List[ICResult] = field(default_factory=list)
    mean_ic_7d: float = 0.0
    mean_win_rate: float = 0.0
    mean_sharpe: float = 0.0
    n_with_edge: int = 0
    verdict: str = "INSUFFICIENT DATA"

    def __post_init__(self) -> None:
        if self.results:
            self._compute_aggregates()

    def _compute_aggregates(self) -> None:
        valid = [r for r in self.results if r.n_signals >= 5]
        if not valid:
            return
        self.mean_ic_7d = float(np.mean([r.ic_7d for r in valid]))
        self.mean_win_rate = float(np.mean([r.win_rate for r in valid]))
        self.mean_sharpe = float(np.mean([r.sharpe for r in valid]))
        self.n_with_edge = sum(1 for r in valid if r.has_edge())

        if self.mean_ic_7d > 0.08:
            self.verdict = "STRONG EDGE"
        elif self.mean_ic_7d > 0.05:
            self.verdict = "MODERATE EDGE"
        elif self.mean_ic_7d > 0.02:
            self.verdict = "WEAK EDGE"
        else:
            self.verdict = "NO EDGE DETECTED"


class SignalICAnalyzer:
    """
    Computes Information Coefficient between
    historical price/volume-only scores and forward returns.

    Usage:
        analyzer = SignalICAnalyzer()
        report = analyzer.run_analysis(
            symbols=["AAPL", "MSFT", "NVDA"],
            lookback_days=252,
        )
        analyzer.render_streamlit(report)
    """

    def __init__(self) -> None:
        self.score_threshold = 5.0
        self.horizons = [1, 3, 7, 14]

    def compute_ic_for_symbol(
        self,
        symbol: str,
        lookback_days: int = 252,
    ) -> Optional[ICResult]:
        """
        For a given symbol:
        1. Get price history for lookback period
        2. Score each sample date using _historical_score (no external APIs)
        3. Measure actual forward returns
        4. Compute IC = corr(score, forward_return)
        """
        try:
            from scipy.stats import spearmanr

            from trading.data.price_cache import get_history

            period = f"{int(lookback_days)}d"
            hist = get_history(symbol, period=period, interval="1d")
            if hist is None or len(hist) < 30:
                logger.warning("Insufficient history for %s", symbol)
                return None

            col_map = {c.lower(): c for c in hist.columns}
            close_col = col_map.get("close", hist.columns[0])

            prices = pd.to_numeric(hist[close_col], errors="coerce").dropna()
            if len(prices) < 30:
                return None

            # Denser sampling when history is short (still avoids scoring every bar).
            step = 2 if len(hist) < 120 else 5

            scores = []
            dates = []

            for i in range(0, len(hist) - 20, step):
                slice_hist = hist.iloc[: i + 1]
                if len(slice_hist) < 20:
                    continue
                try:
                    score_val = _historical_score(slice_hist)
                    if score_val <= 0:
                        continue
                    scores.append(float(score_val))
                    dates.append(hist.index[i])
                except Exception:
                    continue

            if len(scores) < 10:
                logger.warning(
                    "Too few scores for %s: %d",
                    symbol,
                    len(scores),
                )
                return None

            scores_series = pd.Series(scores, index=dates)

            ic_values: Dict[int, float] = {}
            for h in self.horizons:
                fwd_returns = []
                valid_scores = []

                for date, score in scores_series.items():
                    future_idx = int(prices.index.searchsorted(date))
                    if future_idx >= len(prices) or future_idx + h >= len(prices):
                        continue

                    price_now = float(prices.iloc[future_idx])
                    price_future = float(prices.iloc[future_idx + h])

                    if price_now > 0:
                        ret = (price_future - price_now) / price_now
                        fwd_returns.append(ret)
                        valid_scores.append(score)

                if len(fwd_returns) < 5:
                    ic_values[h] = 0.0
                    continue

                ic, _ = spearmanr(valid_scores, fwd_returns)
                ic_values[h] = float(ic) if not np.isnan(ic) else 0.0

            fwd_7d = []
            valid_sc = []
            for date, score in scores_series.items():
                future_idx = int(prices.index.searchsorted(date))
                if future_idx >= len(prices) or future_idx + 7 >= len(prices):
                    continue
                price_now = float(prices.iloc[future_idx])
                price_future = float(prices.iloc[future_idx + 7])
                if price_now > 0:
                    ret = (price_future - price_now) / price_now
                    fwd_7d.append(ret)
                    valid_sc.append(score)

            if not fwd_7d:
                return None

            fwd_arr = np.array(fwd_7d, dtype=float)
            win_rate = float(np.mean(fwd_arr > 0))
            mean_ret = float(np.mean(fwd_arr))

            strategy_returns = [
                r for s, r in zip(valid_sc, fwd_7d) if s > self.score_threshold
            ]
            if len(strategy_returns) >= 5:
                sr = float(np.mean(strategy_returns))
                ss = float(np.std(strategy_returns))
                sharpe = float(sr / ss * np.sqrt(52)) if ss > 0 else 0.0
            else:
                sharpe = 0.0

            high_conv = [r for s, r in zip(valid_sc, fwd_7d) if s > 6.0]
            hit_above_6 = (
                float(np.mean([r > 0 for r in high_conv])) if high_conv else 0.0
            )

            return ICResult(
                symbol=symbol,
                n_signals=len(scores),
                ic_1d=ic_values.get(1, 0.0),
                ic_3d=ic_values.get(3, 0.0),
                ic_7d=ic_values.get(7, 0.0),
                ic_14d=ic_values.get(14, 0.0),
                win_rate=win_rate,
                mean_return=mean_ret,
                sharpe=sharpe,
                hit_rate_above_6=hit_above_6,
            )

        except Exception as e:
            logger.warning("IC analysis failed for %s: %s", symbol, e)
            return None

    def run_analysis(
        self,
        symbols: List[str],
        lookback_days: int = 252,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> ICReport:
        """Run IC analysis across multiple symbols."""
        results = []
        total = len(symbols)

        for i, symbol in enumerate(symbols):
            logger.info("IC analysis: %s (%d/%d)", symbol, i + 1, total)
            result = self.compute_ic_for_symbol(symbol, lookback_days)
            if result:
                results.append(result)
                logger.info(result.summary())
            if progress_callback:
                progress_callback(i + 1, total)

        return ICReport(results=results)

    def render_streamlit(self, report: ICReport) -> None:
        """Render IC analysis results in Streamlit."""
        try:
            import plotly.graph_objects as go
            import streamlit as st

            if not report.results:
                st.warning("No IC results available. Run analysis first.")
                return

            verdict_color = {
                "STRONG EDGE": "success",
                "MODERATE EDGE": "success",
                "WEAK EDGE": "warning",
                "NO EDGE DETECTED": "error",
                "INSUFFICIENT DATA": "warning",
            }.get(report.verdict, "info")

            getattr(st, verdict_color)(
                f"**Signal Edge: {report.verdict}** — "
                f"Mean IC(7d) = {report.mean_ic_7d:.3f} | "
                f"Win Rate = {report.mean_win_rate:.1%} | "
                f"Sharpe = {report.mean_sharpe:.2f}"
            )

            with st.expander("What does IC mean?", expanded=False):
                st.markdown(
                    """
**Information Coefficient (IC)** measures
correlation between the historical price/volume
score (no live APIs) and actual forward returns.

| IC Value | Interpretation |
|---|---|
| > 0.10 | Exceptional edge |
| 0.05–0.10 | Strong edge |
| 0.02–0.05 | Moderate edge |
| < 0.02 | No significant edge |

**Win Rate** = % of signals where the stock
went up over 7 days after the signal.
A coin flip = 50%. You need > 52% to have edge.

**Sharpe** = risk-adjusted return of a strategy
that buys when score > 5.0. > 0.5 is good.
"""
                )

            st.markdown("### IC by horizon")
            valid = [r for r in report.results if r.n_signals >= 10]
            if valid:
                horizons = [1, 3, 7, 14]
                mean_ics = [
                    float(np.mean([getattr(r, f"ic_{h}d") for r in valid]))
                    for h in horizons
                ]

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=horizons,
                        y=mean_ics,
                        mode="lines+markers",
                        name="Mean IC",
                        line=dict(color="#00d4ff", width=2),
                        marker=dict(size=8),
                    )
                )
                fig.add_hline(
                    y=0.05,
                    line_dash="dash",
                    line_color="green",
                    annotation_text="Edge threshold",
                )
                fig.add_hline(
                    y=0,
                    line_dash="solid",
                    line_color="gray",
                    opacity=0.3,
                )
                fig.update_layout(
                    template="plotly_dark",
                    height=300,
                    xaxis_title="Horizon (days)",
                    yaxis_title="Mean IC",
                    showlegend=False,
                )
                st.plotly_chart(fig, width="stretch")

            st.markdown("### Per-symbol results")
            rows = []
            for r in sorted(report.results, key=lambda x: x.ic_7d, reverse=True):
                rows.append(
                    {
                        "Symbol": r.symbol,
                        "Signals": r.n_signals,
                        "IC (1d)": f"{r.ic_1d:.3f}",
                        "IC (7d)": f"{r.ic_7d:.3f}",
                        "IC (14d)": f"{r.ic_14d:.3f}",
                        "Win Rate": f"{r.win_rate:.1%}",
                        "Sharpe": f"{r.sharpe:.2f}",
                        "High-Conv Win": f"{r.hit_rate_above_6:.1%}",
                        "Edge": (
                            "✅"
                            if r.has_edge()
                            else "🔄"
                            if r.is_contrarian
                            else "❌"
                        ),
                    }
                )

            st.dataframe(
                pd.DataFrame(rows),
                width="stretch",
                hide_index=True,
            )

        except Exception as e:
            try:
                import streamlit as st

                st.caption(f"IC visualization unavailable: {e}")
            except Exception:
                pass


def get_ic_analyzer() -> SignalICAnalyzer:
    return SignalICAnalyzer()
