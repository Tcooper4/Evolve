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
            abs(self.ic_7d) > 0.05
            and self.win_rate > 0.52
            and self.n_signals >= 10
        )

    def summary(self) -> str:
        edge = "✅ EDGE DETECTED" if self.has_edge() else "❌ NO EDGE"
        return (
            f"{self.symbol}: IC(7d)={self.ic_7d:.3f}, "
            f"Win={self.win_rate:.1%}, "
            f"Sharpe={self.sharpe:.2f} — {edge}"
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
    AI Score signals and forward returns.

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
        2. Score each day using AI Score
        3. Measure actual forward returns
        4. Compute IC = corr(score, forward_return)
        """
        try:
            from scipy.stats import spearmanr

            from trading.analysis.ai_score import compute_ai_score
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
                    score_result = compute_ai_score(symbol, slice_hist)
                    score_val = (
                        score_result.get("overall_score", 0)
                        if isinstance(score_result, dict)
                        else float(score_result or 0)
                    )
                    if score_val > 0:
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
correlation between your AI Score and actual
forward returns.

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
that buys when AI Score > 5.0. > 0.5 is good.
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
                        "Edge": "✅" if r.has_edge() else "❌",
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
