"""
Econometric Diagnostics
========================
INTEGRATION NOTES:
- Drop into: trading/analysis/econometric_diagnostics.py
- Wire into: Analyze page "Diagnostics" lab (API: /api/diagnostics/{symbol};
  legacy alias /api/causal/{symbol})
- Call pattern:
    from trading.analysis.econometric_diagnostics import EconometricDiagnostics
    diag = EconometricDiagnostics(symbol, hist_df)
    results = diag.run_all()
    diag.render_streamlit()  # renders full diagnostics UI

Dependencies: statsmodels (already in requirements)

Honesty: this module runs stationarity / WN / ACF / ARCH / normality /
lag-order / structural-break diagnostics. It does **not** implement
Granger causality or any other causal-identification procedure.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EconometricDiagnostics:
    """
    Suite of econometric diagnostic tests for a single return series.

    Tests included:
    - Stationarity: ADF, KPSS
    - White noise: Ljung-Box
    - Autocorrelation: ACF, PACF
    - Heteroskedasticity: ARCH effects
    - Normality: Jarque-Bera
    - Structural breaks: Chow test approximation
    - Optimal lag selection: AIC/BIC

    Not included (do not claim otherwise): Granger causality or other
    causal identification vs SPY / a benchmark.
    """

    def __init__(
        self,
        symbol: str,
        data: pd.DataFrame,
    ):
        self.symbol = symbol
        self.data = data
        self._results: Dict[str, Any] = {}

        # Resolve close column
        _col_map = {c.lower(): c for c in data.columns}
        self.close_col = _col_map.get("close", list(
            data.select_dtypes(include="number").columns
        )[0])

        self.prices = data[self.close_col].dropna()
        self.returns = self.prices.pct_change().dropna()
        self.log_returns = np.log(self.prices / self.prices.shift(1)).dropna()

    def run_all(self) -> Dict[str, Any]:
        """Run all diagnostic tests and return results dict."""
        self._results = {
            "symbol": self.symbol,
            "n_observations": len(self.prices),
            "stationarity": self.test_stationarity(),
            "white_noise": self.test_white_noise(),
            "autocorrelation": self.compute_acf_pacf(),
            "arch_effects": self.test_arch_effects(),
            "normality": self.test_normality(),
            "optimal_lags": self.select_optimal_lags(),
            "structural_breaks": self.detect_structural_breaks(),
            "summary": {},
        }
        self._results["summary"] = self._generate_summary()
        return self._results

    def test_stationarity(self) -> Dict[str, Any]:
        """ADF and KPSS stationarity tests."""
        result = {}
        try:
            from statsmodels.tsa.stattools import adfuller, kpss

            # ADF test — H0: unit root (non-stationary)
            adf_result = adfuller(self.prices, autolag="AIC")
            result["adf"] = {
                "statistic": round(float(adf_result[0]), 4),
                "p_value": round(float(adf_result[1]), 4),
                "critical_values": {
                    k: round(v, 4)
                    for k, v in adf_result[4].items()
                },
                "is_stationary": adf_result[1] < 0.05,
                "interpretation": (
                    "Stationary (reject unit root)"
                    if adf_result[1] < 0.05
                    else "Non-stationary (unit root present)"
                ),
            }

            # ADF on returns
            adf_returns = adfuller(self.returns, autolag="AIC")
            result["adf_returns"] = {
                "statistic": round(float(adf_returns[0]), 4),
                "p_value": round(float(adf_returns[1]), 4),
                "is_stationary": adf_returns[1] < 0.05,
            }

            # KPSS test — H0: stationary
            try:
                kpss_result = kpss(self.prices, regression="c", nlags="auto")
                result["kpss"] = {
                    "statistic": round(float(kpss_result[0]), 4),
                    "p_value": round(float(kpss_result[1]), 4),
                    "is_stationary": kpss_result[1] > 0.05,
                    "interpretation": (
                        "Stationary (fail to reject)"
                        if kpss_result[1] > 0.05
                        else "Non-stationary (reject stationarity)"
                    ),
                }
            except Exception as e:
                logger.debug("KPSS test failed: %s", e)
                result["kpss"] = {"error": str(e)}

        except ImportError:
            result["error"] = "statsmodels not available"
        except Exception as e:
            logger.warning("Stationarity test failed: %s", e)
            result["error"] = str(e)

        return result

    def test_white_noise(self, lags: int = 20) -> Dict[str, Any]:
        """Ljung-Box white noise test on returns."""
        result = {}
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox

            lb_result = acorr_ljungbox(
                self.returns, lags=[lags], return_df=True
            )
            p_value = float(lb_result["lb_pvalue"].iloc[0])
            stat = float(lb_result["lb_stat"].iloc[0])

            result = {
                "statistic": round(stat, 4),
                "p_value": round(p_value, 4),
                "lags_tested": lags,
                "is_white_noise": p_value > 0.05,
                "interpretation": (
                    "Returns appear to be white noise (no autocorrelation)"
                    if p_value > 0.05
                    else "Significant autocorrelation detected in returns"
                ),
            }

            # Also test squared returns for ARCH
            lb_sq = acorr_ljungbox(
                self.returns ** 2, lags=[lags], return_df=True
            )
            result["squared_returns_p_value"] = round(
                float(lb_sq["lb_pvalue"].iloc[0]), 4
            )
            result["volatility_clustering"] = (
                float(lb_sq["lb_pvalue"].iloc[0]) < 0.05
            )

        except ImportError:
            result["error"] = "statsmodels not available"
        except Exception as e:
            logger.warning("White noise test failed: %s", e)
            result["error"] = str(e)

        return result

    def compute_acf_pacf(
        self, nlags: int = 30
    ) -> Dict[str, Any]:
        """Compute ACF and PACF values for plotting."""
        result = {}
        try:
            from statsmodels.tsa.stattools import acf, pacf

            acf_vals, acf_confint = acf(
                self.returns,
                nlags=nlags,
                alpha=0.05,
                fft=True,
            )
            pacf_vals, pacf_confint = pacf(
                self.returns,
                nlags=nlags,
                alpha=0.05,
            )

            conf_level = 1.96 / np.sqrt(len(self.returns))

            result = {
                "lags": list(range(nlags + 1)),
                "acf": [round(v, 4) for v in acf_vals],
                "pacf": [round(v, 4) for v in pacf_vals],
                "confidence_interval": round(conf_level, 4),
                "significant_acf_lags": [
                    i for i, v in enumerate(acf_vals)
                    if i > 0 and abs(v) > conf_level
                ],
                "significant_pacf_lags": [
                    i for i, v in enumerate(pacf_vals)
                    if i > 0 and abs(v) > conf_level
                ],
                "suggested_ar_order": self._suggest_ar_order(pacf_vals, conf_level),
                "suggested_ma_order": self._suggest_ma_order(acf_vals, conf_level),
            }

        except ImportError:
            result["error"] = "statsmodels not available"
        except Exception as e:
            logger.warning("ACF/PACF computation failed: %s", e)
            result["error"] = str(e)

        return result

    def test_arch_effects(self, lags: int = 10) -> Dict[str, Any]:
        """Test for ARCH effects (volatility clustering)."""
        result = {}
        try:
            from statsmodels.stats.diagnostic import het_arch

            arch_result = het_arch(self.returns, nlags=lags)
            p_value = float(arch_result[1])

            result = {
                "lm_statistic": round(float(arch_result[0]), 4),
                "p_value": round(p_value, 4),
                "f_statistic": round(float(arch_result[2]), 4),
                "f_p_value": round(float(arch_result[3]), 4),
                "arch_effects_present": p_value < 0.05,
                "interpretation": (
                    "ARCH effects present — GARCH model recommended"
                    if p_value < 0.05
                    else "No significant ARCH effects"
                ),
            }

        except ImportError:
            result["error"] = "statsmodels not available"
        except Exception as e:
            logger.warning("ARCH test failed: %s", e)
            result["error"] = str(e)

        return result

    def test_normality(self) -> Dict[str, Any]:
        """Jarque-Bera normality test on returns."""
        result = {}
        try:
            from scipy import stats

            jb_stat, jb_p = stats.jarque_bera(self.returns)
            skewness = float(stats.skew(self.returns))
            kurtosis = float(stats.kurtosis(self.returns))

            result = {
                "jarque_bera_statistic": round(float(jb_stat), 4),
                "p_value": round(float(jb_p), 4),
                "skewness": round(skewness, 4),
                "excess_kurtosis": round(kurtosis, 4),
                "is_normal": jb_p > 0.05,
                "interpretation": (
                    "Returns are approximately normal"
                    if jb_p > 0.05
                    else f"Returns are non-normal (skew={skewness:.2f}, "
                    f"excess kurtosis={kurtosis:.2f})"
                ),
                "fat_tails": kurtosis > 1.0,
                "negative_skew": skewness < -0.5,
            }

        except ImportError:
            result["error"] = "scipy not available"
        except Exception as e:
            logger.warning("Normality test failed: %s", e)
            result["error"] = str(e)

        return result

    def select_optimal_lags(self, max_lags: int = 10) -> Dict[str, Any]:
        """Select optimal lag order using AIC and BIC."""
        result = {}
        try:
            from statsmodels.tsa.ar_model import AutoReg
            from statsmodels.tsa.stattools import arma_order_select_ic

            aic_scores = {}
            bic_scores = {}

            for lag in range(1, max_lags + 1):
                try:
                    model = AutoReg(
                        self.returns, lags=lag, old_names=False
                    ).fit()
                    aic_scores[lag] = model.aic
                    bic_scores[lag] = model.bic
                except Exception:
                    continue

            if aic_scores:
                optimal_aic = min(aic_scores, key=aic_scores.get)
                optimal_bic = min(bic_scores, key=bic_scores.get)

                result = {
                    "optimal_lag_aic": optimal_aic,
                    "optimal_lag_bic": optimal_bic,
                    "recommended_lag": optimal_bic,  # BIC penalizes more
                    "aic_scores": {
                        k: round(v, 2) for k, v in aic_scores.items()
                    },
                    "bic_scores": {
                        k: round(v, 2) for k, v in bic_scores.items()
                    },
                    "interpretation": (
                        f"BIC suggests {optimal_bic} lag(s), "
                        f"AIC suggests {optimal_aic} lag(s)"
                    ),
                }
            else:
                result["error"] = "Could not fit AR models"

        except ImportError:
            result["error"] = "statsmodels not available"
        except Exception as e:
            logger.warning("Lag selection failed: %s", e)
            result["error"] = str(e)

        return result

    def detect_structural_breaks(self) -> Dict[str, Any]:
        """Detect potential structural breaks using rolling statistics."""
        result = {}
        try:
            window = min(60, len(self.returns) // 4)
            if window < 10:
                return {"error": "Insufficient data for structural break detection"}

            rolling_mean = self.returns.rolling(window).mean()
            rolling_std = self.returns.rolling(window).std()

            # Z-score of rolling mean vs overall mean
            overall_mean = self.returns.mean()
            overall_std = self.returns.std()

            z_scores = (rolling_mean - overall_mean) / (
                overall_std / np.sqrt(window)
            )

            # Find potential break points (|z| > 2.5)
            breaks = z_scores[abs(z_scores) > 2.5].dropna()

            result = {
                "n_potential_breaks": len(breaks),
                "break_dates": [
                    str(d.date()) for d in breaks.index[:5]
                ] if len(breaks) > 0 else [],
                "rolling_window": window,
                "regime_changes_detected": len(breaks) > 0,
                "interpretation": (
                    f"{len(breaks)} potential structural break(s) detected"
                    if len(breaks) > 0
                    else "No significant structural breaks detected"
                ),
                "rolling_volatility": {
                    "current": round(float(rolling_std.iloc[-1]), 6)
                    if not rolling_std.empty else None,
                    "mean": round(float(rolling_std.mean()), 6),
                    "is_elevated": float(rolling_std.iloc[-1]) >
                    float(rolling_std.mean()) * 1.5
                    if not rolling_std.empty else False,
                },
            }

        except Exception as e:
            logger.warning("Structural break detection failed: %s", e)
            result["error"] = str(e)

        return result

    def _suggest_ar_order(
        self, pacf_vals: np.ndarray, conf_level: float
    ) -> int:
        """Suggest AR order from PACF cutoff."""
        for i in range(len(pacf_vals) - 1, 0, -1):
            if abs(pacf_vals[i]) > conf_level:
                return min(i, 5)
        return 1

    def _suggest_ma_order(
        self, acf_vals: np.ndarray, conf_level: float
    ) -> int:
        """Suggest MA order from ACF cutoff."""
        for i in range(len(acf_vals) - 1, 0, -1):
            if abs(acf_vals[i]) > conf_level:
                return min(i, 5)
        return 1

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate plain-English summary of all diagnostics."""
        flags = []
        recommendations = []

        # Stationarity
        stat = self._results.get("stationarity", {})
        if stat.get("adf", {}).get("is_stationary") is False:
            flags.append("⚠️ Price series is non-stationary")
            recommendations.append(
                "Difference the series before modeling"
            )

        # White noise
        wn = self._results.get("white_noise", {})
        if not wn.get("is_white_noise", True):
            flags.append("📊 Significant autocorrelation in returns")
            recommendations.append(
                "Consider ARIMA or lag-based models"
            )
        if wn.get("volatility_clustering"):
            flags.append("📈 Volatility clustering detected")
            recommendations.append("GARCH model recommended for volatility")

        # Normality
        norm = self._results.get("normality", {})
        if norm.get("fat_tails"):
            flags.append("🔔 Fat tails in return distribution")
            recommendations.append(
                "Use robust risk metrics (CVaR vs VaR)"
            )
        if norm.get("negative_skew"):
            flags.append("📉 Negative skewness — downside risk elevated")

        # ARCH
        arch = self._results.get("arch_effects", {})
        if arch.get("arch_effects_present"):
            flags.append("🌊 ARCH effects present")

        return {
            "flags": flags,
            "recommendations": recommendations,
            "overall_complexity": (
                "High" if len(flags) >= 3
                else "Medium" if len(flags) >= 1
                else "Low"
            ),
        }

    def render_streamlit(self) -> None:
        """Render full diagnostics UI in Streamlit."""
        try:
            import plotly.graph_objects as go
            import streamlit as st

            if not self._results:
                self.run_all()

            st.subheader(f"📊 Econometric Diagnostics — {self.symbol}")

            # Summary flags
            summary = self._results.get("summary", {})
            if summary.get("flags"):
                for flag in summary["flags"]:
                    st.warning(flag)
            if summary.get("recommendations"):
                with st.expander("💡 Recommendations"):
                    for rec in summary["recommendations"]:
                        st.write(f"• {rec}")

            tab1, tab2, tab3, tab4 = st.tabs([
                "Stationarity", "ACF/PACF",
                "Distributions", "Advanced"
            ])

            with tab1:
                self._render_stationarity_tab(st)

            with tab2:
                self._render_acf_pacf_tab(st, go)

            with tab3:
                self._render_distribution_tab(st, go)

            with tab4:
                self._render_advanced_tab(st)

        except Exception as e:
            try:
                import streamlit as st
                st.caption(f"Diagnostics unavailable: {e}")
            except Exception:
                pass

    def _render_stationarity_tab(self, st) -> None:
        """Render stationarity test results."""
        stat = self._results.get("stationarity", {})

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**ADF Test (Price)**")
            adf = stat.get("adf", {})
            if "error" not in adf:
                status = "✅ Stationary" if adf.get("is_stationary") else "❌ Non-stationary"
                st.metric("Result", status)
                st.metric("p-value", f"{adf.get('p_value', 'N/A'):.4f}")
                st.caption(adf.get("interpretation", ""))

        with col2:
            st.markdown("**ADF Test (Returns)**")
            adf_r = stat.get("adf_returns", {})
            if "error" not in adf_r:
                status = "✅ Stationary" if adf_r.get("is_stationary") else "❌ Non-stationary"
                st.metric("Result", status)
                st.metric("p-value", f"{adf_r.get('p_value', 'N/A'):.4f}")

        # White noise
        st.markdown("---")
        st.markdown("**Ljung-Box White Noise Test**")
        wn = self._results.get("white_noise", {})
        if "error" not in wn:
            col1, col2, col3 = st.columns(3)
            col1.metric(
                "White Noise",
                "✅ Yes" if wn.get("is_white_noise") else "❌ No"
            )
            col2.metric("p-value", f"{wn.get('p_value', 'N/A'):.4f}")
            col3.metric(
                "Vol Clustering",
                "⚠️ Yes" if wn.get("volatility_clustering") else "✅ No"
            )
            st.caption(wn.get("interpretation", ""))

        # Optimal lags
        st.markdown("---")
        st.markdown("**Optimal Lag Selection**")
        lags = self._results.get("optimal_lags", {})
        if "error" not in lags:
            col1, col2 = st.columns(2)
            col1.metric("Optimal Lag (BIC)", lags.get("optimal_lag_bic"))
            col2.metric("Optimal Lag (AIC)", lags.get("optimal_lag_aic"))
            st.caption(lags.get("interpretation", ""))

    def _render_acf_pacf_tab(self, st, go) -> None:
        """Render ACF and PACF plots."""
        acf_data = self._results.get("autocorrelation", {})
        if "error" in acf_data:
            st.caption(f"ACF/PACF unavailable: {acf_data['error']}")
            return

        lags = acf_data.get("lags", [])
        acf_vals = acf_data.get("acf", [])
        pacf_vals = acf_data.get("pacf", [])
        conf = acf_data.get("confidence_interval", 0.1)

        # ACF plot
        fig_acf = go.Figure()
        fig_acf.add_trace(go.Bar(
            x=lags[1:], y=acf_vals[1:],
            name="ACF", marker_color="#00d4ff"
        ))
        fig_acf.add_hline(y=conf, line_dash="dash",
                          line_color="red", opacity=0.5)
        fig_acf.add_hline(y=-conf, line_dash="dash",
                          line_color="red", opacity=0.5)
        fig_acf.update_layout(
            title="Autocorrelation Function (ACF)",
            template="plotly_dark",
            height=300,
            xaxis_title="Lag",
            yaxis_title="Correlation",
        )
        st.plotly_chart(fig_acf, width='stretch')

        # PACF plot
        fig_pacf = go.Figure()
        fig_pacf.add_trace(go.Bar(
            x=lags[1:], y=pacf_vals[1:],
            name="PACF", marker_color="#7b61ff"
        ))
        fig_pacf.add_hline(y=conf, line_dash="dash",
                           line_color="red", opacity=0.5)
        fig_pacf.add_hline(y=-conf, line_dash="dash",
                           line_color="red", opacity=0.5)
        fig_pacf.update_layout(
            title="Partial Autocorrelation Function (PACF)",
            template="plotly_dark",
            height=300,
            xaxis_title="Lag",
            yaxis_title="Partial Correlation",
        )
        st.plotly_chart(fig_pacf, width='stretch')

        sig_acf = acf_data.get("significant_acf_lags", [])
        sig_pacf = acf_data.get("significant_pacf_lags", [])
        if sig_acf:
            st.caption(f"Significant ACF lags: {sig_acf[:10]}")
        if sig_pacf:
            st.caption(f"Significant PACF lags: {sig_pacf[:10]}")

    def _render_distribution_tab(self, st, go) -> None:
        """Render return distribution analysis."""
        norm = self._results.get("normality", {})
        arch = self._results.get("arch_effects", {})

        # Return distribution histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=self.returns.values,
            nbinsx=50,
            name="Returns",
            marker_color="#00d4ff",
            opacity=0.7,
            histnorm="probability density",
        ))

        # Overlay normal distribution
        from scipy import stats as scipy_stats
        x_range = np.linspace(
            self.returns.min(), self.returns.max(), 100
        )
        normal_pdf = scipy_stats.norm.pdf(
            x_range,
            self.returns.mean(),
            self.returns.std()
        )
        fig.add_trace(go.Scatter(
            x=x_range, y=normal_pdf,
            name="Normal fit",
            line=dict(color="red", dash="dash"),
        ))
        fig.update_layout(
            title="Return Distribution",
            template="plotly_dark",
            height=350,
            xaxis_title="Daily Return",
            yaxis_title="Density",
        )
        st.plotly_chart(fig, width='stretch')

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Skewness", f"{norm.get('skewness', 'N/A'):.3f}")
        col2.metric("Excess Kurtosis", f"{norm.get('excess_kurtosis', 'N/A'):.3f}")
        col3.metric("JB p-value", f"{norm.get('p_value', 'N/A'):.4f}")
        col4.metric("Normal", "✅" if norm.get("is_normal") else "❌")

        if "error" not in arch:
            st.markdown("**ARCH Effects**")
            col1, col2 = st.columns(2)
            col1.metric(
                "ARCH Effects",
                "⚠️ Present" if arch.get("arch_effects_present") else "✅ None"
            )
            col2.metric("p-value", f"{arch.get('p_value', 'N/A'):.4f}")
            st.caption(arch.get("interpretation", ""))

    def _render_advanced_tab(self, st) -> None:
        """Render structural breaks and advanced stats."""
        breaks = self._results.get("structural_breaks", {})

        st.markdown("**Structural Break Detection**")
        if "error" not in breaks:
            col1, col2 = st.columns(2)
            col1.metric(
                "Potential Breaks",
                breaks.get("n_potential_breaks", 0)
            )
            vol = breaks.get("rolling_volatility", {})
            col2.metric(
                "Current Vol",
                f"{vol.get('current', 0):.4f}" if vol.get("current") else "N/A",
                delta="Elevated" if vol.get("is_elevated") else "Normal",
                delta_color="inverse" if vol.get("is_elevated") else "normal",
            )
            if breaks.get("break_dates"):
                st.caption(
                    f"Break dates: {', '.join(breaks['break_dates'])}"
                )
            st.caption(breaks.get("interpretation", ""))


def run_stationarity_tests(data: pd.DataFrame) -> Dict[str, Any]:
    """ADF/KPSS on price level for router hooks; safe defaults if statsmodels fails."""
    try:
        if data is None or data.empty:
            return {"is_stationary": True}
        diag = EconometricDiagnostics("_wf", data)
        st_out = diag.test_stationarity()
        adf = st_out.get("adf") or {}
        return {
            "is_stationary": bool(adf.get("is_stationary", True)),
            "stationarity": st_out,
        }
    except Exception as e:
        logger.warning("run_stationarity_tests failed: %s", e)
        return {"is_stationary": True, "error": str(e)}
