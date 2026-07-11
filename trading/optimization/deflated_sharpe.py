# -*- coding: utf-8 -*-
"""Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

When an optimizer evaluates N parameter combinations and reports the
best Sharpe, that number is inflated by selection bias: even pure noise
produces an impressive-looking maximum across enough trials. The DSR
answers the honest question: *what is the probability that the observed
best Sharpe exceeds the maximum you'd expect from N trials of pure
noise?* — accounting for the number of trials, the variance of trial
results, track length, and non-normality (skew/kurtosis) of returns.

DSR ~ 1.0  -> the edge very likely survives the search process.
DSR ~ 0.5  -> indistinguishable from lucky noise; do not trust it.

References:
    Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe
    Ratio: Correcting for Selection Bias, Backtest Overfitting and
    Non-Normality." Journal of Portfolio Management.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence


def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _phi_inv(p: float) -> float:
    """Standard normal inverse CDF (Acklam's rational approximation;
    max abs error ~1.15e-9, far below what DSR needs)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def expected_max_sharpe(n_trials: int, trials_std: float,
                        trials_mean: float = 0.0) -> float:
    """Expected MAXIMUM Sharpe across n_trials under the null (no skill),
    per Bailey & Lopez de Prado's expression using extreme value theory:

        E[max SR] ~ mean + std * ((1-g)*Z(1-1/N) + g*Z(1-1/(N*e)))

    with g the Euler-Mascheroni constant.
    """
    if n_trials <= 1 or trials_std <= 0:
        return trials_mean
    gamma = 0.5772156649015329
    e = math.e
    z1 = _phi_inv(1.0 - 1.0 / n_trials)
    z2 = _phi_inv(1.0 - 1.0 / (n_trials * e))
    return trials_mean + trials_std * ((1.0 - gamma) * z1 + gamma * z2)


def probabilistic_sharpe(observed_sr: float, benchmark_sr: float,
                         n_obs: int, skew: float = 0.0,
                         kurtosis: float = 3.0) -> float:
    """PSR: probability that the TRUE Sharpe exceeds benchmark_sr, given
    the observed Sharpe over n_obs returns with the given skew/kurtosis
    (kurtosis is the raw Pearson kurtosis; 3.0 = normal)."""
    if n_obs <= 1:
        return 0.5
    denom = math.sqrt(
        max(1e-12,
            1.0 - skew * observed_sr
            + (kurtosis - 1.0) / 4.0 * observed_sr ** 2)
    )
    z = (observed_sr - benchmark_sr) * math.sqrt(n_obs - 1) / denom
    return _phi(z)


def deflated_sharpe_ratio(
    observed_sr: float,
    trial_scores: Sequence[float],
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> Optional[dict]:
    """The headline number: PSR of the observed (best) Sharpe against the
    expected-max-under-null benchmark implied by the search itself.

    Args:
        observed_sr: the selected (best) per-period Sharpe estimate.
        trial_scores: the metric across ALL optimizer evaluations (the
            search history) - its dispersion sets the null benchmark.
        n_obs: number of return observations backing observed_sr.
    """
    scores = [s for s in trial_scores
              if s is not None and math.isfinite(s)]
    if len(scores) < 2 or n_obs <= 1:
        return None
    mean = sum(scores) / len(scores)
    var = sum((s - mean) ** 2 for s in scores) / max(1, len(scores) - 1)
    bench = expected_max_sharpe(len(scores), math.sqrt(var), trials_mean=0.0)
    dsr = probabilistic_sharpe(observed_sr, bench, n_obs, skew, kurtosis)
    return {
        "deflated_sharpe": round(dsr, 4),
        "expected_max_sharpe_under_null": round(bench, 4),
        "n_trials": len(scores),
        "interpretation": (
            "likely real edge (survives the search)" if dsr >= 0.95
            else "uncertain - could be selection luck" if dsr >= 0.75
            else "indistinguishable from lucky noise across this many trials"
        ),
    }
