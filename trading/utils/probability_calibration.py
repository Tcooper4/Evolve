# -*- coding: utf-8 -*-
"""Probability calibration utilities (Platt scaling + optional isotonic).

Infrastructure only — not wired into Analyze, Backtest, Scanner, Kelly,
AI Score, or any live route. Evolve does not currently emit probabilities
that need calibration; when that changes, prefer this module over reaching
for scikit-learn's isotonic path by default.

Why Platt is the default
------------------------
Isotonic regression is well-documented to overfit on small calibration
samples. Platt scaling (logistic map from scores → probabilities) is more
sample-efficient and is the standard choice until roughly 1,000+
calibration points are available. Callers who truly have a large, stable
calibration set may opt into ``method="isotonic"``; below
``MIN_ISOTONIC_SAMPLES`` we emit a warning (and still run if requested).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

MethodName = Literal["platt", "isotonic"]

# Literature / practical threshold: isotonic needs a large sample
MIN_ISOTONIC_SAMPLES = 1000


@dataclass
class CalibrationResult:
    """Fitted calibrator + transformed probabilities + diagnostics."""

    method: str
    calibrated: np.ndarray
    params: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)

    def transform(self, scores: Sequence[float]) -> np.ndarray:
        """Apply the fitted map to new raw scores."""
        s = np.asarray(scores, dtype=float).ravel()
        if self.method == "platt":
            a = float(self.params["a"])
            b = float(self.params["b"])
            return _sigmoid(a * s + b)
        # isotonic
        x = np.asarray(self.params["x"], dtype=float)
        y = np.asarray(self.params["y"], dtype=float)
        return np.interp(s, x, y, left=y[0], right=y[-1])


def brier_score(probs: Sequence[float], labels: Sequence[float]) -> float:
    """Mean squared error between probabilities and binary labels."""
    p = np.asarray(probs, dtype=float).ravel()
    y = np.asarray(labels, dtype=float).ravel()
    if p.size != y.size or p.size == 0:
        raise ValueError("probs and labels must be non-empty and same length")
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    probs: Sequence[float],
    labels: Sequence[float],
    n_bins: int = 10,
) -> float:
    """Binned ECE (equal-width bins on [0, 1])."""
    p = np.asarray(probs, dtype=float).ravel()
    y = np.asarray(labels, dtype=float).ravel()
    if p.size != y.size or p.size == 0:
        raise ValueError("probs and labels must be non-empty and same length")
    n_bins = max(2, int(n_bins))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = float(p.size)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y[mask]))
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_platt(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Fit P(y=1|s) = sigmoid(a*s + b) via logistic regression (sklearn)."""
    from sklearn.linear_model import LogisticRegression

    # Need both classes for a meaningful fit
    uniq = np.unique(labels)
    if uniq.size < 2:
        # Degenerate: map everything toward the observed class rate
        rate = float(np.clip(labels.mean(), 1e-6, 1.0 - 1e-6))
        # sigmoid(b) ≈ rate with a≈0
        b = float(np.log(rate / (1.0 - rate)))
        return 0.0, b

    x = scores.reshape(-1, 1)
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        fit_intercept=True,
    )
    clf.fit(x, labels.astype(int))
    a = float(clf.coef_.ravel()[0])
    b = float(clf.intercept_.ravel()[0])
    return a, b


def _fit_isotonic(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(scores, labels)
    # Store the piecewise map at unique sorted score anchors
    order = np.argsort(scores)
    x = scores[order]
    y = np.asarray(iso.transform(x), dtype=float)
    # Collapse duplicate x for stable interp
    uniq_x, idx = np.unique(x, return_index=True)
    return uniq_x, y[idx]


def calibrate_probabilities(
    scores: Sequence[float],
    labels: Sequence[Union[int, float, bool]],
    method: MethodName = "platt",
    *,
    n_bins: int = 10,
    min_isotonic_samples: int = MIN_ISOTONIC_SAMPLES,
    warn: bool = True,
) -> CalibrationResult:
    """Fit a calibrator on ``(scores, labels)`` and return calibrated probs.

    Parameters
    ----------
    scores :
        Raw model scores or uncalibrated probabilities in a comparable scale.
        For Platt, any real-valued score works; for fair Brier/ECE comparison
        against the raw values, pass values already in [0, 1] when possible.
    labels :
        Binary outcomes (0/1).
    method :
        ``"platt"`` (default) or ``"isotonic"``.
    n_bins :
        Bins for ECE diagnostics.
    min_isotonic_samples :
        Soft threshold; isotonic below this emits a warning.
    warn :
        If False, suppress the low-sample isotonic warning (tests may set this).

    Returns
    -------
    CalibrationResult
        Includes ``calibrated`` array aligned with ``scores``, fit ``params``,
        any ``warnings``, and Brier/ECE before vs after.
    """
    s = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(labels, dtype=float).ravel()
    if s.size == 0 or y.size == 0:
        raise ValueError("scores and labels must be non-empty")
    if s.size != y.size:
        raise ValueError(
            f"scores/labels length mismatch: {s.size} vs {y.size}"
        )
    if not np.all(np.isfinite(s)):
        raise ValueError("scores contain non-finite values")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("labels must be binary 0/1")

    method_l = str(method).strip().lower()
    if method_l not in ("platt", "isotonic"):
        raise ValueError("method must be 'platt' or 'isotonic'")

    notes: List[str] = []
    # Treat raw scores as probs for "before" metrics when they look like probs;
    # otherwise clip to [0,1] for a comparable baseline only.
    raw_as_prob = np.clip(s, 0.0, 1.0)
    before = {
        "brier": brier_score(raw_as_prob, y),
        "ece": expected_calibration_error(raw_as_prob, y, n_bins=n_bins),
    }

    params: Dict[str, Any] = {}
    if method_l == "platt":
        a, b = _fit_platt(s, y)
        calibrated = _sigmoid(a * s + b)
        params = {"a": a, "b": b}
    else:
        n = int(s.size)
        if n < int(min_isotonic_samples):
            msg = (
                f"isotonic calibration with n={n} < {min_isotonic_samples} "
                "is prone to overfitting; prefer method='platt' until you have "
                "roughly 1000+ calibration points"
            )
            notes.append(msg)
            if warn:
                warnings.warn(msg, UserWarning, stacklevel=2)
                logger.warning(msg)
        x_map, y_map = _fit_isotonic(s, y)
        calibrated = np.interp(s, x_map, y_map, left=y_map[0], right=y_map[-1])
        params = {"x": x_map, "y": y_map, "n_samples": n}

    calibrated = np.clip(calibrated, 0.0, 1.0)
    after = {
        "brier": brier_score(calibrated, y),
        "ece": expected_calibration_error(calibrated, y, n_bins=n_bins),
    }

    return CalibrationResult(
        method=method_l,
        calibrated=calibrated,
        params=params,
        warnings=notes,
        metrics_before=before,
        metrics_after=after,
    )


__all__ = [
    "MIN_ISOTONIC_SAMPLES",
    "CalibrationResult",
    "brier_score",
    "expected_calibration_error",
    "calibrate_probabilities",
]
