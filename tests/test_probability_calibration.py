# -*- coding: utf-8 -*-
"""Tests for trading.utils.probability_calibration (Platt / isotonic)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from trading.utils.probability_calibration import (
    MIN_ISOTONIC_SAMPLES,
    brier_score,
    calibrate_probabilities,
    expected_calibration_error,
)


def _miscalibrated_scores(n: int = 400, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic classifier: true probs distorted by a known monotonic map.

    True p ~ Beta; labels ~ Bern(p); reported score = p^2 (under-confident
    on highs / over-confident on lows relative to identity — a classic
    miscalibration shape that Platt can partially undo).
    """
    rng = np.random.default_rng(seed)
    true_p = rng.beta(2.0, 2.0, size=n)
    labels = (rng.random(n) < true_p).astype(float)
    # Squaring pushes mass toward 0 → systematically miscalibrated
    scores = true_p ** 2
    return scores, labels


class TestMetrics:
    def test_brier_perfect(self):
        assert brier_score([0, 1, 0.5], [0, 1, 0.5]) == pytest.approx(0.0)

    def test_ece_perfect(self):
        # Constant 0.5 with half positives in that bin → ECE ~ 0
        probs = [0.5] * 100
        labels = [0] * 50 + [1] * 50
        assert expected_calibration_error(probs, labels, n_bins=10) == pytest.approx(0.0)


class TestPlatt:
    def test_improves_brier_on_miscalibrated_scores(self):
        scores, labels = _miscalibrated_scores(500, seed=11)
        raw_brier = brier_score(scores, labels)
        result = calibrate_probabilities(scores, labels, method="platt")
        assert result.method == "platt"
        assert "a" in result.params and "b" in result.params
        assert result.metrics_after["brier"] < raw_brier
        assert result.metrics_after["brier"] <= result.metrics_before["brier"]
        # Calibrated outputs are valid probabilities
        assert np.all(result.calibrated >= 0.0) and np.all(result.calibrated <= 1.0)

    def test_improves_ece_on_miscalibrated_scores(self):
        scores, labels = _miscalibrated_scores(600, seed=23)
        result = calibrate_probabilities(scores, labels, method="platt", n_bins=10)
        assert result.metrics_after["ece"] <= result.metrics_before["ece"] + 1e-9

    def test_transform_matches_calibrated(self):
        scores, labels = _miscalibrated_scores(200, seed=3)
        result = calibrate_probabilities(scores, labels, method="platt")
        again = result.transform(scores)
        np.testing.assert_allclose(again, result.calibrated, rtol=1e-6, atol=1e-6)

    def test_rejects_bad_labels(self):
        with pytest.raises(ValueError, match="binary"):
            calibrate_probabilities([0.1, 0.9], [0, 2])

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="mismatch"):
            calibrate_probabilities([0.1, 0.2], [0])


class TestIsotonicGuard:
    def test_low_sample_warning(self):
        scores, labels = _miscalibrated_scores(80, seed=5)
        assert len(scores) < MIN_ISOTONIC_SAMPLES
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = calibrate_probabilities(
                scores, labels, method="isotonic", warn=True
            )
        assert result.method == "isotonic"
        assert any("isotonic" in str(w.message).lower() for w in caught)
        assert any("overfit" in m.lower() or "platt" in m.lower() for m in result.warnings)

    def test_platt_default_no_isotonic_warning(self):
        scores, labels = _miscalibrated_scores(80, seed=5)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            calibrate_probabilities(scores, labels, method="platt")
        iso_warns = [w for w in caught if "isotonic" in str(w.message).lower()]
        assert iso_warns == []
