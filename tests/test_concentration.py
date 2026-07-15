# -*- coding: utf-8 -*-
"""Hand-verifiable concentration / correlation flags."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.portfolio.concentration import (
    HIGH_PAIRWISE_CORR,
    build_concentration_report,
    flag_high_correlation_pairs,
)


class TestHighCorrThreshold:
    def test_threshold_is_seventy_with_variance_rationale(self):
        assert HIGH_PAIRWISE_CORR == pytest.approx(0.70)
        # rho^2 shared variance ~49% at the cutoff
        assert HIGH_PAIRWISE_CORR ** 2 == pytest.approx(0.49)


class TestFlagPairsHand:
    def test_perfect_correlation_flagged(self):
        # Constructed: B = A exactly → rho=1
        n = 80
        a = np.linspace(100, 120, n)
        b = a.copy()
        hist = {
            "AAA": pd.Series(a),
            "BBB": pd.Series(b),
        }
        rep = build_concentration_report(["AAA", "BBB"], hist)
        assert rep["success"] is True
        pairs = rep["high_pairs"]
        assert len(pairs) >= 1
        assert pairs[0]["abs_correlation"] >= 0.99
        assert "AAA" in pairs[0]["message"] and "BBB" in pairs[0]["message"]
        assert "diversification" in pairs[0]["message"].lower()

    def test_uncorrelated_not_flagged(self):
        # Hand: orthogonalize returns → corr ~ 0
        rng = np.random.default_rng(0)
        n = 200
        ra = rng.normal(0, 0.01, n)
        rb = rng.normal(0, 0.01, n)
        rb = rb - ra * (np.dot(rb, ra) / np.dot(ra, ra))
        pa = 100 * np.cumprod(1 + ra)
        pb = 100 * np.cumprod(1 + rb)
        hist = {"XXX": pd.Series(pa), "YYY": pd.Series(pb)}
        rep = build_concentration_report(["XXX", "YYY"], hist)
        assert rep["success"] is True
        assert rep["high_pairs"] == []
        mat = rep["matrix"]
        rho = abs(float(mat["XXX"]["YYY"]))
        assert rho < HIGH_PAIRWISE_CORR

    def test_matrix_flag_helper_hand_numbers(self):
        corr = pd.DataFrame(
            [[1.0, 0.85, 0.10],
             [0.85, 1.0, 0.05],
             [0.10, 0.05, 1.0]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        flags = flag_high_correlation_pairs(corr, threshold=0.70)
        assert len(flags) == 1
        assert {flags[0]["symbol_a"], flags[0]["symbol_b"]} == {"A", "B"}
        assert flags[0]["correlation"] == pytest.approx(0.85)
