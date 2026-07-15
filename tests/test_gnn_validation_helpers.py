# -*- coding: utf-8 -*-
"""Hand-checkable helpers for GNN OOS harness (no long real-data run)."""

from __future__ import annotations

import numpy as np

from trading.models.gnn_validation import _directional_hits


class TestDirectionalHits:
    def test_perfect_direction(self):
        # anchor 100, preds up then up; actuals up then up
        hits = _directional_hits(
            100.0,
            np.array([101.0, 102.0]),
            np.array([101.5, 103.0]),
        )
        assert hits == [1.0, 1.0]

    def test_wrong_direction(self):
        hits = _directional_hits(
            100.0,
            np.array([110.0]),
            np.array([90.0]),
        )
        assert hits == [0.0]
