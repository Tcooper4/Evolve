# -*- coding: utf-8 -*-
"""Hand-check IC → weight blend math (unit, no live DB required)."""

from __future__ import annotations

import pytest


def test_ic_shift_normalize_and_60_40_blend_hand():
    """Documented conversion: shift IC+1, normalize, blend 60% IC / 40% static."""
    # Fake ICs: momentum much stronger than others
    ics = {
        "technical": 0.0,
        "momentum": 0.5,
        "sentiment": -0.2,
        "fundamental": 0.1,
    }
    static = {
        "technical": 0.30,
        "momentum": 0.35,
        "sentiment": 0.20,
        "fundamental": 0.15,
    }
    dims = list(ics)
    shifted = {d: max(0.01, ics[d] + 1.0) for d in dims}
    # tech:1.0 mom:1.5 sent:0.8 fund:1.1 → sum 4.4
    assert shifted["momentum"] == pytest.approx(1.5)
    assert shifted["sentiment"] == pytest.approx(0.8)
    total = sum(shifted.values())
    assert total == pytest.approx(4.4)
    ic_w = {d: shifted[d] / total for d in dims}
    assert ic_w["momentum"] == pytest.approx(1.5 / 4.4)
    blended = {d: 0.60 * ic_w[d] + 0.40 * static[d] for d in dims}
    # Highest IC dimension ends above lowest IC dimension after blend
    assert blended["momentum"] > blended["sentiment"]
    assert blended["momentum"] > blended["technical"]
    renorm = sum(blended.values())
    final = {d: round(blended[d] / renorm, 4) for d in dims}
    assert abs(sum(final.values()) - 1.0) < 1e-3
    assert final["momentum"] > final["sentiment"]
    # Hand values (pre-renorm): mom IC w=1.5/4.4≈0.3409 → blend≈0.3445
    assert blended["momentum"] == pytest.approx(0.6 * (1.5 / 4.4) + 0.4 * 0.35)
