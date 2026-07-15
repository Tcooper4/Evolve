# -*- coding: utf-8 -*-
"""Options structure research guide — GEX/skew → structure mapping."""

from __future__ import annotations

from trading.analysis.options_structure_overlay import (
    STRUCTURE_CALL_CREDIT,
    STRUCTURE_IRON_CONDOR,
    STRUCTURE_PUT_CREDIT,
    STRUCTURE_WAIT,
    pick_options_structure,
)


class TestPickOptionsStructure:
    def test_long_gamma_defaults_to_iron_condor(self):
        p = pick_options_structure(regime_short="long_gamma")
        assert p["structure"] == STRUCTURE_IRON_CONDOR
        assert p["mark_text"] == "IC"

    def test_short_gamma_waits(self):
        p = pick_options_structure(regime_short="short_gamma")
        assert p["structure"] == STRUCTURE_WAIT
        assert "short premium" in p["rationale"].lower() or "Wait" in p["label"]

    def test_near_flip_waits(self):
        p = pick_options_structure(regime_short="near_flip")
        assert p["structure"] == STRUCTURE_WAIT

    def test_put_smirk_above_flip_prefers_pcs(self):
        p = pick_options_structure(
            regime_short="long_gamma",
            skew_shape="put_smirk",
            skew_diff=0.05,
            spot=100.0,
            gamma_flip=95.0,
        )
        assert p["structure"] == STRUCTURE_PUT_CREDIT
        assert p["mark_text"] == "PCS"

    def test_call_smirk_below_flip_prefers_ccs(self):
        p = pick_options_structure(
            regime_short="long_gamma",
            skew_shape="call_smirk",
            skew_diff=-0.05,
            spot=100.0,
            gamma_flip=105.0,
        )
        assert p["structure"] == STRUCTURE_CALL_CREDIT
        assert p["mark_text"] == "CCS"
