# -*- coding: utf-8 -*-
"""Options structure research guide — GEX/skew → structure mapping."""

from __future__ import annotations

from trading.analysis.options_structure_overlay import (
    DISCLOSURE,
    STRUCTURE_CALL_CREDIT,
    STRUCTURE_IRON_CONDOR,
    STRUCTURE_MAPPING_NOTE,
    STRUCTURE_MAPPING_VALIDATED,
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
            risk_tolerance="moderate",
            allow_undefined_risk=True,
        )
        assert p["structure"] == STRUCTURE_CALL_CREDIT
        assert p["mark_text"] == "CCS"


class TestConservativeDeprioritizesUndefined:
    def test_conservative_flags_undefined_without_hiding(self):
        cons = pick_options_structure(
            regime_short="long_gamma",
            risk_tolerance="conservative",
            allow_undefined_risk=False,
        )
        agg = pick_options_structure(
            regime_short="long_gamma",
            risk_tolerance="aggressive",
            allow_undefined_risk=True,
        )
        assert cons["structure"] == STRUCTURE_IRON_CONDOR  # lead stays defined
        noted = cons.get("also_noted") or []
        undef = [x for x in noted if x.get("risk_class") == "undefined"]
        assert undef, "undefined-risk must remain visible"
        assert all(x.get("deprioritized") is True for x in undef)
        assert cons.get("conservative_note")
        # Aggressive with allow may leave undefined not deprioritized
        agg_undef = [
            x for x in (agg.get("also_noted") or [])
            if x.get("risk_class") == "undefined"
        ]
        assert agg_undef
        assert any(not x.get("deprioritized") for x in agg_undef)


class TestStructureMappingHonesty:
    """Phase 3 — scoping only; mapping branches unchanged."""

    def test_mapping_not_evolve_validated(self):
        assert STRUCTURE_MAPPING_VALIDATED is False
        assert "not" in STRUCTURE_MAPPING_NOTE.lower()
        assert "validated" in STRUCTURE_MAPPING_NOTE.lower()
        assert "research" in DISCLOSURE.lower()
        assert "EVOLVE_GEX_SNAPSHOT_LOG" in DISCLOSURE

    def test_every_pick_carries_unvalidated_meta(self):
        for regime in ("long_gamma", "short_gamma", "near_flip"):
            p = pick_options_structure(regime_short=regime)
            assert p["mapping_validated"] is False
            assert p["mapping_basis"] == "research_default"
            assert "mapping_note" in p

    def test_truth_table_unchanged_by_honesty_meta(self):
        # Hand table — same as before Phase 3 disclosure work
        assert (
            pick_options_structure(regime_short="long_gamma")["structure"]
            == STRUCTURE_IRON_CONDOR
        )
        assert (
            pick_options_structure(regime_short="short_gamma")["structure"]
            == STRUCTURE_WAIT
        )
        assert (
            pick_options_structure(regime_short="near_flip")["structure"]
            == STRUCTURE_WAIT
        )
