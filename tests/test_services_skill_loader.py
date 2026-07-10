# -*- coding: utf-8 -*-
"""Tests for the Agent Skills loader (trading/services/skill_loader.py)."""

import pytest

from trading.services.skill_loader import (
    MAX_SKILLS_PER_TURN,
    _parse_skill_md,
    discover_skills,
    match_skills,
    render_skills_context,
)


class TestParsing:
    def test_valid_skill_parses(self):
        s = _parse_skill_md(
            "---\nname: demo\ndescription: d\ntriggers: alpha, beta\n---\nBody here"
        )
        assert s is not None
        assert s.name == "demo"
        assert s.triggers == ["alpha", "beta"]
        assert s.body == "Body here"

    def test_missing_frontmatter_or_name_skipped(self):
        assert _parse_skill_md("no frontmatter") is None
        assert _parse_skill_md("---\ndescription: x\n---\nbody") is None


class TestRepoSkills:
    def test_bundled_skills_discovered(self):
        names = {s.name for s in discover_skills(use_cache=False)}
        assert {
            "signal-interpretation",
            "position-sizing-and-risk",
            "optimizer-results-review",
        } <= names

    @pytest.mark.parametrize(
        "message,expected",
        [
            ("what does this RSI divergence mean", "signal-interpretation"),
            ("are these optimized parameters overfit?", "optimizer-results-review"),
            ("how much size for this 0DTE spread", "position-sizing-and-risk"),
        ],
    )
    def test_routing(self, message, expected):
        assert expected in {s.name for s in match_skills(message)}

    def test_cap_and_no_match(self):
        assert render_skills_context("hello there") == ""
        # A message hitting many triggers still loads at most the cap.
        msg = "optimize the rsi signal sizing risk overfit parameters macd"
        assert len(match_skills(msg)) <= MAX_SKILLS_PER_TURN

    def test_render_never_raises(self, tmp_path):
        # Broken skills dir -> '' rather than an exception.
        bad = tmp_path / "skills" / "broken"
        bad.mkdir(parents=True)
        (bad / "SKILL.md").write_text("---\nname broken frontmatter")
        assert render_skills_context("rsi", skills_dir=tmp_path / "skills") == ""
