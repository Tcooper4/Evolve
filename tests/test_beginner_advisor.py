# -*- coding: utf-8 -*-
"""Guided-analyst layer for zero-background users: the beginner-advisor
skill must fire on plain-language investing questions and stay quiet on
expert queries; the chat system prompt must carry the audience-adaptation
contract."""

from trading.services.skill_loader import render_skills_context


class TestBeginnerAdvisorSkill:
    def test_fires_on_novice_phrasings(self):
        for q in (
            "what stocks should I consider buying today?",
            "I'm new to investing, where should I put my money?",
            "is now a good time to buy?",
            "what do you recommend for a beginner",
        ):
            assert "Guided analyst" in render_skills_context(q), q

    def test_silent_on_expert_queries(self):
        for q in (
            "run a grid search on the rsi strategy for SPY",
            "show me the sharpe of my last backtest",
        ):
            assert "Guided analyst" not in render_skills_context(q), q

    def test_playbook_mandates_the_pipeline_and_risk_framing(self):
        ctx = render_skills_context("what should i buy today")
        for required in ("detect_market_regime", "scan_universe",
                         "Lead with the loss", "Pros and cons",
                         "What to watch"):
            assert required in ctx, required


class TestSystemPromptContract:
    def test_audience_adaptation_present(self):
        from trading.services.chat_nl_service import EVOLVE_CHAT_SYSTEM_PROMPT as P
        assert "Audience adaptation" in P
        assert "Lead with the loss" in P
        assert "never as guarantees" in P
