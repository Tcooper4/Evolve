# -*- coding: utf-8 -*-
"""Shared chat turn (trading/services/chat_turn.py): ONE tool-calling
brain for both frontends. Verifies graceful degradation and that
executor placeholder strings are never surfaced as successful replies."""

from trading.services.chat_turn import run_chat_turn, STANDARD_TOOLS


class TestSharedChatTurn:
    def test_no_llm_degrades_to_clean_error(self):
        r = run_chat_turn("hello")
        assert r["success"] is False
        assert "API key" in (r["error"] or "")
        assert r["tool_captions"] == []

    def test_shape_is_stable(self):
        r = run_chat_turn("hello")
        assert set(r) == {"success", "reply", "tool_captions", "error"}

    def test_standard_toolkit_matches_streamlit_chat(self):
        # The Streamlit page and the API must offer the SAME tools;
        # this list is the single source of truth for both.
        assert "get_ai_score" in STANDARD_TOOLS
        assert "run_backtest" in STANDARD_TOOLS
        # registry-derived after the 2026-07 connection pass: the chat
        # surface IS the platform registry (16 tools), so capabilities
        # can never silently diverge between frontends again
        assert "detect_market_regime" in STANDARD_TOOLS
        assert "get_portfolio_allocation" in STANDARD_TOOLS
        assert "retune_strategies" in STANDARD_TOOLS
        assert len(STANDARD_TOOLS) >= 14

    def test_placeholder_never_surfaces_as_reply(self, monkeypatch):
        import trading.services.chat_turn as ct

        class FakeRes:
            text = "No response."
            tool_captions = []

        import agents.llm.tool_executor as te
        monkeypatch.setattr(te, "execute_with_tools",
                            lambda **kw: FakeRes())
        r = ct.run_chat_turn("hello")
        assert r["reply"] != "No response."
