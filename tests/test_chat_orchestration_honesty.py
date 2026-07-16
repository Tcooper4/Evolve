# -*- coding: utf-8 -*-
"""Chat orchestration + honesty tests against the live chat_turn path.

NOT a DSR/edge suite — chat is an orchestration layer. These tests prove:
  1) beginner-advisor tool SEQUENCE (order matters), not just skill fire
  2) mid-turn tool failure is honest + placeholder filter still works
  3) stated risk framing actually reaches the model context payload
  4) recommendation tracker retrieval works through the chat tool loop
     across a separate simulated turn/session identity
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

# Canonical beginner-advisor order from skills/beginner-advisor/SKILL.md
BEGINNER_PIPELINE = (
    "detect_market_regime",
    "scan_universe",
    "get_ai_score",
    "get_news",
    "get_risk_metrics",
)


def _router_json(names: List[str]) -> str:
    calls = [{"name": n, "arguments": {"symbol": "AAPL"}} for n in names]
    return json.dumps({"tool_calls": calls})


def _patch_llms(monkeypatch, *, simple_fn, chat_fn):
    """Patch where tool_executor imports from (inside execute_with_tools)."""
    import agents.llm.active_llm_calls as alc

    monkeypatch.setattr(alc, "call_active_llm_simple", simple_fn)
    monkeypatch.setattr(alc, "call_active_llm_chat", chat_fn)


class TestBeginnerPipelineSequence:
    """Router returns the full playbook; executor must preserve order.

    Honesty note: MAX_TOOLS_PER_TURN=3 so only the first three run in one
    turn — but those three must still be regime → scan → score, never
    score-before-regime.
    """

    def test_executed_prefix_preserves_skill_order(self, monkeypatch):
        import agents.llm.tool_executor as te
        from agents.llm.tool_executor import MAX_TOOLS_PER_TURN, execute_with_tools
        from trading.services.skill_loader import render_skills_context

        msg = "what should I buy today"
        skills = render_skills_context(msg)
        assert "detect_market_regime" in skills
        assert "scan_universe" in skills
        for i, name in enumerate(BEGINNER_PIPELINE):
            assert name in skills, name
            if i > 0:
                assert skills.index(BEGINNER_PIPELINE[i - 1]) < skills.index(name)

        call_order: List[str] = []

        def fake_simple(prompt: str, max_tokens: int = 512) -> str:
            return _router_json(list(BEGINNER_PIPELINE))

        def fake_chat(*_a, **_k) -> str:
            return (
                "Here are a couple of researched ideas with downside first — "
                "not a guarantee."
            )

        def spy_run(name, fn, arguments, timeout_s):
            call_order.append(str(name))
            return {"success": True, "symbol": "AAPL", "results": [], "items": []}

        _patch_llms(monkeypatch, simple_fn=fake_simple, chat_fn=fake_chat)
        monkeypatch.setattr(te, "_run_tool", spy_run)

        res = execute_with_tools(
            user_message=msg,
            context_block="",
            conversation_messages=[],
            system_prompt="test",
            platform_context_suffix=skills,
            available_tools=list(BEGINNER_PIPELINE),
        )

        assert call_order == list(BEGINNER_PIPELINE[:MAX_TOOLS_PER_TURN])
        assert call_order[0] == "detect_market_regime"
        assert call_order[1] == "scan_universe"
        assert call_order[2] == "get_ai_score"
        assert call_order.index("get_ai_score") > call_order.index(
            "detect_market_regime"
        )
        assert res.text and "guarantee" in res.text.lower()
        assert len(call_order) == MAX_TOOLS_PER_TURN == 3

    def test_wrong_router_order_is_observable(self, monkeypatch):
        """Hand-check: executor preserves router order (does not reshuffle)."""
        import agents.llm.tool_executor as te
        from agents.llm.tool_executor import execute_with_tools

        call_order: List[str] = []
        bad_order = ["get_ai_score", "detect_market_regime", "scan_universe"]

        _patch_llms(
            monkeypatch,
            simple_fn=lambda *a, **k: _router_json(bad_order),
            chat_fn=lambda *a, **k: "ok",
        )

        def spy_run(name, fn, arguments, timeout_s):
            call_order.append(str(name))
            return {"success": True}

        monkeypatch.setattr(te, "_run_tool", spy_run)
        execute_with_tools(
            user_message="what should i buy",
            context_block="",
            conversation_messages=[],
            system_prompt="test",
            available_tools=list(BEGINNER_PIPELINE),
        )
        assert call_order[0] == "get_ai_score"
        assert call_order.index("get_ai_score") < call_order.index(
            "detect_market_regime"
        )


class TestToolFailureHonestyAndPlaceholder:
    def test_mid_turn_failure_acknowledged_and_turn_completes(self, monkeypatch):
        import agents.llm.tool_executor as te
        from agents.llm.tool_executor import execute_with_tools
        from trading.services import chat_turn as ct

        names = ["detect_market_regime", "scan_universe", "get_ai_score"]
        captured_augmented: Dict[str, Any] = {}

        def fake_chat(system_prompt, context, conv, user_message, max_tokens=2048):
            captured_augmented["context"] = context or ""
            if "scan_universe: error" in (context or ""):
                return (
                    "I couldn't finish the full scan just now "
                    "(scan_universe failed: injected boom), but regime looked ok."
                )
            return "unexpected"

        def boom(**_kwargs):
            raise RuntimeError("injected boom")

        def ok(**_kwargs):
            return {"success": True, "regime": "risk_on"}

        # Keep real _run_tool (catches exceptions → success:False);
        # inject a raising tool via the registry map.
        monkeypatch.setattr(
            te,
            "_registry_map",
            lambda: {
                "detect_market_regime": (ok, {}),
                "scan_universe": (boom, {}),
                "get_ai_score": (ok, {}),
            },
        )
        _patch_llms(
            monkeypatch,
            simple_fn=lambda *a, **k: _router_json(names),
            chat_fn=fake_chat,
        )

        res = execute_with_tools(
            user_message="what should i buy today",
            context_block="",
            conversation_messages=[],
            system_prompt="test",
            available_tools=names,
        )
        assert "couldn't finish" in res.text.lower() or "failed" in res.text.lower()
        assert any("scan_universe failed" in c for c in res.tool_captions)
        assert "scan_universe: error — injected boom" in captured_augmented["context"]
        assert "## Live platform tool results" in captured_augmented["context"]

        monkeypatch.setattr(te, "execute_with_tools", lambda **kw: res)
        out = ct.run_chat_turn("what should i buy today")
        assert out["success"] is True
        assert "scan" in (out["reply"] or "").lower() or "failed" in (
            out["reply"] or ""
        ).lower()

    def test_placeholder_after_failure_does_not_fake_success(self, monkeypatch):
        """Exercise chat_turn's documented 'No response.' filter after tools."""
        import agents.llm.active_llm_calls as alc
        import agents.llm.tool_executor as te
        from trading.services import chat_turn as ct

        class FakeRes:
            text = "No response."
            tool_captions = ["scan_universe failed · boom"]

        monkeypatch.setattr(te, "execute_with_tools", lambda **kw: FakeRes())
        monkeypatch.setattr(
            alc, "call_active_llm_chat", lambda *a, **k: "No response."
        )

        out = ct.run_chat_turn("what should i buy today")
        assert out["success"] is False
        assert out["reply"] is None
        assert "API key" in (out["error"] or "")
        assert out["reply"] != "No response."


class TestRiskProfileReachesModelPayload:
    def test_conservative_moderate_aggressive_payloads_differ(self, monkeypatch):
        from config.user_store import save_user_preferences
        from trading.services import chat_turn as ct

        captured: Dict[str, str] = {}

        class FakeRes:
            text = "Framed reply for testing."
            tool_captions: List[str] = []

        def capture_execute(**kw):
            captured["context_block"] = kw.get("context_block") or ""
            captured["system_prompt"] = kw.get("system_prompt") or ""
            return FakeRes()

        import agents.llm.tool_executor as te

        monkeypatch.setattr(te, "execute_with_tools", capture_execute)

        payloads: Dict[str, str] = {}
        for level in ("conservative", "moderate", "aggressive"):
            sid = f"user:chat_orch_risk_{level}"
            save_user_preferences(sid, {"risk_tolerance": level})
            captured.clear()
            out = ct.run_chat_turn("what should i buy", session_id=sid)
            assert out["success"] is True
            ctx = captured["context_block"]
            assert "[Stated risk profile" in ctx
            assert f"risk_tolerance={level}" in ctx
            assert "Audience adaptation" in captured["system_prompt"]
            assert "conservative → lead harder with downside" in captured[
                "system_prompt"
            ]
            payloads[level] = ctx

        assert payloads["conservative"] != payloads["moderate"]
        assert payloads["moderate"] != payloads["aggressive"]
        assert payloads["conservative"] != payloads["aggressive"]

        assert "lead harder with downside" in payloads["conservative"].lower()
        assert "defined-risk" in payloads["conservative"].lower()
        assert "balanced" in payloads["moderate"].lower()
        assert "soft-pedal" in payloads["aggressive"].lower() or (
            "standard analytical" in payloads["aggressive"].lower()
        )


class TestRecommendationRetrievalViaChat:
    def test_separate_turn_retrieves_tracked_outcome(self, monkeypatch, tmp_path):
        """Track + record_real_outcome, then a NEW chat turn pulls via tool."""
        import trading.portfolio.paper_portfolio as PP
        import agents.llm.tool_executor as te
        from agents.llm.tool_executor import execute_with_tools
        from trading.services import chat_turn as ct

        monkeypatch.setattr(PP, "DB_PATH", tmp_path / "pp_chat.db")
        sid = "user:chat_orch_rec"
        monkeypatch.setenv("EVOLVE_SESSION_ID", sid)

        p = PP.PaperPortfolio(user_id=sid)
        tracked = p.track_recommendation(
            "AAPL",
            source="analyze",
            score=7.5,
            price_at_rec=190.0,
            note="July idea",
            capture_guidance=False,
        )
        assert tracked.get("success") is True
        out = p.record_real_outcome(
            tracked["id"],
            real_pnl=140.0,
            real_strategy="shares",
            real_acted=True,
            real_notes="closed last month",
        )
        assert out.get("success") is True
        assert out.get("won") is True

        captured: Dict[str, str] = {}

        def synth(system_prompt, context, conv, user_message, max_tokens=2048):
            captured["context"] = context or ""
            assert "AAPL" in (context or "")
            assert "140" in (context or "") or "real_pnl" in (context or "")
            return (
                "Your AAPL idea from last month made about $140 on the "
                "real outcome you recorded — not a guarantee of future results."
            )

        _patch_llms(
            monkeypatch,
            simple_fn=lambda *a, **k: _router_json(["get_recommendations"]),
            chat_fn=synth,
        )

        res = execute_with_tools(
            user_message="how did my AAPL idea from last month do?",
            context_block="",
            conversation_messages=[],
            system_prompt="test",
            available_tools=["get_recommendations"],
        )
        assert res.text
        assert "AAPL" in res.text
        assert "140" in res.text
        assert "get_recommendations" in captured["context"]

        class WrapRes:
            text = res.text
            tool_captions = list(res.tool_captions)

        monkeypatch.setattr(te, "execute_with_tools", lambda **kw: WrapRes())
        turn = ct.run_chat_turn(
            "how did my AAPL idea from last month do?",
            session_id=sid,
        )
        assert turn["success"] is True
        assert "AAPL" in (turn["reply"] or "")
