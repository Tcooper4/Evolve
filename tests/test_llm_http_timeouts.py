# -*- coding: utf-8 -*-
"""Explicit timeouts on research HTTP + Anthropic call sites."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_research_agent_http_timeouts_passed():
    from trading.agents.research_agent import (
        SEARCH_HTTP_TIMEOUT_S,
        ResearchAgent,
    )

    assert 10.0 <= SEARCH_HTTP_TIMEOUT_S <= 15.0

    agent = ResearchAgent()
    seen: list[float] = []

    def _fake_get(url, *args, **kwargs):
        seen.append(float(kwargs.get("timeout") or 0))
        resp = MagicMock()
        resp.status_code = 200
        if "github" in url:
            resp.json.return_value = {"items": []}
        else:
            resp.text = (
                '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">'
                "</feed>"
            )
        return resp

    with patch("trading.agents.research_agent.requests.get", side_effect=_fake_get):
        assert agent.search_github("momentum", max_results=1) == []
        # Force keyword path (no vector search)
        if hasattr(agent, "_vector_search"):
            delattr(agent, "_vector_search")
        assert agent.search_arxiv("momentum", max_results=1) == []

    assert seen == [SEARCH_HTTP_TIMEOUT_S, SEARCH_HTTP_TIMEOUT_S]


def test_research_agent_http_timeout_returns_empty():
    from trading.agents.research_agent import ResearchAgent
    import requests

    agent = ResearchAgent()
    with patch(
        "trading.agents.research_agent.requests.get",
        side_effect=requests.Timeout("hang"),
    ):
        assert agent.search_github("x") == []
        assert agent.search_arxiv("x") == []


def test_anthropic_clients_receive_use_case_timeouts():
    """Contract: each listed Anthropic() construction passes an explicit timeout."""
    checks = [
        (
            ROOT / "trading" / "agents" / "research_agent.py",
            "ANTHROPIC_TIMEOUT_S",
            45.0,
        ),
        (
            ROOT / "trading" / "agents" / "enhanced_prompt_router.py",
            "ANTHROPIC_INTENT_TIMEOUT_S",
            20.0,
        ),
        (
            ROOT / "trading" / "services" / "news_context.py",
            "ANTHROPIC_TIMEOUT_S",
            15.0,
        ),
        (
            ROOT / "agents" / "llm" / "llm_interface.py",
            "ANTHROPIC_TIMEOUT_S",
            45.0,
        ),
    ]
    for path, const_name, expected in checks:
        src = path.read_text(encoding="utf-8", errors="replace")
        assert f"{const_name} = {expected}" in src, path.name
        # Every Anthropic( call site must pass timeout=<const> nearby
        idx = 0
        found = 0
        while True:
            pos = src.find("Anthropic(", idx)
            if pos < 0:
                break
            window = src[pos : pos + 180]
            assert f"timeout={const_name}" in window, (
                f"{path.name}: Anthropic() missing timeout={const_name} near {window!r}"
            )
            found += 1
            idx = pos + 1
        assert found >= 1, path.name


def test_research_agent_anthropic_timeout_wired(monkeypatch):
    from trading.agents.research_agent import ANTHROPIC_TIMEOUT_S, ResearchAgent

    agent = ResearchAgent()
    agent.anthropic_api_key = "test-key"
    captured: dict = {}

    class _FakeAnthropic:
        def __init__(self, api_key=None, timeout=None):
            captured["timeout"] = timeout
            captured["api_key"] = api_key

        class messages:
            @staticmethod
            def create(**kwargs):
                block = MagicMock()
                block.text = "ok summary"
                msg = MagicMock()
                msg.content = [block]
                return msg

    import sys
    fake_mod = MagicMock()
    fake_mod.Anthropic = _FakeAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)

    out = agent._summarize("some paper text about momentum")
    assert out == "ok summary"
    assert captured["timeout"] == ANTHROPIC_TIMEOUT_S
