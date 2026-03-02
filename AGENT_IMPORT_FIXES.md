# Agent import fixes: model_creator_agent and prompt_router_agent

This document records the fixes applied to remove references to the two missing agent modules and to use the canonical agents from AGENT_ARCHITECTURE.md.

## (1) model_creator_agent to model_builder_agent / ModelBuilderAgent

**Findings:** No file in the live codebase imports from trading.agents.model_creator_agent. agents/llm/agent.py already uses ModelBuilderAgent from trading.agents.model_builder_agent. The file trading/agents/model_creator_agent.py does not exist.

**Actions taken:** None required. No imports were changed; no file was deleted.

**Verification:** No live Python file references the module name model_creator_agent.

## (2) prompt_router_agent to enhanced_prompt_router / EnhancedPromptRouterAgent

**Findings:** One file imported from trading.agents.prompt_router_agent: tests/test_router.py. Other files already use EnhancedPromptRouterAgent from trading.agents.enhanced_prompt_router. The file trading/agents/prompt_router_agent.py does not exist.

**Actions taken:** In tests/test_router.py: Replaced import with EnhancedPromptRouterAgent from trading.agents.enhanced_prompt_router. Updated tests to use parse_intent and route instead of detect_intent, route_to_agent, process_request, and agent_registry; skipped test_agent_registry; updated test_end_to_end_routing to mock parse_intent and assert on route() result.

**Verification:** No live Python file references the module name prompt_router_agent.

## (3) Verification

Grep for trading.agents.model_creator_agent and trading.agents.prompt_router_agent in *.py: no matches. No file in the live codebase (excluding _dead_code/) references these module names after the fix.

## Summary

- model_creator_agent: Replacement = model_builder_agent / ModelBuilderAgent. Files updated: None. File deleted: N/A.
- prompt_router_agent: Replacement = enhanced_prompt_router / EnhancedPromptRouterAgent. Files updated: tests/test_router.py. File deleted: N/A.

References: AGENT_ARCHITECTURE.md.
