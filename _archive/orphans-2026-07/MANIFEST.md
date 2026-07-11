# Archived orphans — 2026-07-10 review (evidence manifest)

Every file here was archived only after ALL of the following held:
(1) zero live importers, verified mechanically (AST import graph from all
entry points — app.py, all pages, MCP server, scripts — PLUS a
quoted-string scan for dynamic imports); (2) exclusion of the codebase's
filesystem-discovery loaders, which auto-import whole directories and make
static analysis lie (strategy registry globs `trading/strategies/*.py`;
forecast router globs `*_model.py`; provider manager globs providers;
agents barrel exposes lazy getters) — anything reachable through those was
exonerated and kept; (3) an individual read of the file itself; and
(4) the archival verified by re-importing every touched package, running
the test battery (identical pass/fail with the change stashed vs applied —
zero regressions), and smoke-running all seven pages.

Nothing was deleted. `git mv` back restores any file.

## Files and per-file evidence

- **rl_trainer.py** — self-described empty placeholder: "remains as an
  empty placeholder so imports resolve." Zero importers exist, so its only
  stated purpose is void. Only mentions anywhere are regex patterns inside
  old audit-verification scripts.
- **upgrader/** — broken package: `__init__` imports
  `trading.agents.upgrader.agent`, which does not exist; the package has
  never been importable (`ModuleNotFoundError` verified). Zero references.
- **agents_implementations/** (`agents/implementations/`, incl.
  `model_benchmarker.py`) — broken package: `__init__` imports
  `implementation_generator` and `research_fetcher`, neither exists;
  unimportable (verified). Decisively, `agents/__init__.py` sets
  `ModelBenchmarker = None` — the codebase itself already declared it dead.
- **agents_llm_providers/** (`agents/llm_providers/`) — empty shell: both
  providers it try-imports are gone, `__all__` ends empty. The one test
  referencing it try/excepts the import. Live LLM access is
  `agents/llm/active_llm_calls.py` + `config/llm_config.py`.
- **trading_integration/** (`trading/integration/`) — 3-line empty
  tombstone package. Two test files import `meta_agent_manager`,
  `model_registry`, `service_mesh` from it — none exist (those tests are
  stale and skip via try/except).
- **math_utils.py** — a THIRD copy of the Sharpe/max-drawdown/win-rate/
  profit-factor/Calmar/beta/alpha formulas with zero importers. Canonical
  implementations are `utils/performance_metrics.py` +
  `utils/risk_metrics.py` (consolidated 2026-07-10). A stray uncoordinated
  formula copy is how divergent numbers happen; archiving it removes that
  hazard.
- **persistent_memory.py** — a parallel, abandoned Redis/vector-store
  memory system (961 lines) with zero importers. The canonical
  `trading/memory/memory_store.py` (MemoryStore) has 19 live importers.
- **position_sizing_engine.py** (from `trading/risk/`) — a SECOND
  `PositionSizingEngine` class duplicating a subset of
  `trading/backtesting/position_sizing.py` (the one the live Backtester
  uses and the 2026-07-10 depth pass verified method-by-method). Zero
  importers; even its module-level singleton was never imported.
- **base_service.py / model_builder_service.py /
  performance_critic_service.py** — a Redis pub/sub microservice trio
  (base_service hard-imports `redis`). Zero external callers; the live
  equivalents are the agent implementations (`ModelBuilderAgent`,
  `PerformanceCriticAgent` — the latter's instantiation was fixed by the
  audit and is registered in the live agents barrel).
- **provider_manager.py** — a data-provider discovery mechanism
  (globs a providers dir and imports each) that nothing ever invokes:
  zero calls to `get_provider_manager`/`ProviderManager` anywhere. Live
  code imports providers directly (e.g. `yfinance_provider`).
- **leaderboard_dashboard.py** — a standalone Streamlit dashboard never
  mounted in `pages/` (unreachable UI). The live leaderboard is
  `trading/agents/agent_leaderboard.py`, used by `agent_manager` and
  exposed via the agents barrel; this file was a separate viewer for it.
- **strategy_dispatcher.py** — zero references. Live strategy dispatch is
  the strategy registry (filesystem discovery) plus `StrategyGatekeeper`
  in the agent path.

## Deliberately NOT archived (reviewed, kept)

- **`trading/optimization/rsi_optimizer.py`** — zero live importers and
  superseded by the general optimizer stack, BUT
  `tests/strategies/test_rsi_strategy.py` hard-imports it at module top
  alongside live `rsi_signals` coverage; archiving it would break
  collection of a live module's tests. Kept; flagged as superseded.
- **`data/streaming_pipeline.py`** (1,166 lines) — substantial,
  self-contained websocket streaming implementation; imports clean;
  plausible future feature. Kept; flagged.
- **`trading/agents/prompt_response_validator.py`**,
  **`trading/risk/risk_analyzer.py`**, **`trading/nlp/prompt_processor.py`**
  — test-only today (working code with passing-relevant tests);
  prompt_processor is superseded by `EnhancedPromptRouterAgent` in the
  live chat path but is part of the live `trading/nlp` package. Kept;
  flagged for a future keep-or-wire decision.

## Stale tests noted (not caused by, and not fixed in, this pass)

- `tests/test_optimization/test_backtest_optimizer.py` and
  `tests/test_optimization/test_hyperparameter_tuner.py` import
  `trading.optimization.backtest_optimizer` / a tuner module that do not
  exist on this branch — they fail collection identically with or without
  this archival.
