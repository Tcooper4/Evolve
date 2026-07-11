# Second orphan disposition pass — 2026-07 (session close)

Decisions on the five kept-but-flagged files from TECHNICAL_DEBT.md,
after re-checking importers with the web/ backend now in the graph:

- **trading/nlp/prompt_processor.py** → ARCHIVED here. Zero importers
  outside itself across trading/, agents/, pages/, components/, web/;
  its role (prompt intent parsing) is served by the live
  enhanced_prompt_router path used by the Chat page.
- **data/streaming_pipeline.py** → KEPT. New, concrete justification:
  the React frontend now has a websocket quote stream
  (web/backend /ws/quote) implemented as a polling loop; the pipeline is
  the designed upgrade path to true push streaming in the rewrite.
- **trading/optimization/rsi_optimizer.py** → ARCHIVED here (reversal of
  the earlier keep). Investigated for revival: BaseOptimizer's
  log_results/plot_results became abstract after this class was written,
  so it has been UNINSTANTIABLE - its sole consumer
  (tests/strategies/test_rsi_strategy.py) failed at setUpClass forever,
  and beneath that the suite is stale at three more independent layers
  (lowercase-column fixtures vs 'Close', numpy-typing assertions, result
  shape). Test archived to _archive/stale_tests_2026_07/ as a matched
  pair. RSI optimization is served by the live optimizer cluster
  (strategy_param_spaces + grid/genetic/PSO/Bayesian).
- **trading/risk/risk_analyzer.py** → KEPT (reachable via lazy/report
  paths; received the per-user API-key reroute this session — archiving
  a file that just got a correctness patch would be self-contradictory;
  revisit after the live-data session shows whether its outputs surface).
- **trading/agents/prompt_response_validator.py** → ARCHIVED here.
  20 functions of schema validation with ZERO importers; wiring unused
  validation into the (working, tested) shared chat loop would add risk
  without a driver. Revive if tool-output validation becomes a need.
- **data/streaming_pipeline.py** keep UPGRADED to evidence-based: it is
  real push infrastructure (Polygon wss + Finnhub/Alpaca provider
  abstraction, in-memory cache), not another poller - it becomes the
  true-push source for /ws/quote when a paid feed key exists; the
  current 5s polling websocket is correct for free yfinance data.

Restore any file with: git mv _archive/orphans-2026-07-b/<file> <original path>

## ExecutionAgent determination (2026-07, requested review)
- **trading/agents/execution/execution_agent.py** (619L) → ARCHIVED to
  `_archive/trading/agents/execution/`, REUNITING it with its own
  dependencies: an earlier sweep (51a081a) archived
  execution_providers.py and position_manager.py but stranded the agent
  that imports them, leaving it permanently broken. Evidence for
  archive over revive: (1) zero consumers besides lazy-loader entries;
  (2) its purpose is LIVE broker order routing (ExecutionMode /
  create_execution_provider adapters), which is explicitly outside the
  platform's declared paper-only scope; (3) paper trading is served by
  the Backtester + trade handler paths. Revival = restore all three
  files together and add a broker key - a deliberate future decision,
  not a default.
