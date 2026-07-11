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
- **prompt_response_validator** → already gone from the live tree
  (archived by an earlier sweep); no action.

Restore any file with: git mv _archive/orphans-2026-07-b/<file> <original path>
