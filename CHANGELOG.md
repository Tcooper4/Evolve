# Changelog

## Fable session (2026-07)

- **Strategy Optimizer tab** on Backtest page: grid search / genetic / PSO / Bayesian parameter search for all six strategies against real history, with baseline comparison, convergence chart, and one-click apply into the strategy backtest
- Canonical per-strategy parameter spaces (`strategy_param_spaces.py`) with cross-parameter constraints; SelfTuningOptimizer now wired with real bounds (was inert)
- Fixed CCI signal generation (crashed on every call), SMA silent zero-signals on lowercase input, registry parameter path (3 breakages), grid-search premature early stopping
- Theme overhaul: design tokens, Inter + JetBrains Mono tabular numerals, layered surfaces, interaction transitions, reduced-motion support
- 30 new execution-level tests

## v3.40.0 (2026-04)

- Consistency audit fixes: Home chat eight tools, `sp100` loads `data/universes/sp100.json`, consensus default ten models with per-model timeouts, dead “Model Lab” / “Forecasting” strings → Analyze, `filelock` / `cvxpy` in requirements, Transformer smoke test, aligned option/social/scanner caches to 300s, model comparison + quick forecast lists, `CHANGELOG` / `TECHNICAL_DEBT` / manifest update

## v3.39.0 (2026-04)

- Onboarding redesign: no-key bypass, Reddit auth, capability preview
- Feature status strip on Home page

## v3.38.0 (2026-04)

- Removed all simulation fallbacks from external_signals, fallback_provider, forecast_router
- Added caching singletons: ESM, MacroFactors, ForecastRouter in briefing
- @st.cache_data on ai_score, options_flow, social_sentiment
- Data quality transparency in deep dive UI

## v3.37.0 (2026-04)

- Advanced Tools UI redesign: 5 pages, 29 issues fixed
- Analyze 11→5 tabs, Scanner 4→2 tabs, Trade 4→3 tabs, Backtest 0→3 tabs
- 3 new agent tools: pattern analysis, backtest, options sentiment
- Consistent page headers

## v3.36.0 (2026-04)

- Wired 11 agent gaps: chart patterns→AI Score, options→AI Score, walk-forward→confidence, stationarity→model selection, Monte Carlo→briefing, pairs→scanner, paper trade button, strategy comparison→briefing, alpha attribution→Trade, social sentiment failure handling, sidebar navigation

## v3.35.0 (2026-04)

- Final archive pass: 101 true orphans moved
- v2 codebase map: 394 nodes, 842 edges, 4 reachability types

## v3.34.0 (2026-04)

- Fixed market_regime_agent abstract class bug
- Accurate node graph built from v2 map

## v3.33.0 (2026-04)

- Barrel cleanup pass 2
- Data providers audit
- Execution engine archived

## v3.32.0 (2026-04)

- Barrel __init__ cleanup
- Config/logging/risk deduplication
- Session state pruning added

## v3.29.0–v3.31.0 (2026-04)

- 20 new modules wired into live system
- Dead code archived: 161+ files
- Three archive passes with safe script
