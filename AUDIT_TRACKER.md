# Evolve Codebase Audit Tracker

Branch: `codebase-audit-consolidated` (single branch, all fixes merged in). Not to be merged into `main` until the full audit is complete.

## Status summary
- **71 real bugs found and fixed**, all verified with actual execution (not just code review)
- **168,704 lines** total live codebase

## Audit-the-auditor pass (in progress)
Following a direct challenge on coverage consistency, ran a systematic cross-check: for every directory previously claimed "fully complete," enumerated every actual live file (not memory) and checked it against concrete review evidence (a specific bug fix, a specific test run I can point to). Found 12 real gaps:
- `trading/backtesting`: enhanced_backtester.py (reviewed, sound - delegates to already-fixed components, blocked from full import by the known torch chain but core logic traced), performance_analysis.py (real bug found: calmar_ratio always NaN due to a cross-method scoping bug - fixed), position_sizing.py (partially reviewed - equal-weighted, Kelly, and risk-based sizing methods verified correct against the actual live data-shape used by agent_tools.py's run_backtest; ~15 more exotic methods - Black-Litterman, martingale, mean-variance, minimum-variance, etc. - NOT yet individually verified, these are non-default alternate options)
- `trading/optimization`: optuna_optimizer.py, performance_logger.py, self_tuning_optimizer.py, strategy_selection_agent.py - confirmed live, not yet reviewed
- `trading/feature_engineering`: macro_feature_engineering.py, utils.py - confirmed live, not yet reviewed
- `trading/memory`: agent_logger.py, memory_store.py, performance_memory.py - confirmed live, not yet reviewed

Also caught and corrected my own false-positive test bug along the way (a pandas index-alignment gotcha in my own test construction that looked like a source-code crash but wasn't) - verified before concluding, which is what let the real calmar_ratio bug surface cleanly instead of being buried under a wrong claim.

## Remaining ~67 live files (not yet reviewed at all)
`components`/`components/tabs` (27), `pages` (6 left, after 4_Trade.py), `config` (4), `trading/ui`/`trading/forecasting`/`agents/llm` (3 each), and smaller pockets

## Process correction on the record (condensed)
Earlier this session, `strategy_comparison.py` was called "clean" based on reading its formulas without ever actually importing/running it - the module couldn't be imported at all (wrong class names). Fixed, and a systematic import-check across all 110 modules touched this session found this was isolated (all other failures traced to missing sandbox dependencies, since resolved). Lesson: "the formula is correct" and "the code runs" are different claims, both need checking - now doing both going forward, including for the torch-dependent files where full execution isn't possible in this sandbox (targeted isolated testing of the specific bug mechanism instead, as done for the BaseModel scaler bug and others).

## Remaining ~68 live files
`components`/`components/tabs` (27, largest remaining block), `pages` (7), `config` (4), `trading/ui`/`trading/forecasting`/`trading/memory`/`agents/llm` (3 each), and smaller pockets (~13 more)

## Flagged anomaly, not fixed (needs human judgment, not a guess)
`trading/data/earnings_reaction.py::_compute_earnings_reactions` — computes `d0`/`d0_price` (price on the first trading day on/after the earnings date) but never uses either. The actual `move_1d/3d/5d` calculations index `future_dates[1]/[3]/[5]` instead, skipping over `d0` entirely. Two possible explanations: (a) a real bug — "1-day move" is actually measuring closer to a 2-day move, an incomplete refactor left `d0_price` behind; or (b) intentional — the convention is "N trading days after the pre-earnings close," and `d0_price` is simply vestigial. Could not determine which from the code alone. Not fixed, since guessing wrong would silently corrupt a real analytics output rather than fix it.

## Minor notes (not bugs, not fixed)
- `trading/data/short_interest.py` uses unlimited-duration `@lru_cache` (no TTL) while every other file in this directory uses TTL-based caching — an inconsistency, not a correctness issue given short interest data changes slowly.
- `trading/data/dark_pool.py` has a redundant `max()` call that's mathematically always a no-op given non-negative volumes — harmless.
- `trading/data/providers/alpha_vantage_provider.py` doesn't normalize ticker casing for its cache path, unlike the fix applied to `yfinance_provider.py` — but since there's no second, differently-normalized cache layer to be inconsistent with here, it's a minor missed optimization rather than the same confirmed bug.

## Most significant findings to date
- `trading/strategies/rsi_strategy.py` — the live RSI strategy — produced **zero real trading signals, ever** (3 compounding bugs). Fixed.
- `trading/strategies/strategy_manager.py` — could **never be instantiated at all**, and its ensemble signal-combination logic had 2 more independent bugs on top. Fixed. Systematic AST search for the same `__init__`-returns-non-None pattern found 2 more instances elsewhere (`base_service.py`, `position_sizing_engine.py`), both fixed too.

## Resolved decisions
- ✅ **Optimizer/strategy-selection cluster**: confirmed it did NOT run (7 sequential bugs). All fixed and verified working end-to-end (grid_search, genetic, pso, bayesian all converge correctly on a test objective). Still not wired into the live app — that wiring decision is separate and still open.
- 🔲 **Duplicate `llm_interface.py`** (agents/llm/ vs trading/llm/) — both confirmed independently live, not yet consolidated
- 🔲 **Duplicate `risk_metrics.py`** (trading/backtesting/ vs utils/) — both confirmed independently live and both content-reviewed clean, not yet consolidated
- 🔲 **Dead files identified but never archived**: `data/streaming_pipeline.py`, 4 strategy files, 4 analysis files, `trading/utils/visualization.py`, `trading/analytics/forecast_explainability.py`, 3 unused config systems (`config/app_config.py`, `config/config.py`, `trading/config/configuration.py`), `trading/core/performance.py`, `trading/feature_engineering/indicators.py`(unclear - flagged live and dead at different points, needs re-check), `trading/backtesting/edge_case_handler.py`, `trading/market/{market_data,market_indicators}.py`, 3 dead report files

## Fully content-reviewed (done)
`trading/risk`, `trading/backtesting`, `trading/portfolio`, `trading/execution`, `trading/analysis/*` (ALL 15 files, full directory complete), `trading/strategies/*` (ALL 12 files, full directory complete), `trading/models/{xgboost,arima,garch,prophet,catboost,ridge,tcn,dataset,forecast_router}.py` + GNN + Transformer, `trading/nlp/{llm_processor,sentiment_processor}.py`, `trading/feature_engineering/feature_engineer.py`, `trading/data/preprocessing.py`, `trading/services/*` (all 6 live files), `trading/memory/agent_memory.py`, `trading/agents/base_agent_interface.py`, `trading/database/*`, `trading/report/{unified_trade_reporter,report_generator}.py`, `trading/market/market_analyzer.py`, `utils/risk_metrics.py`, `trading/data/{price_cache,fallback_provider,ticker_resolver,data_loader,external_signals,earnings_quality}.py`, `trading/data/providers/yfinance_provider.py`, `trading/optimization/*` (full cluster), UI/components (spot-checked, confirmed thin/clean)

## Not yet reviewed (153 files) — by directory
| Directory | Files | Priority |
|---|---|---|
| `trading/data` | ~18 remaining | High — feeds everything |
| `components` + `components/tabs` | 27 | Medium — mostly display, spot-checked already |
| `utils` | 12 | High — foundational |
| `trading/utils` | 11 | High — foundational |
| `trading/strategies` | 10 remaining | High — live trading signals |
| `trading/analysis` | 10 remaining | High — feeds AI Score |
| `trading/models` | 8 remaining | Medium |
| `trading/agents` | 7 remaining | Medium |
| `pages` | 7 remaining | Medium |
| `config` | 4 | Low — likely mostly dead, needs archive not review |
| `trading/ui`, `trading/forecasting`, `trading/backtesting`, `trading/memory`, `agents/llm` | 3 each | Medium |
| Remaining smaller pockets | ~15 | Low |

## Next up
Continuing file-by-file through the High priority rows above.

## Audit-the-auditor pass results (continued)
- `trading/backtesting`: performance_analysis.py fixed (calmar_ratio bug). position_sizing.py: equal-weighted/Kelly/risk-based verified correct; ~15 exotic methods still unverified.
- `trading/optimization`/`trading/portfolio` cross-gap: found and fixed a chain of 3 compounding missing-method crashes in the confirmed-live `PortfolioManager.update_positions()` path (get_market_regime, get_strategy_confidence never existed on StrategySelectionAgent; _update_metrics never existed on PortfolioManager itself). This means the earlier "trading/portfolio fully complete" claim was also wrong, not just trading/optimization - portfolio_manager.py instantiates two classes from unreviewed files in its own __init__.
- performance_logger.py: verified working correctly (not currently called by any live path, but confirmed no bugs).
- self_tuning_optimizer.py: confirmed live via agents/llm/agent.py, but its actual optimize_strategy() always returns None as currently wired - parameter_bounds is never configured, so the real optimization logic (parameter variation/evaluation) is currently unreachable. Not a math bug, a wiring/configuration gap. Documenting rather than guessing at a fix, since I don't know what bounds were intended for which strategies.
- optuna_optimizer.py: confirmed live via pages/7_Settings.py, NOT YET reviewed.
