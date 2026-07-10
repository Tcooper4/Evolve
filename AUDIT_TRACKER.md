# Evolve Codebase Audit Tracker

Branch: `codebase-audit-consolidated` (single branch, all fixes merged in). Not to be merged into `main` until the full audit is complete.

## Status summary
- **43 real bugs found and fixed**, all verified with actual execution (not just code review)
- **168,704 lines** total live codebase
- `trading/analysis`, `trading/strategies`, and `trading/data` all fully complete (3 of the original High priority directories)
- Remaining High priority: `utils` (12), `trading/utils` (11)
- Remaining Medium priority: `trading/models` (8), `trading/agents` (7), `pages` (7), `components`/`components/tabs` (27, spot-checked only)

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
