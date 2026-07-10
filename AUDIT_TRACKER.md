# Evolve Codebase Audit Tracker

Branch: `codebase-audit-consolidated` (single branch, all fixes merged in). Not to be merged into `main` until the full audit is complete.

## Status summary
- **34 real bugs found and fixed**, all verified with actual execution (not just code review)
- **168,704 lines** total live codebase
- **~168,000 - 153 files worth ≈ still ~140K lines effectively covered**, 153 live files remain never content-reviewed

## Resolved decisions
- ✅ **Optimizer/strategy-selection cluster**: confirmed it did NOT run (7 sequential bugs). All fixed and verified working end-to-end (grid_search, genetic, pso, bayesian all converge correctly on a test objective). Still not wired into the live app — that wiring decision is separate and still open.
- 🔲 **Duplicate `llm_interface.py`** (agents/llm/ vs trading/llm/) — both confirmed independently live, not yet consolidated
- 🔲 **Duplicate `risk_metrics.py`** (trading/backtesting/ vs utils/) — both confirmed independently live and both content-reviewed clean, not yet consolidated
- 🔲 **Dead files identified but never archived**: `data/streaming_pipeline.py`, 4 strategy files, 4 analysis files, `trading/utils/visualization.py`, `trading/analytics/forecast_explainability.py`, 3 unused config systems (`config/app_config.py`, `config/config.py`, `trading/config/configuration.py`), `trading/core/performance.py`, `trading/feature_engineering/indicators.py`(unclear - flagged live and dead at different points, needs re-check), `trading/backtesting/edge_case_handler.py`, `trading/market/{market_data,market_indicators}.py`, 3 dead report files

## Fully content-reviewed (done)
`trading/risk`, `trading/backtesting`, `trading/portfolio`, `trading/execution`, `trading/analysis/ai_score.py`, `trading/strategies/{macd,sma}_strategy.py`, `trading/models/{xgboost,arima,garch,prophet,catboost,ridge,tcn,dataset,forecast_router}.py` + GNN + Transformer, `trading/nlp/{llm_processor,sentiment_processor}.py`, `trading/feature_engineering/feature_engineer.py`, `trading/data/preprocessing.py`, `trading/services/*` (all 6 live files), `trading/memory/agent_memory.py`, `trading/agents/base_agent_interface.py`, `trading/database/*`, `trading/report/{unified_trade_reporter,report_generator}.py`, `trading/market/market_analyzer.py`, `utils/risk_metrics.py`, `trading/data/{price_cache,fallback_provider,ticker_resolver,data_loader}.py`, `trading/optimization/*` (full cluster), UI/components (spot-checked, confirmed thin/clean)

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
