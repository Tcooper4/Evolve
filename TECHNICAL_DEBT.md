# Technical Debt Register

Last updated: 2026-07-10 (Fable session)

## Known issues (not blocking)

### AI Score

- ML blend is applied as second-layer on overall score, not proportional dimension reweighting. Documents say "40% ML" which is accurate for final score but not for dimension budget.
- Short interest bump (+1.5/+2.0) can dominate momentum dimension before weighting.

### Codebase

- 48 files blocked from archive by barrel __init__.py imports — need barrel refactor to fully clean
- 37 barrel-referenced files in live tree require deeper refactor to remove

### Data

- Twitter sentiment always simulated (no real Twitter API path — requires paid API)
- Reddit sentiment falls back to public JSON API when PRAW credentials not set (rate limited at ~60 req/min)
- Macro indicators from ExternalSignalsManager (FRED path) not implemented — uses MacroFactors via yfinance instead

### UI

- tab_quick_forecast.py is 1623 lines — candidate for further modularization
- agents/llm/agent.py is ~2800 lines — candidate for splitting

### Kept-but-flagged (working code, zero live wiring — decide: wire or archive)

- `data/streaming_pipeline.py` (1,166L websocket streaming) — self-contained, imports clean, unused; plausible future live-data feature
- `trading/agents/prompt_response_validator.py` — schema validation for strategy/prompt responses; could be wired into the chat pipeline as a quality gate
- `trading/risk/risk_analyzer.py` — test-covered; overlaps advanced_risk
- `trading/nlp/prompt_processor.py` — superseded in the live chat path by EnhancedPromptRouterAgent
- `trading/optimization/rsi_optimizer.py` — superseded by the general optimizer stack; kept because tests/strategies/test_rsi_strategy.py hard-imports it alongside live rsi_signals coverage
- Stale tests importing nonexistent modules: tests/test_optimization/test_backtest_optimizer.py, test_hyperparameter_tuner.py (fail collection; pre-existing)

## Resolved (formerly in debt)

- risk_metrics duplication: consolidated into utils/risk_metrics.py (backtesting copy removed); flat-series Sharpe/Sortino blowup root-fixed (Fable session)
- llm_interface "duplication": trading/llm copy is a deliberate re-export bridge, not a duplicate — no action needed
- 17 one-off repair/audit scripts moved out of tests/ root to _archive/scripts/tests-oneoffs/ (zero importers each, verified)

- SHAP explainability (installed)
- Trade duration in Trade.to_dict()
- GNN multi-asset tab crash
- Ridge MAPE=100% warning
- Transformer import (TransformerForecaster)
- Walk-forward run_walk_forward=False
- AI Score col_map normalization
- Sharpe excess_returns std fix
- Abstract class instantiation (market_regime_agent)
- Simulation fallbacks removed
- ExternalSignalsManager singleton
- ForecastRouter singleton in briefing
