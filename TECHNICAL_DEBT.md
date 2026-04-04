# Technical Debt Register

Last updated: v3.40.0

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

## Resolved (formerly in debt)

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
