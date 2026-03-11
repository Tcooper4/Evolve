# Technical Debt

Generated from codebase grep for TODO, FIXME, HACK, XXX, TEMP, temporary, placeholder.

| File | Line | Comment | Priority |
|------|------|---------|----------|
| pages/7_Performance.py | 278 | sample_returns = None  # TODO: load from strategy logger or trade history | Medium |
| pages/7_Performance.py | 1447 | sample_returns = None  # TODO: load from strategy logger or trade history | Medium |
| pages/7_Performance.py | 1451 | benchmark_returns = None  # TODO: load benchmark from data provider | Medium |
| pages/7_Performance.py | 2000 | sample_returns = None  # TODO: load from strategy logger or trade history | Medium |
| pages/9_Reports.py | 228 | has_report_data = False  # TODO: set True when real data from backtest/logs exists | Medium |
| pages/6_Risk_Management.py | 266 | # TODO: get actual returns from portfolio manager / FallbackDataProvider / strategy logs | Medium |
| trading/strategies/hybrid_engine.py | 324 | # This is a placeholder for more sophisticated correlation analysis | Low |
| trading/backtesting/edge_case_handler.py | 219 | # Create performance metrics placeholder | Low |
| trading/models/tcn_model.py | 516 | "confidence": 0.8,  # Placeholder confidence | Low |
| trading/models/lstm_model.py | 1615 | "confidence": np.full(horizon, 0.8),  # Placeholder confidence | Low |
| trading/config/__init__.py | 112 | except Exception as _unused_var:  # Placeholder, flake8 ignore | Low |
| trading/backtesting/enhanced_backtester.py | 570 | _unused_var = trade  # Placeholder, flake8 ignore | Low |

Priority: High = blocks correctness or security; Medium = missing feature or data source; Low = cosmetic or placeholder.

---

## Resolved (Sessions 1–19)

- **RESOLVED**: RSI calculation (Wilder's smoothing) — Sessions 1–5
- **RESOLVED**: Look-ahead bias in backtester
- **RESOLVED**: Division by zero across 83+ files
- **RESOLVED**: Reports fake data — Session 15
- **RESOLVED**: Walk-forward wiring — Session 15
- **RESOLVED**: Forecasting page blank tabs — Session 19
- **RESOLVED**: Model registry duplicates — Session 14
- **RESOLVED**: TransformerForecaster missing — Session 14
- **RESOLVED**: AI Score not wired — Session 16

## Resolved (Sessions 23–28)

- **RESOLVED**: GNN multi-asset tab crash
- **RESOLVED**: SHAP explainability (wired, requires pip install shap)
- **RESOLVED**: Trade.to_dict() missing entry/exit dates
- **RESOLVED**: Benchmark returns zero Series
- **RESOLVED**: Factor model unverified (confirmed working, wired)
- **RESOLVED**: Placeholder liquidity/Greek/factor analytics
- **RESOLVED**: Strategy correlation random matrix
- **RESOLVED**: Onboarding Cloud session_id bug
- **RESOLVED**: Advanced orders not wired to ExecutionAgent
- **RESOLVED**: Automated execution loop missing
- **RESOLVED**: Portfolio partial close/risk levels info-only
- **RESOLVED**: Reports email not wired
- **RESOLVED**: ArxivResearchFetcher not surfaced in UI
- **RESOLVED**: Strategy lifecycle hard-coded
- **RESOLVED**: Auto-pause rules not persisted
- **RESOLVED**: Task Orchestrator not initialized

## Resolved (Sessions 29–33)

- **RESOLVED**: TaskOrchestrator slow init (39s → <1s lazy init)
- **RESOLVED**: session_id entropy bug (full key hash, no truncation)
- **RESOLVED**: yfinance DatetimeArray type error
- **RESOLVED**: SentimentFetcher cache_model_operation ttl signature
- **RESOLVED**: Startup noise (all INFO→DEBUG at source)
- **RESOLVED**: cache_management, model_validation, strategy_backtesting added to TaskType

## Resolved (Sessions 34–36, v1.4.0 audit)

- **RESOLVED**: All 14 HIGH severity audit findings
- **RESOLVED**: All MEDIUM severity audit findings (iloc guards, set_page_config,
  lru_cache, decrypt safety, bare excepts in Admin, app shutdown, execution_engine)

## Open (non-blocking)

- **OPEN**: pickle.load from cache files (accepted risk — internal paths only)
- **OPEN**: SQLite __del__ in strategy_switcher.py and task_memory.py (low risk singletons)
- **OPEN**: Options flow overlay (requires data source decision)
- **OPEN**: Chart pattern detection (head & shoulders, triangles, S/R levels)
- **OPEN**: Startup noise via trading.memory direct import path (minor)
- **OPEN**: Admin maintenance operations simulated (backup/restore/optimize)
- **OPEN**: Ridge MAPE=100% warning (non-blocking)
- **OPEN**: lru_cache replaced with TTL dict in `earnings_reaction`
- **OPEN**: `.cache/lstm` directory missing warning
