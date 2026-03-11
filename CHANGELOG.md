# Evolve Changelog

## v1.0.0 — 2026-03-09

### Major Features

- 13-page Bloomberg Terminal-style trading platform
- 10 ML forecasting models (ARIMA, XGBoost, Ridge, CatBoost, Prophet, LSTM, TCN, Ensemble, GARCH, GNN)
- Real-time market data via yfinance with 5-min caching
- Paper trading execution engine with simulated fill at live market price
- Multi-model walk-forward validation
- Monte Carlo risk simulation
- SHAP explainability for tree models

### Bug Fixes (Sessions 6–8)

- Fixed forecasting math: all models now output price-space forecasts (previously returned normalized ratios)
- Fixed ARIMA continuity discontinuity (>2% gap correction)
- Fixed Ridge flat forecast (autoregressive feature propagation)
- Removed all hardcoded fake performance data
- Fixed DuplicateWidgetID on Strategy Testing Save buttons
- Fixed `Position()` constructor argument mismatch in execution engine
- Fixed Python 3.10 f-string syntax error in `implementation_generator.py`
- Fixed Windows cp1252 encoding errors in logger calls
- Added GARCH model support via `arch` package

### Improvements

- Home page: live Market Pulse (SPY/QQQ/IWM/VIX/GLD/BTC) and Top Movers
- Risk Management: VaR/CVaR/Sharpe/Sortino/Beta grid and composite Risk Score
- Portfolio: backtest simulation snapshot and correlation demo mode
- All Forecasting tabs now surface errors instead of going blank
- Stack traces hidden behind "Show technical details" toggles for end users

## [1.2.0] — 2026-03-10

### Added
- AI Score composite signal (Technical 30%, Momentum 35%, Sentiment 20%, Fundamental 15%) wired into Forecasting and Chat
- Market Scanner (`pages/13_Scanner.py`) — 6 filter types, 58-stock universe, AI Score ranking, drill-down panel
- Multi-timeframe chart component (Daily/Weekly/Monthly)
- Earnings reaction tracker — historical EPS surprise vs price reaction for up to 8 quarters
- Pre/post market prices on Home page Market Pulse tiles
- Home page "Top Opportunities" quick scan (cached 15min)
- Performance Attribution panel in Reports (P&L distribution, profit factor, trade duration)
- Walk-forward validator wired into Strategy Testing

### Fixed
- Forecasting page: AI Model Selection, Model Comparison, Market Analysis, and Monte Carlo tabs no longer render blank
- Reports page: removed hardcoded fake return/trade data; now shows session_state backtest results or empty state
- Performance page: pnl column normalization guard; polyfit SVD error wrapped
- Alerts page: inline NotificationSystem initialization
- Admin page: psutil system health metrics; port check for Agent API
- earnings_reaction: replaced lru_cache with TTL dict cache
- Home page scan: cached in session_state (15min TTL) to prevent re-downloading on every page refresh

### Changed
- Error visibility: all Forecasting tabs now surface tracebacks instead of silently failing
- Page-level error boundary added to Forecasting, Strategy Testing, Performance pages

## [1.3.0] — 2026-03-10

### Fixed
- Onboarding: session_id now derived from API key hash — Cloud iframe
  localStorage restriction no longer causes key loss on page refresh
- GNN multi-asset tab: try/except wrapper prevents page crash
- Trade.to_dict(): entry_date, exit_date, duration_days now included
- Performance Attribution: benchmark returns now fetched from yfinance
  (SPY/QQQ/DIA) instead of zero Series
- Startup noise: JSON INFO loggers suppressed in non-Streamlit contexts

### Added
- SHAP explainability wired in Model Lab (pip install shap to enable)
- Advanced orders (bracket/trailing/OCO/conditional/multi-leg) wired
  to ExecutionAgent with graceful degradation
- Automated execution loop: polls active strategies on each rerun,
  respects emergency stop, daily limits, confidence thresholds
- ArxivResearchFetcher surfaced as Research Browser tab in Model Lab
- Portfolio partial close and risk level updates now mutate positions
- Reports email delivery wired to NotificationService
- Factor model (factor_attribution_pct) wired into Risk page factor
  decomposition with SPY regression fallback
- Liquidity risk uses real ADV data from yfinance
- Greek exposure uses real equity delta (1.0) instead of random values
- Strategy correlation uses real trade history when available
- Strategy lifecycle tracker queries memory store instead of hard-coded
- Auto-pause rules persisted to memory store
- Task Orchestrator activated (core/orchestrator modules initialized)

### Changed
- Model Lab: added Research Browser tab (12 tabs total)

## [1.3.1] — 2026-03-10

### Fixed
- TaskOrchestrator: optional agent init failures no longer print warnings
- TaskType: alert_manager added back as legacy compatibility value
- SentimentFetcher: cache_model_operation ttl argument removed (use ttl_hours)
- Startup noise: core trading system INFO logs suppressed in all
  non-Streamlit import paths

## [1.3.2] — 2026-03-10

### Fixed
- TaskOrchestrator: __init__ now completes in <1s (was 39s);
  heavy initialization deferred to explicit start() / ensure_initialized() call
- Onboarding: session_id entropy bug — was hashing only first 16
  chars of API key (2 chars of entropy for Anthropic keys);
  now hashes full key before truncating hash output
- yfinance: DatetimeArray type coercion in yfinance_provider.py
  and data_loader.py prevents date range errors
- SentimentFetcher: cache_model_operation ttl_hours now float-typed;
  per-entry TTL stored in .meta sidecar files
- Startup noise: INFO log calls changed to DEBUG at source in
  trading.config.settings, trading.config.enhanced_settings,
  and trading core init — 0 INFO lines in non-Streamlit contexts
- TaskType: added cache_management, model_validation,
  strategy_backtesting for legacy task compatibility

## [1.4.0] — 2026-03-10

### Security & Stability (audit-driven)
- Fixed CCXT_AVAILABLE logic inversion on ImportError
- Removed hardcoded JWT secret — now env var with secrets.token_hex fallback
- Guarded exec() in safe_executor with assertion + trust documentation
- Fixed Backtester.run() — method was missing, causing silent backtest failures
- Fixed execution_engine get_execution_summary() KeyError on missing result key
- Added zero guards: report_generator running_max, backtester initial_cash,
  strategy_selector negative_returns division
- SQLite connections in history_logger now use context managers
- Removed st.set_page_config() from leaderboard_dashboard and institutional_dashboard
- Fixed create_strategy lru_cache — kwargs no longer cause cache collisions
- Wrapped cipher.decrypt in try/except — corrupted keys return empty gracefully
- Replaced bare except: pass in Admin, execution_engine, app.py shutdown
- Added iloc empty guards across backtest_utils, market_analyzer, portfolio,
  performance, forecast_router, strategy_fallback, position_sizing, agent
