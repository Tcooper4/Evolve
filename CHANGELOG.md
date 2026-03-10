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

