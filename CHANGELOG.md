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

