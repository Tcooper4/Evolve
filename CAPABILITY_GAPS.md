# Evolve — Institutional Quant Capability Gaps

**Date:** 2025-03-08  
**Scope:** Audit vs institutional platforms (QuantConnect, Bloomberg, Two Sigma, Renaissance).  
Status: **PRESENT** = exists in codebase; **MISSING/WEAK** = documented below with priority.

---

## Risk Metrics

| Capability | Status | Notes |
|------------|--------|--------|
| CVaR / Expected Shortfall | PRESENT | risk_metrics.py, portfolio_simulator, backtesting/risk_metrics, Risk page |
| VaR | PRESENT | risk_metrics, risk_manager, backtesting, Risk page |
| Monte Carlo (portfolio + price paths) | PRESENT | backtesting/monte_carlo.py, analysis/monte_carlo.py (GBM fan chart), Risk + Forecasting tabs |
| Beta / market beta / CAPM | PRESENT | risk_metrics, performance_analysis, gatekeeper, prompts |
| Information ratio | PRESENT | backtesting/risk_metrics, report, performance_summarizer |
| Sortino / Calmar / Omega | PRESENT | safe_math, risk_metrics, gatekeeper, backtest_common |
| Stress test / scenario analysis | PRESENT | risk_manager, report_export_engine, Stress Test tab (Risk page) |
| Correlation matrix (rolling) | PRESENT | risk_metrics.compute_correlation_matrix, Risk/Portfolio heatmaps (real data) |
| Omega ratio | WEAK | Not a dedicated function; add if needed (downside vs upside probability). |

**Gap (Low priority):** Dedicated **Omega ratio** for full downside/upside comparison.  
- **Why:** Some funds prefer Omega over Sortino.  
- **Difficulty:** Easy.  
- **Package:** None (numpy).  
- **Priority:** Low.

---

## Portfolio Optimization

| Capability | Status | Notes |
|------------|--------|--------|
| Efficient frontier / mean-variance | PRESENT | portfolio_optimizer (CVXPY), PyPortfolioOpt fallback (PART 6) |
| Black-Litterman | PRESENT | portfolio_optimizer |
| Risk parity / equal risk contribution | PRESENT | portfolio_optimizer |
| Max Sharpe / min volatility / min CVaR | PRESENT | portfolio_optimizer |

No critical gaps.

---

## Alpha / Factor

| Capability | Status | Notes |
|------------|--------|--------|
| Momentum (e.g. 12-1) | PRESENT | factor_model, strategies, feature_engineer, market_analyzer |
| Value (P/B, P/E, book) | WEAK | factor_model has momentum/reversal/vol/volume; no P/B, P/E in factor model |
| Quality (ROE, ROA, margin) | WEAK | Not in factor_model; alpha_attribution has some |
| Low vol / realized vol | PRESENT | factor_model (volatility), strategies, risk |
| Mean reversion / pairs / cointegration | PRESENT | strategies, backtesting, market_analyzer |

**Gap:** **Value factor (P/B, P/E)** and **Quality factor (ROE, ROA, gross margin)** in factor_model.  
- **Why:** Standard Fama-French / quant factor attribution.  
- **Difficulty:** Medium (need fundamental data source).  
- **Package:** yfinance (limited fundamentals) or external API.  
- **Priority:** Medium.

---

## Market Regime

| Capability | Status | Notes |
|------------|--------|--------|
| Regime detection / bull-bear | PRESENT | market_regime_agent, regime_detection_agent, strategy_selector |
| HMM / Gaussian Mixture | WEAK | GARCH/volatility used; no explicit HMM in codebase |
| VIX / fear index / vol regime | PRESENT | risk_analyzer, report, config, strategies |

**Gap:** **Explicit HMM (Hidden Markov) or Gaussian Mixture** for regime states.  
- **Why:** Clean regime labels (e.g. 3 states: low/med/high vol).  
- **Difficulty:** Medium.  
- **Package:** hmmlearn or sklearn.mixture.  
- **Priority:** Medium.

---

## Technical Indicators

| Capability | Status | Notes |
|------------|--------|--------|
| RSI, MACD, Bollinger | PRESENT | feature_engineer, market_indicators, strategies |
| ATR | PRESENT | atr_strategy, risk_calculator, market_indicators |
| OBV | PRESENT | market_indicators, strategies |
| Stochastic | PRESENT | market_indicators, strategies |
| Ichimoku | WEAK | Referenced in some files; not central. |
| Fibonacci | WEAK | Few references. |
| VWAP | PRESENT | portfolio_simulator, market_indicators, execution |
| Williams %R | PRESENT | market_indicators |
| CCI | PRESENT | cci_strategy, market_indicators |

No critical gaps; Ichimoku/Fibonacci could be expanded if needed.

---

## Macro / Alternative Data

| Capability | Status | Notes |
|------------|--------|--------|
| FRED / federal reserve | PRESENT | macro_data_integration, macro_feature_engineering |
| Earnings / EPS / calendar / surprise | WEAK | options_forecaster, external_signals; no dedicated earnings calendar |
| Options flow / put-call / implied vol | PRESENT | options_forecaster, options module |
| Short interest / short float / days to cover | WEAK | Scattered references; no dedicated pipeline |
| Insider trading / SEC filings | WEAK | sentiment_bridge, demo_reasoning; no full pipeline |

**Gaps:**  
1. **Earnings calendar + surprise** — Dedicated pipeline for earnings dates and surprise vs estimate.  
   - **Why:** Core for event-driven and fundamental strategies.  
   - **Difficulty:** Medium.  
   - **Package:** yfinance, finviz, or earnings API.  
   - **Priority:** High.  

2. **Short interest / days to cover** — Structured feed.  
   - **Why:** Sentiment and squeeze signals.  
   - **Difficulty:** Medium.  
   - **Package:** yfinance or data vendor.  
   - **Priority:** Medium.  

3. **Insider trading / SEC filings** — Structured pipeline.  
   - **Why:** Institutional alpha source.  
   - **Difficulty:** Hard.  
   - **Package:** SEC EDGAR API or vendor.  
   - **Priority:** Medium.

---

## Summary Table (Missing or Weak Only)

| Capability | Why it matters | Difficulty | Package | Priority |
|------------|----------------|------------|---------|----------|
| Omega ratio | Downside/upside ratio preferred by some funds | Easy | None | Low |
| Value factor (P/B, P/E) in factor model | Standard factor attribution | Medium | yfinance / API | Medium |
| Quality factor (ROE, ROA) in factor model | Standard factor attribution | Medium | yfinance / API | Medium |
| HMM / GMM regime | Clean regime states | Medium | hmmlearn / sklearn | Medium |
| Earnings calendar + surprise | Event-driven and fundamental alpha | Medium | yfinance / API | High |
| Short interest pipeline | Sentiment and squeeze signals | Medium | yfinance / vendor | Medium |
| Insider / SEC pipeline | Institutional alpha | Hard | EDGAR / vendor | Medium |

---

*End of Capability Gaps.*
