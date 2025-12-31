# PAGE 5: ⚠️ RISK MANAGEMENT & ANALYSIS
**Complete Integration Guide for Cursor AI**

---

## 📋 OVERVIEW

**Target File:** `pages/5_Risk_Management.py`  
**Merges:** risk_preview_dashboard.py + Monte_Carlo_Simulation.py  
**Tabs:** 5 tabs  
**Estimated Time:** 8-10 hours  
**Priority:** HIGH

### Features Preserved:
✅ Real-time risk metrics (VaR, CVaR)  
✅ Portfolio volatility tracking  
✅ Monte Carlo simulation  
✅ Stress testing  
✅ Scenario analysis  
✅ Risk limit monitoring  
✅ Tail risk analysis

---

## CURSOR PROMPT 5.1 - Create Page Structure

```
Create pages/5_Risk_Management.py with 5-tab structure

BACKEND IMPORTS:
from trading.risk.risk_manager import RiskManager
from trading.risk.advanced_risk import AdvancedRiskAnalytics
from trading.backtesting.monte_carlo import MonteCarloSimulator

TABS:
1. Risk Dashboard
2. VaR Analysis
3. Monte Carlo Simulation
4. Stress Testing
5. Advanced Analytics

Initialize with portfolio risk manager.
```

---

## CURSOR PROMPT 5.2 - Implement Risk Dashboard (Tab 1)

```
Implement Tab 1 (Risk Dashboard) in pages/5_Risk_Management.py

Real-time risk monitoring dashboard with:
- Risk gauge (Low/Medium/High)
- Key metrics cards (VaR, volatility, beta, max DD)
- Risk limits status (green/yellow/red indicators)
- Risk alerts feed
- Portfolio heat map
- Risk by position
- Historical risk trend

Update metrics every 30 seconds for live mode.
```

---

## CURSOR PROMPT 5.3 - Implement VaR Analysis (Tab 2)

```
Implement Tab 2 (VaR Analysis) in pages/5_Risk_Management.py

Comprehensive Value at Risk analysis:
- VaR calculation methods selector (Historical, Parametric, Monte Carlo)
- Confidence level slider (90%, 95%, 99%)
- Time horizon selector (1-day, 10-day, 1-month)
- VaR chart (distribution with VaR line)
- CVaR (Conditional VaR) calculation
- Expected Shortfall
- VaR backtesting (actual vs predicted losses)
- Component VaR (by position)

Show VaR in both $ and % terms.
```

---

## CURSOR PROMPT 5.4 - Implement Monte Carlo (Tab 3)

```
Implement Tab 3 (Monte Carlo Simulation) in pages/5_Risk_Management.py

Monte Carlo portfolio simulation:
- Number of simulations slider (1k, 10k, 100k)
- Time horizon (days, weeks, months)
- Simulation parameters configuration
- Run button with progress bar
- Results:
  * Distribution of outcomes (histogram)
  * Percentile outcomes (P10, P50, P90)
  * Probability of profit/loss
  * Maximum drawdown distribution
  * Path visualization (sample paths)
- Export simulation results

Use multiprocessing for speed.
```

---

## CURSOR PROMPT 5.5 - Implement Stress Testing (Tab 4)

```
Implement Tab 4 (Stress Testing) in pages/5_Risk_Management.py

Portfolio stress testing:
- Historical scenarios dropdown:
  * 2008 Financial Crisis
  * 2020 COVID Crash
  * 1987 Black Monday
  * 2000 Dot-com Bubble
  * Custom scenario
- Factor stress tests (rates, volatility, correlations)
- Scenario builder (manual shock inputs)
- Results display:
  * Portfolio impact ($, %)
  * Position-level impact
  * Recovery time estimate
- Scenario comparison

Make scenarios configurable.
```

---

## CURSOR PROMPT 5.6 - Implement Advanced Analytics (Tab 5)

```
Implement Tab 5 (Advanced Analytics) in pages/5_Risk_Management.py

Advanced risk analytics:
- Correlation matrix (interactive heatmap)
- Factor decomposition
- Tail risk metrics (skewness, kurtosis)
- Liquidity risk analysis
- Concentration risk
- Greek exposure (for options)
- Risk-adjusted return metrics
- Rolling risk metrics charts

Use plotly for interactive visualizations.
```

---

## ✅ PAGE 5 CHECKLIST

- [ ] File created: pages/5_Risk_Management.py
- [ ] All 5 tabs implemented
- [ ] Risk dashboard real-time
- [ ] VaR calculations accurate
- [ ] Monte Carlo simulation works
- [ ] Stress tests functional
- [ ] Advanced analytics complete
- [ ] All visualizations display
- [ ] Performance optimized
- [ ] Committed to git

---

## 🚀 COMMIT COMMAND

```bash
git add pages/5_Risk_Management.py
git commit -m "feat(page-5): Implement Risk Management & Analysis with 5 tabs

- Tab 1: Real-time risk dashboard with alerts
- Tab 2: Comprehensive VaR analysis
- Tab 3: Monte Carlo simulation
- Tab 4: Historical and custom stress testing
- Tab 5: Advanced risk analytics

Merges 2 pages with enhanced risk tools"
```

---

**Next:** PAGE_06_PERFORMANCE_INTEGRATION.md
