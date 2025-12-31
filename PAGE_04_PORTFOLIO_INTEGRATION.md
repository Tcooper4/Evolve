# PAGE 4: 📊 PORTFOLIO & POSITIONS
**Complete Integration Guide for Cursor AI**

---

## 📋 OVERVIEW

**Target File:** `pages/4_Portfolio.py`  
**Status:** ALREADY FUNCTIONAL (portfolio_dashboard.py)  
**Action:** ENHANCE & RENAME  
**Tabs:** 5 tabs  
**Estimated Time:** 4-6 hours  
**Priority:** HIGH

### Current Features (Already Working):
✅ Current positions display  
✅ Portfolio value tracking  
✅ P&L tracking  
✅ Basic allocation chart

### Features to Add:
✅ Portfolio optimization (Markowitz, Black-Litterman)  
✅ Rebalancing tools  
✅ Tax lot tracking  
✅ Dividend tracking  
✅ Attribution analysis  
✅ Benchmark comparison

---

## CURSOR PROMPT 4.1 - Rename and Enhance Structure

```
Copy pages/portfolio_dashboard.py to pages/4_Portfolio.py and add tab structure:

TABS:
1. Overview (current implementation)
2. Positions (detailed view)
3. Performance (historical, attribution)
4. Optimization (portfolio optimizer, rebalancing)
5. Tax & Accounting (tax lots, dividends)

Keep existing functionality in Tab 1, add new features in other tabs.
```

---

## CURSOR PROMPT 4.2 - Enhance Overview Tab

```
Enhance Tab 1 (Overview) in pages/4_Portfolio.py

Current content is good. Add:
- Asset allocation pie chart (by sector, asset class)
- Risk metrics (portfolio beta, volatility)
- Correlation heatmap with major indices
- Quick rebalance suggestions

Keep existing position table and P&L display.
```

---

## CURSOR PROMPT 4.3 - Implement Detailed Positions Tab

```
Implement Tab 2 (Positions) in pages/4_Portfolio.py

ADD IMPORT:
from trading.portfolio.position_manager import PositionManager

Features:
- Expandable position cards
- Position-level metrics (Greeks for options, beta, etc.)
- Individual position P&L charts
- Position history
- Quick close buttons with confirmation
- Partial close interface
- Position notes/tags

Make it more detailed than overview.
```

---

## CURSOR PROMPT 4.4 - Implement Performance Tab

```
Implement Tab 3 (Performance) in pages/4_Portfolio.py

ADD IMPORTS:
from trading.analytics.performance_attribution import PerformanceAttribution
from trading.evaluation.benchmark_comparison import BenchmarkComparison

Features:
- Historical portfolio value chart
- Returns by period (daily, monthly, yearly)
- Benchmark comparison (SPY, user-selected)
- Performance attribution (what drove returns)
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Rolling metrics
```

---

## CURSOR PROMPT 4.5 - Implement Optimization Tab

```
Implement Tab 4 (Optimization) in pages/4_Portfolio.py

ADD IMPORT:
from trading.optimization.portfolio_optimizer import PortfolioOptimizer

Features:
- Modern Portfolio Theory optimization
- Efficient frontier visualization
- Target return/risk optimization
- Rebalancing recommendations
- What-if scenarios
- Constraint configuration (min/max weights)
- Black-Litterman model option

Interactive and visual.
```

---

## CURSOR PROMPT 4.6 - Implement Tax & Accounting Tab

```
Implement Tab 5 (Tax & Accounting) in pages/4_Portfolio.py

Features:
- Tax lot tracking (FIFO, LIFO, specific ID)
- Realized gains/losses by period
- Unrealized gains/losses
- Dividend history and projections
- Tax loss harvesting opportunities
- Wash sale warnings
- Export for tax filing

Comprehensive tax tools.
```

---

## ✅ PAGE 4 CHECKLIST

- [ ] Renamed from portfolio_dashboard.py
- [ ] All 5 tabs implemented
- [ ] Overview enhanced
- [ ] Detailed positions work
- [ ] Performance tracking complete
- [ ] Optimization functional
- [ ] Tax tools operational
- [ ] All charts display
- [ ] Export functions work
- [ ] Committed to git

---

## 🚀 COMMIT COMMAND

```bash
git mv pages/portfolio_dashboard.py pages/4_Portfolio.py
git add pages/4_Portfolio.py
git commit -m "feat(page-4): Enhance Portfolio & Positions with 5 tabs

- Tab 1: Enhanced overview with allocation and risk
- Tab 2: Detailed positions with individual metrics
- Tab 3: Performance attribution and benchmarking
- Tab 4: Portfolio optimization and rebalancing
- Tab 5: Tax lot tracking and dividend management

Builds on existing functional portfolio dashboard"
```

---

**Next:** PAGE_05_RISK_MANAGEMENT_INTEGRATION.md
