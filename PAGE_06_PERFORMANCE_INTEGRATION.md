# PAGE 6: 📉 PERFORMANCE & HISTORY
**Complete Integration Guide for Cursor AI**

---

## 📋 OVERVIEW

**Target File:** `pages/6_Performance.py`  
**Merges:** 7_Strategy_Performance.py + 6_Strategy_History.py + 10_Strategy_Health_Dashboard.py  
**Tabs:** 5 tabs  
**Estimated Time:** 8-10 hours  
**Priority:** MEDIUM

### Features Preserved:
✅ Strategy performance metrics  
✅ Trade-by-trade history  
✅ Strategy health scoring  
✅ Performance attribution  
✅ Rolling metrics  
✅ Regime-based analysis

---

## CURSOR PROMPT 6.1 - Create Page Structure

```
Create pages/6_Performance.py with 5-tab structure

BACKEND IMPORTS:
from trading.memory.strategy_logger import StrategyLogger
from trading.evaluation.metrics import PerformanceMetrics
from trading.evaluation.model_evaluator import ModelEvaluator
from trading.analytics.alpha_attribution_engine import AlphaAttributionEngine

TABS:
1. Performance Summary
2. Detailed History
3. Strategy Health
4. Attribution Analysis
5. Advanced Analytics

Load historical data on page load.
```

---

## CURSOR PROMPT 6.2 - Implement Performance Summary (Tab 1)

```
Implement Tab 1 (Performance Summary) in pages/6_Performance.py

Summary dashboard showing:
- Overall portfolio metrics (total return, Sharpe, max DD)
- Strategy comparison table:
  * Strategy name
  * Total return
  * Sharpe ratio
  * Win rate
  * Number of trades
  * Status (active/paused)
- Best/worst performers highlight
- Performance trend chart (last 30/90/365 days)
- Top trades (by P&L)

Allow filtering by date range, strategy type.
```

---

## CURSOR PROMPT 6.3 - Implement Detailed History (Tab 2)

```
Implement Tab 2 (Detailed History) in pages/6_Performance.py

Complete trade history with:
- Searchable/filterable trade table:
  * Entry date/time
  * Exit date/time
  * Symbol
  * Strategy
  * Direction (Long/Short)
  * Entry price
  * Exit price
  * P&L ($, %)
  * Holding period
- Trade details modal (click to expand)
- Trade reasoning/notes
- Related market conditions
- Export to CSV/Excel
- Trade calendar view option

Use ag-grid or similar for advanced table.
```

---

## CURSOR PROMPT 6.4 - Implement Strategy Health (Tab 3)

```
Implement Tab 3 (Strategy Health) in pages/6_Performance.py

Strategy health monitoring:
- Health score for each strategy (0-100)
- Health indicators:
  * Performance degradation detection
  * Win rate trend
  * Drawdown severity
  * Trade frequency
  * Slippage increase
- Traffic light system (green/yellow/red)
- Recommended actions
- Strategy lifecycle tracker
- Auto-pause triggers configuration

Visual health dashboard with actionable insights.
```

---

## CURSOR PROMPT 6.5 - Implement Attribution Analysis (Tab 4)

```
Implement Tab 4 (Attribution Analysis) in pages/6_Performance.py

Performance attribution tools:
- Return decomposition:
  * Alpha vs Beta
  * Strategy contribution
  * Asset contribution
  * Sector contribution
- Factor attribution
- Time-based attribution (daily/weekly/monthly)
- Attribution waterfall chart
- Comparison to benchmark

Show what's driving portfolio returns.
```

---

## CURSOR PROMPT 6.6 - Implement Advanced Analytics (Tab 5)

```
Implement Tab 5 (Advanced Analytics) in pages/6_Performance.py

Advanced performance analytics:
- Rolling Sharpe ratio chart
- Rolling maximum drawdown
- Drawdown periods analysis
- Recovery time statistics
- Trade distribution analysis:
  * P&L histogram
  * Win/loss distribution
  * Holding period distribution
- Regime-based performance:
  * Bull market performance
  * Bear market performance
  * Sideways market performance
- Correlation analysis between strategies

Interactive charts with filters.
```

---

## ✅ PAGE 6 CHECKLIST

- [ ] File created: pages/6_Performance.py
- [ ] All 5 tabs implemented
- [ ] Performance summary displays
- [ ] Trade history searchable
- [ ] Strategy health monitoring works
- [ ] Attribution analysis accurate
- [ ] Advanced analytics complete
- [ ] Charts interactive
- [ ] Export functions work
- [ ] Committed to git

---

## 🚀 COMMIT COMMAND

```bash
git add pages/6_Performance.py
git commit -m "feat(page-6): Implement Performance & History with 5 tabs

- Tab 1: Performance summary and comparison
- Tab 2: Detailed trade history with search
- Tab 3: Strategy health monitoring
- Tab 4: Performance attribution analysis
- Tab 5: Advanced analytics and distributions

Merges 3 performance-related pages"
```

---

**Next:** PAGE_07_MODEL_LAB_INTEGRATION.md
