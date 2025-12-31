# PAGE 3: 💰 TRADE EXECUTION & ORDER MANAGEMENT
**Complete Integration Guide for Cursor AI**

---

## 📋 OVERVIEW

**Target File:** `pages/3_Trade_Execution.py`  
**Merges:** 3_Trade_Execution.py (standalone - already one page)  
**Sections:** 5 sections (not tabs, vertical layout)  
**Estimated Time:** 6-8 hours  
**Priority:** CRITICAL

### Features Preserved:
✅ Manual trade entry (market, limit, stop orders)  
✅ Automated strategy execution  
✅ Order status tracking  
✅ Bracket orders (TP + SL)  
✅ Trailing stops  
✅ Multiple broker support  
✅ Paper/Live trading toggle  
✅ Pre-trade risk checks  
✅ Position sizing calculator

---

## CURSOR PROMPT 3.1 - Create Page Structure

```
Create pages/3_Trade_Execution.py with execution interface

BACKEND IMPORTS:
from trading.execution.execution_engine import ExecutionEngine
from trading.portfolio.portfolio_manager import PortfolioManager
from trading.execution.broker_adapter import BrokerAdapter
from trading.risk.risk_control import RiskControl

PAGE STRUCTURE:
- Quick Trade section (simple order entry)
- Advanced Orders section (bracket, conditional, etc.)
- Automated Execution section (connect strategies)
- Order Management section (active orders table)
- Execution Analytics section (fill analysis)

Use vertical sections, not tabs. Include paper/live toggle at top.
```

---

## CURSOR PROMPT 3.2 - Implement Quick Trade

```
Implement quick trade section with:
- Symbol input with validation
- Buy/Sell toggle
- Quantity input
- Market/Limit selector  
- Limit price (conditional on order type)
- Position size calculator
- Pre-trade risk check display
- Submit button with confirmation dialog
- Order confirmation display

Include real-time validation and error handling.
```

---

## CURSOR PROMPT 3.3 - Implement Advanced Orders

```
Implement advanced order types:
- Bracket orders (entry + TP + SL)
- Trailing stop orders
- Conditional orders (if-then)
- Multi-leg orders
- OCO (One Cancels Other)
- Good-til-Cancelled vs Day orders

Use expandable sections for each order type.
```

---

## CURSOR PROMPT 3.4 - Implement Automated Execution

```
Create interface to connect strategies to live trading:
- Load available strategies from registry
- Configure auto-execution parameters
- Set safety limits (max orders/day, max loss, etc.)
- Start/Stop toggles
- Real-time execution log
- Emergency stop button

Include comprehensive safety checks.
```

---

## CURSOR PROMPT 3.5 - Implement Order Management

```
Create order management dashboard:
- Active orders table (refresh every 5 seconds)
- Order modification interface
- Batch cancellation
- Order history
- Fill notifications
- Order status tracking

Use st.dataframe with auto-refresh.
```

---

## CURSOR PROMPT 3.6 - Implement Execution Analytics

```
Add execution quality analytics:
- Slippage analysis
- Fill rate statistics
- Price improvement metrics
- Execution time analysis
- Comparison to VWAP/TWAP

Display in charts and tables.
```

---

## ✅ PAGE 3 CHECKLIST

- [ ] Quick trade works
- [ ] Advanced orders functional
- [ ] Automated execution operational
- [ ] Order management real-time
- [ ] Analytics display
- [ ] Paper trading mode works
- [ ] Risk checks in place
- [ ] Confirmations work
- [ ] Committed to git

---

## 🚀 COMMIT COMMAND

```bash
git add pages/3_Trade_Execution.py
git commit -m "feat(page-3): Implement Trade Execution & Order Management

- Quick trade interface
- Advanced order types (bracket, trailing, conditional)
- Automated strategy execution
- Real-time order management
- Execution quality analytics

Critical feature for live trading"
```

---

**Next:** PAGE_04_PORTFOLIO_INTEGRATION.md
