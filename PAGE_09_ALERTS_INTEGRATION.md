# PAGE 9: 🔔 ALERTS & NOTIFICATIONS
**Complete Integration Guide for Cursor AI**

---

## 📋 OVERVIEW

**Target File:** `pages/9_Alerts.py`  
**Merges:** 18_Alerts.py (standalone)  
**Tabs:** 5 tabs  
**Estimated Time:** 6-8 hours  
**Priority:** MEDIUM

### Features Preserved:
✅ Price alerts  
✅ Strategy signal alerts  
✅ Risk limit alerts  
✅ Portfolio alerts  
✅ Multiple notification channels (email, SMS, Telegram, Slack, webhook)  
✅ Alert conditions builder  
✅ Alert history

---

## CURSOR PROMPT 9.1 - Create Page Structure

```
Create pages/9_Alerts.py with 5-tab structure

BACKEND IMPORTS:
from trading.monitoring.health_check import HealthMonitor
from system.infra.agents.alert_manager import AlertManager
from trading.utils.notification_system import NotificationSystem

TABS:
1. Active Alerts
2. Create Alert
3. Alert Templates
4. Alert History
5. Notification Settings

Initialize alert manager and notification system.
```

---

## CURSOR PROMPT 9.2 - Implement Active Alerts (Tab 1)

```
Implement Tab 1 (Active Alerts) in pages/9_Alerts.py

Currently configured alerts dashboard:
- Alerts table:
  * Alert name
  * Type (price, strategy, risk, etc.)
  * Condition
  * Status (active/paused)
  * Last triggered
  * Enable/disable toggle
  * Edit button
  * Delete button
- Quick filters (by type, status)
- Bulk actions (enable/disable all, delete)
- Alert test button (trigger manually)
- Recent triggers feed

Real-time alert monitoring.
```

---

## CURSOR PROMPT 9.3 - Implement Create Alert (Tab 2)

```
Implement Tab 2 (Create Alert) in pages/9_Alerts.py

Alert creation wizard:
- Alert type selector:
  * Price Alert
  * Technical Indicator Alert
  * Strategy Signal Alert
  * Risk Limit Alert
  * Portfolio Alert
  * Custom Condition
- Condition builder (dynamic based on type):
  * Symbol selection
  * Condition operator (>, <, =, crosses above, etc.)
  * Threshold value
  * Frequency (once, daily, continuous)
- Notification settings:
  * Channels (email, SMS, Telegram, Slack, webhook)
  * Message template
  * Priority level
- Alert name and description
- Test alert button
- Create button

User-friendly alert builder.
```

---

## CURSOR PROMPT 9.4 - Implement Alert Templates (Tab 3)

```
Implement Tab 3 (Alert Templates) in pages/9_Alerts.py

Pre-built and custom templates:
- Pre-built templates:
  * "Price drops 5%"
  * "New all-time high"
  * "RSI overbought/oversold"
  * "Large volume spike"
  * "Position loss exceeds threshold"
  * "Daily loss limit reached"
- Template library (user-created)
- Use template button
- Edit template
- Delete template
- Share template
- Import/export templates

Speed up alert creation.
```

---

## CURSOR PROMPT 9.5 - Implement Alert History (Tab 4)

```
Implement Tab 4 (Alert History) in pages/9_Alerts.py

Triggered alerts log:
- History table:
  * Trigger timestamp
  * Alert name
  * Condition that triggered
  * Value at trigger
  * Notification sent
  * Action taken (if any)
- Search and filter
- Date range selector
- Export to CSV
- Alert effectiveness analysis:
  * Most triggered alerts
  * False positive rate
  * Response time
- Visualizations (alerts over time)

Learn from alert history.
```

---

## CURSOR PROMPT 9.6 - Implement Notification Settings (Tab 5)

```
Implement Tab 5 (Notification Settings) in pages/9_Alerts.py

Configure notification channels:
- Email settings:
  * SMTP configuration
  * From address
  * Test email button
- SMS settings:
  * Twilio credentials
  * Phone number
  * Test SMS button
- Telegram:
  * Bot token
  * Chat ID
  * Test message button
- Slack:
  * Webhook URL
  * Channel
  * Test notification button
- Webhook:
  * Custom webhook URLs
  * Payload template
  * Test webhook button
- Global notification rules:
  * Quiet hours
  * Max notifications per hour
  * Priority filtering

Complete notification configuration.
```

---

## ✅ PAGE 9 CHECKLIST

- [ ] File created: pages/9_Alerts.py
- [ ] All 5 tabs implemented
- [ ] Active alerts dashboard works
- [ ] Alert creation functional
- [ ] Templates available
- [ ] History logging works
- [ ] All notification channels configured
- [ ] Test functions work
- [ ] Alerts trigger correctly
- [ ] Committed to git

---

## 🚀 COMMIT COMMAND

```bash
git add pages/9_Alerts.py
git commit -m "feat(page-9): Implement Alerts & Notifications with 5 tabs

- Tab 1: Active alerts monitoring
- Tab 2: Alert creation wizard
- Tab 3: Alert templates library
- Tab 4: Alert history and analytics
- Tab 5: Multi-channel notification settings

Complete alerting system for live trading"
```

---

**Next:** PAGE_10_ADMIN_INTEGRATION.md
