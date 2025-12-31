# PAGE 10: ⚙️ SYSTEM ADMINISTRATION
**Complete Integration Guide for Cursor AI**

---

## 📋 OVERVIEW

**Target File:** `pages/10_Admin.py`  
**Merges:** 19_Admin_Panel.py + 8_Agent_Management.py + 9_System_Monitoring.py + 5_System_Scorecard.py  
**Tabs:** 6 tabs  
**Estimated Time:** 8-10 hours  
**Priority:** MEDIUM

### Features Preserved:
✅ System configuration  
✅ User management  
✅ API key management  
✅ Broker connections  
✅ AI agent management  
✅ System health monitoring  
✅ Resource usage tracking  
✅ Logs and debugging

---

## CURSOR PROMPT 10.1 - Create Page Structure

```
Create pages/10_Admin.py with 6-tab structure

BACKEND IMPORTS:
from trading.config.enhanced_settings import EnhancedSettings
from trading.agents.agent_registry import AgentRegistry
from trading.monitoring.health_check import SystemHealthMonitor
from trading.utils.system_status import SystemStatus

TABS:
1. System Dashboard
2. Configuration
3. AI Agents
4. System Monitoring
5. Logs & Debugging
6. Maintenance

Add authentication check (admin only).
```

---

## CURSOR PROMPT 10.2 - Implement System Dashboard (Tab 1)

```
Implement Tab 1 (System Dashboard) in pages/10_Admin.py

System overview scorecard:
- Overall system health (0-100 score)
- Status indicators:
  * Database (green/yellow/red)
  * API connections (green/yellow/red)
  * Broker connections (green/yellow/red)
  * Data providers (green/yellow/red)
  * Agents (active count)
- Quick stats:
  * Uptime
  * Total trades today
  * Active strategies
  * System load
- Recent system events feed
- Quick actions:
  * Restart services
  * Clear cache
  * Run health check

High-level system overview.
```

---

## CURSOR PROMPT 10.3 - Implement Configuration (Tab 2)

```
Implement Tab 2 (Configuration) in pages/10_Admin.py

System settings management:
- General Settings section:
  * System name
  * Timezone
  * Base currency
  * Trading hours
- API Keys section:
  * Alpha Vantage
  * Finnhub
  * Polygon
  * OpenAI
  * (Masked display, reveal button)
- Broker Connections:
  * Alpaca (paper/live)
  * Binance
  * Interactive Brokers
  * Connection status
  * Test connection button
- Feature Flags:
  * Enable/disable features
  * Beta features toggle
- Database settings
- Save button with confirmation

Secure configuration management.
```

---

## CURSOR PROMPT 10.4 - Implement AI Agents (Tab 3)

```
Implement Tab 3 (AI Agents) in pages/10_Admin.py

Agent management dashboard:
- Agent registry table:
  * Agent name
  * Type (model selector, optimizer, etc.)
  * Status (active/paused/error)
  * Last run
  * Performance score
  * Enable/disable toggle
  * Configure button
- Agent details panel (expandable):
  * Description
  * Configuration
  * Execution history
  * Performance metrics
- Add new agent button
- Agent logs
- Agent testing interface

Full agent lifecycle management.
```

---

## CURSOR PROMPT 10.5 - Implement System Monitoring (Tab 4)

```
Implement Tab 4 (System Monitoring) in pages/10_Admin.py

Real-time system monitoring:
- Resource usage:
  * CPU usage (gauge + chart)
  * Memory usage (gauge + chart)
  * Disk usage (gauge)
  * Network I/O
- Service status:
  * Web server
  * Database
  * Cache (Redis)
  * Task queue
- API rate limits:
  * Requests used/remaining
  * Reset time
  * Per-provider breakdown
- Performance metrics:
  * Response times
  * Query times
  * Error rates
- Auto-refresh every 5 seconds

Real-time system health.
```

---

## CURSOR PROMPT 10.6 - Implement Logs & Debugging (Tab 5)

```
Implement Tab 5 (Logs & Debugging) in pages/10_Admin.py

Log viewing and debugging:
- Log level selector (DEBUG, INFO, WARNING, ERROR)
- Log source selector (all, app, trading, data, etc.)
- Live log tail (auto-scrolling)
- Log search
- Date/time filter
- Download logs button
- Error summary:
  * Error count by type
  * Most common errors
  * Error trend
- Debugging tools:
  * Clear cache
  * Reset session
  * Force garbage collection
  * Test database connection

Debug production issues.
```

---

## CURSOR PROMPT 10.7 - Implement Maintenance (Tab 6)

```
Implement Tab 6 (Maintenance) in pages/10_Admin.py

System maintenance tools:
- Database:
  * Backup database
  * Restore from backup
  * Optimize database
  * Vacuum database
  * Database size
- Cache management:
  * Clear all cache
  * Clear specific cache
  * Cache hit rate
- Data cleanup:
  * Delete old logs (older than X days)
  * Archive old data
  * Clean temp files
- System updates:
  * Check for updates
  * Installed version
  * Update history
- Scheduled maintenance:
  * Configure backup schedule
  * Configure cleanup schedule

Keep system running smoothly.
```

---

## ✅ PAGE 10 CHECKLIST

- [ ] File created: pages/10_Admin.py
- [ ] All 6 tabs implemented
- [ ] System dashboard shows health
- [ ] Configuration management works
- [ ] Agent management functional
- [ ] Real-time monitoring displays
- [ ] Logs viewable and searchable
- [ ] Maintenance tools operational
- [ ] Admin authentication in place
- [ ] All actions have confirmations
- [ ] Committed to git

---

## 🚀 COMMIT COMMAND

```bash
git add pages/10_Admin.py
git commit -m "feat(page-10): Implement System Administration with 6 tabs

- Tab 1: System health dashboard
- Tab 2: Configuration and API key management
- Tab 3: AI agent lifecycle management
- Tab 4: Real-time system monitoring
- Tab 5: Logs and debugging tools
- Tab 6: Database and cache maintenance

Merges 4 admin/monitoring pages into unified admin panel"
```

---

## 🎉 ALL 10 PAGES COMPLETE!

You now have complete integration guides for all 10 streamlined pages!

**Next Step:** Create master integration tracker and summary document.
