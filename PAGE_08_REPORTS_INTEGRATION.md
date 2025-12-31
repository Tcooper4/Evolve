# PAGE 8: 📄 REPORTS & EXPORTS
**Complete Integration Guide for Cursor AI**

---

## 📋 OVERVIEW

**Target File:** `pages/8_Reports.py`  
**Merges:** Reports.py (standalone)  
**Tabs:** 4 tabs  
**Estimated Time:** 6-8 hours  
**Priority:** MEDIUM

### Features Preserved:
✅ Trading performance reports  
✅ Risk analysis reports  
✅ Portfolio summary reports  
✅ Custom report builder  
✅ Scheduled reports  
✅ Multiple export formats (PDF, Excel, HTML)  
✅ Email delivery

---

## CURSOR PROMPT 8.1 - Create Page Structure

```
Create pages/8_Reports.py with 4-tab structure

BACKEND IMPORTS:
from trading.report.report_generator import ReportGenerator
from trading.report.export_report import ExportReport
from trading.report.report_export_engine import ReportExportEngine

TABS:
1. Quick Reports
2. Custom Report Builder
3. Scheduled Reports
4. Report Library

Initialize report generator on page load.
```

---

## CURSOR PROMPT 8.2 - Implement Quick Reports (Tab 1)

```
Implement Tab 1 (Quick Reports) in pages/8_Reports.py

Pre-built report templates:
- Report type selector:
  * Daily Performance Report
  * Weekly Summary
  * Monthly Performance Report
  * Quarterly Review
  * Annual Report
  * Risk Analysis Report
  * Portfolio Summary
  * Trade Journal
  * Tax Report
- Date range selector
- Generate button
- Report preview in streamlit
- Export options:
  * PDF
  * Excel
  * HTML
  * Email delivery

One-click report generation.
```

---

## CURSOR PROMPT 8.3 - Implement Custom Report Builder (Tab 2)

```
Implement Tab 2 (Custom Report Builder) in pages/8_Reports.py

Build custom reports:
- Section selector (drag-and-drop or checklist):
  * Executive Summary
  * Performance Metrics
  * Portfolio Holdings
  * Trade History
  * Risk Analysis
  * Charts (equity curve, allocation, etc.)
  * Custom text sections
- Configuration for each section:
  * Date ranges
  * Metrics to include
  * Chart types
- Report template saving
- Report branding (logo, colors)
- Preview button
- Generate button

Full customization capabilities.
```

---

## CURSOR PROMPT 8.4 - Implement Scheduled Reports (Tab 3)

```
Implement Tab 3 (Scheduled Reports) in pages/8_Reports.py

Automated report scheduling:
- Create schedule interface:
  * Report type selection
  * Frequency (daily, weekly, monthly)
  * Day/time selection
  * Recipients (email list)
  * Format selection
- Active schedules table:
  * Schedule name
  * Report type
  * Frequency
  * Next run
  * Status
  * Edit/Delete buttons
- Schedule testing (send test report)
- Schedule history
- Enable/disable toggles

Automation for recurring reports.
```

---

## CURSOR PROMPT 8.5 - Implement Report Library (Tab 4)

```
Implement Tab 4 (Report Library) in pages/8_Reports.py

Previously generated reports:
- Reports table:
  * Report name
  * Type
  * Generation date
  * File size
  * Actions (view, download, share, delete)
- Search and filter
- Sort by date, type, name
- Preview report inline
- Re-generate with updated data
- Share report (generate link)
- Batch operations (download multiple, delete old)

Report archive and management.
```

---

## ✅ PAGE 8 CHECKLIST

- [ ] File created: pages/8_Reports.py
- [ ] All 4 tabs implemented
- [ ] Quick reports generate correctly
- [ ] Custom builder works
- [ ] Scheduled reports functional
- [ ] Report library operational
- [ ] PDF export works
- [ ] Excel export works
- [ ] Email delivery works
- [ ] Committed to git

---

## 🚀 COMMIT COMMAND

```bash
git add pages/8_Reports.py
git commit -m "feat(page-8): Implement Reports & Exports with 4 tabs

- Tab 1: Pre-built report templates
- Tab 2: Custom report builder
- Tab 3: Automated scheduled reports
- Tab 4: Report library and management

Full reporting suite with multiple export formats"
```

---

**Next:** PAGE_09_ALERTS_INTEGRATION.md
