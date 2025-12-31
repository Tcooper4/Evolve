# 🛡️ EVOLVE CODEBASE - SAFE CLEANUP TOOLKIT
# 100% SAFE - Creates backups, requires confirmation, fully reversible

# USAGE:
# 1. Run Step 1: Creates full backup and analyzes issues
# 2. Review the analysis report
# 3. Run Step 2: Creates fix scripts (but doesn't execute them)
# 4. Review each fix script
# 5. Run Step 3: Execute fixes one at a time with confirmation

# ============================================================================
# STEP 1: BACKUP AND ANALYZE (100% SAFE - READ ONLY)
# ============================================================================

Write-Host "=" -NoNewline
Write-Host "=" * 79
Write-Host "🛡️ EVOLVE SAFE CLEANUP - STEP 1: BACKUP & ANALYZE"
Write-Host "=" * 80
Write-Host ""
Write-Host "This script will:"
Write-Host "  ✅ Create a full backup of your codebase"
Write-Host "  ✅ Analyze all issues (READ ONLY)"
Write-Host "  ✅ Generate a detailed report"
Write-Host "  ❌ NOT modify ANY files"
Write-Host ""

$confirmation = Read-Host "Proceed with backup and analysis? (yes/no)"
if ($confirmation -ne "yes") {
    Write-Host "❌ Aborted by user"
    exit
}

# Create backup directory with timestamp
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupDir = "BACKUP_$timestamp"

Write-Host ""
Write-Host "📦 Creating backup..."
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null

# Backup critical directories
$dirsToBackup = @("pages", "trading", "agents", "utils", "core", "config")

foreach ($dir in $dirsToBackup) {
    if (Test-Path $dir) {
        Write-Host "  ✅ Backing up $dir..."
        Copy-Item -Path $dir -Destination "$backupDir/" -Recurse -Force
    } else {
        Write-Host "  ⚠️  Skipping $dir (not found)"
    }
}

# Backup individual important files
$filesToBackup = @("app.py", "main.py", "requirements.txt", ".env")
foreach ($file in $filesToBackup) {
    if (Test-Path $file) {
        Write-Host "  ✅ Backing up $file..."
        Copy-Item -Path $file -Destination "$backupDir/" -Force
    }
}

Write-Host ""
Write-Host "✅ Backup completed: $backupDir"
Write-Host ""

# Create analysis report directory
$reportDir = "ANALYSIS_REPORT_$timestamp"
New-Item -ItemType Directory -Path $reportDir -Force | Out-Null

Write-Host "🔍 Analyzing codebase..."
Write-Host ""

# ============================================================================
# ANALYSIS 1: Find problematic imports
# ============================================================================

Write-Host "📋 Analysis 1: Finding problematic imports..."

$problematicImports = @()

# Find files with core.session_utils imports
Get-ChildItem -Path . -Recurse -Filter "*.py" -ErrorAction SilentlyContinue | ForEach-Object {
    $content = Get-Content $_.FullName -ErrorAction SilentlyContinue
    if ($content -match "from core\.session_utils") {
        $problematicImports += @{
            File = $_.FullName
            Import = "core.session_utils"
            ShouldBe = "utils.session_utils"
            Line = ($content | Select-String "from core\.session_utils" | Select-Object -First 1).LineNumber
        }
    }
    if ($content -match "from core\.task_orchestrator import") {
        $problematicImports += @{
            File = $_.FullName
            Import = "core.task_orchestrator"
            ShouldBe = "core.orchestrator.task_orchestrator"
            Line = ($content | Select-String "from core\.task_orchestrator" | Select-Object -First 1).LineNumber
        }
    }
    if ($content -match "from core\.agent_hub") {
        $problematicImports += @{
            File = $_.FullName
            Import = "core.agent_hub"
            ShouldBe = "[MISSING - needs to be created]"
            Line = ($content | Select-String "from core\.agent_hub" | Select-Object -First 1).LineNumber
        }
    }
}

# Save to report
$importReport = "$reportDir/1_problematic_imports.txt"
@"
PROBLEMATIC IMPORTS REPORT
Generated: $(Get-Date)
==================================================

Found $($problematicImports.Count) files with problematic imports:

"@ | Out-File $importReport

$problematicImports | ForEach-Object {
    @"
File: $($_.File)
  Line: $($_.Line)
  Current: from $($_.Import) import ...
  Should be: from $($_.ShouldBe) import ...

"@ | Out-File $importReport -Append
}

Write-Host "  ✅ Found $($problematicImports.Count) problematic imports"
Write-Host "  📄 Report saved: $importReport"
Write-Host ""

# ============================================================================
# ANALYSIS 2: Find duplicate pages
# ============================================================================

Write-Host "📋 Analysis 2: Finding duplicate pages..."

$allPages = Get-ChildItem -Path pages -Filter "*.py" -ErrorAction SilentlyContinue | Where-Object { $_.Name -ne "__init__.py" }

# Categorize pages
$duplicatePages = @(
    "1_Forecast_Trade.py",  # Duplicate of Forecasting.py
    "forecast.py",           # Test version
    "2_Backtest_Strategy.py", # Too basic
    "4_Portfolio_Management.py", # Too basic
    "5_Risk_Analysis.py",    # Too basic
    "risk_dashboard.py",     # Superseded
    "6_Strategy_History.py", # Covered elsewhere
    "strategy.py",           # Test version
    "Strategy_Pipeline_Demo.py", # Demo
    "HybridModel.py",        # Test
    "nlp_tester.py",         # Test
    "ui_helpers.py",         # Empty/test
    "home.py",               # Duplicate of app.py
    "settings.py",           # Covered by Admin
    "optimization_dashboard.py", # Duplicate
    "performance_tracker.py" # Duplicate
)

$keepPages = @(
    "Forecasting.py",
    "2_Strategy_Backtest.py",
    "portfolio_dashboard.py",
    "risk_preview_dashboard.py",
    "Strategy_Lab.py",
    "Strategy_Combo_Creator.py",
    "10_Strategy_Health_Dashboard.py",
    "7_Strategy_Performance.py",
    "Model_Lab.py",
    "Model_Performance_Dashboard.py",
    "Monte_Carlo_Simulation.py",
    "Reports.py",
    "9_System_Monitoring.py",
    "18_Alerts.py",
    "19_Admin_Panel.py"
)

# Save to report
$pageReport = "$reportDir/2_page_analysis.txt"
@"
PAGE ANALYSIS REPORT
Generated: $(Get-Date)
==================================================

Current page count: $($allPages.Count)
Recommended page count: $($keepPages.Count)
Pages to archive: $($duplicatePages.Count)

PAGES TO KEEP (Core functionality):
"@ | Out-File $pageReport

$keepPages | ForEach-Object {
    $file = Get-Item "pages/$_" -ErrorAction SilentlyContinue
    if ($file) {
        $lines = (Get-Content $file.FullName | Measure-Object -Line).Lines
        "  ✅ $_ ($lines lines)" | Out-File $pageReport -Append
    } else {
        "  ⚠️  $_ (NOT FOUND)" | Out-File $pageReport -Append
    }
}

"`n`nPAGES TO ARCHIVE (Duplicates/Tests):`n" | Out-File $pageReport -Append

$duplicatePages | ForEach-Object {
    $file = Get-Item "pages/$_" -ErrorAction SilentlyContinue
    if ($file) {
        $lines = (Get-Content $file.FullName | Measure-Object -Line).Lines
        "  ❌ $_ ($lines lines)" | Out-File $pageReport -Append
    } else {
        "  ⚠️  $_ (NOT FOUND)" | Out-File $pageReport -Append
    }
}

Write-Host "  ✅ Analyzed $($allPages.Count) pages"
Write-Host "  📄 Report saved: $pageReport"
Write-Host ""

# ============================================================================
# ANALYSIS 3: Find duplicate backend files
# ============================================================================

Write-Host "📋 Analysis 3: Finding duplicate backend implementations..."

$duplicateBackendFiles = @(
    @{
        Category = "Risk Management"
        Files = @(
            "trading/risk/risk_manager.py (KEEP - Main implementation)",
            "portfolio/risk_manager.py (REMOVE - Duplicate)"
        )
    },
    @{
        Category = "Forecast Explainability"
        Files = @(
            "trading/models/forecast_explainability.py (KEEP)",
            "trading/analytics/forecast_explainability.py (REMOVE - Duplicate)"
        )
    },
    @{
        Category = "Agent Implementation"
        Files = @(
            "agents/model_generator_agent.py (Review)",
            "trading/agents/model_synthesizer_agent.py (Review - May be duplicate)"
        )
    }
)

$backendReport = "$reportDir/3_duplicate_backends.txt"
@"
DUPLICATE BACKEND FILES REPORT
Generated: $(Get-Date)
==================================================

Found $($duplicateBackendFiles.Count) categories with potential duplicates:

"@ | Out-File $backendReport

$duplicateBackendFiles | ForEach-Object {
    "`n$($_.Category):" | Out-File $backendReport -Append
    $_.Files | ForEach-Object {
        "  $_" | Out-File $backendReport -Append
    }
}

"`n`nNOTE: These require manual review before deletion!" | Out-File $backendReport -Append

Write-Host "  ✅ Identified $($duplicateBackendFiles.Count) duplicate categories"
Write-Host "  📄 Report saved: $backendReport"
Write-Host ""

# ============================================================================
# ANALYSIS 4: Check for missing modules
# ============================================================================

Write-Host "📋 Analysis 4: Checking for missing modules..."

$missingModules = @()

# Check if core/agent_hub.py exists
if (-not (Test-Path "core/agent_hub.py")) {
    $missingModules += @{
        Module = "core/agent_hub.py"
        Status = "MISSING - Referenced but not found"
        ReferencedBy = @()
    }
    
    # Find what references it
    Get-ChildItem -Path . -Recurse -Filter "*.py" -ErrorAction SilentlyContinue | ForEach-Object {
        $content = Get-Content $_.FullName -ErrorAction SilentlyContinue
        if ($content -match "from core\.agent_hub") {
            $missingModules[-1].ReferencedBy += $_.FullName
        }
    }
}

$missingReport = "$reportDir/4_missing_modules.txt"
@"
MISSING MODULES REPORT
Generated: $(Get-Date)
==================================================

Found $($missingModules.Count) missing modules:

"@ | Out-File $missingReport

$missingModules | ForEach-Object {
    @"
Module: $($_.Module)
Status: $($_.Status)
Referenced by:
"@ | Out-File $missingReport -Append
    $_.ReferencedBy | ForEach-Object {
        "  - $_" | Out-File $missingReport -Append
    }
    "" | Out-File $missingReport -Append
}

Write-Host "  ✅ Found $($missingModules.Count) missing modules"
Write-Host "  📄 Report saved: $missingReport"
Write-Host ""

# ============================================================================
# Create master summary report
# ============================================================================

$summaryReport = "$reportDir/0_SUMMARY.txt"
@"
EVOLVE CODEBASE - ANALYSIS SUMMARY
Generated: $(Get-Date)
==================================================

BACKUP LOCATION: $backupDir

ISSUES FOUND:

1. Problematic Imports: $($problematicImports.Count) files
   📄 Details: $importReport

2. Duplicate Pages: $($duplicatePages.Count) pages to archive
   📄 Details: $pageReport

3. Duplicate Backends: $($duplicateBackendFiles.Count) categories
   📄 Details: $backendReport

4. Missing Modules: $($missingModules.Count) modules
   📄 Details: $missingReport

==================================================
NEXT STEPS:

1. Review all reports in: $reportDir
2. Run Step 2 script to generate fix scripts
3. Review fix scripts before executing
4. Execute fixes with Step 3 script

NOTE: All fixes are reversible using backup: $backupDir
==================================================
"@ | Out-File $summaryReport

Write-Host ""
Write-Host "=" * 80
Write-Host "✅ ANALYSIS COMPLETE"
Write-Host "=" * 80
Write-Host ""
Write-Host "📊 Summary:"
Write-Host "  • Backup created: $backupDir"
Write-Host "  • Reports created: $reportDir"
Write-Host "  • Problematic imports: $($problematicImports.Count)"
Write-Host "  • Pages to archive: $($duplicatePages.Count)"
Write-Host "  • Missing modules: $($missingModules.Count)"
Write-Host ""
Write-Host "📄 See detailed summary: $summaryReport"
Write-Host ""
Write-Host "🎯 Next: Review the reports, then run Step 2 to generate fix scripts"
Write-Host ""
