# EVOLVE - COMPREHENSIVE ARCHITECTURAL ANALYSIS
# Master script that runs all diagnostics

Write-Host "`n" + "="*80 -ForegroundColor Magenta
Write-Host "EVOLVE TRADING SYSTEM - COMPREHENSIVE ANALYSIS" -ForegroundColor Magenta
Write-Host "="*80 -ForegroundColor Magenta
Write-Host ""
Write-Host "This will run 3 comprehensive analyses:" -ForegroundColor Cyan
Write-Host "  1. Structural Analysis (imports, duplicates, legacy files)" -ForegroundColor White
Write-Host "  2. Dead Code Detection (unused files, orphaned code)" -ForegroundColor White
Write-Host "  3. Code Quality Check (syntax errors, anti-patterns)" -ForegroundColor White
Write-Host ""

$continue = Read-Host "Continue? (Y/n)"
if ($continue -eq 'n' -or $continue -eq 'N') {
    Write-Host "Cancelled." -ForegroundColor Yellow
    exit
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$masterReport = "MASTER_ANALYSIS_$timestamp"
New-Item -Path $masterReport -ItemType Directory -Force | Out-Null

Write-Host "`n📁 All reports will be saved to: $masterReport`n" -ForegroundColor Yellow

# ============================================================================
# 1. STRUCTURAL ANALYSIS
# ============================================================================

Write-Host "`n" + "="*80 -ForegroundColor Cyan
Write-Host "[1/3] STRUCTURAL ANALYSIS" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Cyan

.\run_diagnostics.ps1

# Move reports to master folder
Get-ChildItem -Path "cleanup_reports_*" -Directory | ForEach-Object {
    $sourceDir = $_.FullName
    Get-ChildItem -Path $sourceDir | ForEach-Object {
        Copy-Item $_.FullName "$masterReport\structural_$($_.Name)"
    }
    Remove-Item $sourceDir -Recurse -Force
}

# ============================================================================
# 2. DEAD CODE ANALYSIS
# ============================================================================

Write-Host "`n" + "="*80 -ForegroundColor Cyan
Write-Host "[2/3] DEAD CODE DETECTION" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Cyan

.\analyze_dead_code.ps1

# Move reports
Get-ChildItem -Path "dead_code_analysis_*.txt" | ForEach-Object {
    Move-Item $_.FullName "$masterReport\dead_code_analysis.txt"
}

if (Test-Path "DELETE_CANDIDATES.ps1") {
    Move-Item "DELETE_CANDIDATES.ps1" "$masterReport\DELETE_CANDIDATES.ps1"
}

# ============================================================================
# 3. CODE QUALITY ANALYSIS
# ============================================================================

Write-Host "`n" + "="*80 -ForegroundColor Cyan
Write-Host "[3/3] CODE QUALITY CHECK" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Cyan

.\analyze_code_quality.ps1

# Move reports
Get-ChildItem -Path "code_quality_analysis_*.txt" | ForEach-Object {
    Move-Item $_.FullName "$masterReport\code_quality_analysis.txt"
}

# ============================================================================
# GENERATE MASTER SUMMARY
# ============================================================================

Write-Host "`n📊 Generating master summary..." -ForegroundColor Cyan

# Read all reports
$structuralSummary = Get-Content "$masterReport\structural_0_SUMMARY.txt" -Raw -ErrorAction SilentlyContinue
$deadCodeReport = Get-Content "$masterReport\dead_code_analysis.txt" -Raw -ErrorAction SilentlyContinue
$qualityReport = Get-Content "$masterReport\code_quality_analysis.txt" -Raw -ErrorAction SilentlyContinue

# Extract key metrics
$brokenImports = 0
$duplicateClasses = 0
$legacyFiles = 0
$neverImported = 0
$syntaxErrors = 0

if ($structuralSummary -match "Broken Imports:\s*(\d+)") { $brokenImports = [int]$matches[1] }
if ($structuralSummary -match "Duplicate Classes:\s*(\d+)") { $duplicateClasses = [int]$matches[1] }
if ($structuralSummary -match "Legacy Files:\s*(\d+)") { $legacyFiles = [int]$matches[1] }
if ($deadCodeReport -match "Never Imported:\s*(\d+)") { $neverImported = [int]$matches[1] }
if ($qualityReport -match "Syntax Errors:\s*(\d+)") { $syntaxErrors = [int]$matches[1] }

$totalIssues = $brokenImports + $duplicateClasses + $legacyFiles + $neverImported + $syntaxErrors

# Create master summary
$masterSummary = @"
EVOLVE TRADING SYSTEM - MASTER ANALYSIS SUMMARY
Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
="*80

EXECUTIVE SUMMARY:
="*80

Total Issues Found: $totalIssues

CRITICAL (Fix Immediately):
  ❌ Broken Imports: $brokenImports
  ❌ Syntax Errors: $syntaxErrors

HIGH PRIORITY (Fix Soon):
  ⚠️  Duplicate Classes: $duplicateClasses
  ⚠️  Never Imported Files: $neverImported

MEDIUM PRIORITY (Cleanup):
  🗑️  Legacy Files: $legacyFiles

="*80
SEVERITY ASSESSMENT:
="*80

$(if ($syntaxErrors -gt 0) { "🚨 CRITICAL: Syntax errors found - code won't run!" } else { "✅ No syntax errors" })
$(if ($brokenImports -gt 5) { "🚨 CRITICAL: Many broken imports - system unstable" } elseif ($brokenImports -gt 0) { "⚠️  HIGH: Some broken imports" } else { "✅ No broken imports" })
$(if ($neverImported -gt 50) { "⚠️  HIGH: Significant dead code (~$([math]::Round($neverImported * 100.0 / 962, 1))% of files)" } elseif ($neverImported -gt 20) { "⚠️  MEDIUM: Moderate dead code" } else { "✅ Minimal dead code" })
$(if ($duplicateClasses -gt 10) { "⚠️  HIGH: Significant code duplication" } elseif ($duplicateClasses -gt 0) { "⚠️  MEDIUM: Some duplicate classes" } else { "✅ No duplicates" })

="*80
RECOMMENDED ACTION PLAN:
="*80

PHASE 1: EMERGENCY FIXES (Do Today)
$(if ($syntaxErrors -gt 0) { "  1. Fix $syntaxErrors syntax errors (see code_quality_analysis.txt)" } else { "  ✅ No syntax errors" })
$(if ($brokenImports -gt 0) { "  2. Fix $brokenImports broken imports (see structural_1_broken_imports.txt)" } else { "  ✅ No broken imports" })

PHASE 2: HIGH PRIORITY (This Week)
  3. Review $neverImported never-imported files (see dead_code_analysis.txt)
  4. Delete confirmed dead code (use DELETE_CANDIDATES.ps1 template)
$(if ($duplicateClasses -gt 5) { "  5. Consolidate $duplicateClasses duplicate classes (see structural_2_duplicate_classes.txt)" } else { "  ✅ Minimal duplicates" })

PHASE 3: CLEANUP (Next Week)
$(if ($legacyFiles -gt 0) { "  6. Remove $legacyFiles legacy files (see structural_3_legacy_files.txt)" } else { "  ✅ No legacy files" })
  7. Add missing __init__.py files (see structural_5_missing_init.txt)
  8. Clean up directory structure (see structural_4_duplicate_directories.txt)

="*80
DETAILED REPORTS:
="*80

$masterReport\structural_0_SUMMARY.txt          - Structural issues summary
$masterReport\structural_1_broken_imports.txt   - Import errors
$masterReport\structural_2_duplicate_classes.txt - Duplicate code
$masterReport\structural_3_legacy_files.txt     - Old files to remove
$masterReport\structural_4_duplicate_directories.txt - Structure issues
$masterReport\structural_5_missing_init.txt     - Missing __init__ files
$masterReport\dead_code_analysis.txt            - Unused files analysis
$masterReport\code_quality_analysis.txt         - Code quality issues
$masterReport\DELETE_CANDIDATES.ps1             - Template deletion script

="*80
NEXT STEPS:
="*80

1. Review this summary
2. Read detailed reports for each issue
3. Create git branch: git checkout -b cleanup/architecture
4. Fix critical issues first (syntax, imports)
5. Test after each change
6. Remove dead code carefully
7. Document cleanup in git commits

="*80
METRICS:
="*80

Total Python Files: 962
Issues Per File Average: $([math]::Round($totalIssues / 962.0, 2))
System Health Score: $([math]::Round(100 - ($totalIssues / 962.0 * 10), 1))%

$(if ($totalIssues -lt 50) { 
    "✅ GOOD: System is relatively clean" 
} elseif ($totalIssues -lt 150) { 
    "⚠️  MODERATE: System needs cleanup" 
} else { 
    "🚨 POOR: Significant cleanup required" 
})

="*80

Generated by EVOLVE Architectural Analysis Suite
For questions or issues, review the detailed reports above.

"@

$masterSummary | Out-File "$masterReport\00_MASTER_SUMMARY.txt"

# ============================================================================
# FINAL REPORT
# ============================================================================

Write-Host "`n" + "="*80 -ForegroundColor Magenta
Write-Host "COMPREHENSIVE ANALYSIS COMPLETE!" -ForegroundColor Magenta
Write-Host "="*80 -ForegroundColor Magenta
Write-Host ""

Write-Host "📊 RESULTS:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Critical Issues:" -ForegroundColor Red
Write-Host "    - Syntax Errors: $syntaxErrors" -ForegroundColor $(if ($syntaxErrors -gt 0) { "Red" } else { "Green" })
Write-Host "    - Broken Imports: $brokenImports" -ForegroundColor $(if ($brokenImports -gt 0) { "Red" } else { "Green" })
Write-Host ""
Write-Host "  High Priority:" -ForegroundColor Yellow
Write-Host "    - Never Imported: $neverImported files"
Write-Host "    - Duplicates: $duplicateClasses classes"
Write-Host ""
Write-Host "  Cleanup:" -ForegroundColor Gray
Write-Host "    - Legacy Files: $legacyFiles files"
Write-Host ""

Write-Host "📁 All reports saved to: $masterReport\" -ForegroundColor Cyan
Write-Host ""
Write-Host "📄 START HERE: $masterReport\00_MASTER_SUMMARY.txt" -ForegroundColor Green
Write-Host ""

if ($syntaxErrors -gt 0 -or $brokenImports -gt 0) {
    Write-Host "⚠️  ATTENTION: Critical issues found!" -ForegroundColor Red
    Write-Host "   Fix syntax errors and broken imports immediately." -ForegroundColor Red
    Write-Host ""
}

Write-Host "✅ Next: Read the master summary and start fixing!" -ForegroundColor Green
Write-Host ""

# Open the summary
Write-Host "Opening master summary..." -ForegroundColor Gray
Start-Process notepad.exe "$masterReport\00_MASTER_SUMMARY.txt"
