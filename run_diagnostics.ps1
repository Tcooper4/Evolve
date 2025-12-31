# EVOLVE Architectural Cleanup - Master Diagnostic Script
# Run this first to analyze the codebase

Write-Host "`n" + "="*80 -ForegroundColor Cyan
Write-Host "EVOLVE ARCHITECTURAL CLEANUP - DIAGNOSTIC ANALYSIS" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$reportDir = "cleanup_reports_$timestamp"
New-Item -Path $reportDir -ItemType Directory -Force | Out-Null

Write-Host "📁 Reports will be saved to: $reportDir`n" -ForegroundColor Yellow

# ============================================================================
# 1. BROKEN IMPORTS ANALYSIS
# ============================================================================

Write-Host "🔍 [1/5] Analyzing Broken Imports..." -ForegroundColor Cyan

$brokenImports = @()
$importStats = @{
    'trading.ui' = 0
    'trading.components' = 0
    'ui.components' = 0
    'other' = 0
}

Get-ChildItem -Path . -Include *.py -Recurse | Where-Object {
    $_.FullName -notmatch '\.venv|__pycache__|\.git'
} | ForEach-Object {
    $file = $_.FullName
    $relativePath = $_.FullName -replace [regex]::Escape($PWD), '.'
    $content = Get-Content $file -Raw -ErrorAction SilentlyContinue
    
    if ($content) {
        if ($content -match "from trading\.ui|import trading\.ui") {
            $brokenImports += "❌ $relativePath - imports trading.ui"
            $importStats['trading.ui']++
        }
        if ($content -match "from trading\.components|import trading\.components") {
            $brokenImports += "❌ $relativePath - imports trading.components"
            $importStats['trading.components']++
        }
        if ($content -match "from ui\.components|import ui\.components") {
            $brokenImports += "⚠️  $relativePath - imports ui.components (verify exists)"
            $importStats['ui.components']++
        }
    }
}

Write-Host "   Found: $($brokenImports.Count) broken imports"
Write-Host "   - trading.ui: $($importStats['trading.ui'])"
Write-Host "   - trading.components: $($importStats['trading.components'])"
Write-Host "   - ui.components: $($importStats['ui.components'])`n"

$brokenImports | Out-File "$reportDir/1_broken_imports.txt"

# ============================================================================
# 2. DUPLICATE CLASS ANALYSIS
# ============================================================================

Write-Host "🔍 [2/5] Analyzing Duplicate Classes..." -ForegroundColor Cyan

$classes = @{}

Get-ChildItem -Path . -Include *.py -Recurse | Where-Object {
    $_.FullName -notmatch '\.venv|__pycache__|\.git'
} | ForEach-Object {
    $file = $_.FullName
    $relativePath = $_.FullName -replace [regex]::Escape($PWD), '.'
    $content = Get-Content $file -ErrorAction SilentlyContinue
    
    $content | Select-String "^class (\w+)" | ForEach-Object {
        $className = $_.Matches[0].Groups[1].Value
        
        if (-not $classes.ContainsKey($className)) {
            $classes[$className] = @()
        }
        $classes[$className] += $relativePath
    }
}

$duplicates = $classes.GetEnumerator() | Where-Object { $_.Value.Count -gt 1 }
$duplicateReport = @()

$duplicateReport += "DUPLICATE CLASSES FOUND: $($duplicates.Count)"
$duplicateReport += "="*80
$duplicateReport += ""

foreach ($dup in $duplicates) {
    $duplicateReport += "Class: $($dup.Key)"
    $dup.Value | ForEach-Object { $duplicateReport += "   - $_" }
    $duplicateReport += ""
}

Write-Host "   Found: $($duplicates.Count) duplicate classes`n"

$duplicateReport | Out-File "$reportDir/2_duplicate_classes.txt"

# ============================================================================
# 3. LEGACY FILES ANALYSIS
# ============================================================================

Write-Host "🔍 [3/5] Analyzing Legacy Files..." -ForegroundColor Cyan

$legacyPatterns = @(
    "*_old.py", "*_backup.py", "*_legacy.py", "*_deprecated.py",
    "*_v1.py", "*_v2.py", "*_bak.py", "*.bak", "*_copy.py"
)

$legacyFiles = @()

foreach ($pattern in $legacyPatterns) {
    Get-ChildItem -Path . -Filter $pattern -Recurse | Where-Object {
        $_.FullName -notmatch '\.venv|__pycache__|\.git'
    } | ForEach-Object {
        $relativePath = $_.FullName -replace [regex]::Escape($PWD), '.'
        $legacyFiles += "🗑️  $relativePath"
    }
}

Write-Host "   Found: $($legacyFiles.Count) legacy files`n"

$legacyFiles | Out-File "$reportDir/3_legacy_files.txt"

# ============================================================================
# 4. DIRECTORY STRUCTURE ANALYSIS
# ============================================================================

Write-Host "🔍 [4/5] Analyzing Directory Structure..." -ForegroundColor Cyan

$dirs = @{}

Get-ChildItem -Path . -Directory -Recurse | Where-Object {
    $_.FullName -notmatch '\.venv|__pycache__|\.git|node_modules'
} | ForEach-Object {
    $dirName = $_.Name
    if (-not $dirs.ContainsKey($dirName)) {
        $dirs[$dirName] = @()
    }
    $relativePath = $_.FullName -replace [regex]::Escape($PWD), '.'
    $dirs[$dirName] += $relativePath
}

$dupDirs = $dirs.GetEnumerator() | Where-Object { $_.Value.Count -gt 1 }
$dirReport = @()

$dirReport += "DUPLICATE DIRECTORY NAMES: $($dupDirs.Count)"
$dirReport += "="*80
$dirReport += ""

foreach ($dup in $dupDirs) {
    $dirReport += "Directory: $($dup.Key)"
    $dup.Value | ForEach-Object { $dirReport += "   - $_" }
    $dirReport += ""
}

Write-Host "   Found: $($dupDirs.Count) duplicate directory names`n"

$dirReport | Out-File "$reportDir/4_duplicate_directories.txt"

# ============================================================================
# 5. MISSING __init__.py ANALYSIS
# ============================================================================

Write-Host "🔍 [5/5] Analyzing Missing __init__.py Files..." -ForegroundColor Cyan

$missingInit = @()

Get-ChildItem -Path . -Directory -Recurse | Where-Object {
    $_.FullName -notmatch '\.venv|__pycache__|\.git|node_modules'
} | ForEach-Object {
    $dir = $_
    $relativePath = $dir.FullName -replace [regex]::Escape($PWD), '.'
    
    # Check if directory has .py files
    $pyFiles = Get-ChildItem -Path $dir.FullName -Filter "*.py" -File -ErrorAction SilentlyContinue
    
    if ($pyFiles.Count -gt 0) {
        $initFile = Join-Path $dir.FullName "__init__.py"
        if (-not (Test-Path $initFile)) {
            $missingInit += "❌ $relativePath"
        }
    }
}

Write-Host "   Found: $($missingInit.Count) directories without __init__.py`n"

$missingInit | Out-File "$reportDir/5_missing_init.txt"

# ============================================================================
# GENERATE SUMMARY REPORT
# ============================================================================

Write-Host "📊 Generating Summary Report..." -ForegroundColor Cyan

$summary = @"
EVOLVE ARCHITECTURAL CLEANUP - DIAGNOSTIC SUMMARY
Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
="*80

ISSUES FOUND:
="*80

1. BROKEN IMPORTS: $($brokenImports.Count) files
   - trading.ui: $($importStats['trading.ui']) files
   - trading.components: $($importStats['trading.components']) files
   - ui.components: $($importStats['ui.components']) files

2. DUPLICATE CLASSES: $($duplicates.Count) classes defined in multiple places

3. LEGACY FILES: $($legacyFiles.Count) files with legacy naming patterns

4. DUPLICATE DIRECTORIES: $($dupDirs.Count) directory names appearing multiple times

5. MISSING __init__.py: $($missingInit.Count) Python directories without __init__.py

="*80
PRIORITY ASSESSMENT:
="*80

CRITICAL (Fix Immediately):
$(if ($brokenImports.Count -gt 0) { "❌ BROKEN IMPORTS - System has import errors" } else { "✅ No broken imports" })
$(if ($missingInit.Count -gt 10) { "⚠️  MISSING __init__.py - Many directories need init files" } elseif ($missingInit.Count -gt 0) { "⚠️  Some missing __init__.py files" } else { "✅ All __init__.py files present" })

HIGH (Fix Soon):
$(if ($duplicates.Count -gt 5) { "⚠️  DUPLICATE CLASSES - Significant duplication found" } elseif ($duplicates.Count -gt 0) { "⚠️  Some duplicate classes" } else { "✅ No duplicate classes" })
$(if ($dupDirs.Count -gt 5) { "⚠️  DUPLICATE DIRECTORIES - Structure needs consolidation" } elseif ($dupDirs.Count -gt 0) { "⚠️  Some duplicate directories" } else { "✅ Clean directory structure" })

MEDIUM (Cleanup):
$(if ($legacyFiles.Count -gt 10) { "🗑️  LEGACY FILES - Significant cleanup needed" } elseif ($legacyFiles.Count -gt 0) { "🗑️  Some legacy files to remove" } else { "✅ No legacy files" })

="*80
RECOMMENDED NEXT STEPS:
="*80

1. Review detailed reports in: $reportDir/
2. Fix broken imports (if any)
3. Add missing __init__.py files
4. Review and remove legacy files
5. Consolidate duplicate classes
6. Clean up directory structure

="*80
DETAILED REPORTS:
="*80

$reportDir/1_broken_imports.txt
$reportDir/2_duplicate_classes.txt
$reportDir/3_legacy_files.txt
$reportDir/4_duplicate_directories.txt
$reportDir/5_missing_init.txt

"@

$summary | Out-File "$reportDir/0_SUMMARY.txt"

# ============================================================================
# DISPLAY SUMMARY
# ============================================================================

Write-Host "`n" + "="*80 -ForegroundColor Green
Write-Host "DIAGNOSTIC ANALYSIS COMPLETE" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Green
Write-Host ""

Write-Host "ISSUES FOUND:" -ForegroundColor Yellow
Write-Host "  1. Broken Imports: $($brokenImports.Count)" -ForegroundColor $(if ($brokenImports.Count -gt 0) { "Red" } else { "Green" })
Write-Host "  2. Duplicate Classes: $($duplicates.Count)" -ForegroundColor $(if ($duplicates.Count -gt 5) { "Red" } elseif ($duplicates.Count -gt 0) { "Yellow" } else { "Green" })
Write-Host "  3. Legacy Files: $($legacyFiles.Count)" -ForegroundColor $(if ($legacyFiles.Count -gt 10) { "Yellow" } else { "Green" })
Write-Host "  4. Duplicate Directories: $($dupDirs.Count)" -ForegroundColor $(if ($dupDirs.Count -gt 5) { "Yellow" } else { "Green" })
Write-Host "  5. Missing __init__.py: $($missingInit.Count)" -ForegroundColor $(if ($missingInit.Count -gt 10) { "Yellow" } else { "Green" })
Write-Host ""

Write-Host "📁 All reports saved to: $reportDir" -ForegroundColor Cyan
Write-Host "📄 Read summary: $reportDir/0_SUMMARY.txt" -ForegroundColor Cyan
Write-Host ""

Write-Host "✅ Next step: Review the reports and start with broken imports (if any)" -ForegroundColor Green
Write-Host ""
