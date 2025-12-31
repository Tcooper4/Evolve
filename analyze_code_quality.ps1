# EVOLVE Code Quality Analysis
# Finds syntax errors, undefined variables, and other issues

Write-Host "`n" + "="*80 -ForegroundColor Cyan
Write-Host "CODE QUALITY ANALYSIS - Finding Broken Code" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$reportFile = "code_quality_analysis_$timestamp.txt"

# ============================================================================
# Check if Python is available
# ============================================================================

$pythonCmd = $null
foreach ($cmd in @('python', 'python3', 'py')) {
    try {
        $version = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $pythonCmd = $cmd
            Write-Host "✅ Found Python: $version" -ForegroundColor Green
            break
        }
    }
    catch {
        continue
    }
}

if (-not $pythonCmd) {
    Write-Host "❌ Python not found! Install Python to run this analysis." -ForegroundColor Red
    exit 1
}

# ============================================================================
# STEP 1: Syntax Check All Files
# ============================================================================

Write-Host "`n🔍 [1/3] Checking Python syntax..." -ForegroundColor Yellow

$syntaxErrors = @()
$fileCount = 0

Get-ChildItem -Path . -Include *.py -Recurse | Where-Object {
    $_.FullName -notmatch '\.venv|__pycache__|\.git'
} | ForEach-Object {
    $file = $_
    $relativePath = $file.FullName -replace [regex]::Escape($PWD), '.'
    $fileCount++
    
    # Try to compile the Python file
    $result = & $pythonCmd -m py_compile "$($file.FullName)" 2>&1
    
    if ($LASTEXITCODE -ne 0) {
        $syntaxErrors += @{
            'file' = $relativePath
            'error' = $result -join "`n"
        }
    }
    
    if ($fileCount % 50 -eq 0) {
        Write-Host "   Checked $fileCount files..." -ForegroundColor Gray
    }
}

Write-Host "   ✅ Checked $fileCount files"
Write-Host "   Found: $($syntaxErrors.Count) files with syntax errors"

# ============================================================================
# STEP 2: Find Common Code Issues
# ============================================================================

Write-Host "`n🔍 [2/3] Scanning for common issues..." -ForegroundColor Yellow

$issues = @{
    'undefined_names' = @()
    'unused_imports' = @()
    'print_statements' = @()
    'bare_except' = @()
    'mutable_defaults' = @()
}

Get-ChildItem -Path . -Include *.py -Recurse | Where-Object {
    $_.FullName -notmatch '\.venv|__pycache__|\.git'
} | ForEach-Object {
    $file = $_
    $relativePath = $file.FullName -replace [regex]::Escape($PWD), '.'
    $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
    
    if ($content) {
        # Find bare except clauses
        if ($content -match 'except\s*:') {
            $issues['bare_except'] += $relativePath
        }
        
        # Find print statements (should use logging)
        if ($content -match '\bprint\s*\(' -and $relativePath -notlike '*test*') {
            $lineNumbers = @()
            $lines = $content -split "`n"
            for ($i = 0; $i -lt $lines.Count; $i++) {
                if ($lines[$i] -match '\bprint\s*\(') {
                    $lineNumbers += ($i + 1)
                }
            }
            if ($lineNumbers.Count -gt 0) {
                $issues['print_statements'] += "$relativePath (lines: $($lineNumbers -join ', '))"
            }
        }
        
        # Find mutable default arguments
        if ($content -match 'def\s+\w+\([^)]*=\s*\[') {
            $issues['mutable_defaults'] += $relativePath
        }
    }
}

Write-Host "   Found issues:"
Write-Host "     - Bare except: $($issues['bare_except'].Count)"
Write-Host "     - Print statements: $($issues['print_statements'].Count)"
Write-Host "     - Mutable defaults: $($issues['mutable_defaults'].Count)"

# ============================================================================
# STEP 3: Try to run flake8 (if available)
# ============================================================================

Write-Host "`n🔍 [3/3] Running code linter (if available)..." -ForegroundColor Yellow

$flake8Results = $null
$pylintResults = $null

try {
    $flake8Version = & $pythonCmd -m flake8 --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ Found flake8, running analysis..." -ForegroundColor Green
        $flake8Results = & $pythonCmd -m flake8 . --exclude=.venv,__pycache__,.git --count --statistics 2>&1
    }
}
catch {
    Write-Host "   ⚠️  flake8 not installed (optional)" -ForegroundColor Yellow
}

# ============================================================================
# GENERATE REPORT
# ============================================================================

Write-Host "`n📄 Generating report..." -ForegroundColor Cyan

$report = @"
CODE QUALITY ANALYSIS REPORT
Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
="*80

SUMMARY:
="*80

Files Analyzed: $fileCount
Syntax Errors: $($syntaxErrors.Count)
Bare Except Clauses: $($issues['bare_except'].Count)
Print Statements: $($issues['print_statements'].Count)
Mutable Default Args: $($issues['mutable_defaults'].Count)

="*80
CRITICAL: SYNTAX ERRORS ($($syntaxErrors.Count)):
="*80

"@

if ($syntaxErrors.Count -eq 0) {
    $report += "`n✅ No syntax errors found!`n"
}
else {
    foreach ($error in $syntaxErrors) {
        $report += "`n❌ $($error['file'])`n"
        $report += "   Error: $($error['error'])`n"
    }
}

$report += "`n"
$report += "="*80
$report += "`nWARNINGS: CODE QUALITY ISSUES:"
$report += "`n="*80
$report += "`n"

if ($issues['bare_except'].Count -gt 0) {
    $report += "`n⚠️  BARE EXCEPT CLAUSES ($($issues['bare_except'].Count)):`n"
    $report += "These catch all exceptions, making debugging hard.`n"
    $report += "Replace with specific exceptions (e.g., except ValueError:)`n`n"
    foreach ($file in $issues['bare_except']) {
        $report += "   - $file`n"
    }
}

if ($issues['print_statements'].Count -gt 0) {
    $report += "`n⚠️  PRINT STATEMENTS ($($issues['print_statements'].Count)):`n"
    $report += "Use logging instead of print() for production code.`n`n"
    foreach ($file in ($issues['print_statements'] | Select-Object -First 20)) {
        $report += "   - $file`n"
    }
    if ($issues['print_statements'].Count -gt 20) {
        $report += "   ... and $($issues['print_statements'].Count - 20) more`n"
    }
}

if ($issues['mutable_defaults'].Count -gt 0) {
    $report += "`n⚠️  MUTABLE DEFAULT ARGUMENTS ($($issues['mutable_defaults'].Count)):`n"
    $report += "Using lists/dicts as default args can cause bugs.`n"
    $report += "Use None as default, then initialize in function.`n`n"
    foreach ($file in $issues['mutable_defaults']) {
        $report += "   - $file`n"
    }
}

if ($flake8Results) {
    $report += "`n"
    $report += "="*80
    $report += "`nFLAKE8 LINTER RESULTS:"
    $report += "`n="*80
    $report += "`n"
    $report += $flake8Results -join "`n"
}

$report += "`n"
$report += "="*80
$report += "`nRECOMMENDATIONS:"
$report += "`n="*80
$report += "`n"
$report += "1. Fix all syntax errors immediately`n"
$report += "2. Replace bare 'except:' with specific exceptions`n"
$report += "3. Replace print() with logging.info/debug/error`n"
$report += "4. Fix mutable default arguments`n"
$report += "5. Run: pip install flake8 black pylint (for better analysis)`n"
$report += "6. Run: black . (to auto-format code)`n"
$report += "`n"

$report | Out-File $reportFile

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

Write-Host "`n" + "="*80 -ForegroundColor Green
Write-Host "CODE QUALITY ANALYSIS COMPLETE" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Green
Write-Host ""

Write-Host "FINDINGS:" -ForegroundColor Yellow
Write-Host "  Syntax Errors: $($syntaxErrors.Count)" -ForegroundColor $(if ($syntaxErrors.Count -gt 0) { "Red" } else { "Green" })
Write-Host "  Bare Except: $($issues['bare_except'].Count)" -ForegroundColor $(if ($issues['bare_except'].Count -gt 10) { "Yellow" } else { "Green" })
Write-Host "  Print Statements: $($issues['print_statements'].Count)" -ForegroundColor $(if ($issues['print_statements'].Count -gt 20) { "Yellow" } else { "Green" })
Write-Host "  Mutable Defaults: $($issues['mutable_defaults'].Count)" -ForegroundColor $(if ($issues['mutable_defaults'].Count -gt 5) { "Yellow" } else { "Green" })
Write-Host ""

if ($syntaxErrors.Count -gt 0) {
    Write-Host "❌ CRITICAL: Found files with syntax errors!" -ForegroundColor Red
    Write-Host "   These will cause runtime crashes. Fix immediately!" -ForegroundColor Red
    Write-Host ""
}

Write-Host "📄 Full report: $reportFile" -ForegroundColor Cyan
Write-Host ""
Write-Host "💡 TIP: Install better linters:" -ForegroundColor Cyan
Write-Host "   pip install flake8 black pylint mypy --break-system-packages" -ForegroundColor Gray
Write-Host ""
Write-Host "✅ Next: Fix syntax errors and critical issues" -ForegroundColor Green
Write-Host ""
