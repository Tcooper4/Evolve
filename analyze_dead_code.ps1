# EVOLVE Dead Code Analysis
# Finds files that aren't imported anywhere (potential dead code)

Write-Host "`n" + "="*80 -ForegroundColor Cyan
Write-Host "DEAD CODE ANALYSIS - Finding Unused Files" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$reportFile = "dead_code_analysis_$timestamp.txt"

# ============================================================================
# STEP 1: Build Import Graph
# ============================================================================

Write-Host "📊 [1/4] Building import graph..." -ForegroundColor Yellow

$allPythonFiles = @{}
$importGraph = @{}  # Maps each file to what it imports
$importedBy = @{}   # Maps each file to what imports it

# Get all Python files
Get-ChildItem -Path . -Include *.py -Recurse | Where-Object {
    $_.FullName -notmatch '\.venv|__pycache__|\.git|node_modules'
} | ForEach-Object {
    $relativePath = $_.FullName -replace [regex]::Escape($PWD), '.'
    $relativePath = $relativePath -replace '^\.\\', '' -replace '^\./', ''
    $allPythonFiles[$relativePath] = $true
    $importGraph[$relativePath] = @()
    $importedBy[$relativePath] = @()
}

Write-Host "   Found: $($allPythonFiles.Count) Python files"

# ============================================================================
# STEP 2: Analyze Imports
# ============================================================================

Write-Host "🔍 [2/4] Analyzing imports..." -ForegroundColor Yellow

$importPatterns = @(
    'from\s+(\S+)\s+import',     # from X import Y
    'import\s+(\S+)',              # import X
    'from\s+\.\s*import',          # from . import
    'from\s+\.\.+\s*(\S+)'         # from .. import
)

foreach ($file in $allPythonFiles.Keys) {
    $fullPath = Join-Path $PWD $file
    $content = Get-Content $fullPath -Raw -ErrorAction SilentlyContinue
    
    if ($content) {
        # Find all imports in this file
        foreach ($pattern in $importPatterns) {
            $matches = [regex]::Matches($content, $pattern)
            
            foreach ($match in $matches) {
                if ($match.Groups.Count -gt 1) {
                    $importedModule = $match.Groups[1].Value
                    
                    # Convert module path to file path
                    $potentialFile = $importedModule -replace '\.', '/'
                    
                    # Try different extensions
                    $candidates = @(
                        "$potentialFile.py",
                        "$potentialFile/__init__.py"
                    )
                    
                    foreach ($candidate in $candidates) {
                        if ($allPythonFiles.ContainsKey($candidate)) {
                            # Found the imported file!
                            $importGraph[$file] += $candidate
                            $importedBy[$candidate] += $file
                            break
                        }
                    }
                }
            }
        }
    }
}

# ============================================================================
# STEP 3: Find Never-Imported Files
# ============================================================================

Write-Host "🎯 [3/4] Finding never-imported files..." -ForegroundColor Yellow

$neverImported = @()
$entryPoints = @(
    'main.py',
    'app.py',
    'start_orchestrator.py',
    'setup.py'
)

foreach ($file in $allPythonFiles.Keys) {
    $isEntryPoint = $false
    foreach ($entry in $entryPoints) {
        if ($file -like "*$entry") {
            $isEntryPoint = $true
            break
        }
    }
    
    # Skip entry points and test files
    if (-not $isEntryPoint -and $file -notlike '*test*.py' -and $file -notlike '*__init__.py') {
        if ($importedBy[$file].Count -eq 0) {
            $neverImported += $file
        }
    }
}

Write-Host "   Found: $($neverImported.Count) files never imported"

# ============================================================================
# STEP 4: Find Isolated Clusters
# ============================================================================

Write-Host "🔍 [4/4] Finding isolated code clusters..." -ForegroundColor Yellow

# Find files that import each other but are never imported by main code
$isolatedClusters = @{}
$clusterNumber = 0

foreach ($file in $neverImported) {
    # Check if this file imports anything
    $imports = $importGraph[$file]
    
    if ($imports.Count -gt 0) {
        # This file imports something - might be part of isolated cluster
        $cluster = @($file)
        
        # Follow the import chain
        $toExplore = [System.Collections.ArrayList]@($imports)
        $explored = @{}
        
        while ($toExplore.Count -gt 0) {
            $current = $toExplore[0]
            $toExplore.RemoveAt(0)
            
            if (-not $explored.ContainsKey($current) -and $neverImported -contains $current) {
                $explored[$current] = $true
                $cluster += $current
                
                # Add what this imports
                foreach ($imp in $importGraph[$current]) {
                    if (-not $explored.ContainsKey($imp)) {
                        [void]$toExplore.Add($imp)
                    }
                }
            }
        }
        
        if ($cluster.Count -gt 1) {
            $isolatedClusters["Cluster_$clusterNumber"] = $cluster
            $clusterNumber++
        }
    }
}

Write-Host "   Found: $($isolatedClusters.Count) isolated clusters"

# ============================================================================
# GENERATE REPORT
# ============================================================================

Write-Host "`n📄 Generating report..." -ForegroundColor Cyan

$report = @"
DEAD CODE ANALYSIS REPORT
Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
="*80

SUMMARY:
="*80

Total Python Files: $($allPythonFiles.Count)
Never Imported: $($neverImported.Count)
Isolated Clusters: $($isolatedClusters.Count)

="*80
NEVER IMPORTED FILES ($($neverImported.Count)):
="*80

These files exist but are never imported by any other file.
They might be:
  - Dead code (can be deleted)
  - Entry points (main.py, etc.) - these are expected
  - Standalone scripts
  - Orphaned legacy code

"@

# Group by directory
$byDirectory = @{}
foreach ($file in $neverImported) {
    $dir = Split-Path $file -Parent
    if (-not $dir) { $dir = "." }
    
    if (-not $byDirectory.ContainsKey($dir)) {
        $byDirectory[$dir] = @()
    }
    $byDirectory[$dir] += $file
}

foreach ($dir in ($byDirectory.Keys | Sort-Object)) {
    $report += "`n📁 $dir/`n"
    foreach ($file in ($byDirectory[$dir] | Sort-Object)) {
        $fileName = Split-Path $file -Leaf
        $report += "   ❌ $fileName`n"
    }
}

if ($isolatedClusters.Count -gt 0) {
    $report += "`n"
    $report += "="*80
    $report += "`nISOLATED CODE CLUSTERS ($($isolatedClusters.Count)):"
    $report += "`n="*80
    $report += "`n"
    $report += "`nThese are groups of files that import each other but are never"
    $report += "`nimported by the main application. Likely dead code."
    $report += "`n"
    
    foreach ($clusterName in ($isolatedClusters.Keys | Sort-Object)) {
        $cluster = $isolatedClusters[$clusterName]
        $report += "`n$clusterName ($($cluster.Count) files):`n"
        foreach ($file in $cluster) {
            $report += "   - $file`n"
        }
    }
}

$report += "`n"
$report += "="*80
$report += "`nHOW TO USE THIS REPORT:"
$report += "`n="*80
$report += "`n"
$report += "1. Review each 'Never Imported' file`n"
$report += "2. Search codebase for any dynamic imports (importlib, __import__)`n"
$report += "3. Check if file is a standalone script`n"
$report += "4. If truly unused, mark for deletion`n"
$report += "5. Review isolated clusters - likely can delete entire cluster`n"
$report += "`n"
$report += "CAUTION:`n"
$report += "  - Don't delete files loaded via config (yaml/json)`n"
$report += "  - Don't delete files loaded dynamically at runtime`n"
$report += "  - Don't delete entry points (main.py, app.py, etc.)`n"
$report += "  - Don't delete test files (covered separately)`n"
$report += "`n"

$report | Out-File $reportFile

# ============================================================================
# ADDITIONAL ANALYSIS: Find Files With No Functions/Classes
# ============================================================================

Write-Host "📊 Finding empty/minimal files..." -ForegroundColor Yellow

$emptyFiles = @()
$minimalFiles = @()

foreach ($file in $allPythonFiles.Keys) {
    $fullPath = Join-Path $PWD $file
    $content = Get-Content $fullPath -Raw -ErrorAction SilentlyContinue
    
    if ($content) {
        # Remove comments and whitespace
        $codeOnly = $content -replace '#.*', '' -replace '^\s+', '' -replace '\s+$', ''
        
        $hasFunction = $codeOnly -match 'def\s+\w+'
        $hasClass = $codeOnly -match 'class\s+\w+'
        $lineCount = ($content -split "`n").Count
        
        if (-not $hasFunction -and -not $hasClass) {
            if ($lineCount -lt 10) {
                $emptyFiles += "$file ($lineCount lines)"
            }
            elseif ($lineCount -lt 50) {
                $minimalFiles += "$file ($lineCount lines)"
            }
        }
    }
}

$emptyReport = "`n"
$emptyReport += "="*80
$emptyReport += "`nEMPTY/MINIMAL FILES:"
$emptyReport += "`n="*80
$emptyReport += "`n"
$emptyReport += "`nFiles with no functions or classes (potential dead code):`n"
$emptyReport += "`n"
$emptyReport += "Very Small (<10 lines): $($emptyFiles.Count)`n"
foreach ($file in $emptyFiles) {
    $emptyReport += "   🗑️  $file`n"
}
$emptyReport += "`n"
$emptyReport += "Minimal (10-50 lines): $($minimalFiles.Count)`n"
foreach ($file in $minimalFiles) {
    $emptyReport += "   ⚠️  $file`n"
}

$emptyReport | Out-File $reportFile -Append

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

Write-Host "`n" + "="*80 -ForegroundColor Green
Write-Host "DEAD CODE ANALYSIS COMPLETE" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Green
Write-Host ""

Write-Host "FINDINGS:" -ForegroundColor Yellow
Write-Host "  Never Imported: $($neverImported.Count) files" -ForegroundColor $(if ($neverImported.Count -gt 20) { "Red" } elseif ($neverImported.Count -gt 5) { "Yellow" } else { "Green" })
Write-Host "  Isolated Clusters: $($isolatedClusters.Count) groups" -ForegroundColor $(if ($isolatedClusters.Count -gt 0) { "Yellow" } else { "Green" })
Write-Host "  Empty Files: $($emptyFiles.Count) files" -ForegroundColor $(if ($emptyFiles.Count -gt 10) { "Yellow" } else { "Green" })
Write-Host "  Minimal Files: $($minimalFiles.Count) files" -ForegroundColor $(if ($minimalFiles.Count -gt 20) { "Yellow" } else { "Green" })
Write-Host ""

if ($neverImported.Count -gt 0) {
    Write-Host "⚠️  WARNING: Found files that are never imported!" -ForegroundColor Yellow
    Write-Host "   These could be dead code or standalone scripts." -ForegroundColor Yellow
    Write-Host ""
}

if ($isolatedClusters.Count -gt 0) {
    Write-Host "⚠️  WARNING: Found isolated code clusters!" -ForegroundColor Yellow
    Write-Host "   These groups of files import each other but nothing imports them." -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "📄 Full report: $reportFile" -ForegroundColor Cyan
Write-Host ""
Write-Host "✅ Next: Review the report and identify dead code to remove" -ForegroundColor Green
Write-Host ""

# ============================================================================
# GENERATE DELETION SCRIPT (TEMPLATE)
# ============================================================================

$deleteScript = "# GENERATED: Files to potentially delete`n"
$deleteScript += "# REVIEW CAREFULLY before running!`n`n"
$deleteScript += "# Uncomment to delete:`n`n"

foreach ($file in $neverImported) {
    $deleteScript += "# Remove-Item '$file' -Force`n"
}

$deleteScript | Out-File "DELETE_CANDIDATES.ps1"

Write-Host "📝 Deletion template created: DELETE_CANDIDATES.ps1" -ForegroundColor Cyan
Write-Host "   (All commands commented out - review before using)" -ForegroundColor Gray
Write-Host ""
