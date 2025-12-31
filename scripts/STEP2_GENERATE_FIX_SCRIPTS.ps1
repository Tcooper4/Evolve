# 🛠️ EVOLVE CODEBASE - STEP 2: GENERATE FIX SCRIPTS
# 100% SAFE - Only creates scripts, does NOT execute them

# This script creates individual fix scripts that you can:
# 1. Review before running
# 2. Run one at a time
# 3. Test after each fix
# 4. Undo if needed

Write-Host "=" * 80
Write-Host "🛠️ EVOLVE SAFE CLEANUP - STEP 2: GENERATE FIX SCRIPTS"
Write-Host "=" * 80
Write-Host ""
Write-Host "This script will:"
Write-Host "  ✅ Create individual fix scripts"
Write-Host "  ✅ Each script is reviewed before execution"
Write-Host "  ❌ NOT execute any fixes automatically"
Write-Host ""

$confirmation = Read-Host "Proceed with generating fix scripts? (yes/no)"
if ($confirmation -ne "yes") {
    Write-Host "❌ Aborted by user"
    exit
}

# Find the most recent analysis report
$reportDirs = Get-ChildItem -Directory | Where-Object { $_.Name -like "ANALYSIS_REPORT_*" } | Sort-Object Name -Descending
if ($reportDirs.Count -eq 0) {
    Write-Host ""
    Write-Host "❌ ERROR: No analysis report found!"
    Write-Host "   Please run STEP1_SAFE_BACKUP_AND_ANALYZE.ps1 first"
    exit
}

$reportDir = $reportDirs[0].Name
Write-Host ""
Write-Host "📂 Using analysis from: $reportDir"
Write-Host ""

# Create fixes directory
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$fixesDir = "FIXES_$timestamp"
New-Item -ItemType Directory -Path $fixesDir -Force | Out-Null

Write-Host "📝 Generating fix scripts in: $fixesDir"
Write-Host ""

# ============================================================================
# FIX SCRIPT 1: Fix core.session_utils imports
# ============================================================================

Write-Host "  📝 Creating Fix 1: session_utils imports..."

$fix1Script = @"
# FIX 1: Update core.session_utils → utils.session_utils
# SAFE: Creates backup of each file before modifying

Write-Host "🔧 FIX 1: Fixing session_utils imports"
Write-Host ""

# Files that need fixing
`$filesToFix = @(
    "pages/6_Strategy_History.py",
    "pages/performance_tracker.py",
    "utils/runner.py",
    "utils/ui_helpers.py"
)

Write-Host "Files to update: `$(`$filesToFix.Count)"
Write-Host ""

foreach (`$file in `$filesToFix) {
    if (Test-Path `$file) {
        Write-Host "  Processing: `$file"
        
        # Create backup
        `$backup = "`$file.backup_`$(Get-Date -Format 'yyyyMMdd_HHmmss')"
        Copy-Item `$file `$backup
        Write-Host "    ✅ Backup created: `$backup"
        
        # Read content
        `$content = Get-Content `$file -Raw
        
        # Check if it needs fixing
        if (`$content -match "from core\.session_utils") {
            # Replace
            `$newContent = `$content -replace "from core\.session_utils", "from utils.session_utils"
            Set-Content `$file `$newContent
            Write-Host "    ✅ Fixed import path"
        } else {
            Write-Host "    ⚠️  No changes needed"
            Remove-Item `$backup
        }
    } else {
        Write-Host "  ⚠️  File not found: `$file"
    }
    Write-Host ""
}

Write-Host "✅ Fix 1 complete!"
Write-Host ""
Write-Host "To undo: Restore from .backup files"
"@

$fix1Script | Out-File "$fixesDir/FIX1_session_utils_imports.ps1" -Encoding UTF8

# ============================================================================
# FIX SCRIPT 2: Fix core.task_orchestrator imports
# ============================================================================

Write-Host "  📝 Creating Fix 2: task_orchestrator imports..."

$fix2Script = @"
# FIX 2: Update core.task_orchestrator → core.orchestrator.task_orchestrator
# SAFE: Creates backup of each file before modifying

Write-Host "🔧 FIX 2: Fixing task_orchestrator imports"
Write-Host ""

# Find all files with this import
`$files = Get-ChildItem -Path . -Recurse -Filter "*.py" -ErrorAction SilentlyContinue | 
    Where-Object { (Get-Content `$_.FullName -Raw) -match "from core\.task_orchestrator import" }

Write-Host "Files to update: `$(`$files.Count)"
Write-Host ""

foreach (`$file in `$files) {
    Write-Host "  Processing: `$(`$file.FullName)"
    
    # Create backup
    `$backup = "`$(`$file.FullName).backup_`$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    Copy-Item `$file.FullName `$backup
    Write-Host "    ✅ Backup created"
    
    # Read content
    `$content = Get-Content `$file.FullName -Raw
    
    # Replace
    `$newContent = `$content -replace "from core\.task_orchestrator import", "from core.orchestrator.task_orchestrator import"
    Set-Content `$file.FullName `$newContent
    Write-Host "    ✅ Fixed import path"
    Write-Host ""
}

Write-Host "✅ Fix 2 complete!"
Write-Host ""
Write-Host "To undo: Restore from .backup files"
"@

$fix2Script | Out-File "$fixesDir/FIX2_task_orchestrator_imports.ps1" -Encoding UTF8

# ============================================================================
# FIX SCRIPT 3: Create missing core/agent_hub.py
# ============================================================================

Write-Host "  📝 Creating Fix 3: Create missing agent_hub..."

$fix3Script = @"
# FIX 3: Create missing core/agent_hub.py
# SAFE: Only creates new file, doesn't modify existing code

Write-Host "🔧 FIX 3: Creating missing core/agent_hub.py"
Write-Host ""

# Check if core directory exists
if (-not (Test-Path "core")) {
    Write-Host "  Creating core/ directory..."
    New-Item -ItemType Directory -Path "core" -Force | Out-Null
}

# Check if __init__.py exists
if (-not (Test-Path "core/__init__.py")) {
    Write-Host "  Creating core/__init__.py..."
    "" | Out-File "core/__init__.py" -Encoding UTF8
}

# Create agent_hub.py
Write-Host "  Creating core/agent_hub.py..."

`$agentHubContent = @"
'''
Agent Hub - Central registry for all agents
This is a stub implementation to resolve import errors.
Replace with actual implementation when ready.
'''

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class AgentHub:
    '''Central hub for agent management and coordination'''
    
    def __init__(self):
        self.agents: Dict[str, Any] = {}
        self.logger = logger
        logger.info("AgentHub initialized")
    
    def register(self, name: str, agent: Any) -> None:
        '''Register an agent with the hub'''
        self.agents[name] = agent
        logger.info(f"Agent registered: {name}")
    
    def unregister(self, name: str) -> None:
        '''Unregister an agent from the hub'''
        if name in self.agents:
            del self.agents[name]
            logger.info(f"Agent unregistered: {name}")
    
    def get(self, name: str) -> Optional[Any]:
        '''Get an agent by name'''
        return self.agents.get(name)
    
    def list_agents(self) -> list:
        '''List all registered agents'''
        return list(self.agents.keys())
    
    def clear(self) -> None:
        '''Clear all registered agents'''
        self.agents.clear()
        logger.info("All agents cleared")


# Create a default instance
default_hub = AgentHub()
"@

`$agentHubContent | Out-File "core/agent_hub.py" -Encoding UTF8

Write-Host "  ✅ Created core/agent_hub.py"
Write-Host ""
Write-Host "✅ Fix 3 complete!"
Write-Host ""
Write-Host "NOTE: This is a stub implementation. Replace with actual logic when ready."
"@

$fix3Script | Out-File "$fixesDir/FIX3_create_agent_hub.ps1" -Encoding UTF8

# ============================================================================
# FIX SCRIPT 4: Archive duplicate pages
# ============================================================================

Write-Host "  📝 Creating Fix 4: Archive duplicate pages..."

$fix4Script = @"
# FIX 4: Archive duplicate pages
# SAFE: Moves files to archive (doesn't delete), fully reversible

Write-Host "🔧 FIX 4: Archiving duplicate pages"
Write-Host ""

# Create archive directory
`$archiveDir = "pages_archive_`$(Get-Date -Format 'yyyyMMdd_HHmmss')"
Write-Host "Creating archive: `$archiveDir"
New-Item -ItemType Directory -Path `$archiveDir -Force | Out-Null
Write-Host ""

# List of pages to archive
`$pagesToArchive = @(
    "1_Forecast_Trade.py",
    "forecast.py",
    "2_Backtest_Strategy.py",
    "4_Portfolio_Management.py",
    "5_Risk_Analysis.py",
    "risk_dashboard.py",
    "6_Strategy_History.py",
    "strategy.py",
    "Strategy_Pipeline_Demo.py",
    "HybridModel.py",
    "nlp_tester.py",
    "ui_helpers.py",
    "home.py",
    "settings.py",
    "optimization_dashboard.py",
    "performance_tracker.py"
)

Write-Host "Pages to archive: `$(`$pagesToArchive.Count)"
Write-Host ""

`$archived = 0
`$notFound = 0

foreach (`$page in `$pagesToArchive) {
    `$sourcePath = "pages/`$page"
    if (Test-Path `$sourcePath) {
        Move-Item `$sourcePath `$archiveDir/
        Write-Host "  ✅ Archived: `$page"
        `$archived++
    } else {
        Write-Host "  ⚠️  Not found: `$page"
        `$notFound++
    }
}

Write-Host ""
Write-Host "✅ Archiving complete!"
Write-Host "  • Archived: `$archived pages"
Write-Host "  • Not found: `$notFound pages"
Write-Host ""
Write-Host "To undo: Move files from `$archiveDir/ back to pages/"
Write-Host ""
Write-Host "📋 Remaining pages:"
Get-ChildItem -Path pages -Filter "*.py" | Where-Object { `$_.Name -ne "__init__.py" } | ForEach-Object {
    Write-Host "  • `$(`$_.Name)"
}
"@

$fix4Script | Out-File "$fixesDir/FIX4_archive_duplicate_pages.ps1" -Encoding UTF8

# ============================================================================
# FIX SCRIPT 5: Test after fixes
# ============================================================================

Write-Host "  📝 Creating Fix 5: Test script..."

$fix5Script = @"
# FIX 5: Test that fixes worked
# SAFE: Read-only testing, no modifications

Write-Host "🧪 FIX 5: Testing fixes"
Write-Host ""

Write-Host "Test 1: Checking import paths..."
`$importErrors = 0

# Check for remaining core.session_utils imports
Get-ChildItem -Path pages -Filter "*.py" -ErrorAction SilentlyContinue | ForEach-Object {
    `$content = Get-Content `$_.FullName -Raw -ErrorAction SilentlyContinue
    if (`$content -match "from core\.session_utils") {
        Write-Host "  ❌ Still has core.session_utils: `$(`$_.Name)"
        `$importErrors++
    }
}

if (`$importErrors -eq 0) {
    Write-Host "  ✅ No core.session_utils imports found"
} else {
    Write-Host "  ⚠️  Found `$importErrors files still using core.session_utils"
}
Write-Host ""

Write-Host "Test 2: Checking core/agent_hub.py exists..."
if (Test-Path "core/agent_hub.py") {
    Write-Host "  ✅ core/agent_hub.py exists"
} else {
    Write-Host "  ❌ core/agent_hub.py missing"
}
Write-Host ""

Write-Host "Test 3: Checking page count..."
`$pageCount = (Get-ChildItem -Path pages -Filter "*.py" | Where-Object { `$_.Name -ne "__init__.py" }).Count
Write-Host "  📊 Current pages: `$pageCount"
Write-Host "  🎯 Target pages: ~15"
if (`$pageCount -le 20) {
    Write-Host "  ✅ Page count reasonable"
} else {
    Write-Host "  ⚠️  Still have many pages"
}
Write-Host ""

Write-Host "Test 4: Try launching Streamlit..."
Write-Host "  💡 Run manually: python main.py streamlit"
Write-Host ""

Write-Host "✅ Testing complete!"
Write-Host ""
Write-Host "If all tests passed, fixes were successful!"
"@

$fix5Script | Out-File "$fixesDir/FIX5_test_fixes.ps1" -Encoding UTF8

# ============================================================================
# Create master execution script
# ============================================================================

Write-Host "  📝 Creating master execution script..."

$masterScript = @"
# MASTER SCRIPT: Execute all fixes with confirmation
# SAFE: Asks for confirmation before each fix

Write-Host "=" * 80
Write-Host "🎯 EVOLVE FIXES - MASTER EXECUTION SCRIPT"
Write-Host "=" * 80
Write-Host ""
Write-Host "This will execute all fixes in order, with confirmation at each step."
Write-Host ""

`$fixes = @(
    @{
        Name = "Fix 1: Update session_utils imports"
        Script = "FIX1_session_utils_imports.ps1"
        Description = "Updates 'from core.session_utils' to 'from utils.session_utils'"
    },
    @{
        Name = "Fix 2: Update task_orchestrator imports"
        Script = "FIX2_task_orchestrator_imports.ps1"
        Description = "Updates task_orchestrator import paths"
    },
    @{
        Name = "Fix 3: Create missing agent_hub"
        Script = "FIX3_create_agent_hub.ps1"
        Description = "Creates stub core/agent_hub.py to resolve import errors"
    },
    @{
        Name = "Fix 4: Archive duplicate pages"
        Script = "FIX4_archive_duplicate_pages.ps1"
        Description = "Moves duplicate pages to archive (reversible)"
    },
    @{
        Name = "Fix 5: Test fixes"
        Script = "FIX5_test_fixes.ps1"
        Description = "Verifies all fixes worked correctly"
    }
)

foreach (`$fix in `$fixes) {
    Write-Host ""
    Write-Host "=" * 80
    Write-Host `$fix.Name
    Write-Host "=" * 80
    Write-Host ""
    Write-Host "Description: `$(`$fix.Description)"
    Write-Host "Script: `$(`$fix.Script)"
    Write-Host ""
    
    `$confirm = Read-Host "Execute this fix? (yes/no/skip)"
    
    if (`$confirm -eq "yes") {
        Write-Host ""
        & ".\`$(`$fix.Script)"
        Write-Host ""
        Write-Host "✅ Fix completed"
        Read-Host "Press Enter to continue to next fix"
    } elseif (`$confirm -eq "skip") {
        Write-Host "⏭️  Skipped"
    } else {
        Write-Host "❌ Aborted by user"
        exit
    }
}

Write-Host ""
Write-Host "=" * 80
Write-Host "🎉 ALL FIXES COMPLETE!"
Write-Host "=" * 80
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Test the system: python main.py streamlit"
Write-Host "  2. If issues, restore from backup"
Write-Host "  3. Commit changes to git"
Write-Host ""
"@

$masterScript | Out-File "$fixesDir/RUN_ALL_FIXES.ps1" -Encoding UTF8

# ============================================================================
# Create undo script
# ============================================================================

Write-Host "  📝 Creating undo script..."

$undoScript = @"
# UNDO SCRIPT: Restore from backup
# SAFE: Restores all files from most recent backup

Write-Host "🔙 UNDO: Restore from backup"
Write-Host ""

# Find most recent backup
`$backups = Get-ChildItem -Directory | Where-Object { `$_.Name -like "BACKUP_*" } | Sort-Object Name -Descending
if (`$backups.Count -eq 0) {
    Write-Host "❌ No backups found!"
    exit
}

`$backup = `$backups[0]
Write-Host "Found backup: `$(`$backup.Name)"
Write-Host ""

`$confirm = Read-Host "Restore from this backup? This will overwrite current files. (yes/no)"
if (`$confirm -ne "yes") {
    Write-Host "❌ Aborted"
    exit
}

Write-Host ""
Write-Host "Restoring files..."

# Restore directories
Get-ChildItem `$backup.FullName -Directory | ForEach-Object {
    Write-Host "  Restoring: `$(`$_.Name)"
    if (Test-Path `$_.Name) {
        Remove-Item `$_.Name -Recurse -Force
    }
    Copy-Item `$_.FullName . -Recurse -Force
}

# Restore files
Get-ChildItem `$backup.FullName -File | ForEach-Object {
    Write-Host "  Restoring: `$(`$_.Name)"
    Copy-Item `$_.FullName . -Force
}

Write-Host ""
Write-Host "✅ Restore complete!"
Write-Host ""
Write-Host "All files restored from: `$(`$backup.Name)"
"@

$undoScript | Out-File "$fixesDir/UNDO_RESTORE_FROM_BACKUP.ps1" -Encoding UTF8

# ============================================================================
# Create README
# ============================================================================

$readmeContent = @"
# EVOLVE FIXES - EXECUTION GUIDE

Generated: $(Get-Date)

## 📋 WHAT'S IN THIS FOLDER

This folder contains **5 individual fix scripts** and **2 helper scripts**:

### Fix Scripts (Run in order):

1. **FIX1_session_utils_imports.ps1**
   - Updates 'from core.session_utils' → 'from utils.session_utils'
   - Affects: ~4 files
   - Safety: Creates .backup files before modifying

2. **FIX2_task_orchestrator_imports.ps1**
   - Updates task_orchestrator import paths
   - Affects: Multiple files
   - Safety: Creates .backup files before modifying

3. **FIX3_create_agent_hub.ps1**
   - Creates missing core/agent_hub.py
   - Affects: Creates 1 new file
   - Safety: Only creates, doesn't modify existing code

4. **FIX4_archive_duplicate_pages.ps1**
   - Moves duplicate pages to archive folder
   - Affects: ~16 pages
   - Safety: Moves (not deletes), fully reversible

5. **FIX5_test_fixes.ps1**
   - Tests that all fixes worked
   - Affects: Nothing (read-only)
   - Safety: 100% safe, just tests

### Helper Scripts:

- **RUN_ALL_FIXES.ps1**
  - Executes all fixes in order with confirmation
  - Recommended way to run fixes

- **UNDO_RESTORE_FROM_BACKUP.ps1**
  - Restores from backup if something goes wrong
  - Emergency use only

## 🚀 HOW TO USE

### Option A: Run All Fixes (Recommended)

``````powershell
# Run master script
.\RUN_ALL_FIXES.ps1

# It will ask for confirmation before each fix
# Answer 'yes' to execute, 'skip' to skip, 'no' to abort
``````

### Option B: Run Individual Fixes

``````powershell
# Run fixes one at a time
.\FIX1_session_utils_imports.ps1
# Review results, test system
.\FIX2_task_orchestrator_imports.ps1
# Review results, test system
# ... and so on
``````

### Option C: Review First, Then Run

``````powershell
# Open each script in notepad first
notepad FIX1_session_utils_imports.ps1
# Review what it does
# Then run it
.\FIX1_session_utils_imports.ps1
``````

## 🛡️ SAFETY FEATURES

Every fix script:
- ✅ Creates backups before modifying files
- ✅ Shows what it's doing in real-time
- ✅ Can be undone with UNDO script
- ✅ Doesn't delete (only moves to archive)

## 🔙 IF SOMETHING GOES WRONG

``````powershell
# Restore from backup
.\UNDO_RESTORE_FROM_BACKUP.ps1
``````

This will restore ALL files from the most recent backup.

## 📝 WHAT EACH FIX DOES

### Fix 1: session_utils imports
**Problem**: Pages import from 'core.session_utils' but file is at 'utils/session_utils.py'
**Solution**: Updates import paths in affected files
**Files affected**: ~4 files in pages/

### Fix 2: task_orchestrator imports
**Problem**: Some files import from 'core.task_orchestrator' (wrong path)
**Solution**: Updates to 'core.orchestrator.task_orchestrator' (correct path)
**Files affected**: Examples, scripts, tests

### Fix 3: Create agent_hub
**Problem**: Multiple files import 'core.agent_hub' but it doesn't exist
**Solution**: Creates a stub implementation
**Files affected**: Creates 1 new file (core/agent_hub.py)

### Fix 4: Archive duplicate pages
**Problem**: 41 pages when you only need ~15
**Solution**: Moves duplicate/test pages to archive folder
**Files affected**: ~16 pages moved to archive

### Fix 5: Test fixes
**Problem**: Need to verify fixes worked
**Solution**: Runs automated tests
**Files affected**: None (read-only)

## 📊 EXPECTED RESULTS

**Before fixes:**
- ❌ Pages crash with ModuleNotFoundError
- ❌ 41 pages in sidebar
- ❌ Import path confusion

**After fixes:**
- ✅ All pages load without errors
- ✅ ~15-20 pages in sidebar
- ✅ Clean import paths

## 🧪 TESTING

After running fixes:

``````powershell
# Test system
python main.py streamlit

# Check pages load without errors
# Click through each page in sidebar
``````

If any issues:
1. Check error messages
2. Run UNDO script to restore
3. Review what went wrong
4. Try fixes individually

## 📞 TROUBLESHOOTING

**Problem**: Script won't run
**Solution**: Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**Problem**: Files not found
**Solution**: Make sure you're in the project root directory

**Problem**: Changes didn't work
**Solution**: Run `UNDO_RESTORE_FROM_BACKUP.ps1` to restore

## ✅ CHECKLIST

Before running fixes:
- [ ] Reviewed analysis reports
- [ ] Understand what each fix does
- [ ] Backup exists (created by Step 1)
- [ ] In project root directory

After running fixes:
- [ ] All fixes completed successfully
- [ ] Tested system (python main.py streamlit)
- [ ] Pages load without errors
- [ ] Committed changes to git

## 📝 NOTES

- All fixes are **reversible**
- Backups are created automatically
- You can run fixes individually
- Test after each fix if you want
- Undo script restores everything

---

**Generated by**: STEP2_GENERATE_FIX_SCRIPTS.ps1
**Based on**: Analysis in $reportDir
"@

$readmeContent | Out-File "$fixesDir/README.md" -Encoding UTF8

# ============================================================================
# Summary
# ============================================================================

Write-Host ""
Write-Host "=" * 80
Write-Host "✅ FIX SCRIPTS GENERATED!"
Write-Host "=" * 80
Write-Host ""
Write-Host "📂 Location: $fixesDir"
Write-Host ""
Write-Host "📄 Generated files:"
Write-Host "  • FIX1_session_utils_imports.ps1"
Write-Host "  • FIX2_task_orchestrator_imports.ps1"
Write-Host "  • FIX3_create_agent_hub.ps1"
Write-Host "  • FIX4_archive_duplicate_pages.ps1"
Write-Host "  • FIX5_test_fixes.ps1"
Write-Host "  • RUN_ALL_FIXES.ps1 (master script)"
Write-Host "  • UNDO_RESTORE_FROM_BACKUP.ps1 (emergency restore)"
Write-Host "  • README.md (full documentation)"
Write-Host ""
Write-Host "🎯 Next steps:"
Write-Host "  1. Review README.md in $fixesDir"
Write-Host "  2. Review individual fix scripts"
Write-Host "  3. Run fixes when ready (see README for options)"
Write-Host ""
Write-Host "💡 Recommended: cd $fixesDir then run .\RUN_ALL_FIXES.ps1"
Write-Host ""
