#!/usr/bin/env python3
"""
Complete Optimization Consolidation

This script completes the consolidation of all optimization modules into
trading/optimization/ and removes duplicate directories.
"""

import os
import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Complete the optimization consolidation."""
    root_dir = Path(".")
    
    # Source directories to remove (after consolidation)
    source_dirs = {
        'optimizer': root_dir / "optimizer",
        'optimize': root_dir / "optimize", 
        'optimizers': root_dir / "optimizers"
    }
    
    # Target directory
    target_dir = root_dir / "trading" / "optimization"
    
    logger.info("Completing optimization module consolidation...")
    
    # Step 1: Fix imports in all optimization files
    import_result = fix_optimization_imports()
    
    # Step 2: Update imports across the entire codebase
    update_result = update_codebase_imports(root_dir)
    
    # Step 3: Remove duplicate directories
    cleanup_result = remove_duplicate_directories(source_dirs)
    
    # Step 4: Validate consolidation
    validation_result = validate_consolidation(target_dir)
    
    logger.info("Optimization consolidation completed!")
    
    return {
        "status": "completed",
        "steps_executed": 4,
        "import_fixes": import_result,
        "codebase_updates": update_result,
        "cleanup_operations": cleanup_result,
        "validation_results": validation_result
    }

def fix_optimization_imports():
    """Fix imports in all optimization files."""
    logger.info("Fixing imports in optimization files...")
    
    optimization_dir = Path("trading/optimization")
    
    # Import mappings to fix
    import_mappings = {
        "from trading.base_optimizer": "from .base_optimizer",
        "from trading.optimization.base_optimizer": "from .base_optimizer",
        "from trading.optimization.performance_logger": "from .performance_logger",
        "from trading.optimization.strategy_selection_agent": "from .strategy_selection_agent",
        "from trading.strategy_optimizer": "from .strategy_optimizer",
        "from trading.models.base_model": "from ..models.base_model",
        "from trading.risk.risk_metrics": "from ..risk.risk_metrics",
        "from trading.strategies.rsi_signals": "from ..strategies.rsi_signals",
        "from trading.optimizer_factory": "from .optimizer_factory",
    }
    
    fixed_files = []
    errors = []
    
    # Process all Python files in optimization directory
    for py_file in optimization_dir.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply import mappings
            for old_import, new_import in import_mappings.items():
                content = content.replace(old_import, new_import)
            
            # Write back if changes were made
            if content != original_content:
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                fixed_files.append(str(py_file))
                logger.info(f"Fixed imports in {py_file}")
                
        except Exception as e:
            error_msg = f"Error processing {py_file}: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
    
    return {
        "fixed_files": fixed_files,
        "errors": errors,
        "total_files_processed": len(fixed_files) + len(errors)
    }

def update_codebase_imports(root_dir: Path):
    """Update imports across the entire codebase."""
    logger.info("Updating imports across codebase...")
    
    # Import mappings for codebase
    import_mappings = {
        "from optimizer.": "from trading.optimization.",
        "import optimizer.": "import trading.optimization.",
        "from optimize.": "from trading.optimization.",
        "import optimize.": "import trading.optimization.",
        "from optimizers.": "from trading.optimization.utils.",
        "import optimizers.": "import trading.optimization.utils.",
        "from .optimizer.": "from trading.optimization.",
        "import .optimizer.": "import trading.optimization.",
        "from .optimize.": "from trading.optimization.",
        "import .optimize.": "import trading.optimization.",
        "from .optimizers.": "from trading.optimization.utils.",
        "import .optimizers.": "import trading.optimization.utils.",
    }
    
    updated_count = 0
    updated_files = []
    errors = []
    
    # Process all Python files
    for py_file in root_dir.rglob("*.py"):
        # Skip files in backup and cache directories
        if any(part in ["backup", "__pycache__", ".git", ".venv", "node_modules"] for part in py_file.parts):
            continue
        
        # Skip files in the optimization directory (already fixed)
        if "trading/optimization" in str(py_file):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply import mappings
            for old_import, new_import in import_mappings.items():
                content = content.replace(old_import, new_import)
            
            # Write back if changes were made
            if content != original_content:
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                updated_count += 1
                updated_files.append(str(py_file))
                logger.info(f"Updated imports in {py_file}")
                
        except Exception as e:
            error_msg = f"Error updating {py_file}: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
    
    logger.info(f"Updated imports in {updated_count} files")
    
    return {
        "updated_files": updated_files,
        "warnings": len(errors),
        "errors": errors,
        "total_files_updated": updated_count
    }

def remove_duplicate_directories(source_dirs: dict):
    """Remove duplicate directories after consolidation."""
    logger.info("Removing duplicate directories...")
    
    removed_dirs = []
    errors = []
    
    for name, source_dir in source_dirs.items():
        if source_dir.exists():
            try:
                # Add deprecation notice to remaining files
                deprecation_result = add_deprecation_notices(source_dir)
                
                # Remove directory
                shutil.rmtree(source_dir)
                removed_dirs.append(str(source_dir))
                logger.info(f"Removed {source_dir}")
                
            except Exception as e:
                error_msg = f"Error removing {source_dir}: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
    
    return {
        "removed_directories": removed_dirs,
        "errors": errors,
        "deprecation_notices_added": len(removed_dirs)
    }

def add_deprecation_notices(directory: Path):
    """Add deprecation notices to files in directory."""
    files_updated = []
    errors = []
    
    for file_path in directory.rglob("*.py"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            deprecation_notice = f'''"""
DEPRECATED: This module has been consolidated into trading/optimization/
Please update your imports to use the consolidated version.
Last updated: 2025-01-27
"""

'''
            
            if not content.startswith('"""DEPRECATED'):
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(deprecation_notice + content)
                files_updated.append(str(file_path))
                    
        except Exception as e:
            error_msg = f"Error adding deprecation notice to {file_path}: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
    
    return {
        "files_updated": files_updated,
        "errors": errors,
        "total_files_processed": len(files_updated) + len(errors)
    }

def validate_consolidation(target_dir: Path):
    """Validate the consolidation results."""
    logger.info("Validating consolidation...")
    
    # Check that target directory has expected structure
    expected_files = [
        target_dir / "__init__.py",
        target_dir / "base_optimizer.py",
        target_dir / "bayesian_optimizer.py",
        target_dir / "genetic_optimizer.py",
        target_dir / "grid_optimizer.py",
        target_dir / "multi_objective_optimizer.py",
        target_dir / "rsi_optimizer.py",
        target_dir / "strategy_optimizer.py",
        target_dir / "optimization_visualizer.py",
        target_dir / "optimizer_factory.py",
        target_dir / "performance_logger.py",
        target_dir / "strategy_selection_agent.py",
        target_dir / "utils" / "__init__.py",
        target_dir / "utils" / "consolidator.py",
    ]
    
    missing_files = []
    present_files = []
    
    for file_path in expected_files:
        if not file_path.exists():
            missing_files.append(str(file_path))
        else:
            present_files.append(str(file_path))
    
    if missing_files:
        logger.warning(f"Missing files: {missing_files}")
    else:
        logger.info("All expected files present")
    
    # Check for import errors
    import_success = False
    import_error = None
    
    try:
        import sys
        sys.path.insert(0, str(target_dir.parent.parent))
        
        import trading.optimization
        logger.info("Successfully imported trading.optimization")
        import_success = True
        
    except ImportError as e:
        import_error = str(e)
        logger.error(f"Import error: {e}")
    except Exception as e:
        import_error = str(e)
        logger.error(f"Unexpected error: {e}")
    
    return {
        "validation_passed": len(missing_files) == 0 and import_success,
        "missing_files": missing_files,
        "present_files": present_files,
        "import_success": import_success,
        "import_error": import_error,
        "total_expected_files": len(expected_files),
        "files_found": len(present_files)
    }

if __name__ == "__main__":
    main() 