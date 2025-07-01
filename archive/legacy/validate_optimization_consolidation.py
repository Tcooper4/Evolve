#!/usr/bin/env python3
"""
Validate Optimization Consolidation

This script validates that the optimization consolidation is complete
and all imports are working correctly.
"""

import sys
from pathlib import Path
import importlib

def main():
    """Validate the optimization consolidation."""
    print("🔍 Validating Optimization Consolidation")
    print("=" * 50)
    
    # Check if duplicate directories still exist
    duplicate_dirs = ["optimize", "optimizer", "optimizers"]
    existing_duplicates = []
    
    for dir_name in duplicate_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            existing_duplicates.append(dir_name)
            print(f"⚠️  Duplicate directory still exists: {dir_name}")
    
    if not existing_duplicates:
        print("✅ All duplicate directories removed")
    
    # Check if main optimization directory exists
    main_opt_dir = Path("trading/optimization")
    if not main_opt_dir.exists():
        print("❌ Main optimization directory missing")
        return False
    
    print("✅ Main optimization directory exists")
    
    # Check expected files
    expected_files = [
        "base_optimizer.py",
        "bayesian_optimizer.py", 
        "genetic_optimizer.py",
        "grid_optimizer.py",
        "multi_objective_optimizer.py",
        "rsi_optimizer.py",
        "strategy_optimizer.py",
        "optimization_visualizer.py",
        "optimizer_factory.py",
        "performance_logger.py",
        "strategy_selection_agent.py",
        "__init__.py"
    ]
    
    missing_files = []
    for file_name in expected_files:
        file_path = main_opt_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
    else:
        print("✅ All expected files present")
    
    # Check utils directory
    utils_dir = main_opt_dir / "utils"
    if utils_dir.exists():
        consolidator_file = utils_dir / "consolidator.py"
        if consolidator_file.exists():
            print("✅ Utils directory with consolidator exists")
        else:
            print("❌ Consolidator file missing from utils")
    else:
        print("❌ Utils directory missing")
    
    # Test imports
    print("\n🧪 Testing imports...")
    
    try:
        # Test main module import
        import trading.optimization
        print("✅ Main optimization module imported")
        
        # Test specific imports
        from trading.optimization import BaseOptimizer, BayesianOptimizer, GeneticOptimizer
        print("✅ Core optimizers imported")
        
        from trading.optimization import RSIOptimizer, StrategyOptimizer
        print("✅ Strategy optimizers imported")
        
        from trading.optimization import OptimizationVisualizer, OptimizerFactory
        print("✅ Utility classes imported")
        
        from trading.optimization.utils.consolidator import OptimizerConsolidator
        print("✅ Consolidator imported")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    # Check for any remaining old imports in the codebase
    print("\n🔍 Checking for remaining old imports...")
    
    old_imports_found = []
    search_patterns = [
        "from optimizer.",
        "import optimizer.",
        "from optimize.",
        "import optimize.",
        "from optimizers.",
        "import optimizers.",
    ]
    
    # Search in Python files
    for py_file in Path(".").rglob("*.py"):
        if any(part in ["backup", "__pycache__", ".git", ".venv"] for part in py_file.parts):
            continue
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for pattern in search_patterns:
                if pattern in content:
                    old_imports_found.append(f"{py_file}: {pattern}")
                    
        except Exception:
            continue
    
    if old_imports_found:
        print("⚠️  Found remaining old imports:")
        for old_import in old_imports_found[:10]:  # Show first 10
            print(f"   {old_import}")
        if len(old_imports_found) > 10:
            print(f"   ... and {len(old_imports_found) - 10} more")
    else:
        print("✅ No old imports found")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Consolidation Summary:")
    
    if existing_duplicates:
        print(f"❌ {len(existing_duplicates)} duplicate directories still exist")
    else:
        print("✅ All duplicate directories removed")
    
    if missing_files:
        print(f"❌ {len(missing_files)} files missing from main module")
    else:
        print("✅ All expected files present")
    
    if old_imports_found:
        print(f"⚠️  {len(old_imports_found)} old imports still found")
    else:
        print("✅ All imports updated")
    
    print("\n🎯 Next Steps:")
    if existing_duplicates or missing_files or old_imports_found:
        print("   - Remove remaining duplicate directories")
        print("   - Add missing files")
        print("   - Update remaining old imports")
        print("   - Test all optimization functionality")
    else:
        print("   - ✅ Consolidation complete!")
        print("   - Test optimization functionality in UI")
        print("   - Update documentation")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 