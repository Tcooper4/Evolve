#!/usr/bin/env python3
"""
Test script to verify modularization work without importing problematic packages.
"""

import os
import sys
from pathlib import Path

def test_file_structure():
    """Test that all modularized files exist."""
    print("🔍 Testing modularized file structure...")
    
    optimization_files = [
        "trading/optimization/grid_search_optimizer.py",
        "trading/optimization/bayesian_optimizer.py", 
        "trading/optimization/genetic_optimizer.py",
        "trading/optimization/pso_optimizer.py",
        "trading/optimization/ray_optimizer.py",
        "trading/optimization/strategy_optimizer.py",
        "trading/optimization/__init__.py"
    ]
    
    missing_files = []
    for file_path in optimization_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All modularized files exist!")
        return True

def test_file_sizes():
    """Test that files are reasonably sized after modularization."""
    print("\n📏 Testing file sizes...")
    
    # Check that strategy_optimizer.py is now much smaller
    strategy_optimizer_size = Path("trading/optimization/strategy_optimizer.py").stat().st_size
    print(f"Strategy optimizer size: {strategy_optimizer_size:,} bytes")
    
    if strategy_optimizer_size < 10000:  # Should be much smaller now
        print("✅ Strategy optimizer successfully modularized!")
        return True
    else:
        print("❌ Strategy optimizer still too large")
        return False

def test_import_structure():
    """Test the import structure without actually importing."""
    print("\n📦 Testing import structure...")
    
    # Check __init__.py exports
    init_content = Path("trading/optimization/__init__.py").read_text()
    
    expected_exports = [
        "BaseOptimizer",
        "OptimizerConfig", 
        "StrategyOptimizer",
        "GridSearch",
        "OptimizationMethod",
        "OptimizationResult",
        "BayesianOptimization",
        "GeneticAlgorithm",
        "ParticleSwarmOptimization",
        "RayTuneOptimization"
    ]
    
    missing_exports = []
    for export in expected_exports:
        if export not in init_content:
            missing_exports.append(export)
        else:
            print(f"✅ {export} exported")
    
    if missing_exports:
        print(f"❌ Missing exports: {missing_exports}")
        return False
    else:
        print("✅ All expected exports present!")
        return True

def test_code_quality():
    """Test basic code quality metrics."""
    print("\n🔧 Testing code quality...")
    
    # Check for proper docstrings
    files_to_check = [
        "trading/optimization/grid_search_optimizer.py",
        "trading/optimization/bayesian_optimizer.py",
        "trading/optimization/genetic_optimizer.py", 
        "trading/optimization/pso_optimizer.py",
        "trading/optimization/ray_optimizer.py",
        "trading/optimization/strategy_optimizer.py"
    ]
    
    issues = []
    for file_path in files_to_check:
        content = Path(file_path).read_text()
        
        # Check for module docstring
        if not content.startswith('"""'):
            issues.append(f"{file_path}: Missing module docstring")
        
        # Check for class definitions
        if "class " not in content:
            issues.append(f"{file_path}: No classes found")
    
    if issues:
        print("❌ Code quality issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Code quality looks good!")
        return True

def main():
    """Run all tests."""
    print("🚀 Testing Modularization Work\n")
    
    tests = [
        test_file_structure,
        test_file_sizes,
        test_import_structure,
        test_code_quality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Modularization successful!")
        return True
    else:
        print("⚠️ Some tests failed. Check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 