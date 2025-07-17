#!/usr/bin/env python3
"""
Test script to verify modularization work without importing problematic packages.
"""

import os
import sys
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

def test_file_structure():
    """Test that all modularized files exist."""
    logger.info("🔍 Testing modularized file structure...")
    
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
            logger.info(f"✅ {file_path}")
    
    if missing_files:
        logger.error(f"❌ Missing files: {missing_files}")
        return False
    else:
        logger.info("✅ All modularized files exist!")
        return True

def test_file_sizes():
    """Test that files are reasonably sized after modularization."""
    logger.info("\n📏 Testing file sizes...")
    
    # Check that strategy_optimizer.py is now much smaller
    strategy_optimizer_size = Path("trading/optimization/strategy_optimizer.py").stat().st_size
    logger.info(f"Strategy optimizer size: {strategy_optimizer_size:,} bytes")
    
    if strategy_optimizer_size < 10000:  # Should be much smaller now
        logger.info("✅ Strategy optimizer successfully modularized!")
        return True
    else:
        logger.error("❌ Strategy optimizer still too large")
        return False

def test_import_structure():
    """Test the import structure without actually importing."""
    logger.info("\n📦 Testing import structure...")
    
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
            logger.info(f"✅ {export} exported")
    
    if missing_exports:
        logger.error(f"❌ Missing exports: {missing_exports}")
        return False
    else:
        logger.info("✅ All expected exports present!")
        return True

def test_code_quality():
    """Test basic code quality metrics."""
    logger.info("\n🔧 Testing code quality...")
    
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
        logger.error("❌ Code quality issues:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    else:
        logger.info("✅ Code quality looks good!")
        return True

def main():
    """Run all tests."""
    logger.info("🚀 Testing Modularization Work\n")
    
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
            logger.error(f"❌ Test failed with error: {e}")
    
    logger.info(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Modularization successful!")
        return True
    else:
        logger.warning("⚠️ Some tests failed. Check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 