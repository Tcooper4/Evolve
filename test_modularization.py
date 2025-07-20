"""
Test Modularization

This test validates that the optimization module has been properly
modularized into smaller, more manageable files.
"""

import logging
from pathlib import Path

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

        # Check for function definitions
        if "def " not in content:
            issues.append(f"{file_path}: No functions found")

    if issues:
        logger.error(f"❌ Code quality issues: {issues}")
        return False
    else:
        logger.info("✅ Code quality checks passed!")
        return True


def test_dependencies():
    """Test that dependencies are properly separated."""
    logger.info("\n🔗 Testing dependencies...")

    # Check that each file has reasonable dependencies
    files_to_check = [
        "trading/optimization/grid_search_optimizer.py",
        "trading/optimization/bayesian_optimizer.py",
        "trading/optimization/genetic_optimizer.py",
        "trading/optimization/pso_optimizer.py",
        "trading/optimization/ray_optimizer.py"
    ]

    dependency_issues = []
    for file_path in files_to_check:
        content = Path(file_path).read_text()
        
        # Check for excessive imports
        import_lines = [line for line in content.split('\n') if line.strip().startswith('import') or line.strip().startswith('from')]
        
        if len(import_lines) > 10:  # Too many imports
            dependency_issues.append(f"{file_path}: Too many imports ({len(import_lines)})")

    if dependency_issues:
        logger.warning(f"⚠️ Dependency issues: {dependency_issues}")
        return False
    else:
        logger.info("✅ Dependencies look good!")
        return True


def main():
    """Main test function."""
    logger.info("🚀 Starting Modularization Test Suite")
    logger.info("=" * 50)

    test_results = []

    # Run all tests
    tests = [
        ("File Structure", test_file_structure),
        ("File Sizes", test_file_sizes),
        ("Import Structure", test_import_structure),
        ("Code Quality", test_code_quality),
        ("Dependencies", test_dependencies)
    ]

    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} test failed with exception: {e}")
            test_results.append((test_name, False))

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{status}: {test_name}")
        if result:
            passed += 1

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 Modularization completed successfully!")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 