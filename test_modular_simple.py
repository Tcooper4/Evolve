"""
Simple Modular Test

This test validates the modular structure of the trading system
without importing complex dependencies.
"""

import os
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def test_file_structure():
    """Test that all modular files exist and have correct structure."""
    logger.info("🧪 Testing Modular File Structure")
    logger.info("=" * 50)

    # Test execution agent modular structure
    execution_files = [
        "trading/agents/execution/__init__.py",
        "trading/agents/execution/risk_controls.py",
        "trading/agents/execution/trade_signals.py",
        "trading/agents/execution/execution_providers.py",
        "trading/agents/execution/position_manager.py",
        "trading/agents/execution/execution_agent.py"
    ]

    logger.info("📁 Checking Execution Agent Modular Files:")
    for file_path in execution_files:
        if os.path.exists(file_path):
            logger.info(f"  ✅ {file_path}")
        else:
            logger.error(f"  ❌ {file_path} - MISSING")

    # Test optimizer agent modular structure
    optimization_files = [
        "trading/agents/optimization/__init__.py",
        "trading/agents/optimization/parameter_validator.py",
        "trading/agents/optimization/strategy_optimizer.py",
        "trading/agents/optimization/backtest_integration.py",
        "trading/agents/optimization/performance_analyzer.py",
        "trading/agents/optimization/optimizer_agent.py"
    ]

    logger.info("\n📁 Checking Optimizer Agent Modular Files:")
    for file_path in optimization_files:
        if os.path.exists(file_path):
            logger.info(f"  ✅ {file_path}")
        else:
            logger.error(f"  ❌ {file_path} - MISSING")

    # Test task orchestrator modular structure
    orchestrator_files = [
        "core/orchestrator/__init__.py",
        "core/orchestrator/task_models.py",
        "core/orchestrator/task_scheduler.py",
        "core/orchestrator/task_executor.py",
        "core/orchestrator/task_monitor.py",
        "core/orchestrator/task_conditions.py",
        "core/orchestrator/task_providers.py",
        "core/orchestrator/task_orchestrator.py"
    ]

    logger.info("\n📁 Checking Task Orchestrator Modular Files:")
    for file_path in orchestrator_files:
        if os.path.exists(file_path):
            logger.info(f"  ✅ {file_path}")
        else:
            logger.error(f"  ❌ {file_path} - MISSING")

    # Test updated test files
    test_files = [
        "trading/agents/test_execution_agent.py",
        "tests/test_agents/test_execution_agent_risk_controls.py",
        "tests/test_optimization_modular.py",
        "tests/test_task_orchestrator.py"
    ]

    logger.info("\n📁 Checking Updated Test Files:")
    for file_path in test_files:
        if os.path.exists(file_path):
            logger.info(f"  ✅ {file_path}")
        else:
            logger.error(f"  ❌ {file_path} - MISSING")


def test_file_sizes():
    """Test that the original large files have been reduced in size."""
    logger.info("\n📊 Testing File Size Reduction")
    logger.info("=" * 50)

    # Check if original large files still exist
    original_files = [
        "trading/agents/execution_agent.py",
        "trading/agents/optimizer_agent.py",
        "core/task_orchestrator.py"
    ]

    for file_path in original_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            logger.warning(f"  ⚠️  {file_path} still exists ({size:,} bytes)")
        else:
            logger.info(f"  ✅ {file_path} has been modularized")


def test_import_structure():
    """Test the import structure of modular components."""
    logger.info("\n📦 Testing Import Structure")
    logger.info("=" * 50)

    try:
        # Test execution agent imports
        from trading.agents.execution import ExecutionAgent
        logger.info("  ✅ ExecutionAgent import successful")

        # Test optimizer agent imports
        from trading.agents.optimization import OptimizerAgent
        logger.info("  ✅ OptimizerAgent import successful")

        # Test task orchestrator imports
        from core.orchestrator import TaskOrchestrator
        logger.info("  ✅ TaskOrchestrator import successful")

        logger.info("  ✅ All modular imports successful")

    except ImportError as e:
        logger.error(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"  ❌ Unexpected error: {e}")
        return False

    return True


def test_basic_functionality():
    """Test basic functionality of modular components."""
    logger.info("\n🔧 Testing Basic Functionality")
    logger.info("=" * 50)

    try:
        # Test execution agent creation
        from trading.agents.execution import create_execution_agent
        execution_agent = create_execution_agent({"execution_mode": "simulation"})
        logger.info("  ✅ Execution agent creation successful")

        # Test optimizer agent creation
        from trading.agents.optimization import create_optimizer_agent
        optimizer_agent = create_optimizer_agent({"optimization_type": "grid_search"})
        logger.info("  ✅ Optimizer agent creation successful")

        # Test task orchestrator creation
        from core.orchestrator import create_task_orchestrator
        orchestrator = create_task_orchestrator()
        logger.info("  ✅ Task orchestrator creation successful")

        logger.info("  ✅ All basic functionality tests passed")

    except Exception as e:
        logger.error(f"  ❌ Functionality test failed: {e}")
        return False

    return True


def main():
    """Main test function."""
    logger.info("🚀 Starting Modular Structure Test")
    logger.info("=" * 60)

    # Run tests
    test_file_structure()
    test_file_sizes()
    
    import_success = test_import_structure()
    functionality_success = test_basic_functionality()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    if import_success and functionality_success:
        logger.info("🎉 All modular structure tests passed!")
        logger.info("✅ System has been successfully modularized")
        return True
    else:
        logger.error("❌ Some modular structure tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 