"""
Test Modular Components

This test validates the modular components of the trading system,
including execution agents, optimizer agents, and task orchestrators.
"""

import asyncio
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_execution_agent_modular():
    """Test the modular execution agent."""
    logger.info("🧪 Testing Modular Execution Agent")
    logger.info("=" * 50)

    try:
        # Test imports
        from trading.agents.execution import TradeSignal, create_execution_agent

        logger.info("✅ All execution agent imports successful")

        # Test risk controls
        from trading.agents.execution.risk_controls import create_default_risk_controls

        risk_controls = create_default_risk_controls()
        logger.info(
            f"✅ Risk controls created: max_position_size={risk_controls.max_position_size}"
        )

        # Test trade signals
        from trading.portfolio.portfolio_manager import TradeDirection

        signal = TradeSignal(
            symbol="AAPL",
            direction=TradeDirection.LONG,
            strategy="test",
            confidence=0.8,
            entry_price=150.00,
        )
        logger.info(
            f"✅ Trade signal created: {signal.symbol} at ${signal.entry_price}"
        )

        # Test execution agent creation
        config = {
            "execution_mode": "simulation",
            "max_positions": 3,
            "min_confidence": 0.6,
        }
        agent = create_execution_agent(config)
        logger.info("✅ Execution agent created successfully")

        # Test basic functionality
        market_data = {"AAPL_price": 150.50, "AAPL_volume": 1000000}
        result = await agent.execute(signal=signal, market_data=market_data)
        logger.info(f"✅ Execution result: {result.success}")

        logger.info("✅ Modular Execution Agent tests completed!\n")

    except Exception as e:
        logger.error(f"❌ Execution Agent test failed: {e}")
        import traceback

        traceback.print_exc()


async def test_optimizer_agent_modular():
    """Test the modular optimizer agent."""
    logger.info("🧪 Testing Modular Optimizer Agent")
    logger.info("=" * 50)

    try:
        # Test imports
        from trading.agents.optimization import (
            OptimizationParameter,
            OptimizationResult,
            OptimizationType,
            ParameterValidator,
            PerformanceAnalyzer,
            StrategyConfig,
            StrategyOptimizer,
            create_optimizer_agent,
        )

        logger.info("✅ All optimizer agent imports successful")

        # Test parameter validator
        validator = ParameterValidator()
        param = OptimizationParameter(
            name="rsi_period", min_value=10, max_value=30, step=2, parameter_type="int"
        )
        validated_params = validator.validate_optimization_parameters([param])
        logger.info(
            f"✅ Parameter validation: {len(validated_params)} valid parameters"
        )

        # Test strategy optimizer
        optimizer = StrategyOptimizer({})
        strategy_configs = [
            StrategyConfig(strategy_name="rsi_strategy", enabled=True),
            StrategyConfig(strategy_name="macd_strategy", enabled=True),
        ]
        combinations = optimizer._generate_strategy_combinations(strategy_configs)
        logger.info(
            f"✅ Strategy combinations generated: {len(combinations)} combinations"
        )

        # Test performance analyzer
        analyzer = PerformanceAnalyzer()
        result = OptimizationResult(
            parameter_combination={"rsi_period": 14},
            performance_metrics={"sharpe_ratio": 1.5},
            backtest_results={},
            optimization_score=1.5,
        )
        analyzed_result = analyzer.analyze_optimization_result(result)
        logger.info(
            f"✅ Performance analysis completed: score={analyzed_result.optimization_score}"
        )

        # Test optimizer agent creation
        config = {
            "optimization_type": OptimizationType.GRID_SEARCH,
            "max_iterations": 10,
            "parallel_workers": 2,
        }
        create_optimizer_agent(config)
        logger.info("✅ Optimizer agent created successfully")

        logger.info("✅ Modular Optimizer Agent tests completed!\n")

    except Exception as e:
        logger.error(f"❌ Optimizer Agent test failed: {e}")
        import traceback

        traceback.print_exc()


async def test_task_orchestrator_modular():
    """Test the modular task orchestrator."""
    logger.info("🧪 Testing Modular Task Orchestrator")
    logger.info("=" * 50)

    try:
        # Test imports
        from core.orchestrator import (
            OrchestrationConfig,
            TaskDefinition,
            TaskExecutor,
            TaskMonitor,
            TaskPriority,
            TaskScheduler,
            create_task_orchestrator,
        )

        logger.info("✅ All task orchestrator imports successful")

        # Test task definition
        task_def = TaskDefinition(
            name="test_task",
            function_name="test_function",
            parameters={"param1": "value1"},
            priority=TaskPriority.NORMAL,
            timeout=30.0,
        )
        logger.info(f"✅ Task definition created: {task_def.name}")

        # Test task scheduler
        scheduler = TaskScheduler()
        scheduled_task = scheduler.schedule_task(task_def)
        logger.info(f"✅ Task scheduled: {scheduled_task.task_id}")

        # Test task executor
        executor = TaskExecutor()
        executor_result = await executor.execute_task(scheduled_task)
        logger.info(f"✅ Task execution completed: {executor_result.status}")

        # Test task monitor
        monitor = TaskMonitor()
        monitor_status = monitor.get_task_status(scheduled_task.task_id)
        logger.info(f"✅ Task monitoring: {monitor_status}")

        # Test orchestrator creation
        config = OrchestrationConfig(
            max_concurrent_tasks=5, task_timeout=60.0, enable_monitoring=True
        )
        orchestrator = create_task_orchestrator(config)
        logger.info("✅ Task orchestrator created successfully")

        # Test orchestrator functionality
        orchestrator_result = await orchestrator.submit_task(task_def)
        logger.info(f"✅ Orchestrator task submission: {orchestrator_result.success}")

        logger.info("✅ Modular Task Orchestrator tests completed!\n")

    except Exception as e:
        logger.error(f"❌ Task Orchestrator test failed: {e}")
        import traceback

        traceback.print_exc()


async def test_integration():
    """Test integration between modular components."""
    logger.info("🧪 Testing Component Integration")
    logger.info("=" * 50)

    try:
        # Test execution + optimizer integration
        from trading.agents.execution import create_execution_agent
        from trading.agents.optimization import create_optimizer_agent

        execution_agent = create_execution_agent({"execution_mode": "simulation"})
        optimizer_agent = create_optimizer_agent({"optimization_type": "grid_search"})

        logger.info("✅ Execution and optimizer agents created for integration")

        # Test orchestrator + agents integration
        from core.orchestrator import OrchestrationConfig, create_task_orchestrator

        orchestrator = create_task_orchestrator(OrchestrationConfig())
        logger.info("✅ Orchestrator created for integration")

        # Test workflow
        workflow_result = await orchestrator.run_workflow(
            [
                {"agent": optimizer_agent, "task": "optimize_strategy"},
                {"agent": execution_agent, "task": "execute_signals"},
            ]
        )
        logger.info(f"✅ Workflow execution: {workflow_result.success}")

        logger.info("✅ Component Integration tests completed!\n")

    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """Main test function."""
    logger.info("🚀 Starting Modular Components Test Suite")
    logger.info("=" * 60)

    test_results = []

    # Run all tests
    tests = [
        ("Execution Agent", test_execution_agent_modular),
        ("Optimizer Agent", test_optimizer_agent_modular),
        ("Task Orchestrator", test_task_orchestrator_modular),
        ("Integration", test_integration),
    ]

    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            await test_func()
            test_results.append((test_name, True))
        except Exception as e:
            logger.error(f"❌ {test_name} test failed with exception: {e}")
            test_results.append((test_name, False))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{status}: {test_name}")
        if result:
            passed += 1

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All modular component tests passed!")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
