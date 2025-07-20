#!/usr/bin/env python3
"""
Test Modular Components

This script tests all the modularized components of the Evolve trading platform:
- Execution Agent (modular)
- Optimizer Agent (modular)
- Task Orchestrator (modular)
"""

import asyncio
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


async def test_execution_agent_modular():
    """Test the modular execution agent."""
    logger.info("🧪 Testing Modular Execution Agent")
    logger.info("=" * 50)

    try:
        # Test imports
        from trading.agents.execution import (
            ExecutionAgent, create_execution_agent,
            RiskControls, RiskThreshold, RiskThresholdType,
            TradeSignal, ExecutionRequest, ExecutionResult,
            ExecutionProvider, SimulationProvider,
            PositionManager, ExitEvent, ExitReason
        )
        logger.info("✅ All execution agent imports successful")

        # Test risk controls
        from trading.agents.execution.risk_controls import create_default_risk_controls
        risk_controls = create_default_risk_controls()
        logger.info(f"✅ Risk controls created: max_position_size={risk_controls.max_position_size}")

        # Test trade signals
        from trading.portfolio.portfolio_manager import TradeDirection
        signal = TradeSignal(
            symbol="AAPL",
            direction=TradeDirection.LONG,
            strategy="test",
            confidence=0.8,
            entry_price=150.00
        )
        logger.info(f"✅ Trade signal created: {signal.symbol} at ${signal.entry_price}")

        # Test execution agent creation
        config = {
            "execution_mode": "simulation",
            "max_positions": 3,
            "min_confidence": 0.6
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
            OptimizerAgent, create_optimizer_agent,
            ParameterValidator, OptimizationParameter,
            StrategyOptimizer, StrategyConfig, OptimizationType, OptimizationMetric,
            BacktestIntegration, PerformanceAnalyzer, OptimizationResult
        )
        logger.info("✅ All optimizer agent imports successful")

        # Test parameter validator
        validator = ParameterValidator()
        param = OptimizationParameter(
            name="rsi_period",
            min_value=10,
            max_value=30,
            step=2,
            parameter_type="int"
        )
        validated_params = validator.validate_optimization_parameters([param])
        logger.info(f"✅ Parameter validation: {len(validated_params)} valid parameters")

        # Test strategy optimizer
        optimizer = StrategyOptimizer({})
        strategy_configs = [
            StrategyConfig(strategy_name="rsi_strategy", enabled=True),
            StrategyConfig(strategy_name="macd_strategy", enabled=True)
        ]
        combinations = optimizer._generate_strategy_combinations(strategy_configs)
        logger.info(f"✅ Strategy combinations generated: {len(combinations)} combinations")

        # Test performance analyzer
        analyzer = PerformanceAnalyzer()
        result = OptimizationResult(
            parameter_combination={"rsi_period": 14},
            performance_metrics={"sharpe_ratio": 1.5},
            backtest_results={},
            optimization_score=1.5
        )
        analyzer.add_optimization_result(result)
        stats = analyzer.get_optimization_stats()
        logger.info(f"✅ Performance analysis: {stats['total_optimizations']} optimizations")

        # Test optimizer agent creation
        config = {
            "optimizer_config": {},
            "backtest_config": {},
            "performance_config": {}
        }
        agent = create_optimizer_agent(config)
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
            TaskOrchestrator, create_task_orchestrator, start_orchestrator,
            TaskScheduler, TaskConfig, TaskStatus, TaskPriority, TaskType,
            TaskExecutor, TaskExecution,
            TaskMonitor, AgentStatus,
            TaskConditions,
            TaskProvider, AgentTaskProvider
        )
        logger.info("✅ All task orchestrator imports successful")

        # Test task models
        task_config = TaskConfig(
            name="test_task",
            task_type=TaskType.SYSTEM_HEALTH,
            enabled=True,
            interval_minutes=5,
            priority=TaskPriority.MEDIUM
        )
        logger.info(f"✅ Task config created: {task_config.name}")

        # Test task scheduler
        scheduler = TaskScheduler()
        scheduler.add_task(task_config)
        scheduled_tasks = scheduler.get_scheduled_tasks()
        logger.info(f"✅ Task scheduler: {len(scheduled_tasks)} scheduled tasks")

        # Test task executor
        executor = TaskExecutor()
        logger.info("✅ Task executor created")

        # Test task monitor
        monitor = TaskMonitor()
        health_status = await monitor.check_system_health()
        logger.info(f"✅ Task monitor: health score {health_status.get('overall_health', 0):.2f}")

        # Test task conditions
        conditions = TaskConditions()
        available_conditions = conditions.get_available_conditions()
        logger.info(f"✅ Task conditions: {len(available_conditions)} available conditions")

        # Test task orchestrator creation
        orchestrator = create_task_orchestrator()
        logger.info("✅ Task orchestrator created successfully")

        # Test orchestrator start/stop
        await orchestrator.start()
        logger.info("✅ Task orchestrator started")

        system_status = orchestrator.get_system_status()
        logger.info(f"✅ System status: {system_status.get('orchestrator_status', 'unknown')}")

        await orchestrator.stop()
        logger.info("✅ Task orchestrator stopped")

        logger.info("✅ Modular Task Orchestrator tests completed!\n")

    except Exception as e:
        logger.error(f"❌ Task Orchestrator test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_integration():
    """Test integration between modular components."""
    logger.info("🧪 Testing Integration Between Modular Components")
    logger.info("=" * 50)

    try:
        # Test execution agent with orchestrator
        from trading.agents.execution import create_execution_agent
        from core.orchestrator import create_task_orchestrator

        # Create components
        execution_agent = create_execution_agent({"execution_mode": "simulation"})
        orchestrator = create_task_orchestrator()

        logger.info("✅ Integration test: Components created successfully")

        # Test that components can work together
        await orchestrator.start()

        # Execute a task that would use the execution agent
        task_id = await orchestrator.execute_task_now("execution", {"test": True})
        logger.info(f"✅ Integration test: Task executed with ID {task_id}")

        await orchestrator.stop()

        logger.info("✅ Integration tests completed!\n")

    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all modular component tests."""
    logger.info("🚀 Starting Modular Component Tests")
    logger.info("=" * 60)

    # Test each modular component
    await test_execution_agent_modular()
    await test_optimizer_agent_modular()
    await test_task_orchestrator_modular()

    # Test integration
    await test_integration()

    logger.info("🎉 All Modular Component Tests Completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

