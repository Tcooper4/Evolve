"""
Test Task Agent

This test validates the TaskAgent functionality including direct usage,
agent controller integration, and prompt router integration.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_task_agent():
    """Test the TaskAgent directly."""
    logger.info("=" * 60)
    logger.info("TASK AGENT TEST SUITE")
    logger.info("=" * 60)

    try:
        from agents.task_agent import TaskAgent, TaskType, execute_task

        # Test 1: Direct TaskAgent usage
        logger.info("Testing direct TaskAgent usage...")
        agent = TaskAgent()

        result = await agent.execute_task(
            prompt="Build a model for AAPL forecasting that achieves high accuracy",
            task_type=TaskType.MODEL_BUILD,
            parameters={"symbol": "AAPL", "model_type": "lstm"},
            max_depth=3,
            performance_threshold=0.7
        )

        logger.info(f"✅ TaskAgent result: {result.success}")
        logger.info(f"   Performance score: {result.performance_score:.3f}")
        logger.info(f"   Message: {result.message}")

        # Test 2: Convenience function
        logger.info("\nTesting convenience function...")
        result2 = await execute_task(
            prompt="Optimize the model parameters until performance improves",
            task_type=TaskType.MODEL_UPDATE,
            parameters={"model_type": "xgboost"},
            max_depth=2
        )

        logger.info(f"✅ Convenience function result: {result2.success}")
        logger.info(f"   Performance score: {result2.performance_score:.3f}")

        return True

    except Exception as e:
        logger.error(f"❌ TaskAgent test failed: {e}")
        return False


async def test_agent_controller_integration():
    """Test TaskAgent integration with AgentController."""
    logger.info("\n" + "=" * 60)
    logger.info("AGENT CONTROLLER INTEGRATION TEST")
    logger.info("=" * 60)

    try:
        from agents.agent_controller import get_agent_controller

        controller = get_agent_controller()

        # Test task workflow
        logger.info("Testing task workflow through agent controller...")
        result = await controller.execute_workflow(
            "task",
            prompt="Build and iteratively improve a model for TSLA prediction",
            task_type="model_build",
            parameters={"symbol": "TSLA", "model_type": "ensemble"},
            max_depth=3,
            performance_threshold=0.6
        )

        logger.info(f"✅ Agent controller task result: {result.success}")
        logger.info(f"   Workflow type: {result.workflow_type}")
        logger.info(f"   Execution time: {result.execution_time:.2f}s")

        return True

    except Exception as e:
        logger.error(f"❌ Agent controller integration test failed: {e}")
        return False


async def test_prompt_router_integration():
    """Test TaskAgent integration with PromptRouter."""
    logger.info("\n" + "=" * 60)
    logger.info("PROMPT ROUTER INTEGRATION TEST")
    logger.info("=" * 60)

    try:
        from routing.prompt_router import PromptRouter

        router = PromptRouter()

        # Test complex task routing
        logger.info("Testing complex task routing...")
        result = await router.route_prompt(
            "Build a model that iteratively improves until it achieves 80% accuracy for BTC forecasting"
        )

        logger.info(f"✅ Prompt router result: {result['success']}")
        logger.info(f"   Routing type: {result.get('routing_type', 'unknown')}")
        logger.info(f"   Task type: {result.get('task_type', 'unknown')}")

        # Test task execution through router
        if result.get('task_type') == 'task':
            logger.info("Testing task execution through router...")
            task_result = await router.execute_routed_task(result)
            logger.info(f"✅ Routed task execution: {task_result.get('success', False)}")

        return True

    except Exception as e:
        logger.error(f"❌ Prompt router integration test failed: {e}")
        return False


async def test_task_iteration():
    """Test iterative task execution."""
    logger.info("\n" + "=" * 60)
    logger.info("TASK ITERATION TEST")
    logger.info("=" * 60)

    try:
        from agents.task_agent import TaskAgent, TaskType

        agent = TaskAgent()

        # Test iterative improvement
        logger.info("Testing iterative task improvement...")
        initial_result = await agent.execute_task(
            prompt="Create a basic model for ETH price prediction",
            task_type=TaskType.MODEL_BUILD,
            parameters={"symbol": "ETH", "model_type": "linear"},
            max_depth=1
        )

        logger.info(f"✅ Initial result: {initial_result.success}")
        logger.info(f"   Performance: {initial_result.performance_score:.3f}")

        # Test improvement iteration
        if initial_result.success:
            improvement_result = await agent.execute_task(
                prompt="Improve the model performance by 20%",
                task_type=TaskType.MODEL_UPDATE,
                parameters={"base_model": "previous_result"},
                max_depth=2,
                performance_threshold=initial_result.performance_score * 1.2
            )

            logger.info(f"✅ Improvement result: {improvement_result.success}")
            logger.info(f"   New performance: {improvement_result.performance_score:.3f}")

        return True

    except Exception as e:
        logger.error(f"❌ Task iteration test failed: {e}")
        return False


async def test_error_handling():
    """Test error handling in TaskAgent."""
    logger.info("\n" + "=" * 60)
    logger.info("ERROR HANDLING TEST")
    logger.info("=" * 60)

    try:
        from agents.task_agent import TaskAgent, TaskType

        agent = TaskAgent()

        # Test with invalid parameters
        logger.info("Testing invalid parameters...")
        result = await agent.execute_task(
            prompt="Test with invalid parameters",
            task_type=TaskType.MODEL_BUILD,
            parameters={"invalid_param": "invalid_value"},
            max_depth=0  # Invalid depth
        )

        logger.info(f"✅ Error handling result: {not result.success}")
        logger.info(f"   Error message: {result.message}")

        # Test with impossible performance threshold
        logger.info("Testing impossible performance threshold...")
        result2 = await agent.execute_task(
            prompt="Achieve 200% accuracy",
            task_type=TaskType.MODEL_BUILD,
            parameters={"symbol": "TEST"},
            performance_threshold=2.0  # Impossible threshold
        )

        logger.info(f"✅ Impossible threshold handling: {not result2.success}")

        return True

    except Exception as e:
        logger.error(f"❌ Error handling test failed: {e}")
        return False


async def main():
    """Main test function."""
    logger.info("🚀 Starting Task Agent Test Suite")
    logger.info("=" * 60)

    test_results = []

    # Run all tests
    tests = [
        ("Direct TaskAgent", test_task_agent),
        ("Agent Controller Integration", test_agent_controller_integration),
        ("Prompt Router Integration", test_prompt_router_integration),
        ("Task Iteration", test_task_iteration),
        ("Error Handling", test_error_handling)
    ]

    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            result = await test_func()
            test_results.append((test_name, result))
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
        logger.info("🎉 All Task Agent tests passed!")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1) 