#!/usr/bin/env python3
"""
Test script for TaskAgent integration.

This script tests the TaskAgent functionality and its integration
with the agent controller and prompt router.
"""

import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
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

        # Test workflow routing
        logger.info("\nTesting workflow routing...")
        result2 = await router.route_prompt(
            "Build a new LSTM model for forecasting"
        )

        logger.info(f"✅ Workflow routing result: {result2['success']}")
        logger.info(f"   Routing type: {result2.get('routing_type', 'unknown')}")

        return True

    except Exception as e:
        logger.error(f"❌ Prompt router integration test failed: {e}")
        return False


async def main():
    """Main test function."""
    logger.info("Starting TaskAgent integration tests...")

    start_time = datetime.now()

    # Run all tests
    task_agent_success = await test_task_agent()
    controller_success = await test_agent_controller_integration()
    router_success = await test_prompt_router_integration()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"TaskAgent Tests: {'✅ PASSED' if task_agent_success else '❌ FAILED'}")
    logger.info(f"Agent Controller Integration: {'✅ PASSED' if controller_success else '❌ FAILED'}")
    logger.info(f"Prompt Router Integration: {'✅ PASSED' if router_success else '❌ FAILED'}")
    logger.info(f"Total Duration: {duration:.2f} seconds")

    all_passed = task_agent_success and controller_success and router_success
    logger.info(f"{'🎉 ALL TESTS PASSED!' if all_passed else '❌ SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)

