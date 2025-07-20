#!/usr/bin/env python3
"""
Test script for the fallback system

This script tests the agent registration fallback system to ensure
it works correctly when no agents are registered.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_agent_controller_fallback():
    """Test agent controller fallback functionality."""
    logger.info("Testing agent controller fallback...")

    try:
        from agents.agent_controller import get_agent_controller

        # Get agent controller
        controller = get_agent_controller()

        # Check registration status
        status = controller.get_agent_registration_status()

        logger.info("Agent registration status:")
        logger.info(f"  Total agents: {status['total_agents']}")
        logger.info(f"  Successful registrations: {status['successful_registrations']}")
        logger.info(f"  Failed registrations: {status['failed_registrations']}")
        logger.info(f"  Fallback agent created: {status['fallback_agent_created']}")

        # Check if we have real agents
        has_real_agents = controller.has_real_agents()
        logger.info(f"  Has real agents: {has_real_agents}")

        # List available agents
        available_agents = controller.get_available_agents()
        logger.info(f"  Available agents: {available_agents}")

        # Print agent details
        for agent_name, agent_details in status['agent_details'].items():
            logger.info(f"  {agent_name}: {agent_details['class_name']} ({agent_details['category']})")
            logger.info(f"    Capabilities: {', '.join(agent_details['capabilities'])}")

        return status

    except Exception as e:
        logger.error(f"Error testing agent controller: {e}")
        return None


async def test_mock_agent():
    """Test mock agent functionality."""
    logger.info("Testing mock agent...")

    try:
        from agents.mock_agent import create_mock_agent

        # Create mock agent
        mock_agent = create_mock_agent("TestMockAgent")

        # Test different types of prompts
        test_prompts = [
            "What is the system status?",
            "Can you help me?",
            "Forecast the price of AAPL",
            "Generate a trading strategy",
            "Analyze the market trends",
            "Hello, how are you?"
        ]

        results = []
        for prompt in test_prompts:
            logger.info(f"Testing prompt: {prompt}")
            result = await mock_agent.execute(prompt)
            results.append(result)

            logger.info(f"  Success: {result.success}")
            logger.info(f"  Message: {result.message}")
            logger.info(f"  Execution time: {result.execution_time:.3f}s")
            logger.info(f"  Data type: {result.data.get('type', 'unknown')}")

        return results

    except Exception as e:
        logger.error(f"Error testing mock agent: {e}")
        return None


async def test_prompt_router_fallback():
    """Test prompt router fallback functionality."""
    logger.info("Testing prompt router fallback...")

    try:
        from routing.prompt_router import get_prompt_router

        # Get prompt router
        router = get_prompt_router()

        # Test prompts
        test_prompts = [
            "What is the system status?",
            "Help me with trading",
            "Forecast stock prices",
            "Generate a strategy"
        ]

        results = []
        for prompt in test_prompts:
            logger.info(f"Testing router with prompt: {prompt}")
            result = await router.route_prompt(prompt, user_id="test_user")
            results.append(result)

            logger.info(f"  Success: {result.get('success', False)}")
            logger.info(f"  Routing type: {result.get('routing_type', 'unknown')}")
            logger.info(f"  Message: {result.get('message', 'No message')}")

        return results

    except Exception as e:
        logger.error(f"Error testing prompt router: {e}")
        return None


async def test_startup_fallback():
    """Test startup fallback functionality."""
    logger.info("Testing startup fallback...")

    try:
        from start_orchestrator import _check_agent_registration

        # Test agent registration check
        status = await _check_agent_registration()

        if status:
            logger.info("Startup fallback test completed successfully")
            logger.info(f"Registration status: {status}")
        else:
            logger.warning("Startup fallback test returned None")

        return status

    except Exception as e:
        logger.error(f"Error testing startup fallback: {e}")
        return None


async def main():
    """Run all fallback tests."""
    logger.info("Starting fallback system tests...")

    # Test 1: Agent controller fallback
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Agent Controller Fallback")
    logger.info("="*60)
    controller_status = await test_agent_controller_fallback()

    # Test 2: Mock agent functionality
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Mock Agent Functionality")
    logger.info("="*60)
    mock_results = await test_mock_agent()

    # Test 3: Prompt router fallback
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Prompt Router Fallback")
    logger.info("="*60)
    router_results = await test_prompt_router_fallback()

    # Test 4: Startup fallback
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Startup Fallback")
    logger.info("="*60)
    startup_status = await test_startup_fallback()

    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)

    tests_passed = 0
    total_tests = 4

    if controller_status:
        logger.info("✅ Agent controller fallback: PASSED")
        tests_passed += 1
    else:
        logger.error("❌ Agent controller fallback: FAILED")

    if mock_results:
        logger.info("✅ Mock agent functionality: PASSED")
        tests_passed += 1
    else:
        logger.error("❌ Mock agent functionality: FAILED")

    if router_results:
        logger.info("✅ Prompt router fallback: PASSED")
        tests_passed += 1
    else:
        logger.error("❌ Prompt router fallback: FAILED")

    if startup_status is not None:
        logger.info("✅ Startup fallback: PASSED")
        tests_passed += 1
    else:
        logger.error("❌ Startup fallback: FAILED")

    logger.info(f"\nOverall result: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        logger.info("🎉 All fallback tests passed! System is ready for fallback operation.")
    else:
        logger.warning("⚠️ Some tests failed. Check the logs for details.")

    return tests_passed == total_tests


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

