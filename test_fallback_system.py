"""
Test Fallback System

This test validates the fallback system functionality including
agent controller fallbacks, mock agents, and prompt router fallbacks.
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
            "Can you help me with trading?",
            "Forecast the price of AAPL",
            "Generate a trading strategy",
            "Analyze the market trends",
            "Hello, how are you?"
        ]

        results = []
        for prompt in test_prompts:
            logger.info(f"Testing prompt: {prompt}")
            result = await router.route_prompt(prompt)
            results.append(result)

            logger.info(f"  Success: {result.get('success', False)}")
            logger.info(f"  Routing type: {result.get('routing_type', 'unknown')}")
            logger.info(f"  Agent used: {result.get('agent_used', 'unknown')}")
            logger.info(f"  Fallback used: {result.get('fallback_used', False)}")

        return results

    except Exception as e:
        logger.error(f"Error testing prompt router: {e}")
        return None


async def test_startup_fallback():
    """Test startup fallback functionality."""
    logger.info("Testing startup fallback...")

    try:
        from fallback.startup_fallback import StartupFallback

        # Create startup fallback
        startup_fallback = StartupFallback()

        # Test system initialization
        logger.info("Testing system initialization...")
        init_result = await startup_fallback.initialize_system()
        logger.info(f"  Initialization success: {init_result.success}")
        logger.info(f"  Components initialized: {init_result.components_initialized}")
        logger.info(f"  Fallbacks used: {init_result.fallbacks_used}")

        # Test component health check
        logger.info("Testing component health check...")
        health_result = await startup_fallback.check_component_health()
        logger.info(f"  Health check success: {health_result.success}")
        logger.info(f"  Healthy components: {health_result.healthy_components}")
        logger.info(f"  Failed components: {health_result.failed_components}")

        # Test fallback activation
        logger.info("Testing fallback activation...")
        fallback_result = await startup_fallback.activate_fallbacks()
        logger.info(f"  Fallback activation success: {fallback_result.success}")
        logger.info(f"  Fallbacks activated: {fallback_result.activated_fallbacks}")

        return {
            "initialization": init_result,
            "health_check": health_result,
            "fallback_activation": fallback_result
        }

    except Exception as e:
        logger.error(f"Error testing startup fallback: {e}")
        return None


async def test_error_recovery():
    """Test error recovery functionality."""
    logger.info("Testing error recovery...")

    try:
        from fallback.error_recovery import ErrorRecovery

        # Create error recovery
        error_recovery = ErrorRecovery()

        # Test error detection
        logger.info("Testing error detection...")
        error_result = await error_recovery.detect_errors()
        logger.info(f"  Error detection success: {error_result.success}")
        logger.info(f"  Errors detected: {error_result.errors_detected}")

        # Test recovery strategies
        logger.info("Testing recovery strategies...")
        recovery_result = await error_recovery.apply_recovery_strategies()
        logger.info(f"  Recovery success: {recovery_result.success}")
        logger.info(f"  Strategies applied: {recovery_result.strategies_applied}")

        # Test system restoration
        logger.info("Testing system restoration...")
        restoration_result = await error_recovery.restore_system()
        logger.info(f"  Restoration success: {restoration_result.success}")
        logger.info(f"  System restored: {restoration_result.system_restored}")

        return {
            "error_detection": error_result,
            "recovery_strategies": recovery_result,
            "system_restoration": restoration_result
        }

    except Exception as e:
        logger.error(f"Error testing error recovery: {e}")
        return None


async def main():
    """Main test function."""
    logger.info("🚀 Starting Fallback System Test Suite")
    logger.info("=" * 60)

    test_results = []

    # Run all tests
    tests = [
        ("Agent Controller Fallback", test_agent_controller_fallback),
        ("Mock Agent", test_mock_agent),
        ("Prompt Router Fallback", test_prompt_router_fallback),
        ("Startup Fallback", test_startup_fallback),
        ("Error Recovery", test_error_recovery)
    ]

    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            result = await test_func()
            test_results.append((test_name, result is not None))
            
            if result:
                logger.info(f"✅ {test_name} test completed successfully")
            else:
                logger.error(f"❌ {test_name} test failed")
                
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
        logger.info("🎉 All fallback system tests passed!")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1) 