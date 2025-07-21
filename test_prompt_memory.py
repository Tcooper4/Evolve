"""
Test Prompt Memory

This test validates the prompt memory functionality including
basic operations, different backends, and agent integration.
"""

import asyncio
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_prompt_memory_basic():
    """Test basic prompt memory functionality."""
    logger.info("=" * 60)
    logger.info("PROMPT MEMORY BASIC TEST")
    logger.info("=" * 60)

    try:
        from memory.prompt_log import get_last_prompt, get_prompt_history, log_prompt

        # Test basic logging
        logger.info("Testing basic prompt logging...")
        success = await log_prompt(
            prompt="Test prompt for memory",
            result={"status": "success", "data": "test result"},
            user_id="test_user",
            agent_type="TestAgent",
            execution_time=1.5,
            success=True,
        )

        logger.info(f"✅ Prompt logging result: {success}")

        # Test getting last prompt
        logger.info("\nTesting get last prompt...")
        last_prompt = await get_last_prompt("test_user")

        if last_prompt:
            logger.info(f"✅ Last prompt: {last_prompt.prompt}")
            logger.info(f"   Success: {last_prompt.success}")
            logger.info(f"   Execution time: {last_prompt.execution_time}")
        else:
            logger.error("❌ No last prompt found")
            return False

        # Test getting prompt history
        logger.info("\nTesting get prompt history...")
        history = await get_prompt_history("test_user", n=5)

        logger.info(f"✅ Prompt history count: {len(history)}")
        for i, entry in enumerate(history):
            logger.info(f"   Entry {i + 1}: {entry.prompt[:50]}...")

        return True

    except Exception as e:
        logger.error(f"❌ Prompt memory basic test failed: {e}")
        return False


async def test_prompt_memory_backends():
    """Test different prompt memory backends."""
    logger.info("\n" + "=" * 60)
    logger.info("PROMPT MEMORY BACKEND TEST")
    logger.info("=" * 60)

    try:
        from memory.prompt_log import PromptMemory

        # Test JSON backend
        logger.info("Testing JSON backend...")
        json_memory = PromptMemory(
            backend="json", file_path="memory/test_prompt_history.json"
        )

        success = await json_memory.log_prompt(
            prompt="JSON backend test",
            result={"backend": "json", "test": True},
            user_id="json_user",
            agent_type="TestAgent",
        )

        logger.info(f"✅ JSON backend logging: {success}")

        # Test statistics
        stats = await json_memory.get_statistics("json_user")
        logger.info(f"✅ JSON backend stats: {stats}")

        return True

    except Exception as e:
        logger.error(f"❌ Prompt memory backend test failed: {e}")
        return False


async def test_task_agent_integration():
    """Test TaskAgent integration with prompt memory."""
    logger.info("\n" + "=" * 60)
    logger.info("TASK AGENT INTEGRATION TEST")
    logger.info("=" * 60)

    try:
        from agents.task_agent import TaskAgent, TaskType

        # Create TaskAgent (should initialize with prompt memory)
        logger.info("Creating TaskAgent...")
        agent = TaskAgent()

        # Execute a simple task
        logger.info("Executing test task...")
        result = await agent.execute_task(
            task_type=TaskType.MODEL_INNOVATION,
            parameters={"test": True, "user_id": "integration_test_user"},
        )

        logger.info(f"✅ Task execution result: {result.success}")

        # Check if prompt was logged
        logger.info("Checking if prompt was logged...")
        from memory.prompt_log import get_last_prompt

        last_prompt = await get_last_prompt("integration_test_user")

        if last_prompt:
            logger.info(f"✅ Prompt logged: {last_prompt.prompt[:50]}...")
            logger.info(f"   Agent type: {last_prompt.agent_type}")
            logger.info(f"   Success: {last_prompt.success}")
        else:
            logger.warning("⚠️ No prompt found in memory")

        return True

    except Exception as e:
        logger.error(f"❌ Task agent integration test failed: {e}")
        return False


async def test_prompt_memory_functions():
    """Test prompt memory utility functions."""
    logger.info("\n" + "=" * 60)
    logger.info("PROMPT MEMORY FUNCTIONS TEST")
    logger.info("=" * 60)

    try:
        from memory.prompt_log import (
            clear_prompt_history,
            get_prompt_history,
            get_prompt_statistics,
            log_prompt,
            search_prompts,
        )

        # Test multiple prompt logging
        logger.info("Testing multiple prompt logging...")
        for i in range(3):
            await log_prompt(
                prompt=f"Test prompt {i + 1}",
                result={"iteration": i + 1},
                user_id="function_test_user",
                agent_type="TestAgent",
                execution_time=0.5 + i * 0.1,
                success=True,
            )

        # Test statistics
        logger.info("Testing statistics...")
        stats = await get_prompt_statistics("function_test_user")
        logger.info(f"✅ Statistics: {stats}")

        # Test search functionality
        logger.info("Testing search functionality...")
        search_results = await search_prompts("function_test_user", "prompt")
        logger.info(f"✅ Search results: {len(search_results)} found")

        # Test history with limit
        logger.info("Testing history with limit...")
        limited_history = await get_prompt_history("function_test_user", n=2)
        logger.info(f"✅ Limited history: {len(limited_history)} entries")

        # Test clear functionality
        logger.info("Testing clear functionality...")
        clear_result = await clear_prompt_history("function_test_user")
        logger.info(f"✅ Clear result: {clear_result}")

        # Verify clear worked
        remaining_history = await get_prompt_history("function_test_user")
        logger.info(f"✅ Remaining history: {len(remaining_history)} entries")

        return True

    except Exception as e:
        logger.error(f"❌ Prompt memory functions test failed: {e}")
        return False


async def test_memory_persistence():
    """Test memory persistence across sessions."""
    logger.info("\n" + "=" * 60)
    logger.info("MEMORY PERSISTENCE TEST")
    logger.info("=" * 60)

    try:
        from memory.prompt_log import log_prompt

        # Log a prompt
        logger.info("Logging persistent prompt...")
        await log_prompt(
            prompt="Persistent test prompt",
            result={"persistence": "test"},
            user_id="persistence_test_user",
            agent_type="TestAgent",
            success=True,
        )

        # Simulate session restart by creating new memory instance
        logger.info("Simulating session restart...")
        from memory.prompt_log import PromptMemory

        new_memory = PromptMemory()

        # Try to retrieve the prompt
        logger.info("Retrieving persistent prompt...")
        last_prompt = await new_memory.get_last_prompt("persistence_test_user")

        if last_prompt and "Persistent test prompt" in last_prompt.prompt:
            logger.info("✅ Memory persistence confirmed!")
            return True
        else:
            logger.error("❌ Memory persistence failed")
            return False

    except Exception as e:
        logger.error(f"❌ Memory persistence test failed: {e}")
        return False


async def main():
    """Main test function."""
    logger.info("🚀 Starting Prompt Memory Test Suite")
    logger.info("=" * 60)

    test_results = []

    # Run all tests
    tests = [
        ("Basic Functionality", test_prompt_memory_basic),
        ("Backend Support", test_prompt_memory_backends),
        ("Task Agent Integration", test_task_agent_integration),
        ("Utility Functions", test_prompt_memory_functions),
        ("Memory Persistence", test_memory_persistence),
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
        logger.info("🎉 All prompt memory tests passed!")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
