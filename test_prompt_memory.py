#!/usr/bin/env python3
"""
Test script for Prompt Memory integration.

This script tests the prompt memory functionality and its integration
with the TaskAgent.
"""

import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_prompt_memory_basic():
    """Test basic prompt memory functionality."""
    logger.info("=" * 60)
    logger.info("PROMPT MEMORY BASIC TEST")
    logger.info("=" * 60)

    try:
        from memory.prompt_log import get_prompt_memory, log_prompt, get_last_prompt, get_prompt_history

        # Test basic logging
        logger.info("Testing basic prompt logging...")
        success = await log_prompt(
            prompt="Test prompt for memory",
            result={"status": "success", "data": "test result"},
            user_id="test_user",
            agent_type="TestAgent",
            execution_time=1.5,
            success=True
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
            logger.info(f"   Entry {i+1}: {entry.prompt[:50]}...")

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
        json_memory = PromptMemory(backend="json", file_path="memory/test_prompt_history.json")

        success = await json_memory.log_prompt(
            prompt="JSON backend test",
            result={"backend": "json", "test": True},
            user_id="json_user",
            agent_type="TestAgent"
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
        from memory.prompt_log import get_prompt_memory

        # Create TaskAgent (should initialize with prompt memory)
        logger.info("Creating TaskAgent...")
        agent = TaskAgent()

        # Execute a simple task
        logger.info("Executing test task...")
        result = await agent.execute_task(
            prompt="Test task for prompt memory integration",
            task_type=TaskType.MODEL_BUILD,
            parameters={"test": True, "model_type": "test"},
            max_depth=2,
            performance_threshold=0.5
        )

        logger.info(f"✅ Task execution result: {result.success}")
        logger.info(f"   Performance score: {result.performance_score:.3f}")

        # Check prompt memory for the task
        logger.info("\nChecking prompt memory for task...")
        memory = get_prompt_memory()
        task_history = await memory.get_prompt_history("task_agent", n=10)

        logger.info(f"✅ Task prompt history count: {len(task_history)}")

        # Show some entries
        for i, entry in enumerate(task_history[:3]):
            logger.info(f"   Entry {i+1}: {entry.prompt[:60]}...")
            logger.info(f"      Success: {entry.success}, Time: {entry.execution_time:.2f}s")

        # Get statistics
        stats = await memory.get_statistics("task_agent")
        logger.info(f"✅ Task agent stats: {stats}")

        return True

    except Exception as e:
        logger.error(f"❌ Task agent integration test failed: {e}")
        return False


async def test_prompt_memory_functions():
    """Test the convenience functions."""
    logger.info("\n" + "=" * 60)
    logger.info("PROMPT MEMORY FUNCTIONS TEST")
    logger.info("=" * 60)

    try:
        from memory.prompt_log import log_prompt, get_last_prompt, get_prompt_history

        # Test convenience functions
        logger.info("Testing convenience functions...")

        # Log multiple prompts
        for i in range(3):
            await log_prompt(
                prompt=f"Convenience test prompt {i+1}",
                result={"test_number": i+1, "function": "convenience"},
                user_id="convenience_user",
                agent_type="TestAgent",
                execution_time=0.1 * (i+1)
            )

        # Get last prompt
        last = await get_last_prompt("convenience_user")
        logger.info(f"✅ Last prompt: {last.prompt if last else 'None'}")

        # Get history
        history = await get_prompt_history("convenience_user", n=5)
        logger.info(f"✅ History count: {len(history)}")

        return True

    except Exception as e:
        logger.error(f"❌ Prompt memory functions test failed: {e}")
        return False


async def main():
    """Main test function."""
    logger.info("Starting Prompt Memory integration tests...")

    start_time = datetime.now()

    # Run all tests
    basic_success = await test_prompt_memory_basic()
    backend_success = await test_prompt_memory_backends()
    integration_success = await test_task_agent_integration()
    functions_success = await test_prompt_memory_functions()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Basic Functionality: {'✅ PASSED' if basic_success else '❌ FAILED'}")
    logger.info(f"Backend Support: {'✅ PASSED' if backend_success else '❌ FAILED'}")
    logger.info(f"TaskAgent Integration: {'✅ PASSED' if integration_success else '❌ FAILED'}")
    logger.info(f"Convenience Functions: {'✅ PASSED' if functions_success else '❌ FAILED'}")
    logger.info(f"Total Duration: {duration:.2f} seconds")

    all_passed = basic_success and backend_success and integration_success and functions_success
    logger.info(f"{'🎉 ALL TESTS PASSED!' if all_passed else '❌ SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)

