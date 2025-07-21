#!/usr/bin/env python3
"""Test script for Batch 12 imports."""

import logging
import sys

logger = logging.getLogger(__name__)


def test_import(module_name, class_name=None):
    """Test importing a module and optionally a class."""
    try:
        module = __import__(module_name, fromlist=[class_name] if class_name else [])
        if class_name:
            getattr(module, class_name)
        logger.info(f"✓ {module_name}{'.' + class_name if class_name else ''}")
        return True
    except Exception as e:
        logger.error(f"✗ {module_name}{'.' + class_name if class_name else ''}: {e}")
        return False


def main():
    """Test all Batch 12 imports."""
    logger.info("Testing Batch 12 imports...")
    logger.info("=" * 50)

    # Batch 12 modules to test with correct paths
    tests = [
        ("agents.prompt_agent", "PromptAgent"),
        ("trading.agents.agent_manager", "AgentManager"),
        ("agents.llm.model_loader", "ModelLoader"),
        ("agents.task_router", "TaskRouter"),
        ("trading.memory.persistent_memory", "PersistentMemory"),
    ]

    success_count = 0
    total_count = len(tests)

    for module_name, class_name in tests:
        if test_import(module_name, class_name):
            success_count += 1

    logger.info("=" * 50)
    logger.info(f"Results: {success_count}/{total_count} imports successful")

    if success_count == total_count:
        logger.info("🎉 All Batch 12 imports working!")
        return 0
    else:
        logger.error("❌ Some imports failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
