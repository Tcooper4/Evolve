#!/usr/bin/env python3
"""
Test script for configuration fallback system

This script tests the agent configuration fallback system to ensure
it works correctly when config files are missing.
"""

import asyncio
import logging
import shutil
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_agent_config_fallback():
    """Test agent config fallback functionality."""
    logger.info("Testing agent config fallback...")

    try:
        from agents.agent_config import AgentConfig

        # Test loading with missing file
        config = AgentConfig.load_from_file("nonexistent_config.json")

        if config:
            logger.info("✅ AgentConfig fallback works - loaded default config")
            return True
        else:
            logger.error("❌ AgentConfig fallback failed")
            return False

    except Exception as e:
        logger.error(f"❌ AgentConfig fallback test failed: {e}")
        return False


def test_agent_manager_fallback():
    """Test agent manager fallback functionality."""
    logger.info("Testing agent manager fallback...")

    try:
        from trading.agents.agent_manager import (
            AgentManagerConfig,
            EnhancedAgentManager,
        )

        # Create a temporary config path that doesn't exist
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_config_path = Path(temp_dir) / "nonexistent_config.json"

            # Create manager with non-existent config
            config = AgentManagerConfig(config_file=str(temp_config_path))
            manager = EnhancedAgentManager(config)

            # Check if agents were registered
            agents = manager.list_agents()

            if agents:
                logger.info(
                    f"✅ Agent manager fallback works - registered {len(agents)} agents"
                )
                for agent in agents:
                    logger.info(
                        f"  - {agent.get('name', 'Unknown')}: {agent.get('status', 'Unknown')}"
                    )
                return True
            else:
                logger.error("❌ Agent manager fallback failed - no agents registered")
                return False

    except Exception as e:
        logger.error(f"❌ Agent manager fallback test failed: {e}")
        return False


def test_missing_config_file():
    """Test behavior when config file is missing."""
    logger.info("Testing missing config file behavior...")

    try:
        # Backup original config file if it exists
        original_config = Path("trading/agents/agent_config.json")
        backup_config = Path("trading/agents/agent_config.json.backup")

        if original_config.exists():
            shutil.copy2(original_config, backup_config)
            logger.info("Backed up original config file")

        # Remove config file
        if original_config.exists():
            original_config.unlink()
            logger.info("Removed config file for testing")

        # Test agent manager initialization
        from trading.agents.agent_manager import EnhancedAgentManager

        manager = EnhancedAgentManager()
        agents = manager.list_agents()

        if agents:
            logger.info(
                f"✅ Missing config file handled correctly - {len(agents)} agents registered"
            )
            return True
        else:
            logger.error("❌ Missing config file not handled correctly")
            return False

    except Exception as e:
        logger.error(f"❌ Missing config file test failed: {e}")
        return False
    finally:
        # Restore original config file
        if backup_config.exists():
            shutil.copy2(backup_config, original_config)
            backup_config.unlink()
            logger.info("Restored original config file")


def test_fallback_agent_execution():
    """Test that fallback agents can execute requests."""
    logger.info("Testing fallback agent execution...")

    try:
        from trading.agents.agent_manager import EnhancedAgentManager
        from trading.agents.base_agent_interface import AgentResult

        manager = EnhancedAgentManager()

        # Try to execute with a fallback agent
        agents = manager.list_agents()
        if not agents:
            logger.error("❌ No agents available for execution test")
            return False

        # Find a fallback agent
        fallback_agent_name = None
        for agent in agents:
            agent_name = agent.get("name", "")
            if "fallback" in agent_name.lower() or "mock" in agent_name.lower():
                fallback_agent_name = agent_name
                break

        if not fallback_agent_name:
            # Use the first available agent
            fallback_agent_name = agents[0].get("name", "")

        if fallback_agent_name:
            # Execute the agent
            result = asyncio.run(
                manager.execute_agent_with_retry(
                    fallback_agent_name, test_request="Hello, this is a test"
                )
            )

            if isinstance(result, AgentResult) and result.success:
                logger.info(f"✅ Fallback agent execution successful: {result.message}")
                return True
            else:
                logger.error(f"❌ Fallback agent execution failed: {result}")
                return False
        else:
            logger.error("❌ No fallback agent found")
            return False

    except Exception as e:
        logger.error(f"❌ Fallback agent execution test failed: {e}")
        return False


def main():
    """Run all fallback tests."""
    logger.info("=" * 60)
    logger.info("Testing Configuration Fallback System")
    logger.info("=" * 60)

    tests = [
        ("Agent Config Fallback", test_agent_config_fallback),
        ("Agent Manager Fallback", test_agent_manager_fallback),
        ("Missing Config File", test_missing_config_file),
        ("Fallback Agent Execution", test_fallback_agent_execution),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! Fallback system is working correctly.")
        return True
    else:
        logger.error("⚠️ Some tests failed. Check the fallback system implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
