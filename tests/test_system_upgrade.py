#!/usr/bin/env python3
"""
System Upgrade Test Script

This script tests all the upgrades made to the Evolve trading system:
- Validates all methods return proper results
- Tests fallback logic
- Verifies UI integration
- Checks system health monitoring
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SystemUpgradeTester:
    """Test class for validating system upgrades."""

    def __init__(self):
        """Initialize the tester."""
        self.test_results = {}
        self.passed_tests = 0
        self.failed_tests = 0
        self.total_tests = 0

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all system upgrade tests."""
        logger.info("🚀 Starting System Upgrade Tests")

        # Test core modules
        self.test_core_modules()

        # Test agents
        self.test_agents()

        # Test data feed
        self.test_data_feed()

        # Test RL trader
        self.test_rl_trader()

        # Test UI integration
        self.test_ui_integration()

        # Test system health
        self.test_system_health()

        # Generate test report
        return self.generate_test_report()

    def test_core_modules(self):
        """Test core modules (AgentHub, CapabilityRouter)."""
        logger.info("Testing Core Modules...")

        # Test AgentHub
        try:
            from fallback.agent_hub import AgentHub

            agent_hub = AgentHub()

            # Test system health
            health = agent_hub.get_system_health()
            self.assert_test("AgentHub System Health", isinstance(health, dict) and "overall_status" in health)

            # Test agent status
            status = agent_hub.get_agent_status()
            self.assert_test("AgentHub Agent Status", isinstance(status, dict))

            # Test recent interactions
            interactions = agent_hub.get_recent_interactions()
            self.assert_test("AgentHub Recent Interactions", isinstance(interactions, list))

            logger.info("✅ AgentHub tests passed")

        except Exception as e:
            self.assert_test("AgentHub Import", False, str(e))

        # Test CapabilityRouter
        try:
            from fallback.capability_router import (
                get_capability_status,
                get_system_health,
            )

            # Test capability status
            capabilities = get_capability_status()
            self.assert_test("CapabilityRouter Status", isinstance(capabilities, dict))

            # Test system health
            health = get_system_health()
            self.assert_test("CapabilityRouter Health", isinstance(health, dict) and "overall_status" in health)

            logger.info("✅ CapabilityRouter tests passed")

        except Exception as e:
            self.assert_test("CapabilityRouter Import", False, str(e))

    def test_agents(self):
        """Test agent modules."""
        logger.info("Testing Agents...")

        # Test PromptRouterAgent
        try:
            from trading.agents.prompt_router_agent import PromptRouterAgent

            agent = PromptRouterAgent()

            # Test intent parsing
            result = agent.parse_intent("Forecast AAPL for next week")
            self.assert_test(
                "PromptRouter Intent Parsing",
                hasattr(result, "intent") and result.intent in ["forecasting", "forecast"],
            )

            # Test provider status
            providers = agent.get_available_providers()
            self.assert_test("PromptRouter Providers", isinstance(providers, list) and len(providers) > 0)

            # Test system health
            health = agent.get_system_health()
            self.assert_test("PromptRouter Health", isinstance(health, dict) and "overall_status" in health)

            logger.info("✅ PromptRouterAgent tests passed")

        except Exception as e:
            self.assert_test("PromptRouterAgent Import", False, str(e))

    def test_data_feed(self):
        """Test data feed module."""
        logger.info("Testing Data Feed...")

        try:
            from data.live_feed import get_data_feed

            data_feed = get_data_feed()

            # Test provider status
            status = data_feed.get_provider_status()
            self.assert_test("Data Feed Provider Status", isinstance(status, dict))

            # Test system health
            health = data_feed.get_system_health()
            self.assert_test("Data Feed Health", isinstance(health, dict) and "overall_status" in health)

            # Test fallback data generation
            fallback_data = data_feed._get_fallback_historical_data("AAPL", "2023-01-01", "2023-01-31")
            self.assert_test("Data Feed Fallback Historical", fallback_data is not None and not fallback_data.empty)

            fallback_live = data_feed._get_fallback_live_data("AAPL")
            self.assert_test("Data Feed Fallback Live", isinstance(fallback_live, dict) and "symbol" in fallback_live)

            logger.info("✅ Data Feed tests passed")

        except Exception as e:
            self.assert_test("Data Feed Import", False, str(e))

    def test_rl_trader(self):
        """Test RL trader module."""
        logger.info("Testing RL Trader...")

        try:
            from rl.rl_trader import get_rl_trader

            rl_trader = get_rl_trader()

            # Test model status
            status = rl_trader.get_model_status()
            self.assert_test("RL Trader Model Status", isinstance(status, dict))

            # Test system health
            health = rl_trader.get_system_health()
            self.assert_test("RL Trader Health", isinstance(health, dict) and "overall_status" in health)

            # Test training method returns proper result
            import pandas as pd

            dummy_data = pd.DataFrame(
                {
                    "Close": [100, 101, 102, 103, 104],
                    "Open": [99, 100, 101, 102, 103],
                    "High": [101, 102, 103, 104, 105],
                    "Low": [98, 99, 100, 101, 102],
                    "Volume": [1000000] * 5,
                }
            )

            train_result = rl_trader.train_model(dummy_data, total_timesteps=100)
            self.assert_test("RL Trader Training Result", isinstance(train_result, dict) and "success" in train_result)

            logger.info("✅ RL Trader tests passed")

        except Exception as e:
            self.assert_test("RL Trader Import", False, str(e))

    def test_ui_integration(self):
        """Test UI integration components."""
        logger.info("Testing UI Integration...")

        # Test unified interface
        try:
            from interface.unified_interface import UnifiedInterface

            interface = UnifiedInterface()

            # Test system health
            health = interface.get_system_health()
            self.assert_test("Unified Interface Health", isinstance(health, dict) and "overall_status" in health)

            # Test query processing
            result = interface.process_natural_language_query("Forecast AAPL")
            self.assert_test(
                "Unified Interface Query Processing", hasattr(result, "success") and hasattr(result, "data")
            )

            # Test report export
            report_file = interface.export_report("test")
            self.assert_test(
                "Unified Interface Report Export",
                isinstance(report_file, str) and report_file != "report_generation_failed",
            )

            logger.info("✅ Unified Interface tests passed")

        except Exception as e:
            self.assert_test("Unified Interface Import", False, str(e))

    def test_system_health(self):
        """Test overall system health monitoring."""
        logger.info("Testing System Health...")

        # Test that all components return proper health status
        health_checks = [
            ("AgentHub", "fallback.agent_hub", "AgentHub"),
            ("CapabilityRouter", "fallback.capability_router", "get_system_health"),
            ("DataFeed", "data.live_feed", "get_data_feed"),
            ("RLTrader", "rl.rl_trader", "get_rl_trader"),
            ("UnifiedInterface", "interface.unified_interface", "UnifiedInterface"),
        ]

        for name, module_path, component in health_checks:
            try:
                module = __import__(module_path, fromlist=[component])

                if hasattr(module, component):
                    obj = getattr(module, component)
                    if callable(obj):
                        instance = obj()
                    else:
                        instance = obj

                    if hasattr(instance, "get_system_health"):
                        health = instance.get_system_health()
                        self.assert_test(
                            f"{name} Health Method", isinstance(health, dict) and "overall_status" in health
                        )
                    else:
                        self.assert_test(f"{name} Health Method", False, "No get_system_health method")

            except Exception as e:
                self.assert_test(f"{name} Health Check", False, str(e))

    def assert_test(self, test_name: str, condition: bool, error_msg: str = None):
        """Assert a test condition and record the result."""
        self.total_tests += 1

        if condition:
            self.passed_tests += 1
            logger.info(f"✅ {test_name}: PASSED")
            self.test_results[test_name] = {"status": "PASSED", "error": None}
        else:
            self.failed_tests += 1
            logger.error(f"❌ {test_name}: FAILED - {error_msg}")
            self.test_results[test_name] = {"status": "FAILED", "error": error_msg}

    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": self.total_tests,
                "passed_tests": self.passed_tests,
                "failed_tests": self.failed_tests,
                "success_rate": (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0,
            },
            "test_results": self.test_results,
            "upgrade_status": "COMPLETE" if self.failed_tests == 0 else "PARTIAL",
        }

        # Save report
        report_file = f"test_reports/system_upgrade_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        Path(report_file).parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"📋 Test report saved to: {report_file}")
        return report


def main():
    """Main function to run system upgrade tests."""
    print("🚀 Evolve Trading System - Upgrade Test Suite")
    print("=" * 50)

    # Create tester
    tester = SystemUpgradeTester()

    # Run all tests
    report = tester.run_all_tests()

    # Display results
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)

    summary = report["summary"]
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Upgrade Status: {report['upgrade_status']}")

    # Display failed tests
    if summary["failed_tests"] > 0:
        print("\n❌ FAILED TESTS:")
        for test_name, result in report["test_results"].items():
            if result["status"] == "FAILED":
                print(f"  - {test_name}: {result['error']}")

    # Display passed tests
    if summary["passed_tests"] > 0:
        print(f"\n✅ PASSED TESTS: {summary['passed_tests']}")

    print("\n" + "=" * 50)

    if report["upgrade_status"] == "COMPLETE":
        print("🎉 SYSTEM UPGRADE COMPLETED SUCCESSFULLY!")
        print("All components now return proper results and have fallback logic.")
    else:
        print("⚠️ SYSTEM UPGRADE PARTIALLY COMPLETED")
        print("Some components may need additional attention.")

    return report


if __name__ == "__main__":
    main()
