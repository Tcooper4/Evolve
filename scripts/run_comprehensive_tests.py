#!/usr/bin/env python3
"""
Comprehensive test runner for the trading system.

This script runs all tests with proper configuration for pandas-ta compatibility
and ensures all tests pass with meaningful validation.
"""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("test_run.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class ComprehensiveTestRunner:
    """Comprehensive test runner for the trading system."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_results = {}
        self.start_time = time.time()

    def setup_environment(self):
        """Setup the testing environment."""
        logger.info("Setting up testing environment...")

        # Add project root to Python path
        sys.path.insert(0, str(self.project_root))

        # Set environment variables
        os.environ["PYTHONPATH"] = str(self.project_root)
        os.environ["TESTING"] = "true"
        os.environ["MOCK_EXTERNAL_APIS"] = "true"

        logger.info("Environment setup complete")

    def install_dependencies(self):
        """Install required testing dependencies."""
        logger.info("Installing testing dependencies...")

        try:
            # Install pandas-ta and other required packages
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "pandas-ta>=0.3.0",
                    "pytest>=7.0.0",
                    "pytest-mock>=3.10.0",
                    "coverage>=7.0.0",
                    "pytest-cov>=3.0.0",
                ],
                check=True,
                capture_output=True,
            )

            logger.info("Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            raise

    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests."""
        logger.info("Running unit tests...")

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_strategies/",
            "tests/test_agents/",
            "tests/unit/",
            "-v",
            "--tb=short",
            "--maxfail=5",
            "--disable-warnings",
            "--cov=trading",
            "--cov-report=html",
            "--cov-report=term-missing",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except Exception as e:
            logger.error(f"Error running unit tests: {e}")
            return {"success": False, "error": str(e)}

    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        logger.info("Running integration tests...")

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_edge_cases.py",
            "tests/test_performance.py",
            "tests/test_real_world_scenario.py",
            "tests/test_task_integration.py",
            "tests/test_task_dashboard.py",
            "-v",
            "--tb=short",
            "--maxfail=3",
            "--disable-warnings",
            "-m",
            "integration",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except Exception as e:
            logger.error(f"Error running integration tests: {e}")
            return {"success": False, "error": str(e)}

    def run_strategy_tests(self) -> Dict[str, Any]:
        """Run strategy-specific tests."""
        logger.info("Running strategy tests...")

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_strategies/test_rsi.py",
            "tests/test_strategies/test_macd.py",
            "tests/test_strategies/test_bollinger.py",
            "tests/test_strategies/test_sma.py",
            "-v",
            "--tb=short",
            "--maxfail=2",
            "--disable-warnings",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except Exception as e:
            logger.error(f"Error running strategy tests: {e}")
            return {"success": False, "error": str(e)}

    def run_agent_tests(self) -> Dict[str, Any]:
        """Run agent-specific tests."""
        logger.info("Running agent tests...")

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_agents/test_llm_router.py",
            "tests/test_agents/test_self_improving_agent.py",
            "tests/test_router.py",
            "-v",
            "--tb=short",
            "--maxfail=2",
            "--disable-warnings",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except Exception as e:
            logger.error(f"Error running agent tests: {e}")
            return {"success": False, "error": str(e)}

    def run_smoke_tests(self) -> Dict[str, Any]:
        """Run smoke tests."""
        logger.info("Running smoke tests...")

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_app_smoke.py",
            "-v",
            "--tb=short",
            "--disable-warnings",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except Exception as e:
            logger.error(f"Error running smoke tests: {e}")
            return {"success": False, "error": str(e)}

    def validate_pandas_ta_compatibility(self) -> Dict[str, Any]:
        """Validate pandas-ta compatibility."""
        logger.info("Validating pandas-ta compatibility...")

        try:
            import numpy as np
            import pandas as pd
            import pandas_ta as ta

            # Test basic pandas-ta functionality
            test_data = pd.DataFrame({"Close": np.random.normal(100, 2, 100)})

            # Test RSI
            rsi = ta.rsi(test_data["Close"], length=14)
            assert isinstance(rsi, pd.Series)
            assert len(rsi) == len(test_data)

            # Test MACD
            macd = ta.macd(test_data["Close"])
            assert isinstance(macd, pd.DataFrame)
            assert "MACD_12_26_9" in macd.columns

            # Test Bollinger Bands
            bb = ta.bbands(test_data["Close"], length=20, std=2)
            assert isinstance(bb, pd.DataFrame)
            assert "BBL_20_2.0" in bb.columns

            # Test SMA
            sma = ta.sma(test_data["Close"], length=20)
            assert isinstance(sma, pd.Series)
            assert len(sma) == len(test_data)

            return {
                "success": True,
                "message": "pandas-ta compatibility validated successfully",
            }

        except Exception as e:
            logger.error(f"pandas-ta compatibility validation failed: {e}")
            return {"success": False, "error": str(e)}

    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        end_time = time.time()
        duration = end_time - self.start_time

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration,
            "test_results": self.test_results,
            "summary": {
                "total_tests": len(self.test_results),
                "passed_tests": sum(
                    1
                    for result in self.test_results.values()
                    if result.get("success", False)
                ),
                "failed_tests": sum(
                    1
                    for result in self.test_results.values()
                    if not result.get("success", False)
                ),
            },
        }

        # Save report
        report_file = self.project_root / "test_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        return report

    def run_all_tests(self) -> bool:
        """Run all tests and return overall success."""
        logger.info("Starting comprehensive test run...")

        try:
            # Setup environment
            self.setup_environment()

            # Install dependencies
            self.install_dependencies()

            # Validate pandas-ta compatibility
            self.test_results[
                "pandas_ta_compatibility"
            ] = self.validate_pandas_ta_compatibility()

            # Run different test categories
            self.test_results["smoke_tests"] = self.run_smoke_tests()
            self.test_results["strategy_tests"] = self.run_strategy_tests()
            self.test_results["agent_tests"] = self.run_agent_tests()
            self.test_results["unit_tests"] = self.run_unit_tests()
            self.test_results["integration_tests"] = self.run_integration_tests()

            # Generate report
            report = self.generate_test_report()

            # Log results
            logger.info(
                f"Test run completed in {report['duration_seconds']:.2f} seconds"
            )
            logger.info(f"Total tests: {report['summary']['total_tests']}")
            logger.info(f"Passed: {report['summary']['passed_tests']}")
            logger.info(f"Failed: {report['summary']['failed_tests']}")

            # Return overall success
            overall_success = report["summary"]["failed_tests"] == 0
            logger.info(
                f"Overall test result: {'PASSED' if overall_success else 'FAILED'}"
            )

            return overall_success

        except Exception as e:
            logger.error(f"Test run failed with error: {e}")
            return False


def main():
    """Main function."""
    runner = ComprehensiveTestRunner()
    success = runner.run_all_tests()

    if success:
        print("\n🎉 All tests passed successfully!")
        print("✅ pandas-ta compatibility verified")
        print("✅ Strategy tests completed")
        print("✅ Agent tests completed")
        print("✅ Integration tests completed")
        print("✅ Code coverage generated")
        print("\n📊 Check test_report.json for detailed results")
        print("📊 Check htmlcov/ for coverage report")
    else:
        print("\n❌ Some tests failed!")
        print("📊 Check test_report.json for detailed results")
        print("📋 Check test_run.log for detailed logs")
        sys.exit(1)


if __name__ == "__main__":
    main()
