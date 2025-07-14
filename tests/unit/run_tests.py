#!/usr/bin/env python3
"""
Test runner for Evolve system unit tests.

This script runs all unit tests and provides a comprehensive summary
of test results, coverage, and performance metrics.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime


def run_tests_with_pytest(test_dir="tests/unit", verbose=True, coverage=True):
    """
    Run all unit tests using pytest.

    Args:
        test_dir: Directory containing test files
        verbose: Whether to run in verbose mode
        coverage: Whether to generate coverage report

    Returns:
        dict: Test results summary
    """
    print("🧪 Running Evolve System Unit Tests")
    print("=" * 50)

    # Build pytest command
    cmd = ["python", "-m", "pytest", test_dir]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(
            [
                "--cov=trading",
                "--cov=utils",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-report=json:coverage.json",
            ]
        )

    # Add additional options
    cmd.extend(
        ["--tb=short", "--strict-markers", "--disable-warnings", "--durations=10"]
    )

    print(f"Command: {' '.join(cmd)}")
    print()

    # Run tests
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        end_time = time.time()

        # Parse results
        test_results = {
            "success": result.returncode == 0,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration": end_time - start_time,
        }

        return test_results

    except Exception as e:
        return {"success": False, "error": str(e), "duration": time.time() - start_time}


def parse_test_results(stdout):
    """
    Parse pytest output to extract test statistics.

    Args:
        stdout: Pytest stdout output

    Returns:
        dict: Parsed test statistics
    """
    stats = {
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "test_files": [],
    }

    lines = stdout.split("\n")

    for line in lines:
        line = line.strip()

        # Count test results
        if "passed" in line and "failed" in line and "skipped" in line:
            # Extract numbers from summary line
            import re

            numbers = re.findall(r"(\d+)", line)
            if len(numbers) >= 3:
                stats["passed"] = int(numbers[0])
                stats["failed"] = int(numbers[1])
                stats["skipped"] = int(numbers[2])
                stats["total_tests"] = (
                    stats["passed"] + stats["failed"] + stats["skipped"]
                )

        # Extract test file names
        if line.startswith("tests/unit/test_"):
            stats["test_files"].append(line.split("::")[0])

    return stats


def generate_test_summary(test_results, stats):
    """
    Generate a comprehensive test summary.

    Args:
        test_results: Raw test results
        stats: Parsed test statistics

    Returns:
        str: Formatted summary
    """
    summary = []
    summary.append("📊 Test Results Summary")
    summary.append("=" * 50)

    # Overall status
    status = "✅ PASSED" if test_results["success"] else "❌ FAILED"
    summary.append(f"Overall Status: {status}")
    summary.append(f"Duration: {test_results['duration']:.2f} seconds")
    summary.append("")

    # Test statistics
    summary.append("📈 Test Statistics:")
    summary.append(f"  Total Tests: {stats['total_tests']}")
    summary.append(f"  Passed: {stats['passed']} ✅")
    summary.append(f"  Failed: {stats['failed']} ❌")
    summary.append(f"  Skipped: {stats['skipped']} ⏭️")
    summary.append("")

    # Test files
    if stats["test_files"]:
        summary.append("📁 Test Files Executed:")
        for test_file in sorted(set(stats["test_files"])):
            summary.append(f"  • {test_file}")
        summary.append("")

    # Coverage information
    if "coverage.json" in os.listdir("."):
        try:
            with open("coverage.json", "r") as f:
                coverage_data = json.load(f)

            summary.append("📊 Coverage Summary:")
            for file_path, file_data in coverage_data["files"].items():
                if "trading/" in file_path or "utils/" in file_path:
                    coverage_percent = file_data["summary"]["percent_covered"]
                    summary.append(f"  {file_path}: {coverage_percent:.1f}%")
            summary.append("")
        except Exception as e:
            summary.append(f"⚠️  Could not parse coverage data: {e}")
            summary.append("")

    # Error details
    if not test_results["success"] and test_results["stderr"]:
        summary.append("❌ Errors/Warnings:")
        summary.append(test_results["stderr"])
        summary.append("")

    return "\n".join(summary)


def run_individual_test_suites():
    """
    Run individual test suites and report results.

    Returns:
        dict: Results for each test suite
    """
    test_suites = {
        "Forecasting Models": [
            "test_arima_forecaster.py",
            "test_xgboost_forecaster.py",
            "test_lstm_forecaster.py",
            "test_prophet_forecaster.py",
            "test_hybrid_forecaster.py",
        ],
        "Strategy Signals": [
            "test_rsi_signals.py",
            "test_macd_signals.py",
            "test_bollinger_signals.py",
        ],
        "Agents": ["test_prompt_agent.py"],
        "Backtesting": ["test_backtester.py"],
    }

    results = {}

    print("🔍 Running Individual Test Suites")
    print("=" * 50)

    for suite_name, test_files in test_suites.items():
        print(f"\n📋 {suite_name}:")

        suite_results = []
        for test_file in test_files:
            test_path = f"tests/unit/{test_file}"
            if os.path.exists(test_path):
                print(f"  Running {test_file}...", end=" ")

                cmd = ["python", "-m", "pytest", test_path, "-v", "--tb=no"]
                try:
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=60
                    )
                    success = result.returncode == 0
                    status = "✅ PASS" if success else "❌ FAIL"
                    print(status)

                    suite_results.append(
                        {
                            "file": test_file,
                            "success": success,
                            "output": result.stdout,
                            "error": result.stderr,
                        }
                    )
                except subprocess.TimeoutExpired:
                    print("⏰ TIMEOUT")
                    suite_results.append(
                        {
                            "file": test_file,
                            "success": False,
                            "error": "Test timed out after 60 seconds",
                        }
                    )
                except Exception as e:
                    print(f"❌ ERROR: {e}")
                    suite_results.append(
                        {"file": test_file, "success": False, "error": str(e)}
                    )
            else:
                print(f"  ⚠️  {test_file} not found")
                suite_results.append(
                    {"file": test_file, "success": False, "error": "File not found"}
                )

        results[suite_name] = suite_results

    return results


def generate_suite_summary(suite_results):
    """
    Generate summary for individual test suites.

    Args:
        suite_results: Results from individual test suites

    Returns:
        str: Formatted suite summary
    """
    summary = []
    summary.append("\n📊 Individual Test Suite Results")
    summary.append("=" * 50)

    for suite_name, tests in suite_results.items():
        passed = sum(1 for test in tests if test["success"])
        total = len(tests)
        status = "✅" if passed == total else "⚠️" if passed > 0 else "❌"

        summary.append(f"\n{status} {suite_name}: {passed}/{total} tests passed")

        for test in tests:
            test_status = "✅" if test["success"] else "❌"
            summary.append(f"  {test_status} {test['file']}")

            if not test["success"] and test.get("error"):
                summary.append(f"     Error: {test['error']}")

    return "\n".join(summary)


def main():
    """Main test runner function."""
    print("🚀 Evolve System Unit Test Runner")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Check if we're in the right directory
    if not os.path.exists("tests/unit"):
        print("❌ Error: tests/unit directory not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)

    # Run all tests
    test_results = run_tests_with_pytest()

    # Parse results
    stats = parse_test_results(test_results["stdout"])

    # Generate summary
    summary = generate_test_summary(test_results, stats)
    print(summary)

    # Run individual test suites
    suite_results = run_individual_test_suites()
    suite_summary = generate_suite_summary(suite_results)
    print(suite_summary)

    # Final status
    print("\n" + "=" * 50)
    if test_results["success"]:
        print("🎉 All tests completed successfully!")
    else:
        print("⚠️  Some tests failed. Please check the output above.")

    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Save results to file
    results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "test_results": test_results,
                "stats": stats,
                "suite_results": suite_results,
            },
            f,
            indent=2,
        )

    print(f"\n📄 Detailed results saved to: {results_file}")

    return 0 if test_results["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
