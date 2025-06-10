#!/usr/bin/env python3
"""
Test management script.
Provides commands for managing the application's testing process.
"""

import os
import sys
import argparse
import logging
import logging.config
import yaml
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

class TestManager:
    def __init__(self, config_path: str = "config/app_config.yaml"):
        """Initialize the test manager."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger("trading")
        self.test_dir = Path("tests")
        self.coverage_dir = Path("coverage")
        self.coverage_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        """Load application configuration."""
        if not Path(config_path).exists():
            print(f"Error: Configuration file not found: {config_path}")
            sys.exit(1)
        
        with open(config_path) as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        """Initialize logging configuration."""
        log_config_path = Path("config/logging_config.yaml")
        if not log_config_path.exists():
            print("Error: logging_config.yaml not found")
            sys.exit(1)
        
        with open(log_config_path) as f:
            log_config = yaml.safe_load(f)
        
        logging.config.dictConfig(log_config)

    def run_tests(self, test_type: Optional[str] = None, coverage: bool = False):
        """Run tests."""
        self.logger.info("Running tests...")
        
        try:
            # Build pytest command
            cmd = ["pytest"]
            
            if test_type:
                cmd.extend(["-m", test_type])
            
            if coverage:
                cmd.extend([
                    "--cov=trading",
                    "--cov-report=term-missing",
                    "--cov-report=html:coverage/html",
                    "--cov-report=xml:coverage/coverage.xml"
                ])
            
            # Run tests
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Print output
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
            
            if result.returncode == 0:
                self.logger.info("Tests completed successfully")
                return True
            else:
                self.logger.error("Tests failed")
                return False
        except Exception as e:
            self.logger.error(f"Failed to run tests: {e}")
            return False

    def run_linting(self):
        """Run code linting."""
        self.logger.info("Running code linting...")
        
        try:
            # Run flake8
            cmd = ["flake8", "trading", "tests"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.stdout:
                print(result.stdout)
            
            if result.returncode == 0:
                self.logger.info("Linting completed successfully")
                return True
            else:
                self.logger.error("Linting failed")
                return False
        except Exception as e:
            self.logger.error(f"Failed to run linting: {e}")
            return False

    def run_type_checking(self):
        """Run type checking."""
        self.logger.info("Running type checking...")
        
        try:
            # Run mypy
            cmd = ["mypy", "trading", "tests"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.stdout:
                print(result.stdout)
            
            if result.returncode == 0:
                self.logger.info("Type checking completed successfully")
                return True
            else:
                self.logger.error("Type checking failed")
                return False
        except Exception as e:
            self.logger.error(f"Failed to run type checking: {e}")
            return False

    def run_security_checks(self):
        """Run security checks."""
        self.logger.info("Running security checks...")
        
        try:
            # Run bandit
            cmd = ["bandit", "-r", "trading"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.stdout:
                print(result.stdout)
            
            if result.returncode == 0:
                self.logger.info("Security checks completed successfully")
                return True
            else:
                self.logger.error("Security checks failed")
                return False
        except Exception as e:
            self.logger.error(f"Failed to run security checks: {e}")
            return False

    def generate_test_report(self):
        """Generate test report."""
        self.logger.info("Generating test report...")
        
        try:
            # Load test results
            test_results = {
                "timestamp": datetime.now().isoformat(),
                "tests": {
                    "unit": self._run_test_type("unit"),
                    "integration": self._run_test_type("integration"),
                    "e2e": self._run_test_type("e2e")
                },
                "coverage": self._get_coverage(),
                "linting": self._run_linting(),
                "type_checking": self._run_type_checking(),
                "security": self._run_security_checks()
            }
            
            # Save report
            report_file = self.coverage_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, "w") as f:
                json.dump(test_results, f, indent=2)
            
            # Print summary
            print("\nTest Report Summary:")
            print(f"\nTimestamp: {test_results['timestamp']}")
            
            print("\nTest Results:")
            for test_type, result in test_results["tests"].items():
                status = "✓" if result["success"] else "✗"
                print(f"\n{status} {test_type.upper()}")
                print(f"  Passed: {result['passed']}")
                print(f"  Failed: {result['failed']}")
                print(f"  Skipped: {result['skipped']}")
            
            print("\nCoverage:")
            print(f"  Overall: {test_results['coverage']['overall']:.2f}%")
            for module, coverage in test_results["coverage"]["modules"].items():
                print(f"  {module}: {coverage:.2f}%")
            
            print("\nCode Quality:")
            print(f"  Linting: {'✓' if test_results['linting'] else '✗'}")
            print(f"  Type Checking: {'✓' if test_results['type_checking'] else '✗'}")
            print(f"  Security: {'✓' if test_results['security'] else '✗'}")
            
            self.logger.info(f"Test report generated: {report_file}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to generate test report: {e}")
            return False

    def _run_test_type(self, test_type: str) -> Dict[str, Any]:
        """Run tests of a specific type."""
        try:
            cmd = ["pytest", "-m", test_type, "--json-report"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                report = json.loads(result.stdout)
                return {
                    "success": True,
                    "passed": report["summary"]["passed"],
                    "failed": report["summary"]["failed"],
                    "skipped": report["summary"]["skipped"]
                }
            else:
                return {
                    "success": False,
                    "passed": 0,
                    "failed": 0,
                    "skipped": 0
                }
        except:
            return {
                "success": False,
                "passed": 0,
                "failed": 0,
                "skipped": 0
            }

    def _get_coverage(self) -> Dict[str, Any]:
        """Get coverage information."""
        try:
            cmd = ["pytest", "--cov=trading", "--cov-report=json"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                coverage = json.loads(result.stdout)
                return {
                    "overall": coverage["totals"]["percent_covered"],
                    "modules": {
                        module: data["percent_covered"]
                        for module, data in coverage["files"].items()
                    }
                }
            else:
                return {
                    "overall": 0.0,
                    "modules": {}
                }
        except:
            return {
                "overall": 0.0,
                "modules": {}
            }

    def _run_linting(self) -> bool:
        """Run linting and return success status."""
        try:
            cmd = ["flake8", "trading", "tests"]
            result = subprocess.run(cmd, capture_output=True)
            return result.returncode == 0
        except:
            return False

    def _run_type_checking(self) -> bool:
        """Run type checking and return success status."""
        try:
            cmd = ["mypy", "trading", "tests"]
            result = subprocess.run(cmd, capture_output=True)
            return result.returncode == 0
        except:
            return False

    def _run_security_checks(self) -> bool:
        """Run security checks and return success status."""
        try:
            cmd = ["bandit", "-r", "trading"]
            result = subprocess.run(cmd, capture_output=True)
            return result.returncode == 0
        except:
            return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test Manager")
    parser.add_argument(
        "command",
        choices=["test", "lint", "type-check", "security", "report"],
        help="Command to execute"
    )
    parser.add_argument(
        "--test-type",
        choices=["unit", "integration", "e2e"],
        help="Type of tests to run"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    
    args = parser.parse_args()
    manager = TestManager()
    
    commands = {
        "test": lambda: manager.run_tests(args.test_type, args.coverage),
        "lint": manager.run_linting,
        "type-check": manager.run_type_checking,
        "security": manager.run_security_checks,
        "report": manager.generate_test_report
    }
    
    if args.command in commands:
        success = commands[args.command]()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 