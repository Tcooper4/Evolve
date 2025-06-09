#!/usr/bin/env python3

import os
import sys
import yaml
import json
import time
import logging
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestRunner:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.results = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'started_at': None,
            'completed_at': None,
            'suites': []
        }

    def _load_config(self) -> Dict[str, Any]:
        """Load test configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            sys.exit(1)

    def _execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single test step."""
        try:
            response = requests.request(
                method=step['request']['method'],
                url=f"http://localhost:8000{step['request']['url']}",
                json=step['request'].get('body'),
                headers={'Content-Type': 'application/json'}
            )

            # Check status code
            if response.status_code != step['request']['expected_status']:
                return {
                    'status': 'failed',
                    'error': f"Expected status {step['request']['expected_status']}, got {response.status_code}"
                }

            # Check response body if specified
            if 'expected_body' in step['request']:
                expected = step['request']['expected_body']
                actual = response.json()
                if not self._compare_json(expected, actual):
                    return {
                        'status': 'failed',
                        'error': f"Response body mismatch. Expected: {expected}, Got: {actual}"
                    }

            return {'status': 'passed'}

        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _compare_json(self, expected: Dict, actual: Dict) -> bool:
        """Compare JSON objects recursively."""
        if isinstance(expected, dict) and isinstance(actual, dict):
            return all(
                self._compare_json(expected[k], actual[k])
                for k in expected
                if k in actual
            )
        return expected == actual

    def _execute_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a test case."""
        case_result = {
            'name': case['name'],
            'description': case['description'],
            'status': 'running',
            'steps': []
        }

        for step in case['steps']:
            step_result = self._execute_step(step)
            case_result['steps'].append({
                'name': step['name'],
                'status': step_result['status'],
                'error': step_result.get('error')
            })

            if step_result['status'] == 'failed':
                case_result['status'] = 'failed'
                break

        if case_result['status'] == 'running':
            case_result['status'] = 'passed'

        return case_result

    def _execute_suite(self, suite: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a test suite."""
        suite_result = {
            'name': suite['name'],
            'description': suite['description'],
            'status': 'running',
            'cases': []
        }

        if self.config['config']['parallel']:
            with ThreadPoolExecutor(max_workers=self.config['config']['max_parallel']) as executor:
                case_results = list(executor.map(self._execute_case, suite['cases']))
        else:
            case_results = [self._execute_case(case) for case in suite['cases']]

        suite_result['cases'] = case_results
        suite_result['status'] = 'passed' if all(c['status'] == 'passed' for c in case_results) else 'failed'

        return suite_result

    def run(self) -> Dict[str, Any]:
        """Run all test suites."""
        self.results['started_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        self.results['total'] = sum(len(suite['cases']) for suite in self.config['suites'])

        for suite in self.config['suites']:
            suite_result = self._execute_suite(suite)
            self.results['suites'].append(suite_result)

            # Update counts
            for case in suite_result['cases']:
                if case['status'] == 'passed':
                    self.results['passed'] += 1
                elif case['status'] == 'failed':
                    self.results['failed'] += 1
                else:
                    self.results['skipped'] += 1

        self.results['completed_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        return self.results

    def save_results(self, output_path: str):
        """Save test results to a file."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

def main():
    if len(sys.argv) != 3:
        print("Usage: python run_tests.py <config_path> <output_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    output_path = sys.argv[2]

    runner = TestRunner(config_path)
    results = runner.run()
    runner.save_results(output_path)

    # Print summary
    print("\nTest Results Summary:")
    print(f"Total: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Skipped: {results['skipped']}")
    print(f"Duration: {results['completed_at']} - {results['started_at']}")

    # Exit with appropriate status code
    sys.exit(1 if results['failed'] > 0 else 0)

if __name__ == '__main__':
    main() 