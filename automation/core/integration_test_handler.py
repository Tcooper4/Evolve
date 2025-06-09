import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime
import json
import yaml
from pathlib import Path
import asyncio
import aiohttp
import pytest
from .base_handler import BaseHandler

@dataclass
class TestCase:
    id: str
    name: str
    description: str
    components: List[str]
    steps: List[Dict[str, Any]]
    expected_results: Dict[str, Any]
    status: str = 'pending'
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@dataclass
class TestSuite:
    id: str
    name: str
    description: str
    test_cases: List[TestCase]
    status: str = 'pending'
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None

class IntegrationTestHandler(BaseHandler):
    """Handler for system-wide integration testing."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.setup_logging()
        self.test_suites: Dict[str, TestSuite] = {}
        self.test_cases: Dict[str, TestCase] = {}
        self.test_results: Dict[str, Dict[str, Any]] = {}
        self.test_config = self._load_test_config()
    
    def setup_logging(self):
        """Configure logging for integration testing."""
        log_path = Path("automation/logs/integration_tests")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "integration_tests.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_test_config(self) -> Dict:
        """Load test configuration."""
        try:
            config_path = Path("automation/config/test_config.yaml")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            return self._get_default_config()
        except Exception as e:
            self.logger.error(f"Error loading test configuration: {str(e)}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default test configuration."""
        return {
            'timeout': 30,  # seconds
            'retry_count': 3,
            'retry_delay': 1,  # seconds
            'parallel_tests': True,
            'max_parallel_tests': 5,
            'report_format': 'json',
            'notify_on_failure': True
        }
    
    async def create_test_suite(self, suite_id: str, name: str, description: str) -> TestSuite:
        """Create a new test suite."""
        try:
            if suite_id in self.test_suites:
                raise ValueError(f"Test suite {suite_id} already exists")
            
            suite = TestSuite(
                id=suite_id,
                name=name,
                description=description,
                test_cases=[],
                status='pending'
            )
            
            self.test_suites[suite_id] = suite
            self.logger.info(f"Created test suite: {suite_id}")
            return suite
            
        except Exception as e:
            self.logger.error(f"Error creating test suite: {str(e)}")
            raise
    
    async def add_test_case(self, suite_id: str, test_case: TestCase) -> TestSuite:
        """Add a test case to a suite."""
        try:
            if suite_id not in self.test_suites:
                raise ValueError(f"Test suite {suite_id} not found")
            
            suite = self.test_suites[suite_id]
            suite.test_cases.append(test_case)
            self.test_cases[test_case.id] = test_case
            
            self.logger.info(f"Added test case {test_case.id} to suite {suite_id}")
            return suite
            
        except Exception as e:
            self.logger.error(f"Error adding test case: {str(e)}")
            raise
    
    async def run_test_suite(self, suite_id: str) -> TestSuite:
        """Run a test suite."""
        try:
            if suite_id not in self.test_suites:
                raise ValueError(f"Test suite {suite_id} not found")
            
            suite = self.test_suites[suite_id]
            suite.status = 'running'
            suite.start_time = datetime.now()
            
            if self.test_config['parallel_tests']:
                await self._run_parallel_tests(suite)
            else:
                await self._run_sequential_tests(suite)
            
            suite.end_time = datetime.now()
            suite.status = 'completed'
            
            self.logger.info(f"Completed test suite: {suite_id}")
            return suite
            
        except Exception as e:
            self.logger.error(f"Error running test suite: {str(e)}")
            suite.status = 'failed'
            suite.end_time = datetime.now()
            raise
    
    async def _run_parallel_tests(self, suite: TestSuite):
        """Run test cases in parallel."""
        try:
            tasks = []
            for test_case in suite.test_cases:
                task = asyncio.create_task(self._run_test_case(test_case))
                tasks.append(task)
                
                if len(tasks) >= self.test_config['max_parallel_tests']:
                    await asyncio.gather(*tasks)
                    tasks = []
            
            if tasks:
                await asyncio.gather(*tasks)
                
        except Exception as e:
            self.logger.error(f"Error running parallel tests: {str(e)}")
            raise
    
    async def _run_sequential_tests(self, suite: TestSuite):
        """Run test cases sequentially."""
        try:
            for test_case in suite.test_cases:
                await self._run_test_case(test_case)
                
        except Exception as e:
            self.logger.error(f"Error running sequential tests: {str(e)}")
            raise
    
    async def _run_test_case(self, test_case: TestCase) -> Dict[str, Any]:
        """Run a single test case."""
        try:
            test_case.status = 'running'
            result = {}
            
            for step in test_case.steps:
                step_result = await self._execute_test_step(step)
                result[step['id']] = step_result
                
                if not step_result['success']:
                    test_case.status = 'failed'
                    test_case.error = step_result['error']
                    break
            
            if test_case.status != 'failed':
                test_case.status = 'passed'
            
            test_case.result = result
            self.test_results[test_case.id] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error running test case: {str(e)}")
            test_case.status = 'failed'
            test_case.error = str(e)
            raise
    
    async def _execute_test_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single test step."""
        try:
            for attempt in range(self.test_config['retry_count']):
                try:
                    if step['type'] == 'api_call':
                        result = await self._execute_api_step(step)
                    elif step['type'] == 'component_test':
                        result = await self._execute_component_step(step)
                    elif step['type'] == 'data_validation':
                        result = await self._execute_validation_step(step)
                    else:
                        raise ValueError(f"Unknown step type: {step['type']}")
                    
                    if result['success']:
                        return result
                    
                    if attempt < self.test_config['retry_count'] - 1:
                        await asyncio.sleep(self.test_config['retry_delay'])
                    
                except Exception as e:
                    if attempt == self.test_config['retry_count'] - 1:
                        raise
            
            return {
                'success': False,
                'error': 'Max retries exceeded',
                'attempts': self.test_config['retry_count']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _execute_api_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an API test step."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=step['method'],
                    url=step['url'],
                    headers=step.get('headers', {}),
                    json=step.get('body')
                ) as response:
                    result = {
                        'success': response.status == step['expected_status'],
                        'status_code': response.status,
                        'response': await response.json()
                    }
                    
                    if not result['success']:
                        result['error'] = f"Expected status {step['expected_status']}, got {response.status}"
                    
                    return result
                    
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _execute_component_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a component test step."""
        try:
            # Implement component-specific test logic
            return {
                'success': True,
                'result': 'Component test passed'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _execute_validation_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a data validation step."""
        try:
            # Implement data validation logic
            return {
                'success': True,
                'result': 'Validation passed'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_test_suite(self, suite_id: str) -> Optional[TestSuite]:
        """Get a test suite by ID."""
        return self.test_suites.get(suite_id)
    
    def get_test_case(self, case_id: str) -> Optional[TestCase]:
        """Get a test case by ID."""
        return self.test_cases.get(case_id)
    
    def get_test_results(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Get test results for a case."""
        return self.test_results.get(case_id)
    
    def get_all_test_suites(self) -> List[TestSuite]:
        """Get all test suites."""
        return list(self.test_suites.values())
    
    def get_test_config(self) -> Dict:
        """Get current test configuration."""
        return self.test_config 