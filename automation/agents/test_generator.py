import os
import logging
from typing import Dict, List, Optional
import json
from pathlib import Path
import openai
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestGenerator:
    def __init__(self, config: Dict):
        """Initialize the test generator agent."""
        self.config = config
        self.model = config["openai"]["model"]
        self.temperature = config["openai"]["temperature"]
        self.max_tokens = config["openai"]["max_tokens"]
        
        # Load environment variables
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

    async def generate_tests(self, code: str, task: Dict) -> Dict[str, str]:
        """Generate comprehensive tests for the given code."""
        try:
            # Generate different types of tests
            unit_tests = await self._generate_unit_tests(code, task)
            integration_tests = await self._generate_integration_tests(code, task)
            performance_tests = await self._generate_performance_tests(code, task)
            security_tests = await self._generate_security_tests(code, task)
            
            return {
                "unit_tests": unit_tests,
                "integration_tests": integration_tests,
                "performance_tests": performance_tests,
                "security_tests": security_tests
            }
            
        except Exception as e:
            logger.error(f"Error generating tests: {str(e)}")
            raise

    async def _generate_unit_tests(self, code: str, task: Dict) -> str:
        """Generate unit tests."""
        try:
            # Prepare the prompt
            prompt = f"""Generate comprehensive unit tests for the following code:
            {code}
            
            Requirements:
            {json.dumps(task.get('requirements', []), indent=2)}
            
            Include:
            1. Test cases for all functions and methods
            2. Edge cases and boundary conditions
            3. Error cases and exception handling
            4. Mocking of external dependencies
            5. Test fixtures and setup/teardown
            6. Clear test names and descriptions
            
            Use pytest style and include proper assertions."""
            
            # Generate tests
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in writing Python unit tests."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating unit tests: {str(e)}")
            raise

    async def _generate_integration_tests(self, code: str, task: Dict) -> str:
        """Generate integration tests."""
        try:
            # Prepare the prompt
            prompt = f"""Generate integration tests for the following code:
            {code}
            
            Requirements:
            {json.dumps(task.get('requirements', []), indent=2)}
            
            Include:
            1. End-to-end test scenarios
            2. Component interaction tests
            3. Data flow tests
            4. Error propagation tests
            5. Configuration tests
            6. Environment setup/teardown
            
            Use pytest style and include proper assertions."""
            
            # Generate tests
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in writing Python integration tests."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating integration tests: {str(e)}")
            raise

    async def _generate_performance_tests(self, code: str, task: Dict) -> str:
        """Generate performance tests."""
        try:
            # Prepare the prompt
            prompt = f"""Generate performance tests for the following code:
            {code}
            
            Requirements:
            {json.dumps(task.get('requirements', []), indent=2)}
            
            Include:
            1. Load testing scenarios
            2. Stress testing scenarios
            3. Memory usage tests
            4. CPU usage tests
            5. Response time tests
            6. Resource utilization tests
            
            Use pytest-benchmark and include proper assertions."""
            
            # Generate tests
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in writing Python performance tests."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating performance tests: {str(e)}")
            raise

    async def _generate_security_tests(self, code: str, task: Dict) -> str:
        """Generate security tests."""
        try:
            # Prepare the prompt
            prompt = f"""Generate security tests for the following code:
            {code}
            
            Requirements:
            {json.dumps(task.get('requirements', []), indent=2)}
            
            Include:
            1. Input validation tests
            2. Authentication tests
            3. Authorization tests
            4. Data encryption tests
            5. SQL injection tests
            6. XSS tests
            7. CSRF tests
            
            Use pytest and include proper assertions."""
            
            # Generate tests
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in writing Python security tests."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating security tests: {str(e)}")
            raise

    async def generate_test_documentation(self, tests: Dict[str, str]) -> str:
        """Generate documentation for the tests."""
        try:
            # Prepare the prompt
            prompt = f"""Generate documentation for the following tests:
            {json.dumps(tests, indent=2)}
            
            Include:
            1. Test suite overview
            2. Test categories and purposes
            3. Test setup requirements
            4. Test execution instructions
            5. Expected results
            6. Troubleshooting guide"""
            
            # Generate documentation
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert technical writer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating test documentation: {str(e)}")
            raise

    async def analyze_test_coverage(self, code: str, tests: Dict[str, str]) -> Dict:
        """Analyze test coverage and provide recommendations."""
        try:
            # Prepare the prompt
            prompt = f"""Analyze the test coverage for the following code and tests:
            
            Code:
            {code}
            
            Tests:
            {json.dumps(tests, indent=2)}
            
            Provide analysis on:
            1. Code coverage
            2. Test quality
            3. Missing test cases
            4. Edge cases
            5. Performance considerations
            6. Security considerations
            
            Format the response as a JSON object with these categories."""
            
            # Generate analysis
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in test coverage analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error analyzing test coverage: {str(e)}")
            raise 