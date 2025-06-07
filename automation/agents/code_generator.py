import os
import logging
from typing import Dict, List, Optional, Tuple
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

class CodeGenerator:
    def __init__(self, config: Dict):
        """Initialize the code generator agent."""
        self.config = config
        self.model = config["openai"]["model"]
        self.temperature = config["openai"]["temperature"]
        self.max_tokens = config["openai"]["max_tokens"]
        
        # Load environment variables
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

    async def generate_code(self, task: Dict) -> Tuple[str, str]:
        """Generate code based on task requirements."""
        try:
            # Prepare the prompt
            prompt = self._create_prompt(task)
            
            # Generate code
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert Python developer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract code and tests
            code = response.choices[0].message.content
            
            # Generate tests
            tests = await self._generate_tests(code, task)
            
            return code, tests
            
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            raise

    async def refactor_code(self, code: str, requirements: List[str]) -> str:
        """Refactor existing code based on requirements."""
        try:
            # Prepare the prompt
            prompt = f"""Refactor the following code according to these requirements:
            {json.dumps(requirements, indent=2)}
            
            Code to refactor:
            {code}
            
            Provide the refactored code with explanations of the changes made."""
            
            # Generate refactored code
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert Python code refactoring specialist."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error refactoring code: {str(e)}")
            raise

    async def optimize_code(self, code: str) -> str:
        """Optimize code for performance and readability."""
        try:
            # Prepare the prompt
            prompt = f"""Optimize the following code for performance and readability:
            {code}
            
            Provide the optimized code with explanations of the optimizations made."""
            
            # Generate optimized code
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert Python code optimizer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error optimizing code: {str(e)}")
            raise

    async def _generate_tests(self, code: str, task: Dict) -> str:
        """Generate tests for the given code."""
        try:
            # Prepare the prompt
            prompt = f"""Generate comprehensive tests for the following code:
            {code}
            
            Requirements:
            {json.dumps(task.get('requirements', []), indent=2)}
            
            Include unit tests, integration tests, and edge cases."""
            
            # Generate tests
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in writing Python tests."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating tests: {str(e)}")
            raise

    def _create_prompt(self, task: Dict) -> str:
        """Create a prompt for code generation."""
        prompt = f"""Generate Python code for the following task:
        
        Type: {task['type']}
        Description: {task['description']}
        Priority: {task['priority']}
        Dependencies: {', '.join(task['dependencies']) if task['dependencies'] else 'None'}
        
        Requirements:
        {json.dumps(task['requirements'], indent=2)}
        
        Generate the code with:
        1. Proper error handling
        2. Comprehensive documentation
        3. Type hints
        4. Logging
        5. Performance considerations
        6. Security best practices
        
        Include comments explaining complex logic and design decisions."""
        
        return prompt

    async def generate_documentation(self, code: str) -> str:
        """Generate documentation for the given code."""
        try:
            # Prepare the prompt
            prompt = f"""Generate comprehensive documentation for the following code:
            {code}
            
            Include:
            1. Module/class/function documentation
            2. Usage examples
            3. Parameter descriptions
            4. Return value descriptions
            5. Exception descriptions
            6. Performance considerations
            7. Security considerations"""
            
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
            logger.error(f"Error generating documentation: {str(e)}")
            raise

    async def analyze_code_quality(self, code: str) -> Dict:
        """Analyze code quality and provide recommendations."""
        try:
            # Prepare the prompt
            prompt = f"""Analyze the quality of the following code:
            {code}
            
            Provide analysis on:
            1. Code complexity
            2. Maintainability
            3. Performance
            4. Security
            5. Best practices
            6. Potential improvements
            
            Format the response as a JSON object with these categories."""
            
            # Generate analysis
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert code reviewer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error analyzing code quality: {str(e)}")
            raise 