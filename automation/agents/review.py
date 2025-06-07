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

class ReviewAgent:
    def __init__(self, config: Dict):
        """Initialize the review agent."""
        self.config = config
        self.model = config["openai"]["model"]
        self.temperature = config["openai"]["temperature"]
        self.max_tokens = config["openai"]["max_tokens"]
        
        # Load environment variables
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

    async def review_code(self, code: str, task: Dict) -> Dict:
        """Review code and provide feedback."""
        try:
            # Perform different types of reviews
            code_quality = await self._review_code_quality(code)
            security = await self._review_security(code)
            performance = await self._review_performance(code)
            maintainability = await self._review_maintainability(code)
            documentation = await self._review_documentation(code)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations({
                "code_quality": code_quality,
                "security": security,
                "performance": performance,
                "maintainability": maintainability,
                "documentation": documentation
            })
            
            return {
                "code_quality": code_quality,
                "security": security,
                "performance": performance,
                "maintainability": maintainability,
                "documentation": documentation,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error reviewing code: {str(e)}")
            raise

    async def _review_code_quality(self, code: str) -> Dict:
        """Review code quality."""
        try:
            # Prepare the prompt
            prompt = f"""Review the code quality of the following code:
            {code}
            
            Check for:
            1. Code style and formatting
            2. Naming conventions
            3. Code organization
            4. Error handling
            5. Type hints
            6. Code duplication
            7. Complexity
            8. Best practices
            
            Format the response as a JSON object with these categories."""
            
            # Generate review
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in code quality review."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error reviewing code quality: {str(e)}")
            raise

    async def _review_security(self, code: str) -> Dict:
        """Review code security."""
        try:
            # Prepare the prompt
            prompt = f"""Review the security of the following code:
            {code}
            
            Check for:
            1. Input validation
            2. Authentication
            3. Authorization
            4. Data encryption
            5. SQL injection
            6. XSS vulnerabilities
            7. CSRF vulnerabilities
            8. Secure communication
            9. Access control
            10. Common security issues
            
            Format the response as a JSON object with these categories."""
            
            # Generate review
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in code security review."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error reviewing security: {str(e)}")
            raise

    async def _review_performance(self, code: str) -> Dict:
        """Review code performance."""
        try:
            # Prepare the prompt
            prompt = f"""Review the performance of the following code:
            {code}
            
            Check for:
            1. Algorithm efficiency
            2. Memory usage
            3. CPU usage
            4. I/O operations
            5. Database queries
            6. Network calls
            7. Caching
            8. Concurrency
            9. Resource management
            10. Scalability
            
            Format the response as a JSON object with these categories."""
            
            # Generate review
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in code performance review."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error reviewing performance: {str(e)}")
            raise

    async def _review_maintainability(self, code: str) -> Dict:
        """Review code maintainability."""
        try:
            # Prepare the prompt
            prompt = f"""Review the maintainability of the following code:
            {code}
            
            Check for:
            1. Code organization
            2. Modularity
            3. Reusability
            4. Testability
            5. Readability
            6. Documentation
            7. Dependencies
            8. Configuration
            9. Error handling
            10. Logging
            
            Format the response as a JSON object with these categories."""
            
            # Generate review
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in code maintainability review."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error reviewing maintainability: {str(e)}")
            raise

    async def _review_documentation(self, code: str) -> Dict:
        """Review code documentation."""
        try:
            # Prepare the prompt
            prompt = f"""Review the documentation of the following code:
            {code}
            
            Check for:
            1. Module documentation
            2. Class documentation
            3. Function documentation
            4. Parameter documentation
            5. Return value documentation
            6. Exception documentation
            7. Usage examples
            8. Code comments
            9. README
            10. API documentation
            
            Format the response as a JSON object with these categories."""
            
            # Generate review
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in code documentation review."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error reviewing documentation: {str(e)}")
            raise

    async def _generate_recommendations(self, reviews: Dict) -> Dict:
        """Generate recommendations based on reviews."""
        try:
            # Prepare the prompt
            prompt = f"""Generate recommendations based on the following reviews:
            {json.dumps(reviews, indent=2)}
            
            Include:
            1. Priority improvements
            2. Quick wins
            3. Long-term improvements
            4. Best practices
            5. Security enhancements
            6. Performance optimizations
            7. Maintainability improvements
            8. Documentation improvements
            
            Format the response as a JSON object with these categories."""
            
            # Generate recommendations
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in code improvement recommendations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise

    async def generate_review_report(self, review: Dict) -> str:
        """Generate a comprehensive review report."""
        try:
            # Prepare the prompt
            prompt = f"""Generate a comprehensive review report based on the following review:
            {json.dumps(review, indent=2)}
            
            Include:
            1. Executive summary
            2. Detailed findings
            3. Recommendations
            4. Priority improvements
            5. Best practices
            6. Security considerations
            7. Performance considerations
            8. Maintainability considerations
            9. Documentation improvements
            10. Action items"""
            
            # Generate report
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
            logger.error(f"Error generating review report: {str(e)}")
            raise 