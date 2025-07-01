"""Code generation agent using OpenAI's GPT model."""

import logging
import json
import time
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from trading.base_agent import BaseMetaAgent, Task

class CodeGeneratorAgent(BaseMetaAgent):
    """Agent for generating Python code and tests using OpenAI's GPT model."""
    
    def __init__(
        self,
        name: str = "code_generator",
        config: Optional[Dict] = None,
        log_file_path: Optional[Union[str, Path]] = None):
        """Initialize the code generator agent.
        
        Args:
            name: Name of the agent
            config: Configuration dictionary
            log_file_path: Optional custom path for log file
        """
        super().__init__(name, config, log_file_path)
        
        # Default configuration
        self.default_config = {
            "model": "gpt-4-turbo-preview",
            "temperature": 0.7,
            "max_tokens": 2000,
            "system_role": "You are an expert Python developer. Generate clean, well-documented, and testable code.",
            "style_guide": {
                "docstring_format": "google",
                "max_line_length": 88,
                "type_hints": True
            },
            "retry_attempts": 3,
            "retry_delay": 1.0
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **(config or {})}
        
        # Setup OpenAI client
        self._setup_openai()
        
        self.logger.info("Code generator agent initialized")
    
    def _setup_openai(self) -> None:
        """Setup OpenAI client with API key from environment or config."""
        api_key = self.config.get("api_key") or openai.api_key
        if not api_key:
            raise ValueError("OpenAI API key not found in config or environment")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.logger.info(f"OpenAI client initialized with model: {self.config['model']}")
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def _validate_task(self, task: Task) -> None:
        """Validate task dictionary structure.
        
        Args:
            task: Task to validate
            
        Raises:
            ValueError: If task is invalid
        """
        if not isinstance(task.data, dict):
            raise ValueError("Task data must be a dictionary")
            
        required_fields = ["description", "requirements"]
        if not any(field in task.data for field in required_fields):
            raise ValueError(f"Task must include at least one of: {required_fields}")

    def _build_prompt(self, task: Task) -> List[Dict[str, str]]:
        """Build the prompt for code generation.
        
        Args:
            task: Task containing generation requirements
            
        Returns:
            List of message dictionaries for the chat completion
        """
        messages = [
            {"role": "system", "content": self.config["system_role"]}
        ]
        
        # Add style guide if provided
        if "style_guide" in self.config:
            style_guide = json.dumps(self.config["style_guide"], indent=2)
            messages.append({
                "role": "system",
                "content": f"Follow these style guidelines:\n{style_guide}"
            })
        
        # Build task-specific prompt
        prompt = []
        if "description" in task.data:
            prompt.append(f"Description: {task.data['description']}")
        if "requirements" in task.data:
            prompt.append(f"Requirements:\n{task.data['requirements']}")
        if "context" in task.data:
            prompt.append(f"Context:\n{task.data['context']}")
            
        messages.append({
            "role": "user",
            "content": "\n".join(prompt)
        })
        
        return messages
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _call_openai(self, messages: List[Dict[str, str]], stream: bool = False) -> Any:
        """Call OpenAI API with retry logic.
        
        Args:
            messages: List of message dictionaries
            stream: Whether to stream the response
            
        Returns:
            OpenAI API response
            
        Raises:
            openai.OpenAIError: If API call fails
        """
        try:
            response = self.client.chat.completions.create(
                model=self.config["model"],
                messages=messages,
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"],
                stream=stream
            )
            return response
        except openai.RateLimitError:
            self.logger.warning("Rate limit hit, retrying...")
            raise
        except openai.APIError as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    def process_task(self, task: Task) -> Dict[str, Any]:
        """Process a code generation task.
        
        Args:
            task: Task containing generation requirements
            
        Returns:
            Dict containing generated code and metadata
        """
        self._validate_task(task)
        
        # Build and log prompt
        messages = self._build_prompt(task)
        self.logger.debug(f"Prompt: {json.dumps(messages, indent=2)}")
        
        # Call OpenAI API
        start_time = time.time()
        response = self._call_openai(messages)
        duration = time.time() - start_time
        
        # Extract and log response
        content = response.choices[0].message.content
        self.logger.debug(f"Raw response (truncated): {content[:200]}...")
        
        # Parse response into code and tests
        try:
            code, tests = self._parse_response(content)
        except Exception as e:
            self.logger.error(f"Failed to parse response: {str(e)}")
            raise
        
        # Calculate confidence score
        confidence = self._calculate_confidence(response)
        
        return {
            "code": code,
            "tests": tests,
            "metadata": {
                "model": self.config["model"],
                "duration": duration,
                "confidence": confidence,
                "usage": response.usage._asdict() if hasattr(response, "usage") else None
            }
        }
    
    def _parse_response(self, content: str) -> tuple[str, str]:
        """Parse the model response into code and tests.
        
        Args:
            content: Raw response from the model
            
        Returns:
            Tuple of (code, tests)
        """
        # Split on test markers
        parts = content.split("```python")
        if len(parts) < 2:
            raise ValueError("Response does not contain code blocks")
            
        # Extract code and tests
        code = parts[1].split("```")[0].strip()
        tests = parts[2].split("```")[0].strip() if len(parts) > 2 else ""
        
        return {'success': True, 'result': code, tests, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _calculate_confidence(self, response: Any) -> float:
        """Calculate confidence score for the generation.
        
        Args:
            response: OpenAI API response
            
        Returns:
            Confidence score between 0 and 1
        """
        # Use logprobs if available
        if hasattr(response.choices[0], "logprobs"):
            return float(response.choices[0].logprobs.token_logprobs[0])
            
        # Fallback to content length heuristic
        content_length = len(response.choices[0].message.content)
        return min(1.0, content_length / 1000.0)
    
    def refactor_code(
        self,
        code: str,
        style_guide: Optional[Dict] = None,
        generate_diff: bool = False
    ) -> Dict[str, Any]:
        """Refactor code using chain-of-thought prompting.
        
        Args:
            code: Code to refactor
            style_guide: Optional style guide override
            generate_diff: Whether to generate diff output
            
        Returns:
            Dict containing refactored code and metadata
        """
        # Build review prompt
        review_messages = [
            {"role": "system", "content": "You are a code reviewer. Analyze the code and suggest improvements."},
            {"role": "user", "content": f"Review this code:\n```python\n{code}\n```"}
        ]
        
        # Get review
        review = self._call_openai(review_messages)
        self.logger.debug(f"Code review: {review.choices[0].message.content}")
        
        # Build fix prompt
        fix_messages = [
            {"role": "system", "content": "You are a code refactoring expert. Apply the suggested improvements."},
            {"role": "user", "content": f"Refactor this code based on the review:\n```python\n{code}\n```\nReview:\n{review.choices[0].message.content}"}
        ]
        
        # Get refactored code
        refactored = self._call_openai(fix_messages)
        refactored_code = refactored.choices[0].message.content
        
        # Generate diff if requested
        diff = None
        if generate_diff:
            diff_messages = [
                {"role": "system", "content": "You are a diff generator. Show only the changed lines."},
                {"role": "user", "content": f"Generate a diff between:\nOriginal:\n```python\n{code}\n```\nRefactored:\n```python\n{refactored_code}\n```"}
            ]
            diff_response = self._call_openai(diff_messages)
            diff = diff_response.choices[0].message.content
        
        return {
            "original_code": code,
            "refactored_code": refactored_code,
            "review": review.choices[0].message.content,
            "diff": diff,
            "metadata": {
                "model": self.config["model"],
                "style_guide": style_guide or self.config["style_guide"]
            }
        }
    
    def simulate(self, prompt: str) -> Dict[str, Any]:
        """Simulate code generation for testing.
        
        Args:
            prompt: Hardcoded prompt to use
            
        Returns:
            Dict containing mock generation results
        """
        return {
            "code": "# Simulated code\ndef hello():\n    return 'Hello, World!'",
            "tests": "# Simulated tests\ndef test_hello():\n    assert hello() == 'Hello, World!'",
            "metadata": {
                "model": self.config["model"],
                "duration": 0.1,
                "confidence": 0.9,
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            }
        } 