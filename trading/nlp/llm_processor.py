"""LLM processor for natural language interface."""

import os
import json
import logging
from typing import Dict, Any, Optional, Generator, List
import openai
from openai import OpenAI

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add file handler for debug logs
debug_handler = logging.FileHandler('trading/nlp/logs/nlp_debug.log')
debug_handler.setLevel(logging.DEBUG)
debug_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
debug_handler.setFormatter(debug_formatter)
logger.addHandler(debug_handler)

class LLMProcessor:
    """Processes prompts using LLM and handles streaming responses."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LLM processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Load moderation categories
        self.moderation_categories = {
            'hate': True,
            'hate/threatening': True,
            'harassment': True,
            'harassment/threatening': True,
            'self-harm': True,
            'self-harm/intent': True,
            'self-harm/instructions': True,
            'sexual': True,
            'sexual/minors': True,
            'violence': True,
            'violence/graphic': True
        }
        
        logger.info("LLMProcessor initialized with moderation categories")
    
    def process(self, prompt: str) -> str:
        """Process a prompt and get response.
        
        Args:
            prompt: Input prompt string
            
        Returns:
            Response string
        """
        try:
            # Check prompt for unsafe content
            if self.is_unsafe_content(prompt):
                logger.warning("Unsafe content detected in prompt")
                raise ValueError("Prompt contains unsafe content")
            
            # Get response from LLM
            response = self.client.chat.completions.create(
                model=self.config.get('model', 'gpt-4'),
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.get('temperature', 0.7),
                max_tokens=self.config.get('max_tokens', 1000)
            )
            
            # Extract and validate response
            content = response.choices[0].message.content
            
            # Check response for unsafe content
            if self.is_unsafe_content(content):
                logger.warning("Unsafe content detected in response")
                raise ValueError("Response contains unsafe content")
            
            return content
            
        except Exception as e:
            logger.error(f"Error processing prompt: {str(e)}", exc_info=True)
            raise
    
    def process_stream(self, prompt: str) -> Generator[str, None, None]:
        """Process a prompt and stream the response.
        
        Args:
            prompt: Input prompt string
            
        Yields:
            Response chunks
        """
        try:
            # Check prompt for unsafe content
            if self.is_unsafe_content(prompt):
                logger.warning("Unsafe content detected in prompt")
                raise ValueError("Prompt contains unsafe content")
            
            # Get streaming response from LLM
            stream = self.client.chat.completions.create(
                model=self.config.get('model', 'gpt-4'),
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.get('temperature', 0.7),
                max_tokens=self.config.get('max_tokens', 1000),
                stream=True
            )
            
            # Process stream
            buffer = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    buffer += content
                    
                    # Check buffer for unsafe content
                    if self.is_unsafe_content(buffer):
                        logger.warning("Unsafe content detected in stream")
                        raise ValueError("Stream contains unsafe content")
                    
                    yield content
            
        except Exception as e:
            logger.error(f"Error processing stream: {str(e)}", exc_info=True)
            raise
    
    def is_unsafe_content(self, content: str) -> bool:
        """Check if content contains unsafe material.
        
        Args:
            content: Content to check
            
        Returns:
            True if content is unsafe, False otherwise
        """
        try:
            # Get moderation results
            response = self.client.moderations.create(input=content)
            results = response.results[0]
            
            # Check each category
            for category, enabled in self.moderation_categories.items():
                if enabled and getattr(results.categories, category):
                    logger.warning(f"Unsafe content detected in category: {category}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking content safety: {str(e)}", exc_info=True)
            return True  # Fail safe
    
    def validate_json_response(self, response: str) -> Dict[str, Any]:
        """Validate and parse JSON response.
        
        Args:
            response: Response string
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            ValueError: If response is not valid JSON
        """
        try:
            # Try to parse JSON
            data = json.loads(response)
            
            # Validate required fields
            required_fields = ['response', 'confidence']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")
            
            # Validate confidence
            if not isinstance(data['confidence'], (int, float)) or not 0 <= data['confidence'] <= 1:
                raise ValueError("Confidence must be a float between 0 and 1")
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {str(e)}")
            raise ValueError(f"Invalid JSON response: {str(e)}")
        except Exception as e:
            logger.error(f"Error validating response: {str(e)}")
            raise 