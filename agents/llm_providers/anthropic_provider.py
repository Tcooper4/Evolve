"""
Anthropic Claude API Provider
Provides Claude API integration for agent system
"""
import os
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class AnthropicProvider:
    """Anthropic Claude API provider for LLM operations"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Anthropic provider
        
        Args:
            api_key: Anthropic API key (if None, reads from ANTHROPIC_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.client = None
        
        if self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info("Anthropic provider initialized successfully")
            except ImportError:
                logger.error("anthropic package not installed. Run: pip install anthropic")
                self.client = None
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
                self.client = None
        else:
            logger.warning("Anthropic API key not provided. Claude features disabled.")
    
    def is_available(self) -> bool:
        """Check if Anthropic provider is available"""
        return self.client is not None
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4000,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send chat completion request to Claude
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Claude model to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)
            **kwargs: Additional API parameters
            
        Returns:
            Response dict with 'content' key containing the reply
        """
        if not self.is_available():
            raise RuntimeError("Anthropic provider not available. Check API key and installation.")
        
        try:
            # Convert messages to Anthropic format
            # Anthropic expects system message separate from messages
            system_message = None
            chat_messages = []
            
            for msg in messages:
                if msg['role'] == 'system':
                    system_message = msg['content']
                else:
                    chat_messages.append({
                        'role': msg['role'],
                        'content': msg['content']
                    })
            
            # Call Anthropic API
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_message,
                messages=chat_messages,
                **kwargs
            )
            
            # Format response to match OpenAI structure
            return {
                'content': response.content[0].text,
                'model': response.model,
                'usage': {
                    'prompt_tokens': response.usage.input_tokens,
                    'completion_tokens': response.usage.output_tokens,
                    'total_tokens': response.usage.input_tokens + response.usage.output_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

