"""
Local LLM Provider using Ollama
Enables offline operation without external APIs
"""
import os
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class LocalLLMProvider:
    """Local LLM provider using Ollama"""
    
    def __init__(self, model_name: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize local LLM provider
        
        Args:
            model_name: Model to use (default: from LOCAL_LLM_MODEL env var or 'llama3')
            base_url: Ollama server URL (default: from OLLAMA_HOST env var or 'http://localhost:11434')
        """
        self.model_name = model_name or os.getenv('LOCAL_LLM_MODEL', 'llama3')
        self.base_url = base_url or os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        self.client = None
        self.available = False
        
        # Test if Ollama is available
        self._check_availability()
    
    def _check_availability(self) -> None:
        """Check if Ollama is running and available"""
        try:
            import requests
            # Test if Ollama is running
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=2
            )
            if response.status_code == 200:
                self.available = True
                logger.info(f"Local LLM (Ollama) available at {self.base_url}")
            else:
                self.available = False
                logger.warning(f"Ollama not responding (status: {response.status_code})")
        except ImportError:
            self.available = False
            logger.warning("requests package not available. Install with: pip install requests")
        except Exception as e:
            self.available = False
            logger.warning(f"Ollama not available: {e}")
    
    def is_available(self) -> bool:
        """Check if local LLM is available"""
        return self.available
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Send chat completion request to local LLM
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name (overrides default)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Returns:
            Dict with 'content', 'model', and 'usage' keys
            
        Raises:
            RuntimeError: If Ollama is not available
        """
        if not self.is_available():
            raise RuntimeError(
                "Local LLM not available. "
                "Install Ollama: https://ollama.ai and ensure it's running. "
                f"Expected at: {self.base_url}"
            )
        
        try:
            import requests
        except ImportError:
            raise RuntimeError(
                "requests package not available. Install with: pip install requests"
            )
        
        # Format messages for Ollama
        prompt = self._format_messages(messages)
        model_to_use = model or self.model_name
        
        # Call Ollama API
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_to_use,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                },
                timeout=120
            )
            
            response.raise_for_status()
            result = response.json()
            
            return {
                'content': result.get('response', ''),
                'model': model_to_use,
                'usage': {
                    'prompt_tokens': result.get('prompt_eval_count', 0),
                    'completion_tokens': result.get('eval_count', 0),
                    'total_tokens': result.get('prompt_eval_count', 0) + result.get('eval_count', 0)
                }
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Local LLM request failed: {e}")
            raise RuntimeError(f"Failed to communicate with Ollama: {e}")
        except Exception as e:
            logger.error(f"Local LLM error: {e}")
            raise
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a single prompt for Ollama
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")  # Prompt for response
        return "\n\n".join(prompt_parts)
    
    def list_models(self) -> List[str]:
        """List available models from Ollama
        
        Returns:
            List of available model names
        """
        if not self.is_available():
            return []
        
        try:
            import requests
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            models = [model.get('name', '') for model in data.get('models', [])]
            return models
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

