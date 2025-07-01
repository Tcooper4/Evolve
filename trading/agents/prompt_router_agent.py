"""
PromptRouterAgent: Smart prompt router for agent orchestration.
- Detects user intent (forecasting, backtesting, tuning, research)
- Parses arguments using OpenAI, HuggingFace, or regex fallback
- Routes to the correct agent automatically
- Always returns a usable parsed intent
"""

import re
import json
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime
import os

try:
    import openai
except ImportError:
    openai = None

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

from .base_agent_interface import BaseAgent, AgentConfig, AgentResult

logger = logging.getLogger(__name__)

@dataclass
class ParsedIntent:
    """Structured parsed intent result."""
    intent: str
    confidence: float
    args: Dict[str, Any]
    provider: str  # 'openai', 'huggingface', 'regex'
    raw_response: str
    error: Optional[str] = None

class PromptRouterAgent(BaseAgent):
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the prompt router agent.
        
        Args:
            config: Agent configuration
        """
        if config is None:
            config = AgentConfig(
                name="prompt_router",
                enabled=True,
                priority=1,
                custom_config={
                    'openai_api_key': os.getenv('OPENAI_API_KEY'),
                    'huggingface_model': os.getenv('HUGGINGFACE_MODEL', 'gpt2'),
                    'huggingface_api_key': os.getenv('HUGGINGFACE_API_KEY')
                }
            )
        
        super().__init__(config)
        
        # Get configuration
        self.openai_api_key = self.config.custom_config.get('openai_api_key')
        self.huggingface_model = self.config.custom_config.get('huggingface_model', 'gpt2')
        self.huggingface_api_key = self.config.custom_config.get('huggingface_api_key')
        self.hf_pipeline = None
        self.parsing_history = []
        
        # Initialize OpenAI if available
        if openai and self.openai_api_key:
            openai.api_key = self.openai_api_key
            logger.info("✅ OpenAI initialized for prompt routing")
        
        # Initialize HuggingFace if available
        if HUGGINGFACE_AVAILABLE:
            try:
                self._init_huggingface()
                logger.info("✅ HuggingFace initialized for prompt routing")
            except Exception as e:
                logger.warning(f"⚠️ HuggingFace initialization failed: {e}")
        
        # Intent keywords for regex fallback
        self.intent_keywords = {
            'forecasting': ['forecast', 'predict', 'projection', 'future', 'price', 'trend'],
            'backtesting': ['backtest', 'historical', 'simulate', 'performance', 'past', 'test'],
            'tuning': ['tune', 'optimize', 'hyperparameter', 'search', 'bayesian', 'parameter'],
            'research': ['research', 'find', 'paper', 'github', 'arxiv', 'summarize', 'analyze'],
            'portfolio': ['portfolio', 'position', 'holdings', 'allocation', 'balance'],
            'risk': ['risk', 'volatility', 'drawdown', 'var', 'sharpe'],
            'sentiment': ['sentiment', 'news', 'social', 'twitter', 'reddit', 'emotion']
        }
        
        # Argument extraction patterns
        self.arg_patterns = {
            'symbol': r'\b([A-Z]{1,5})\b',
            'date': r'\b(\d{4}-\d{2}-\d{2})\b',
            'number': r'\b(\d+(?:\.\d+)?)\b',
            'percentage': r'\b(\d+(?:\.\d+)?%)\b',
            'timeframe': r'\b(daily|weekly|monthly|yearly|1d|1w|1m|1y)\b',
            'model': r'\b(lstm|transformer|xgboost|prophet|arima|ensemble)\b',
            'strategy': r'\b(momentum|mean_reversion|bollinger|macd|rsi)\b'
        }

    def _init_huggingface(self):
        """Initialize HuggingFace pipeline."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.huggingface_model)
            model = AutoModelForCausalLM.from_pretrained(self.huggingface_model)
            
            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self.hf_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace: {e}")
            self.hf_pipeline = None

    def parse_intent_openai(self, prompt: str) -> Optional[ParsedIntent]:
        """Parse intent using OpenAI GPT-4."""
        if not openai or not self.openai_api_key:
            return None
            
        try:
            system_prompt = """You are an intent classifier for a trading system. 
            Classify the user's intent and extract arguments as JSON.
            Available intents: forecasting, backtesting, tuning, research, portfolio, risk, sentiment
            Return format: {"intent": "intent_name", "confidence": 0.95, "args": {"key": "value"}}"""
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            content = response.choices[0].message['content'].strip()
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                return ParsedIntent(
                    intent=parsed.get('intent', 'unknown'),
                    confidence=parsed.get('confidence', 0.8),
                    args=parsed.get('args', {}),
                    provider='openai',
                    raw_response=content
                )
            
            # Fallback: try to extract intent from text
            intent = self._extract_intent_from_text(content)
            return ParsedIntent(
                intent=intent,
                confidence=0.7,
                args=self._extract_args_regex(prompt),
                provider='openai',
                raw_response=content
            )
            
        except Exception as e:
            logger.warning(f"OpenAI parsing failed: {e}")
            return None

    def parse_intent_huggingface(self, prompt: str) -> Optional[ParsedIntent]:
        """Parse intent using HuggingFace model."""
        if not self.hf_pipeline:
            return None
            
        try:
            # Create a structured prompt for intent classification
            structured_prompt = f"""Task: Classify trading intent
            User input: {prompt}
            Intent:"""
            
            response = self.hf_pipeline(
                structured_prompt,
                max_length=100,
                num_return_sequences=1,
                temperature=0.1,
                do_sample=True
            )
            
            generated_text = response[0]['generated_text']
            
            # Extract intent from generated text
            intent = self._extract_intent_from_text(generated_text)
            args = self._extract_args_regex(prompt)
            
            return ParsedIntent(
                intent=intent,
                confidence=0.6,  # Lower confidence for HF
                args=args,
                provider='huggingface',
                raw_response=generated_text
            )
            
        except Exception as e:
            logger.warning(f"HuggingFace parsing failed: {e}")
            return None

    def parse_intent_regex(self, prompt: str) -> ParsedIntent:
        """Parse intent using regex and keyword matching."""
        prompt_lower = prompt.lower()
        
        # Find the best matching intent
        best_intent = 'unknown'
        best_score = 0
        
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            if score > best_score:
                best_score = score
                best_intent = intent
        
        # Extract arguments using regex patterns
        args = self._extract_args_regex(prompt)
        
        return ParsedIntent(
            intent=best_intent,
            confidence=min(0.5 + (best_score * 0.1), 0.9),  # Scale confidence with keyword matches
            args=args,
            provider='regex',
            raw_response=prompt
        )

    def _extract_intent_from_text(self, text: str) -> str:
        """Extract intent from text using keyword matching."""
        text_lower = text.lower()
        
        for intent, keywords in self.intent_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return intent
        
        return 'unknown'

    def _extract_args_regex(self, prompt: str) -> Dict[str, Any]:
        """Extract arguments using regex patterns."""
        args = {}
        
        for arg_name, pattern in self.arg_patterns.items():
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            if matches:
                if len(matches) == 1:
                    args[arg_name] = matches[0]
                else:
                    args[arg_name] = matches
        
        return args

    def parse_intent(self, prompt: str) -> ParsedIntent:
        """Parse intent using available providers with fallback logic."""
        self._update_status_on_request()
        
        # Try OpenAI first
        if openai and self.openai_api_key:
            result = self.parse_intent_openai(prompt)
            if result and result.confidence > 0.7:
                self._log_parsing_result(prompt, result)
                self._update_status_on_success()
                return result
        
        # Try HuggingFace second
        if self.hf_pipeline:
            result = self.parse_intent_huggingface(prompt)
            if result and result.confidence > 0.6:
                self._log_parsing_result(prompt, result)
                self._update_status_on_success()
                return result
        
        # Fallback to regex
        logger.info("Using regex fallback for intent parsing")
        result = self.parse_intent_regex(prompt)
        self._log_parsing_result(prompt, result)
        self._update_status_on_success()
        return result

    def _log_parsing_result(self, prompt: str, result: ParsedIntent):
        """Log parsing result for monitoring."""
        self.parsing_history.append({
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt,
            'intent': result.intent,
            'confidence': result.confidence,
            'provider': result.provider,
            'args': result.args
        })
        
        # Keep only last 100 entries
        if len(self.parsing_history) > 100:
            self.parsing_history = self.parsing_history[-100:]

    async def execute(self, prompt: str, agents: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Execute the prompt router agent.
        
        Args:
            prompt: User prompt to parse and route
            agents: Optional dictionary of available agents
            
        Returns:
            AgentResult: Result of the execution
        """
        try:
            # Parse the intent
            parsed_intent = self.parse_intent(prompt)
            
            # Route to appropriate agent if available
            if agents and parsed_intent.intent in agents:
                agent = agents[parsed_intent.intent]
                # Here you would call the appropriate agent
                # For now, we'll just return the parsed intent
                pass
            
            return AgentResult(
                success=True,
                data={
                    'intent': parsed_intent.intent,
                    'confidence': parsed_intent.confidence,
                    'args': parsed_intent.args,
                    'provider': parsed_intent.provider,
                    'raw_response': parsed_intent.raw_response
                }
            )
            
        except Exception as e:
            return self.handle_error(e)

    def route_prompt(self, prompt: str, agents: Dict[str, Any]) -> Dict[str, Any]:
        """Route a prompt to the appropriate agent.
        
        Args:
            prompt: User prompt
            agents: Dictionary of available agents
            
        Returns:
            Dictionary with routing information
        """
        parsed_intent = self.parse_intent(prompt)
        
        # Find the best matching agent
        target_agent = None
        if parsed_intent.intent in agents:
            target_agent = agents[parsed_intent.intent]
        else:
            # Try to find a fallback agent
            for intent, agent in agents.items():
                if intent in ['general', 'default', 'fallback']:
                    target_agent = agent
                    break
        
        return {
            'intent': parsed_intent.intent,
            'confidence': parsed_intent.confidence,
            'args': parsed_intent.args,
            'provider': parsed_intent.provider,
            'target_agent': target_agent,
            'timestamp': datetime.now().isoformat()
        }

    def get_available_providers(self) -> List[str]:
        """Get list of available parsing providers."""
        providers = ['regex']  # Always available
        
        if openai and self.openai_api_key:
            providers.append('openai')
        
        if self.hf_pipeline:
            providers.append('huggingface')
        
        return providers

    def get_provider_status(self) -> Dict[str, bool]:
        """Get status of each provider."""
        return {
            'openai': bool(openai and self.openai_api_key),
            'huggingface': bool(self.hf_pipeline),
            'regex': True  # Always available
        }

    def get_parsing_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent parsing history.
        
        Args:
            limit: Number of recent entries to return
            
        Returns:
            List of recent parsing results
        """
        return self.parsing_history[-limit:]

    def get_system_health(self) -> Dict[str, Any]:
        """Get system health information."""
        return {
            'providers': self.get_provider_status(),
            'parsing_history_count': len(self.parsing_history),
            'last_parsing': self.parsing_history[-1] if self.parsing_history else None,
            'status': self.get_status()
        }

    def reset_parsing_history(self) -> None:
        """Reset parsing history."""
        self.parsing_history = []
        logger.info("Parsing history reset") 