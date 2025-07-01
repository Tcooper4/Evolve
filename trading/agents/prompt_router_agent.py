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
    """
    Smart prompt router that selects the best LLM provider and routes user intents.
    
    LLM Selection Priority:
    1. OpenAI GPT-4 (highest quality, most reliable)
    2. HuggingFace models (good quality, local deployment)
    3. Regex fallback (basic pattern matching, always available)
    
    Agent Routing:
    - forecasting: Routes to forecasting agents (LSTM, Transformer, etc.)
    - backtesting: Routes to backtesting engine
    - tuning: Routes to optimization agents
    - research: Routes to research and analysis agents
    - portfolio: Routes to portfolio management agents
    - risk: Routes to risk analysis agents
    - sentiment: Routes to sentiment analysis agents
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the prompt router agent.
        
        Args:
            config: Agent configuration with LLM provider settings
        """
        if config is None:
            config = AgentConfig(
                name="prompt_router",
                enabled=True,
                priority=1,
                custom_config={
                    'openai_api_key': os.getenv('OPENAI_API_KEY'),
                    'huggingface_model': os.getenv('HUGGINGFACE_MODEL', 'gpt2'),
                    'huggingface_api_key': os.getenv('HUGGINGFACE_API_KEY'),
                    'prefer_openai': True,  # Prefer OpenAI over other providers
                    'regex_fallback_enabled': True  # Enable regex fallback
                }
            )
        
        super().__init__(config)
        
        # Get configuration
        self.openai_api_key = self.config.custom_config.get('openai_api_key')
        self.huggingface_model = self.config.custom_config.get('huggingface_model', 'gpt2')
        self.huggingface_api_key = self.config.custom_config.get('huggingface_api_key')
        self.prefer_openai = self.config.custom_config.get('prefer_openai', True)
        self.regex_fallback_enabled = self.config.custom_config.get('regex_fallback_enabled', True)
        
        self.hf_pipeline = None
        self.parsing_history = []
        
        # Initialize LLM providers in order of preference
        self._initialize_providers()
        
        # Intent keywords for regex fallback (simplified and cleaned up)
        self.intent_keywords = {
            'forecasting': ['forecast', 'predict', 'projection', 'future', 'price', 'trend'],
            'backtesting': ['backtest', 'historical', 'simulate', 'performance', 'past', 'test'],
            'tuning': ['tune', 'optimize', 'hyperparameter', 'search', 'bayesian', 'parameter'],
            'research': ['research', 'find', 'paper', 'github', 'arxiv', 'summarize', 'analyze'],
            'portfolio': ['portfolio', 'position', 'holdings', 'allocation', 'balance'],
            'risk': ['risk', 'volatility', 'drawdown', 'var', 'sharpe'],
            'sentiment': ['sentiment', 'news', 'social', 'twitter', 'reddit', 'emotion']
        }
        
        # Argument extraction patterns (simplified)
        self.arg_patterns = {
            'symbol': r'\b([A-Z]{1,5})\b',
            'date': r'\b(\d{4}-\d{2}-\d{2})\b',
            'number': r'\b(\d+(?:\.\d+)?)\b',
            'percentage': r'\b(\d+(?:\.\d+)?%)\b',
            'timeframe': r'\b(daily|weekly|monthly|yearly|1d|1w|1m|1y)\b',
            'model': r'\b(lstm|transformer|xgboost|prophet|arima|ensemble)\b',
            'strategy': r'\b(momentum|mean_reversion|bollinger|macd|rsi)\b'
        }

    def _initialize_providers(self):
        """Initialize LLM providers in order of preference."""
        # Initialize OpenAI if available and preferred
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
        
        # Log available providers
        available_providers = self.get_available_providers()
        logger.info(f"Available LLM providers: {available_providers}")

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
        """
        Parse intent using OpenAI GPT-4 (highest quality).
        
        This method provides the most accurate intent classification and argument extraction.
        Falls back to text-based extraction if JSON parsing fails.
        """
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
            
            # Fallback: extract intent from text
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
        """
        Parse intent using HuggingFace model (good quality, local deployment).
        
        This method provides decent intent classification with local model deployment.
        Useful when OpenAI is not available or for privacy-sensitive applications.
        """
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
                confidence=0.6,  # Lower confidence than OpenAI
                args=args,
                provider='huggingface',
                raw_response=generated_text
            )
            
        except Exception as e:
            logger.warning(f"HuggingFace parsing failed: {e}")
            return None

    def parse_intent_regex(self, prompt: str) -> ParsedIntent:
        """
        Parse intent using regex patterns (basic fallback).
        
        This method provides basic pattern matching when LLM providers are unavailable.
        Always available but with lower accuracy and limited argument extraction.
        """
        if not self.regex_fallback_enabled:
            raise ValueError("Regex fallback is disabled")
        
        # Extract intent using keyword matching
        intent = self._extract_intent_from_text(prompt)
        
        # Extract arguments using regex patterns
        args = self._extract_args_regex(prompt)
        
        return ParsedIntent(
            intent=intent,
            confidence=0.4,  # Lower confidence for regex
            args=args,
            provider='regex',
            raw_response=prompt
        )

    def _extract_intent_from_text(self, text: str) -> str:
        """Extract intent from text using keyword matching."""
        text_lower = text.lower()
        
        # Count keyword matches for each intent
        intent_scores = {}
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # Return intent with highest score, or 'unknown' if no matches
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        else:
            return 'unknown'

    def _extract_args_regex(self, prompt: str) -> Dict[str, Any]:
        """Extract arguments using regex patterns."""
        args = {}
        
        for arg_type, pattern in self.arg_patterns.items():
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            if matches:
                args[arg_type] = matches[0] if len(matches) == 1 else matches
        
        return args

    def parse_intent(self, prompt: str) -> ParsedIntent:
        """
        Parse intent using the best available provider.
        
        Provider selection order:
        1. OpenAI (if available and preferred)
        2. HuggingFace (if available)
        3. Regex fallback (always available)
        
        Args:
            prompt: User prompt to parse
            
        Returns:
            ParsedIntent with intent classification and extracted arguments
            
        Raises:
            ValueError: If prompt is empty or invalid
        """
        # Validate prompt
        if not prompt or not isinstance(prompt, str):
            logger.warning("Empty or invalid prompt provided, returning unknown intent")
            return ParsedIntent(
                intent='unknown',
                confidence=0.0,
                args={},
                provider='regex',
                raw_response='',
                error='Empty or invalid prompt'
            )
        
        # Clean and normalize prompt
        prompt = prompt.strip()
        if len(prompt) < 3:
            logger.warning("Prompt too short, returning unknown intent")
            return ParsedIntent(
                intent='unknown',
                confidence=0.0,
                args={},
                provider='regex',
                raw_response=prompt,
                error='Prompt too short'
            )
        
        # Check for maximum length
        if len(prompt) > 10000:
            logger.warning("Prompt too long, truncating to 10000 characters")
            prompt = prompt[:10000]
        
        try:
            # Try OpenAI first if preferred and available
            if self.prefer_openai:
                result = self.parse_intent_openai(prompt)
                if result:
                    self._log_parsing_result(prompt, result)
                    return result
            
            # Try HuggingFace
            result = self.parse_intent_huggingface(prompt)
            if result:
                self._log_parsing_result(prompt, result)
                return result
            
            # Try OpenAI if not preferred but available
            if not self.prefer_openai:
                result = self.parse_intent_openai(prompt)
                if result:
                    self._log_parsing_result(prompt, result)
                    return result
            
            # Fallback to regex
            result = self.parse_intent_regex(prompt)
            self._log_parsing_result(prompt, result)
            return result
            
        except Exception as e:
            logger.error(f"Error parsing intent: {e}")
            # Return fallback result with error
            return ParsedIntent(
                intent='unknown',
                confidence=0.0,
                args={},
                provider='regex',
                raw_response=prompt,
                error=str(e)
            )

    def _log_parsing_result(self, prompt: str, result: ParsedIntent):
        """Log parsing result for monitoring and debugging."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt[:100] + '...' if len(prompt) > 100 else prompt,
            'intent': result.intent,
            'confidence': result.confidence,
            'provider': result.provider,
            'args_count': len(result.args)
        }
        
        self.parsing_history.append(log_entry)
        
        # Keep only last 100 entries
        if len(self.parsing_history) > 100:
            self.parsing_history = self.parsing_history[-100:]

    async def execute(self, prompt: str, agents: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Execute prompt routing and agent orchestration.
        
        This method:
        1. Parses the user prompt to determine intent
        2. Routes to the appropriate agent based on intent
        3. Returns structured results with routing information
        """
        try:
            # Parse intent from prompt
            parsed_intent = self.parse_intent(prompt)
            
            # Route to appropriate agent
            if agents:
                routing_result = self.route_prompt(prompt, agents)
            else:
                routing_result = {
                    'routed_agent': None,
                    'routing_reason': 'No agents available',
                    'intent': parsed_intent.intent
                }
            
            return AgentResult(
                success=True,
                data={
                    'parsed_intent': parsed_intent,
                    'routing_result': routing_result,
                    'available_providers': self.get_available_providers(),
                    'system_health': self.get_system_health()
                },
                message=f"Intent '{parsed_intent.intent}' parsed with {parsed_intent.confidence:.2f} confidence using {parsed_intent.provider}",
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error in prompt routing: {e}")
            return AgentResult(
                success=False,
                data={'error': str(e)},
                message=f"Prompt routing failed: {e}",
                timestamp=datetime.now().isoformat()
            )

    def route_prompt(self, prompt: str, agents: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route prompt to the most appropriate agent based on parsed intent.
        
        Routing logic:
        - forecasting: Routes to forecasting agents (LSTM, Transformer, etc.)
        - backtesting: Routes to backtesting engine
        - tuning: Routes to optimization agents
        - research: Routes to research and analysis agents
        - portfolio: Routes to portfolio management agents
        - risk: Routes to risk analysis agents
        - sentiment: Routes to sentiment analysis agents
        """
        parsed_intent = self.parse_intent(prompt)
        
        # Define routing rules
        routing_rules = {
            'forecasting': ['lstm_agent', 'transformer_agent', 'ensemble_agent', 'forecast_agent'],
            'backtesting': ['backtest_agent', 'simulation_agent'],
            'tuning': ['optimization_agent', 'hyperparameter_agent', 'bayesian_agent'],
            'research': ['research_agent', 'analysis_agent', 'paper_agent'],
            'portfolio': ['portfolio_agent', 'allocation_agent'],
            'risk': ['risk_agent', 'volatility_agent'],
            'sentiment': ['sentiment_agent', 'news_agent']
        }
        
        # Find available agents for the intent
        available_agents = routing_rules.get(parsed_intent.intent, [])
        routed_agent = None
        
        for agent_name in available_agents:
            if agent_name in agents and agents[agent_name].get('enabled', False):
                routed_agent = agent_name
                break
        
        return {
            'routed_agent': routed_agent,
            'routing_reason': f"Intent '{parsed_intent.intent}' matched to agent '{routed_agent}'" if routed_agent else f"No available agent for intent '{parsed_intent.intent}'",
            'intent': parsed_intent.intent,
            'confidence': parsed_intent.confidence,
            'available_agents': available_agents
        }

    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers."""
        providers = []
        
        if openai and self.openai_api_key:
            providers.append('openai')
        
        if self.hf_pipeline:
            providers.append('huggingface')
        
        if self.regex_fallback_enabled:
            providers.append('regex')
        
        return providers

    def get_provider_status(self) -> Dict[str, bool]:
        """Get status of all LLM providers."""
        return {
            'openai': bool(openai and self.openai_api_key),
            'huggingface': bool(self.hf_pipeline),
            'regex': self.regex_fallback_enabled
        }

    def get_parsing_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent parsing history for monitoring."""
        return self.parsing_history[-limit:] if self.parsing_history else []

    def get_system_health(self) -> Dict[str, Any]:
        """Get system health information."""
        return {
            'available_providers': self.get_available_providers(),
            'provider_status': self.get_provider_status(),
            'parsing_history_count': len(self.parsing_history),
            'prefer_openai': self.prefer_openai,
            'regex_fallback_enabled': self.regex_fallback_enabled
        }

    def reset_parsing_history(self) -> None:
        """Reset parsing history."""
        self.parsing_history = []
        logger.info("Parsing history reset") 