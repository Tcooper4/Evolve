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

class PromptRouterAgent:
    def __init__(self, openai_api_key: Optional[str] = None, 
                 huggingface_model: str = "gpt2", 
                 huggingface_api_key: Optional[str] = None):
        """
        Initialize the prompt router agent.
        
        Args:
            openai_api_key: OpenAI API key
            huggingface_model: HuggingFace model name
            huggingface_api_key: HuggingFace API key
        """
        self.openai_api_key = openai_api_key
        self.huggingface_model = huggingface_model
        self.huggingface_api_key = huggingface_api_key
        self.hf_pipeline = None
        
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
        
        # Find best matching intent
        best_intent = 'unknown'
        best_score = 0
        
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            if score > best_score:
                best_score = score
                best_intent = intent
        
        # Extract arguments using regex patterns
        args = self._extract_args_regex(prompt)
        
        # Calculate confidence based on keyword matches
        confidence = min(0.5 + (best_score * 0.1), 0.9)
        
        return ParsedIntent(
            intent=best_intent,
            confidence=confidence,
            args=args,
            provider='regex',
            raw_response=f"Regex matched intent: {best_intent} with {best_score} keyword matches"
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
        
        # Extract additional context
        if 'forecast' in prompt.lower() or 'predict' in prompt.lower():
            if 'price' in prompt.lower():
                args['target'] = 'price'
            elif 'return' in prompt.lower():
                args['target'] = 'return'
        
        if 'days' in prompt.lower() or 'period' in prompt.lower():
            period_match = re.search(r'(\d+)\s*(days?|weeks?|months?|years?)', prompt.lower())
            if period_match:
                args['period'] = f"{period_match.group(1)} {period_match.group(2)}"
        
        return args

    def parse_intent(self, prompt: str) -> ParsedIntent:
        """Parse intent using the best available provider.
        
        Args:
            prompt: User input prompt
            
        Returns:
            ParsedIntent with intent and arguments
        """
        # Try OpenAI first
        if openai and self.openai_api_key:
            result = self.parse_intent_openai(prompt)
            if result and result.confidence > 0.8:
                logger.info(f"✅ OpenAI parsing successful: {result.intent} (confidence: {result.confidence:.1%})")
                return result
            elif result:
                logger.warning(f"⚠️ OpenAI parsing low confidence: {result.intent} (confidence: {result.confidence:.1%})")
        
        # Try HuggingFace next
        if self.hf_pipeline:
            result = self.parse_intent_huggingface(prompt)
            if result and result.confidence > 0.7:
                logger.info(f"✅ HuggingFace parsing successful: {result.intent} (confidence: {result.confidence:.1%})")
                return result
            elif result:
                logger.warning(f"⚠️ HuggingFace parsing low confidence: {result.intent} (confidence: {result.confidence:.1%})")
        
        # Fallback to regex
        logger.warning("⚠️ Fallback model used due to unavailable capability - using regex parsing")
        result = self.parse_intent_regex(prompt)
        logger.info(f"✅ Regex fallback parsing: {result.intent} (confidence: {result.confidence:.1%})")
        return result

    def route(self, prompt: str, agents: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route the prompt to the correct agent based on intent and arguments.
        
        Args:
            prompt: User prompt
            agents: Dictionary of available agents
            
        Returns:
            Dictionary with routing result and metadata
        """
        parsed_intent = self.parse_intent(prompt)
        
        logger.info(f"Routing intent: {parsed_intent.intent}, args: {parsed_intent.args}")
        
        # Route to appropriate agent
        result = {
            'intent': parsed_intent.intent,
            'confidence': parsed_intent.confidence,
            'args': parsed_intent.args,
            'provider': parsed_intent.provider,
            'raw_response': parsed_intent.raw_response,
            'timestamp': str(datetime.now()),
            'success': False,
            'error': None,
            'result': None
        }
        
        try:
            if parsed_intent.intent == 'forecasting' and 'forecasting' in agents:
                result['result'] = agents['forecasting'].run_forecast(**parsed_intent.args)
                result['success'] = True
            elif parsed_intent.intent == 'backtesting' and 'backtesting' in agents:
                result['result'] = agents['backtesting'].run_backtest(**parsed_intent.args)
                result['success'] = True
            elif parsed_intent.intent == 'tuning' and 'tuning' in agents:
                result['result'] = agents['tuning'].run_tuning(**parsed_intent.args)
                result['success'] = True
            elif parsed_intent.intent == 'research' and 'research' in agents:
                result['result'] = agents['research'].research(**parsed_intent.args)
                result['success'] = True
            elif parsed_intent.intent == 'portfolio' and 'portfolio' in agents:
                result['result'] = agents['portfolio'].analyze_portfolio(**parsed_intent.args)
                result['success'] = True
            elif parsed_intent.intent == 'risk' and 'risk' in agents:
                result['result'] = agents['risk'].analyze_risk(**parsed_intent.args)
                result['success'] = True
            elif parsed_intent.intent == 'sentiment' and 'sentiment' in agents:
                result['result'] = agents['sentiment'].analyze_sentiment(**parsed_intent.args)
                result['success'] = True
            else:
                result['error'] = f'No agent found for intent: {parsed_intent.intent}'
                logger.warning(f"No agent found for intent: {parsed_intent.intent}")
                
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error routing to agent: {e}")
        
        return result

    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers."""
        providers = []
        
        if openai and self.openai_api_key:
            providers.append('openai')
        
        if self.hf_pipeline:
            providers.append('huggingface')
        
        providers.append('regex')  # Always available as fallback
        
        return providers

    def get_provider_status(self) -> Dict[str, bool]:
        """Get status of all LLM providers."""
        return {
            'openai': bool(openai and self.openai_api_key),
            'huggingface': bool(self.hf_pipeline),
            'regex': True  # Always available
        } 