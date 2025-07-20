"""
Parser Engine

This module handles LLM prompt parsing logic with fallback mechanisms and
strategy routing from configuration files.
"""

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Try to import OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OPENAI_AVAILABLE = False

# Try to import PyTorch and transformers
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    TORCH_AVAILABLE = True
    TRANSFORMERS_AVAILABLE = True
    HUGGINGFACE_AVAILABLE = True
except ImportError as e:
    print("âš ï¸ PyTorch/transformers not available. Disabling local LLM features.")
    print(f"   Missing: {e}")
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None
    TORCH_AVAILABLE = False
    TRANSFORMERS_AVAILABLE = False
    HUGGINGFACE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ParsedIntent:
    """Structured parsed intent result."""
    intent: str
    confidence: float
    args: Dict[str, Any]
    provider: str  # 'regex', 'huggingface', 'openai', 'fallback_regex'
    raw_response: str
    error: Optional[str] = None
    json_spec: Optional[Dict[str, Any]] = None


@dataclass
class StrategyRoute:
    """Strategy routing configuration."""
    intent: str
    strategy_name: str
    priority: int
    fallback_strategies: List[str]
    conditions: Dict[str, Any]
    parameters: Dict[str, Any]


class ParserEngine:
    """
    LLM prompt parsing engine with fallback mechanisms and strategy routing.
    
    Features:
    - Regex-based intent detection (fastest)
    - Local LLM processing (HuggingFace)
    - OpenAI fallback (most accurate)
    - Strategy routing from configuration
    - Comprehensive error handling
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        huggingface_model: str = "gpt2",
        strategy_registry_path: str = "config/strategy_registry.json",
        enable_debug_mode: bool = False,
        use_regex_first: bool = True,
        use_local_llm: bool = True,
        use_openai_fallback: bool = True,
    ):
        """
        Initialize the parser engine.

        Args:
            openai_api_key: OpenAI API key
            huggingface_model: HuggingFace model name
            strategy_registry_path: Path to strategy registry file
            enable_debug_mode: Enable JSON spec return for debugging
            use_regex_first: Use regex parsing first (fastest)
            use_local_llm: Use local LLM as fallback
            use_openai_fallback: Use OpenAI as final fallback
        """
        self.openai_api_key = openai_api_key
        self.huggingface_model = huggingface_model
        self.hf_pipeline = None
        self.enable_debug_mode = enable_debug_mode
        
        # LLM selection flags
        self.use_regex_first = use_regex_first
        self.use_local_llm = use_local_llm and HUGGINGFACE_AVAILABLE
        self.use_openai_fallback = use_openai_fallback and OPENAI_AVAILABLE
        
        # Strategy registry
        self.strategy_registry: Dict[str, StrategyRoute] = {}
        self.strategy_registry_path = Path(strategy_registry_path)
        self._load_strategy_registry()
        
        # Initialize patterns
        self._initialize_patterns()
        
        # Initialize LLM providers
        self._initialize_providers()
        
        logger.info(f"ParserEngine initialized with providers: regex={self.use_regex_first}, "
                   f"local_llm={self.use_local_llm}, openai={self.use_openai_fallback}")
    
    def _load_strategy_registry(self) -> None:
        """Load strategy routing configuration from JSON file."""
        try:
            if self.strategy_registry_path.exists():
                with open(self.strategy_registry_path, 'r') as f:
                    registry_data = json.load(f)
                
                for route_data in registry_data.get('routes', []):
                    route = StrategyRoute(
                        intent=route_data['intent'],
                        strategy_name=route_data['strategy_name'],
                        priority=route_data.get('priority', 1),
                        fallback_strategies=route_data.get('fallback_strategies', []),
                        conditions=route_data.get('conditions', {}),
                        parameters=route_data.get('parameters', {})
                    )
                    self.strategy_registry[route.intent] = route
                
                logger.info(f"Loaded {len(self.strategy_registry)} strategy routes")
            else:
                # Create default strategy registry
                self._create_default_strategy_registry()
                
        except Exception as e:
            logger.error(f"Failed to load strategy registry: {e}")
            self._create_default_strategy_registry()
    
    def _create_default_strategy_registry(self) -> None:
        """Create default strategy registry with common patterns."""
        default_routes = [
            {
                'intent': 'forecasting',
                'strategy_name': 'lstm_forecaster',
                'priority': 1,
                'fallback_strategies': ['xgboost_forecaster', 'prophet_forecaster'],
                'conditions': {'min_data_points': 100},
                'parameters': {'horizon': 30, 'confidence_level': 0.95}
            },
            {
                'intent': 'strategy',
                'strategy_name': 'bollinger_strategy',
                'priority': 1,
                'fallback_strategies': ['macd_strategy', 'rsi_strategy'],
                'conditions': {'market_volatility': 'medium'},
                'parameters': {'lookback_period': 20, 'std_dev': 2.0}
            },
            {
                'intent': 'optimization',
                'strategy_name': 'bayesian_optimizer',
                'priority': 1,
                'fallback_strategies': ['grid_search', 'random_search'],
                'conditions': {'max_trials': 100},
                'parameters': {'objective': 'sharpe_ratio', 'timeout': 3600}
            },
            {
                'intent': 'analysis',
                'strategy_name': 'technical_analyzer',
                'priority': 1,
                'fallback_strategies': ['sentiment_analyzer', 'fundamental_analyzer'],
                'conditions': {},
                'parameters': {'indicators': ['rsi', 'macd', 'bollinger']}
            },
            {
                'intent': 'portfolio',
                'strategy_name': 'risk_parity_allocator',
                'priority': 1,
                'fallback_strategies': ['equal_weight', 'market_cap_weight'],
                'conditions': {'min_assets': 3},
                'parameters': {'risk_target': 0.02, 'rebalance_frequency': 'monthly'}
            }
        ]
        
        for route_data in default_routes:
            route = StrategyRoute(
                intent=route_data['intent'],
                strategy_name=route_data['strategy_name'],
                priority=route_data['priority'],
                fallback_strategies=route_data['fallback_strategies'],
                conditions=route_data['conditions'],
                parameters=route_data['parameters']
            )
            self.strategy_registry[route.intent] = route
        
        # Save default registry
        self._save_strategy_registry()
        logger.info("Created default strategy registry")
    
    def _save_strategy_registry(self) -> None:
        """Save strategy registry to JSON file."""
        try:
            self.strategy_registry_path.parent.mkdir(parents=True, exist_ok=True)
            
            registry_data = {
                'version': '1.0',
                'last_updated': datetime.now().isoformat(),
                'routes': []
            }
            
            for route in self.strategy_registry.values():
                route_data = {
                    'intent': route.intent,
                    'strategy_name': route.strategy_name,
                    'priority': route.priority,
                    'fallback_strategies': route.fallback_strategies,
                    'conditions': route.conditions,
                    'parameters': route.parameters
                }
                registry_data['routes'].append(route_data)
            
            with open(self.strategy_registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save strategy registry: {e}")
    
    def _initialize_patterns(self) -> None:
        """Initialize classification and parameter extraction patterns."""
        # Classification patterns
        self.classification_patterns = {
            'forecasting': [
                r'\b(forecast|predict|future|next|upcoming|tomorrow|next week|next month)\b',
                r'\b(price|stock|market|trend|movement|direction)\b',
                r'\b(how much|what will|when will|where will)\b',
            ],
            'strategy': [
                r'\b(strategy|trading|signal|entry|exit|position)\b',
                r'\b(buy|sell|hold|long|short|trade)\b',
                r'\b(rsi|macd|bollinger|moving average|indicator)\b',
            ],
            'analysis': [
                r'\b(analyze|analysis|examine|study|review|assess|evaluate)\b',
                r'\b(performance|metrics|statistics|data|chart|graph)\b',
                r'\b(why|what caused|what happened|explain)\b',
            ],
            'optimization': [
                r'\b(optimize|tune|improve|enhance|better|best|optimal)\b',
                r'\b(parameters|settings|configuration|hyperparameters)\b',
                r'\b(performance|efficiency|accuracy|speed)\b',
            ],
            'portfolio': [
                r'\b(portfolio|allocation|diversification|risk|balance)\b',
                r'\b(asset|investment|holdings|positions|weights)\b',
                r'\b(rebalance|adjust|change|modify)\b',
            ],
            'system': [
                r'\b(system|status|health|monitor|check|diagnose)\b',
                r'\b(error|problem|issue|bug|fix|repair)\b',
                r'\b(restart|stop|start|configure|setup)\b',
            ],
            'investment': [
                r'\b(invest|investment|buy|purchase|acquire)\b',
                r'\b(top stocks|best stocks|recommended|suggest)\b',
                r'\b(what should|which stocks|what to buy|where to invest)\b',
                r'\b(opportunity|potential|growth|returns)\b',
                r'\b(today|now|current|market)\b',
            ],
        }

        # Parameter extraction patterns
        self.parameter_patterns = {
            "symbol": r'\b([A-Z]{1,5})\b',
            "timeframe": r'\b(1m|5m|15m|30m|1h|4h|1d|1w|1M)\b',
            "days": r'\b(\d+)\s*(days?|d)\b',
            "model": r'\b(lstm|arima|xgboost|prophet|ensemble|transformer)\b',
            "strategy": r'\b(rsi|macd|bollinger|sma|ema|custom)\b',
            "risk_level": r'\b(low|medium|high|conservative|aggressive)\b',
        }

        # Intent keywords for regex fallback
        self.intent_keywords = {
            "forecasting": [
                "forecast", "predict", "projection", "future", "price", "trend", "outlook"
            ],
            "backtesting": [
                "backtest", "historical", "simulate", "performance", "past", "test", "simulation"
            ],
            "tuning": [
                "tune", "optimize", "hyperparameter", "search", "bayesian", "parameter", "improve"
            ],
            "research": [
                "research", "find", "paper", "github", "arxiv", "summarize", "analyze", "study"
            ],
            "portfolio": [
                "portfolio", "position", "holdings", "allocation", "balance", "asset"
            ],
            "risk": [
                "risk", "volatility", "drawdown", "var", "sharpe", "danger", "exposure"
            ],
            "sentiment": [
                "sentiment", "news", "social", "twitter", "reddit", "emotion", "mood"
            ],
            "compare_strategies": [
                "compare", "comparison", "versus", "vs", "against", "different", "strategy"
            ],
            "optimize_model": [
                "optimize", "improve", "enhance", "tune", "model", "performance"
            ],
            "debug_forecast": [
                "debug", "fix", "error", "issue", "problem", "forecast", "prediction"
            ],
        }
    
    def _initialize_providers(self) -> None:
        """Initialize LLM providers."""
        # Initialize OpenAI if available
        if self.use_openai_fallback and self.openai_api_key:
            openai.api_key = self.openai_api_key
            logger.info("âœ… OpenAI initialized for prompt parsing")

        # Initialize HuggingFace if available
        if self.use_local_llm:
            try:
                self._init_huggingface()
                logger.info("âœ… HuggingFace initialized for prompt parsing")
            except Exception as e:
                logger.warning(f"âš ï¸ HuggingFace initialization failed: {e}")
                self.use_local_llm = False

    def _init_huggingface(self) -> None:
        """Initialize HuggingFace pipeline."""
        if not HUGGINGFACE_AVAILABLE:
            return
            
        try:
            # Use a smaller model for faster inference
            model_name = "distilgpt2" if self.huggingface_model == "gpt2" else self.huggingface_model
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            self.hf_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device="cpu",  # Use CPU for compatibility
                max_length=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace: {e}")
            self.use_local_llm = False

    def parse_intent(self, prompt: str) -> ParsedIntent:
        """
        Parse intent using the fallback chain: Regex â†’ Local LLM â†’ OpenAI.

        Args:
            prompt: User prompt

        Returns:
            ParsedIntent: Parsed intent with provider information
        """
        # Step 1: Try regex parsing first (fastest)
        if self.use_regex_first:
            try:
                result = self.parse_intent_regex(prompt)
                if result.confidence > 0.7:  # High confidence threshold for regex
                    return result
            except Exception as e:
                logger.warning(f"Regex parsing failed: {e}")

        # Step 2: Try local LLM (medium speed, good accuracy)
        if self.use_local_llm:
            try:
                result = self.parse_intent_huggingface(prompt)
                if result and result.confidence > 0.6:
                    return result
            except Exception as e:
                logger.warning(f"Local LLM parsing failed: {e}")

        # Step 3: Try OpenAI (slowest, most accurate)
        if self.use_openai_fallback:
            try:
                result = self.parse_intent_openai(prompt)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"OpenAI parsing failed: {e}")

        # Fallback regex router if all LLM providers fail
        logger.warning("All LLM providers failed, using enhanced regex fallback")
        return self._fallback_regex_router(prompt)

    def parse_intent_regex(self, prompt: str) -> ParsedIntent:
        """Parse intent using regex patterns."""
        normalized_prompt = prompt.lower()
        
        # Find the best matching intent
        best_intent = "general"
        best_confidence = 0.0
        best_matches = 0
        
        for intent, keywords in self.intent_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in normalized_prompt)
            if matches > best_matches:
                best_matches = matches
                best_intent = intent
                best_confidence = min(0.9, matches / len(keywords) + 0.3)
        
        # Extract arguments using regex
        args = self._extract_args_regex(prompt)
        
        return ParsedIntent(
            intent=best_intent,
            confidence=best_confidence,
            args=args,
            provider="regex",
            raw_response=f"Intent: {best_intent}, Confidence: {best_confidence:.2f}",
        )

    def parse_intent_huggingface(self, prompt: str) -> Optional[ParsedIntent]:
        """Parse intent using local HuggingFace model."""
        if not self.hf_pipeline:
            return None
            
        try:
            # Create a simple prompt for intent classification
            classification_prompt = f"Classify intent: {prompt}\nIntent:"
            
            # Generate response
            response = self.hf_pipeline(
                classification_prompt,
                max_length=len(classification_prompt.split()) + 10,
                do_sample=True,
                temperature=0.3,
                pad_token_id=self.hf_pipeline.tokenizer.eos_token_id,
            )
            
            generated_text = response[0]["generated_text"]
            intent_part = generated_text[len(classification_prompt):].strip()
            
            # Extract intent from generated text
            intent = self._extract_intent_from_text(intent_part)
            confidence = 0.7  # Medium confidence for local LLM
            
            # Extract arguments
            args = self._extract_args_regex(prompt)
            
            return ParsedIntent(
                intent=intent,
                confidence=confidence,
                args=args,
                provider="huggingface",
                raw_response=generated_text,
            )
            
        except Exception as e:
            logger.error(f"HuggingFace parsing error: {e}")
            return None

    def parse_intent_openai(self, prompt: str) -> Optional[ParsedIntent]:
        """Parse intent using OpenAI."""
        if not OPENAI_AVAILABLE or not self.openai_api_key:
            return None
            
        try:
            # Create structured prompt for OpenAI
            system_prompt = """
            You are an intent classification system. Analyze the user's request and return a JSON response with:
            - intent: The primary intent (forecasting, backtesting, tuning, research, portfolio, risk, sentiment, compare_strategies, optimize_model, debug_forecast, general)
            - confidence: Confidence score (0.0 to 1.0)
            - args: Extracted arguments as key-value pairs
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200,
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                parsed = json.loads(response_text)
                intent = parsed.get("intent", "general")
                confidence = float(parsed.get("confidence", 0.8))
                args = parsed.get("args", {})
                
                return ParsedIntent(
                    intent=intent,
                    confidence=confidence,
                    args=args,
                    provider="openai",
                    raw_response=response_text,
                    json_spec=parsed if self.enable_debug_mode else None,
                )
            except json.JSONDecodeError:
                # Fallback: extract intent from text
                intent = self._extract_intent_from_text(response_text)
                return ParsedIntent(
                    intent=intent,
                    confidence=0.8,
                    args={},
                    provider="openai",
                    raw_response=response_text,
                )
                
        except Exception as e:
            logger.error(f"OpenAI parsing error: {e}")
            return None

    def _fallback_regex_router(self, prompt: str) -> ParsedIntent:
        """Enhanced fallback regex router with comprehensive pattern matching."""
        try:
            # Enhanced regex patterns for fallback routing
            fallback_patterns = {
                # Forecasting patterns
                r'\b(forecast|predict|future|trend|outlook)\b.*\b(price|stock|market|value)\b': {
                    'intent': 'forecasting',
                    'confidence': 0.8,
                    'args': {'type': 'price_forecast'}
                },
                r'\b(forecast|predict)\b.*\b(\d+)\s*(days?|weeks?|months?)\b': {
                    'intent': 'forecasting',
                    'confidence': 0.85,
                    'args': {'horizon': 'extracted'}
                },
                
                # Strategy patterns
                r'\b(strategy|strategy|approach|method)\b.*\b(trading|investment|portfolio)\b': {
                    'intent': 'strategy',
                    'confidence': 0.8,
                    'args': {'type': 'trading_strategy'}
                },
                r'\b(buy|sell|hold|position)\b.*\b(signal|recommendation|advice)\b': {
                    'intent': 'strategy',
                    'confidence': 0.9,
                    'args': {'action': 'signal'}
                },
                
                # Analysis patterns
                r'\b(analyze|analysis|examine|study)\b.*\b(market|stock|performance|data)\b': {
                    'intent': 'research',
                    'confidence': 0.8,
                    'args': {'type': 'market_analysis'}
                },
                r'\b(technical|fundamental|sentiment)\b.*\b(analysis|indicator|metric)\b': {
                    'intent': 'research',
                    'confidence': 0.85,
                    'args': {'analysis_type': 'extracted'}
                },
                
                # Optimization patterns
                r'\b(optimize|optimization|improve|enhance)\b.*\b(strategy|portfolio|performance)\b': {
                    'intent': 'tuning',
                    'confidence': 0.8,
                    'args': {'type': 'performance_optimization'}
                },
                r'\b(parameter|hyperparameter|tune|adjust)\b.*\b(model|strategy|algorithm)\b': {
                    'intent': 'tuning',
                    'confidence': 0.85,
                    'args': {'type': 'parameter_tuning'}
                },
                
                # Portfolio patterns
                r'\b(portfolio|allocation|diversification|rebalance)\b': {
                    'intent': 'portfolio',
                    'confidence': 0.8,
                    'args': {'type': 'portfolio_management'}
                },
                r'\b(risk|volatility|sharpe|return)\b.*\b(portfolio|allocation)\b': {
                    'intent': 'risk',
                    'confidence': 0.85,
                    'args': {'type': 'risk_analysis'}
                },
                
                # System patterns
                r'\b(system|status|health|monitor|check)\b': {
                    'intent': 'general',
                    'confidence': 0.8,
                    'args': {'type': 'system_status'}
                },
                r'\b(error|issue|problem|fix|debug)\b': {
                    'intent': 'general',
                    'confidence': 0.9,
                    'args': {'type': 'troubleshooting'}
                },
                
                # General patterns
                r'\b(help|assist|support|guide)\b': {
                    'intent': 'general',
                    'confidence': 0.7,
                    'args': {'type': 'help_request'}
                },
                r'\b(what|how|why|when|where)\b': {
                    'intent': 'general',
                    'confidence': 0.6,
                    'args': {'type': 'question'}
                }
            }
            
            # Test patterns against prompt
            best_match = None
            best_confidence = 0.0
            
            for pattern, config in fallback_patterns.items():
                if re.search(pattern, prompt, re.IGNORECASE):
                    confidence = config['confidence']
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = config
            
            if best_match:
                # Extract additional parameters
                args = best_match['args'].copy()
                
                # Extract time horizons
                time_match = re.search(r'(\d+)\s*(days?|weeks?|months?)', prompt, re.IGNORECASE)
                if time_match:
                    args['horizon'] = int(time_match.group(1))
                    args['horizon_unit'] = time_match.group(2)
                
                # Extract ticker symbols
                ticker_match = re.search(r'\b([A-Z]{1,5})\b', prompt)
                if ticker_match:
                    args['ticker'] = ticker_match.group(1)
                
                # Extract action words
                action_match = re.search(r'\b(buy|sell|hold|long|short)\b', prompt, re.IGNORECASE)
                if action_match:
                    args['action'] = action_match.group(1).lower()
                
                return ParsedIntent(
                    intent=best_match['intent'],
                    confidence=best_confidence,
                    args=args,
                    provider='fallback_regex',
                    raw_response=prompt,
                    error=None
                )
            
            # Default fallback
            return ParsedIntent(
                intent='general',
                confidence=0.5,
                args={'type': 'general_query'},
                provider='fallback_regex',
                raw_response=prompt,
                error=None
            )
            
        except Exception as e:
            logger.error(f"Fallback regex router failed: {e}")
            return ParsedIntent(
                intent='general',
                confidence=0.0,
                args={},
                provider='fallback_regex',
                raw_response=prompt,
                error=str(e)
            )

    def _extract_intent_from_text(self, text: str) -> str:
        """Extract intent from generated text."""
        text_lower = text.lower()
        
        for intent in self.intent_keywords.keys():
            if intent in text_lower:
                return intent
                
        return "general"

    def _extract_args_regex(self, prompt: str) -> Dict[str, Any]:
        """Extract arguments using regex patterns."""
        args = {}
        
        for param_name, pattern in self.parameter_patterns.items():
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            if matches:
                args[param_name] = matches[0] if len(matches) == 1 else matches
        
        return args

    def route_strategy(self, intent: str, args: Dict[str, Any]) -> Optional[StrategyRoute]:
        """
        Route intent to appropriate strategy using registry.
        
        Args:
            intent: Parsed intent
            args: Extracted arguments
            
        Returns:
            StrategyRoute: Best matching strategy route
        """
        if intent not in self.strategy_registry:
            # Try to find best match
            best_match = None
            best_score = 0.0
            
            for route in self.strategy_registry.values():
                score = self._calculate_route_score(route, intent, args)
                if score > best_score:
                    best_score = score
                    best_match = route
            
            return best_match
        
        return self.strategy_registry[intent]

    def _calculate_route_score(self, route: StrategyRoute, intent: str, args: Dict[str, Any]) -> float:
        """Calculate how well a route matches the intent and args."""
        score = 0.0
        
        # Intent match
        if route.intent == intent:
            score += 0.5
        
        # Check conditions
        conditions_met = 0
        total_conditions = len(route.conditions)
        
        for condition_key, condition_value in route.conditions.items():
            if condition_key in args:
                if isinstance(condition_value, (int, float)):
                    if args[condition_key] >= condition_value:
                        conditions_met += 1
                else:
                    if args[condition_key] == condition_value:
                        conditions_met += 1
        
        if total_conditions > 0:
            score += 0.3 * (conditions_met / total_conditions)
        
        # Priority bonus
        score += 0.2 * (1.0 / route.priority)
        
        return score

    def get_fallback_strategies(self, intent: str) -> List[str]:
        """Get fallback strategies for an intent."""
        if intent in self.strategy_registry:
            return self.strategy_registry[intent].fallback_strategies
        return []

    def update_strategy_registry(self, new_routes: List[Dict[str, Any]]) -> None:
        """Update strategy registry with new routes."""
        for route_data in new_routes:
            route = StrategyRoute(
                intent=route_data['intent'],
                strategy_name=route_data['strategy_name'],
                priority=route_data.get('priority', 1),
                fallback_strategies=route_data.get('fallback_strategies', []),
                conditions=route_data.get('conditions', {}),
                parameters=route_data.get('parameters', {})
            )
            self.strategy_registry[route.intent] = route
        
        self._save_strategy_registry()
        logger.info(f"Updated strategy registry with {len(new_routes)} new routes")

    def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary of strategy registry."""
        return {
            'total_routes': len(self.strategy_registry),
            'intents': list(self.strategy_registry.keys()),
            'strategies': list(set(route.strategy_name for route in self.strategy_registry.values())),
            'last_updated': datetime.now().isoformat()
        }


# Convenience function to create a parser engine
def create_parser_engine(
    openai_api_key: Optional[str] = None,
    huggingface_model: str = "gpt2",
    strategy_registry_path: str = "config/strategy_registry.json",
    enable_debug_mode: bool = False,
    use_regex_first: bool = True,
    use_local_llm: bool = True,
    use_openai_fallback: bool = True,
) -> ParserEngine:
    """
    Create a configured parser engine.

    Args:
        openai_api_key: OpenAI API key
        huggingface_model: HuggingFace model name
        strategy_registry_path: Path to strategy registry file
        enable_debug_mode: Enable debug mode
        use_regex_first: Use regex parsing first
        use_local_llm: Use local LLM
        use_openai_fallback: Use OpenAI fallback

    Returns:
        Configured ParserEngine instance
    """
    return ParserEngine(
        openai_api_key=openai_api_key,
        huggingface_model=huggingface_model,
        strategy_registry_path=strategy_registry_path,
        enable_debug_mode=enable_debug_mode,
        use_regex_first=use_regex_first,
        use_local_llm=use_local_llm,
        use_openai_fallback=use_openai_fallback,
    )
