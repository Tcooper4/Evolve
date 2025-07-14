"""
Consolidated Prompt Agent

This module consolidates all prompt routing functionality into a single, comprehensive agent:
- Regex-based intent detection
- Local LLM processing (HuggingFace)
- OpenAI fallback with intelligent routing
- Seamless fallback chain: Regex → Local LLM → OpenAI
- Comprehensive logging and routing logic
- LLM selection flags and performance tracking
"""

import json
import logging
import os
import pickle
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Try to import OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OPENAI_AVAILABLE = False

# Try to import HuggingFace
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

from trading.memory.agent_memory import AgentMemory
from trading.utils.reasoning_logger import (
    ConfidenceLevel,
    DecisionType,
    ReasoningLogger,
)

logger = logging.getLogger(__name__)


class RequestType(Enum):
    """Types of user requests."""
    FORECAST = "forecast"
    STRATEGY = "strategy"
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"
    PORTFOLIO = "portfolio"
    SYSTEM = "system"
    GENERAL = "general"
    INVESTMENT = "investment"
    UNKNOWN = "unknown"


class AgentCapability(Enum):
    """Agent capabilities."""
    FORECASTING = "forecasting"
    STRATEGY_GENERATION = "strategy_generation"
    MARKET_ANALYSIS = "market_analysis"
    OPTIMIZATION = "optimization"
    PORTFOLIO_MANAGEMENT = "portfolio_management"
    SYSTEM_MONITORING = "system_monitoring"
    GENERAL_QUERY = "general_query"


@dataclass
class PromptContext:
    """Context information for prompt processing."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    previous_requests: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedPrompt:
    """Processed prompt information."""
    original_prompt: str
    request_type: RequestType
    confidence: float
    extracted_parameters: Dict[str, Any]
    context: PromptContext
    routing_suggestions: List[str]
    processing_time: float


@dataclass
class ParsedIntent:
    """Structured parsed intent result."""
    intent: str
    confidence: float
    args: Dict[str, Any]
    provider: str  # 'regex', 'huggingface', 'openai'
    raw_response: str
    error: Optional[str] = None
    json_spec: Optional[Dict[str, Any]] = None


@dataclass
class AgentInfo:
    """Information about an available agent."""
    name: str
    capabilities: List[AgentCapability]
    priority: int
    max_concurrent: int
    current_load: int
    success_rate: float
    avg_response_time: float
    last_used: datetime
    is_available: bool = True
    custom_config: Optional[Dict[str, Any]] = None


@dataclass
class RoutingDecision:
    """Routing decision for a user request."""
    request_id: str
    request_type: RequestType
    primary_agent: str
    fallback_agents: List[str]
    confidence: float
    reasoning: str
    expected_response_time: float
    priority: int
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class PromptAgent:
    """
    Consolidated prompt agent with comprehensive routing and fallback capabilities.
    
    Features:
    - Regex-based intent detection (fastest)
    - Local LLM processing (HuggingFace)
    - OpenAI fallback (most accurate)
    - Seamless fallback chain
    - Comprehensive logging and performance tracking
    - Agent capability matching and load balancing
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        huggingface_model: str = "gpt2",
        huggingface_api_key: Optional[str] = None,
        enable_debug_mode: bool = False,
        use_regex_first: bool = True,
        use_local_llm: bool = True,
        use_openai_fallback: bool = True,
    ):
        """
        Initialize the consolidated prompt agent.

        Args:
            openai_api_key: OpenAI API key
            huggingface_model: HuggingFace model name
            huggingface_api_key: HuggingFace API key
            enable_debug_mode: Enable JSON spec return for debugging
            use_regex_first: Use regex parsing first (fastest)
            use_local_llm: Use local LLM as fallback
            use_openai_fallback: Use OpenAI as final fallback
        """
        self.openai_api_key = openai_api_key
        self.huggingface_model = huggingface_model
        self.huggingface_api_key = huggingface_api_key
        self.hf_pipeline = None
        self.enable_debug_mode = enable_debug_mode
        
        # LLM selection flags
        self.use_regex_first = use_regex_first
        self.use_local_llm = use_local_llm and HUGGINGFACE_AVAILABLE
        self.use_openai_fallback = use_openai_fallback and OPENAI_AVAILABLE
        
        # Initialize components
        self.memory = AgentMemory()
        self.reasoning_logger = ReasoningLogger()
        
        # Performance tracking
        self.performance_metrics = {
            "total_requests": 0,
            "successful_routes": 0,
            "avg_response_time": 0.0,
            "provider_usage": {"regex": 0, "huggingface": 0, "openai": 0},
            "agent_usage": {},
        }
        
        # Routing history
        self.routing_history: List[RoutingDecision] = []
        
        # Available agents
        self.available_agents: Dict[str, AgentInfo] = {}
        
        # Initialize LLM providers
        self._initialize_providers()
        
        # Initialize classification patterns
        self._initialize_patterns()
        
        # Initialize agent registry
        self._initialize_agent_registry()
        
        logger.info(f"PromptAgent initialized with providers: regex={self.use_regex_first}, "
                   f"local_llm={self.use_local_llm}, openai={self.use_openai_fallback}")

    def _initialize_providers(self):
        """Initialize LLM providers."""
        # Initialize OpenAI if available
        if self.use_openai_fallback and self.openai_api_key:
            openai.api_key = self.openai_api_key
            logger.info("✅ OpenAI initialized for prompt routing")

        # Initialize HuggingFace if available
        if self.use_local_llm:
            try:
                self._init_huggingface()
                logger.info("✅ HuggingFace initialized for prompt routing")
            except Exception as e:
                logger.warning(f"⚠️ HuggingFace initialization failed: {e}")
                self.use_local_llm = False

    def _init_huggingface(self):
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

    def _initialize_patterns(self):
        """Initialize classification and parameter extraction patterns."""
        # Classification patterns
        self.classification_patterns = {
            RequestType.FORECAST: [
                r"\b(forecast|predict|future|next|upcoming|tomorrow|next week|next month)\b",
                r"\b(price|stock|market|trend|movement|direction)\b",
                r"\b(how much|what will|when will|where will)\b",
            ],
            RequestType.STRATEGY: [
                r"\b(strategy|trading|signal|entry|exit|position)\b",
                r"\b(buy|sell|hold|long|short|trade)\b",
                r"\b(rsi|macd|bollinger|moving average|indicator)\b",
            ],
            RequestType.ANALYSIS: [
                r"\b(analyze|analysis|examine|study|review|assess|evaluate)\b",
                r"\b(performance|metrics|statistics|data|chart|graph)\b",
                r"\b(why|what caused|what happened|explain)\b",
            ],
            RequestType.OPTIMIZATION: [
                r"\b(optimize|tune|improve|enhance|better|best|optimal)\b",
                r"\b(parameters|settings|configuration|hyperparameters)\b",
                r"\b(performance|efficiency|accuracy|speed)\b",
            ],
            RequestType.PORTFOLIO: [
                r"\b(portfolio|allocation|diversification|risk|balance)\b",
                r"\b(asset|investment|holdings|positions|weights)\b",
                r"\b(rebalance|adjust|change|modify)\b",
            ],
            RequestType.SYSTEM: [
                r"\b(system|status|health|monitor|check|diagnose)\b",
                r"\b(error|problem|issue|bug|fix|repair)\b",
                r"\b(restart|stop|start|configure|setup)\b",
            ],
            RequestType.INVESTMENT: [
                r"\b(invest|investment|buy|purchase|acquire)\b",
                r"\b(top stocks|best stocks|recommended|suggest)\b",
                r"\b(what should|which stocks|what to buy|where to invest)\b",
                r"\b(opportunity|potential|growth|returns)\b",
                r"\b(today|now|current|market)\b",
            ],
        }

        # Parameter extraction patterns
        self.parameter_patterns = {
            "symbol": r"\b([A-Z]{1,5})\b",
            "timeframe": r"\b(1m|5m|15m|30m|1h|4h|1d|1w|1M)\b",
            "days": r"\b(\d+)\s*(days?|d)\b",
            "model": r"\b(lstm|arima|xgboost|prophet|ensemble|transformer)\b",
            "strategy": r"\b(rsi|macd|bollinger|sma|ema|custom)\b",
            "risk_level": r"\b(low|medium|high|conservative|aggressive)\b",
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

    def _initialize_agent_registry(self):
        """Initialize the agent registry with available agents."""
        # Define available agents and their capabilities
        agent_definitions = {
            "ForecastAgent": {
                "capabilities": [AgentCapability.FORECASTING],
                "priority": 1,
                "max_concurrent": 5,
                "success_rate": 0.85,
                "avg_response_time": 2.5,
            },
            "StrategyAgent": {
                "capabilities": [AgentCapability.STRATEGY_GENERATION],
                "priority": 2,
                "max_concurrent": 3,
                "success_rate": 0.80,
                "avg_response_time": 1.8,
            },
            "AnalysisAgent": {
                "capabilities": [AgentCapability.MARKET_ANALYSIS],
                "priority": 3,
                "max_concurrent": 4,
                "success_rate": 0.90,
                "avg_response_time": 1.2,
            },
            "OptimizationAgent": {
                "capabilities": [AgentCapability.OPTIMIZATION],
                "priority": 2,
                "max_concurrent": 2,
                "success_rate": 0.75,
                "avg_response_time": 5.0,
            },
            "PortfolioAgent": {
                "capabilities": [AgentCapability.PORTFOLIO_MANAGEMENT],
                "priority": 1,
                "max_concurrent": 3,
                "success_rate": 0.88,
                "avg_response_time": 2.0,
            },
            "SystemAgent": {
                "capabilities": [AgentCapability.SYSTEM_MONITORING],
                "priority": 1,
                "max_concurrent": 2,
                "success_rate": 0.95,
                "avg_response_time": 0.5,
            },
        }

        for agent_name, config in agent_definitions.items():
            self.available_agents[agent_name] = AgentInfo(
                name=agent_name,
                capabilities=config["capabilities"],
                priority=config["priority"],
                max_concurrent=config["max_concurrent"],
                current_load=0,
                success_rate=config["success_rate"],
                avg_response_time=config["avg_response_time"],
                last_used=datetime.now(),
                is_available=True,
            )

    def process_prompt(
        self, prompt: str, context: Optional[PromptContext] = None
    ) -> ProcessedPrompt:
        """
        Process a user prompt and extract information using the fallback chain.

        Args:
            prompt: User's input prompt
            context: Optional context information

        Returns:
            ProcessedPrompt: Processed prompt information
        """
        start_time = datetime.now()

        if context is None:
            context = PromptContext()

        # Normalize prompt
        normalized_prompt = self._normalize_prompt(prompt)

        # Use the fallback chain to parse intent
        parsed_intent = self.parse_intent(prompt)
        
        # Map intent to request type
        request_type = self._map_intent_to_request_type(parsed_intent.intent)
        
        # Extract parameters
        extracted_parameters = self._extract_parameters(normalized_prompt)
        extracted_parameters.update(parsed_intent.args)

        # Generate routing suggestions
        routing_suggestions = self._generate_routing_suggestions(
            request_type, extracted_parameters
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        processed_prompt = ProcessedPrompt(
            original_prompt=prompt,
            request_type=request_type,
            confidence=parsed_intent.confidence,
            extracted_parameters=extracted_parameters,
            context=context,
            routing_suggestions=routing_suggestions,
            processing_time=processing_time,
        )

        logger.info(
            f"Processed prompt: {request_type.value} (confidence: {parsed_intent.confidence:.2f}, "
            f"provider: {parsed_intent.provider})"
        )
        return processed_prompt

    def parse_intent(self, prompt: str) -> ParsedIntent:
        """
        Parse intent using the fallback chain: Regex → Local LLM → OpenAI.

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
                    self.performance_metrics["provider_usage"]["regex"] += 1
                    return result
            except Exception as e:
                logger.warning(f"Regex parsing failed: {e}")

        # Step 2: Try local LLM (medium speed, good accuracy)
        if self.use_local_llm:
            try:
                result = self.parse_intent_huggingface(prompt)
                if result and result.confidence > 0.6:
                    self.performance_metrics["provider_usage"]["huggingface"] += 1
                    return result
            except Exception as e:
                logger.warning(f"Local LLM parsing failed: {e}")

        # Step 3: Try OpenAI (slowest, most accurate)
        if self.use_openai_fallback:
            try:
                result = self.parse_intent_openai(prompt)
                if result:
                    self.performance_metrics["provider_usage"]["openai"] += 1
                    return result
            except Exception as e:
                logger.warning(f"OpenAI parsing failed: {e}")

        # Final fallback: Use regex with lower confidence
        logger.warning("All LLM providers failed, using regex fallback")
        result = self.parse_intent_regex(prompt)
        self.performance_metrics["provider_usage"]["regex"] += 1
        return result

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

    def _map_intent_to_request_type(self, intent: str) -> RequestType:
        """Map parsed intent to request type."""
        intent_mapping = {
            "forecasting": RequestType.FORECAST,
            "backtesting": RequestType.ANALYSIS,
            "tuning": RequestType.OPTIMIZATION,
            "research": RequestType.ANALYSIS,
            "portfolio": RequestType.PORTFOLIO,
            "risk": RequestType.ANALYSIS,
            "sentiment": RequestType.ANALYSIS,
            "compare_strategies": RequestType.STRATEGY,
            "optimize_model": RequestType.OPTIMIZATION,
            "debug_forecast": RequestType.SYSTEM,
        }
        
        return intent_mapping.get(intent, RequestType.GENERAL)

    def _normalize_prompt(self, prompt: str) -> str:
        """Normalize prompt for processing."""
        return prompt.lower().strip()

    def _extract_parameters(self, prompt: str) -> Dict[str, Any]:
        """Extract parameters from prompt."""
        return self._extract_args_regex(prompt)

    def _generate_routing_suggestions(
        self, request_type: RequestType, parameters: Dict[str, Any]
    ) -> List[str]:
        """Generate routing suggestions based on request type and parameters."""
        suggestions = []
        
        # Map request types to suggested agents
        agent_suggestions = {
            RequestType.FORECAST: ["ForecastAgent"],
            RequestType.STRATEGY: ["StrategyAgent"],
            RequestType.ANALYSIS: ["AnalysisAgent"],
            RequestType.OPTIMIZATION: ["OptimizationAgent"],
            RequestType.PORTFOLIO: ["PortfolioAgent"],
            RequestType.SYSTEM: ["SystemAgent"],
            RequestType.INVESTMENT: ["AnalysisAgent", "PortfolioAgent"],
            RequestType.GENERAL: ["AnalysisAgent"],
        }
        
        suggestions.extend(agent_suggestions.get(request_type, ["AnalysisAgent"]))
        
        # Add specific suggestions based on parameters
        if "model" in parameters:
            suggestions.append("OptimizationAgent")
        if "strategy" in parameters:
            suggestions.append("StrategyAgent")
            
        return list(set(suggestions))  # Remove duplicates

    def route_request(
        self, user_request: str, context: Optional[Dict[str, Any]] = None
    ) -> RoutingDecision:
        """
        Route a user request to appropriate agents.

        Args:
            user_request: User's request
            context: Optional context information

        Returns:
            RoutingDecision: Routing decision with agent assignments
        """
        # Process the prompt
        processed = self.process_prompt(user_request)
        
        # Find suitable agents
        suitable_agents = self._find_suitable_agents(processed.request_type, user_request)
        
        # Select primary and fallback agents
        primary_agent, fallback_agents = self._select_agents(suitable_agents, context)
        
        # Calculate confidence and reasoning
        confidence = self._calculate_routing_confidence(
            processed.request_type, primary_agent, suitable_agents
        )
        reasoning = self._generate_routing_reasoning(
            processed.request_type, primary_agent, suitable_agents
        )
        
        # Estimate response time
        expected_response_time = self._estimate_response_time(
            primary_agent, processed.request_type
        )
        
        # Determine priority
        priority = self._determine_priority(processed.request_type, context)
        
        # Create routing decision
        decision = RoutingDecision(
            request_id=f"req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            request_type=processed.request_type,
            primary_agent=primary_agent,
            fallback_agents=fallback_agents,
            confidence=confidence,
            reasoning=reasoning,
            expected_response_time=expected_response_time,
            priority=priority,
            timestamp=datetime.now(),
            metadata={
                "processed_prompt": processed,
                "provider_used": "consolidated",
            },
        )
        
        # Log and update metrics
        self._log_routing_decision(decision)
        self._update_performance_metrics(decision)
        
        return decision

    def _find_suitable_agents(
        self, request_type: RequestType, user_request: str
    ) -> List[AgentInfo]:
        """Find agents suitable for handling the request."""
        suitable_agents = []
        
        # Map request types to capabilities
        capability_mapping = {
            RequestType.FORECAST: AgentCapability.FORECASTING,
            RequestType.STRATEGY: AgentCapability.STRATEGY_GENERATION,
            RequestType.ANALYSIS: AgentCapability.MARKET_ANALYSIS,
            RequestType.OPTIMIZATION: AgentCapability.OPTIMIZATION,
            RequestType.PORTFOLIO: AgentCapability.PORTFOLIO_MANAGEMENT,
            RequestType.SYSTEM: AgentCapability.SYSTEM_MONITORING,
            RequestType.INVESTMENT: AgentCapability.MARKET_ANALYSIS,
            RequestType.GENERAL: AgentCapability.GENERAL_QUERY,
        }
        
        required_capability = capability_mapping.get(request_type, AgentCapability.GENERAL_QUERY)
        
        for agent in self.available_agents.values():
            if (required_capability in agent.capabilities and 
                agent.is_available and 
                agent.current_load < agent.max_concurrent):
                suitable_agents.append(agent)
        
        return suitable_agents

    def _select_agents(
        self, suitable_agents: List[AgentInfo], context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[str]]:
        """Select primary and fallback agents."""
        if not suitable_agents:
            # Fallback to general agent
            return "AnalysisAgent", []
        
        # Sort by score (priority, success rate, load)
        scored_agents = []
        for agent in suitable_agents:
            score = self._calculate_agent_score(agent, context)
            scored_agents.append((agent, score))
        
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        
        # Select primary agent
        primary_agent = scored_agents[0][0].name
        
        # Select fallback agents (top 2 remaining)
        fallback_agents = [agent.name for agent, _ in scored_agents[1:3]]
        
        return primary_agent, fallback_agents

    def _calculate_agent_score(
        self, agent: AgentInfo, context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate agent selection score."""
        # Base score from priority and success rate
        base_score = agent.priority * agent.success_rate
        
        # Load penalty
        load_penalty = agent.current_load / agent.max_concurrent
        
        # Recency bonus (prefer recently used agents)
        time_since_last_use = (datetime.now() - agent.last_used).total_seconds()
        recency_bonus = max(0, 1 - time_since_last_use / 3600)  # Decay over 1 hour
        
        return base_score * (1 - load_penalty) + recency_bonus

    def _calculate_routing_confidence(
        self,
        request_type: RequestType,
        primary_agent: str,
        suitable_agents: List[AgentInfo],
    ) -> float:
        """Calculate routing confidence."""
        if not suitable_agents:
            return 0.3  # Low confidence if no suitable agents
        
        # Base confidence from agent success rate
        primary_agent_info = self.available_agents.get(primary_agent)
        if primary_agent_info:
            base_confidence = primary_agent_info.success_rate
        else:
            base_confidence = 0.5
        
        # Boost confidence if multiple suitable agents available
        agent_count_boost = min(0.2, len(suitable_agents) * 0.1)
        
        return min(0.95, base_confidence + agent_count_boost)

    def _estimate_response_time(
        self, primary_agent: str, request_type: RequestType
    ) -> float:
        """Estimate response time for the request."""
        agent_info = self.available_agents.get(primary_agent)
        if agent_info:
            base_time = agent_info.avg_response_time
        else:
            base_time = 2.0
        
        # Adjust based on request type complexity
        complexity_multipliers = {
            RequestType.FORECAST: 1.5,
            RequestType.STRATEGY: 1.2,
            RequestType.ANALYSIS: 1.0,
            RequestType.OPTIMIZATION: 2.0,
            RequestType.PORTFOLIO: 1.3,
            RequestType.SYSTEM: 0.5,
            RequestType.INVESTMENT: 1.4,
            RequestType.GENERAL: 1.0,
        }
        
        multiplier = complexity_multipliers.get(request_type, 1.0)
        return base_time * multiplier

    def _determine_priority(
        self, request_type: RequestType, context: Optional[Dict[str, Any]] = None
    ) -> int:
        """Determine request priority."""
        # Base priorities by request type
        base_priorities = {
            RequestType.SYSTEM: 1,  # Highest priority
            RequestType.PORTFOLIO: 2,
            RequestType.FORECAST: 3,
            RequestType.STRATEGY: 4,
            RequestType.ANALYSIS: 5,
            RequestType.OPTIMIZATION: 6,
            RequestType.INVESTMENT: 3,
            RequestType.GENERAL: 7,  # Lowest priority
        }
        
        priority = base_priorities.get(request_type, 5)
        
        # Adjust based on context
        if context and context.get("urgent"):
            priority = max(1, priority - 2)
        
        return priority

    def _generate_routing_reasoning(
        self,
        request_type: RequestType,
        primary_agent: str,
        suitable_agents: List[AgentInfo],
    ) -> str:
        """Generate human-readable routing reasoning."""
        reasoning_parts = [
            f"Request type: {request_type.value}",
            f"Primary agent: {primary_agent}",
            f"Suitable agents: {len(suitable_agents)}",
        ]
        
        if suitable_agents:
            agent_names = [agent.name for agent in suitable_agents]
            reasoning_parts.append(f"Available: {', '.join(agent_names)}")
        
        return "; ".join(reasoning_parts)

    def _log_routing_decision(self, decision: RoutingDecision):
        """Log routing decision for analysis."""
        self.routing_history.append(decision)
        
        # Keep only recent history (last 1000 decisions)
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-1000:]
        
        logger.info(
            f"Routing decision: {decision.primary_agent} "
            f"(confidence: {decision.confidence:.2f}, "
            f"priority: {decision.priority})"
        )

    def _update_performance_metrics(self, decision: RoutingDecision):
        """Update performance metrics."""
        self.performance_metrics["total_requests"] += 1
        self.performance_metrics["successful_routes"] += 1
        
        # Update agent usage
        agent_name = decision.primary_agent
        if agent_name not in self.performance_metrics["agent_usage"]:
            self.performance_metrics["agent_usage"][agent_name] = 0
        self.performance_metrics["agent_usage"][agent_name] += 1

    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total_requests = self.performance_metrics["total_requests"]
        
        if total_requests == 0:
            return {
                "total_requests": 0,
                "success_rate": 0.0,
                "avg_response_time": 0.0,
                "provider_usage": {},
                "agent_usage": {},
            }
        
        return {
            "total_requests": total_requests,
            "success_rate": self.performance_metrics["successful_routes"] / total_requests,
            "avg_response_time": self.performance_metrics["avg_response_time"],
            "provider_usage": self.performance_metrics["provider_usage"],
            "agent_usage": self.performance_metrics["agent_usage"],
            "recent_decisions": len(self.routing_history),
        }

    def update_agent_status(
        self, agent_name: str, is_available: bool, current_load: int = 0
    ):
        """Update agent availability and load."""
        if agent_name in self.available_agents:
            self.available_agents[agent_name].is_available = is_available
            self.available_agents[agent_name].current_load = current_load

    def record_agent_performance(
        self, agent_name: str, success: bool, response_time: float
    ):
        """Record agent performance for learning."""
        if agent_name in self.available_agents:
            agent = self.available_agents[agent_name]
            
            # Update success rate (simple moving average)
            if success:
                agent.success_rate = (agent.success_rate * 0.9) + 0.1
            else:
                agent.success_rate = agent.success_rate * 0.9
            
            # Update response time (simple moving average)
            agent.avg_response_time = (agent.avg_response_time * 0.9) + (response_time * 0.1)
            
            # Update last used time
            agent.last_used = datetime.now()

    def handle_prompt(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main entry point for prompt handling.

        Args:
            prompt: User prompt
            context: Optional context

        Returns:
            Dictionary with routing decision and processing results
        """
        try:
            # Process the prompt
            processed = self.process_prompt(prompt)
            
            # Route the request
            routing_decision = self.route_request(prompt, context)
            
            return {
                "success": True,
                "processed_prompt": {
                    "request_type": processed.request_type.value,
                    "confidence": processed.confidence,
                    "extracted_parameters": processed.extracted_parameters,
                    "routing_suggestions": processed.routing_suggestions,
                    "processing_time": processed.processing_time,
                },
                "routing_decision": {
                    "primary_agent": routing_decision.primary_agent,
                    "fallback_agents": routing_decision.fallback_agents,
                    "confidence": routing_decision.confidence,
                    "reasoning": routing_decision.reasoning,
                    "expected_response_time": routing_decision.expected_response_time,
                    "priority": routing_decision.priority,
                },
                "performance_stats": self.get_performance_statistics(),
            }
            
        except Exception as e:
            logger.error(f"Error handling prompt: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_agent": "AnalysisAgent",
            }


# Convenience function to create a prompt agent
def create_prompt_agent(
    openai_api_key: Optional[str] = None,
    huggingface_model: str = "gpt2",
    huggingface_api_key: Optional[str] = None,
    enable_debug_mode: bool = False,
    use_regex_first: bool = True,
    use_local_llm: bool = True,
    use_openai_fallback: bool = True,
) -> PromptAgent:
    """
    Create a configured prompt agent.

    Args:
        openai_api_key: OpenAI API key
        huggingface_model: HuggingFace model name
        huggingface_api_key: HuggingFace API key
        enable_debug_mode: Enable debug mode
        use_regex_first: Use regex parsing first
        use_local_llm: Use local LLM
        use_openai_fallback: Use OpenAI fallback

    Returns:
        Configured PromptAgent instance
    """
    return PromptAgent(
        openai_api_key=openai_api_key,
        huggingface_model=huggingface_model,
        huggingface_api_key=huggingface_api_key,
        enable_debug_mode=enable_debug_mode,
        use_regex_first=use_regex_first,
        use_local_llm=use_local_llm,
        use_openai_fallback=use_openai_fallback,
    ) 