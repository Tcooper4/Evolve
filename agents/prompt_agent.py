"""
Enhanced Prompt Agent with Batch 12 Features

This module consolidates all prompt routing functionality into a single, comprehensive agent:
- Enhanced Hugging Face classification for intent detection
- GPT-4 structured parser with JSON schema validation
- Intelligent fallback chain with confidence scoring
- Comprehensive logging and routing logic
- LLM selection flags and performance tracking
- Memory module integration for persistent learning
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
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

# Try to import HuggingFace with enhanced classification
try:
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline

    HUGGINGFACE_AVAILABLE = True
except ImportError as e:
    print(
        "âš ï¸ HuggingFace libraries not available. Disabling advanced NLP features."
    )
    print(f"   Missing: {e}")
    torch = None
    transformers = None
    sentence_transformers = None
    HUGGINGFACE_AVAILABLE = False

# Try to import Redis for memory
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


# Optional import for prompt trace logging
try:
    from logs.prompt_trace_logger import ActionStatus, PromptTraceLogger
except ImportError:
    # Fallback implementation if module is not available
    # Note: datetime, Enum, and typing imports already available at module level
    # noqa: F811 - These are intentionally redefined for fallback implementation

    class ActionStatus(Enum):
        """Status of prompt actions."""

        PENDING = "pending"
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"
        CANCELLED = "cancelled"
        TIMEOUT = "timeout"

    class PromptTraceLogger:
        """Fallback prompt trace logger."""

        def __init__(self, log_file: Optional[str] = None):
            self.log_file = log_file
            self.traces: Dict[str, Any] = {}

        def start_trace(
            self,
            prompt: str,
            user_id: Optional[str] = None,
            session_id: Optional[str] = None,
        ) -> str:
            return f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        def add_action(
            self,
            trace_id: str,
            action_type: str,
            metadata: Optional[Dict[str, Any]] = None,
        ) -> str:
            return f"action_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        def update_action_status(
            self,
            trace_id: str,
            action_id: str,
            status: ActionStatus,
            result: Optional[Dict[str, Any]] = None,
            error: Optional[str] = None,
        ):
            pass

        def complete_trace(
            self, trace_id: str, metadata: Optional[Dict[str, Any]] = None
        ):
            pass


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
    provider: str  # 'huggingface', 'openai', 'fallback'
    raw_response: str
    error: Optional[str] = None
    json_spec: Optional[Dict[str, Any]] = None
    structured_data: Optional[Dict[str, Any]] = None


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


class EnhancedPromptMemory:
    """Enhanced memory module for persistent prompt â†’ action â†’ outcome learning."""

    def __init__(self, redis_url: Optional[str] = None):
        """Initialize enhanced prompt memory.

        Args:
            redis_url: Redis connection URL
        """
        self.redis_client = None
        self.memory_file = "data/prompt_memory.jsonl"

        # Initialize Redis if available
        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("Connected to Redis for prompt memory")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None

        # Ensure memory file directory exists
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)

        # Initialize sentence transformer for similarity
        self.sentence_transformer = None
        if HUGGINGFACE_AVAILABLE:
            try:
                self.sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Sentence transformer initialized for memory similarity")
            except Exception as e:
                logger.warning(f"Could not initialize sentence transformer: {e}")

    def store_interaction(
        self,
        prompt: str,
        intent: str,
        action: str,
        outcome: Dict[str, Any],
        user_id: Optional[str] = None,
    ):
        """Store a prompt â†’ action â†’ outcome interaction.

        Args:
            prompt: User prompt
            intent: Detected intent
            action: Action taken
            outcome: Outcome of the action
            user_id: Optional user ID
        """
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "intent": intent,
            "action": action,
            "outcome": outcome,
            "user_id": user_id,
            "success": outcome.get("success", False),
        }

        # Store in Redis if available
        if self.redis_client:
            try:
                key = f"prompt_memory:{datetime.now().strftime('%Y%m%d')}"
                self.redis_client.lpush(key, json.dumps(interaction))
                self.redis_client.expire(key, 86400 * 30)  # 30 days
            except Exception as e:
                logger.error(f"Failed to store in Redis: {e}")

        # Store in file
        try:
            with open(self.memory_file, "a") as f:
                f.write(json.dumps(interaction) + "\n")
        except Exception as e:
            logger.error(f"Failed to store in file: {e}")

    def find_similar_prompts(
        self, prompt: str, limit: int = 5, threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find similar past prompts.

        Args:
            prompt: Current prompt
            limit: Maximum number of similar prompts to return
            threshold: Similarity threshold

        Returns:
            List of similar interactions
        """
        if not self.sentence_transformer:
            return []

        try:
            # Get recent interactions
            interactions = self._load_recent_interactions(1000)

            if not interactions:
                return []

            # Compute embeddings
            prompt_embedding = self.sentence_transformer.encode([prompt])[0]
            past_prompts = [i["prompt"] for i in interactions]
            past_embeddings = self.sentence_transformer.encode(past_prompts)

            # Calculate similarities
            similarities = []
            for i, past_embedding in enumerate(past_embeddings):
                similarity = np.dot(prompt_embedding, past_embedding) / (
                    np.linalg.norm(prompt_embedding) * np.linalg.norm(past_embedding)
                )
                if similarity >= threshold:
                    similarities.append((similarity, interactions[i]))

            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[0], reverse=True)
            return [interaction for _, interaction in similarities[:limit]]

        except Exception as e:
            logger.error(f"Error finding similar prompts: {e}")
            return []

    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from past interactions for learning.

        Returns:
            Dictionary with learning insights
        """
        try:
            interactions = self._load_recent_interactions(10000)

            if not interactions:
                return {}

            # Analyze success rates by intent
            intent_success = {}
            action_success = {}

            for interaction in interactions:
                intent = interaction.get("intent", "unknown")
                action = interaction.get("action", "unknown")
                success = interaction.get("success", False)

                if intent not in intent_success:
                    intent_success[intent] = {"total": 0, "successful": 0}
                if action not in action_success:
                    action_success[action] = {"total": 0, "successful": 0}

                intent_success[intent]["total"] += 1
                action_success[action]["total"] += 1

                if success:
                    intent_success[intent]["successful"] += 1
                    action_success[action]["successful"] += 1

            # Calculate success rates
            insights = {
                "intent_success_rates": {},
                "action_success_rates": {},
                "total_interactions": len(interactions),
                "recent_failures": [],
            }

            for intent, stats in intent_success.items():
                insights["intent_success_rates"][intent] = (
                    stats["successful"] / stats["total"]
                )

            for action, stats in action_success.items():
                insights["action_success_rates"][action] = (
                    stats["successful"] / stats["total"]
                )

            # Get recent failures
            recent_failures = [
                i for i in interactions[-100:] if not i.get("success", True)
            ]
            insights["recent_failures"] = recent_failures[:10]

            return insights

        except Exception as e:
            logger.error(f"Error getting learning insights: {e}")
            return {}

    def _load_recent_interactions(self, limit: int) -> List[Dict[str, Any]]:
        """Load recent interactions from storage.

        Args:
            limit: Maximum number of interactions to load

        Returns:
            List of recent interactions
        """
        interactions = []

        # Try Redis first
        if self.redis_client:
            try:
                keys = self.redis_client.keys("prompt_memory:*")
                for key in keys[-7:]:  # Last 7 days
                    items = self.redis_client.lrange(key, 0, -1)
                    for item in items:
                        interactions.append(json.loads(item))
            except Exception as e:
                logger.error(f"Failed to load from Redis: {e}")

        # Load from file
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, "r") as f:
                    for line in f:
                        if line.strip():
                            interactions.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to load from file: {e}")

        # Sort by timestamp and return recent ones
        interactions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return interactions[:limit]


class PromptAgent:
    """
    Enhanced prompt agent with Batch 12 features:
    - Hugging Face classification for intent detection
    - GPT-4 structured parser with JSON schema validation
    - Enhanced memory module for persistent learning
    - Intelligent fallback chain with confidence scoring
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        huggingface_model: str = "microsoft/DialoGPT-medium",
        classification_model: str = "facebook/bart-large-mnli",
        huggingface_api_key: Optional[str] = None,
        enable_debug_mode: bool = False,
        use_huggingface_first: bool = True,
        use_openai_fallback: bool = True,
        max_retries: int = 3,
        retry_backoff_factor: float = 2.0,
        retry_delay_seconds: float = 1.0,
        redis_url: Optional[str] = None,
    ):
        """
        Initialize the enhanced prompt agent.

        Args:
            openai_api_key: OpenAI API key
            huggingface_model: HuggingFace model for text generation
            classification_model: HuggingFace model for intent classification
            huggingface_api_key: HuggingFace API key
            enable_debug_mode: Enable JSON spec return for debugging
            use_huggingface_first: Use HuggingFace classification first
            use_openai_fallback: Use OpenAI as fallback
            max_retries: Maximum number of retry attempts
            retry_backoff_factor: Exponential backoff multiplier
            retry_delay_seconds: Initial delay between retries
            redis_url: Redis URL for memory storage
        """
        # Initialize availability flag
        self.available = True

        self.openai_api_key = openai_api_key
        self.huggingface_model = huggingface_model
        self.classification_model = classification_model
        self.huggingface_api_key = huggingface_api_key
        self.enable_debug_mode = enable_debug_mode

        # LLM selection flags
        self.use_huggingface_first = use_huggingface_first and HUGGINGFACE_AVAILABLE
        self.use_openai_fallback = use_openai_fallback and OPENAI_AVAILABLE

        # Retry configuration
        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor
        self.retry_delay_seconds = retry_delay_seconds

        # Initialize memory module
        self.memory = EnhancedPromptMemory(redis_url)

        # Initialize components with error handling
        try:
            self._initialize_providers()
            self._initialize_classification_labels()
            self._initialize_agent_registry()

            # Performance tracking
            self.performance_metrics = {
                "total_requests": 0,
                "successful_parses": 0,
                "huggingface_parses": 0,
                "openai_parses": 0,
                "fallback_parses": 0,
                "avg_parse_time": 0.0,
                "errors": [],
            }

            # Initialize prompt trace logger
            self.trace_logger = PromptTraceLogger()

            logger.info("Enhanced PromptAgent initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize PromptAgent: {e}")
            self.available = False
            print("âš ï¸ PromptAgent unavailable due to initialization failure")
            print(f"   Error: {e}")

    def _initialize_providers(self):
        """Initialize LLM providers."""
        # Initialize HuggingFace
        if self.use_huggingface_first:
            self._init_huggingface()

        # Initialize OpenAI
        if self.use_openai_fallback and self.openai_api_key:
            try:
                openai.api_key = self.openai_api_key
                logger.info("OpenAI API configured")
            except Exception as e:
                logger.error(f"Failed to configure OpenAI API: {e}")
                self.use_openai_fallback = False

    def _init_huggingface(self):
        """Initialize HuggingFace models."""
        try:
            # Initialize classification pipeline
            self.classifier = pipeline(
                "zero-shot-classification",
                model=self.classification_model,
                device=0 if torch.cuda.is_available() else -1,
            )

            # Initialize text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.huggingface_model,
                device=0 if torch.cuda.is_available() else -1,
            )

            logger.info(
                f"HuggingFace models initialized: {self.classification_model}, {self.huggingface_model}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace models: {e}")
            self.use_huggingface_first = False
            print("âš ï¸ HuggingFace models unavailable due to model load failure")
            print(f"   Error: {e}")

    def _initialize_classification_labels(self):
        """Initialize classification labels for intent detection."""
        self.intent_labels = [
            "forecast stock price",
            "generate trading strategy",
            "analyze market",
            "optimize portfolio",
            "manage positions",
            "system status",
            "general question",
            "investment advice",
        ]

        # Map labels to request types
        self.label_to_request_type = {
            "forecast stock price": RequestType.FORECAST,
            "generate trading strategy": RequestType.STRATEGY,
            "analyze market": RequestType.ANALYSIS,
            "optimize portfolio": RequestType.OPTIMIZATION,
            "manage positions": RequestType.PORTFOLIO,
            "system status": RequestType.SYSTEM,
            "general question": RequestType.GENERAL,
            "investment advice": RequestType.INVESTMENT,
        }

    def _initialize_agent_registry(self):
        """Initialize the registry of available agents."""
        self.available_agents = {}

        agent_definitions = {
            "ModelSelectorAgent": {
                "capabilities": [AgentCapability.FORECASTING],
                "priority": 1,
                "max_concurrent": 3,
                "success_rate": 0.85,
                "avg_response_time": 2.5,
            },
            "StrategySelectorAgent": {
                "capabilities": [AgentCapability.STRATEGY_GENERATION],
                "priority": 1,
                "max_concurrent": 3,
                "success_rate": 0.80,
                "avg_response_time": 3.0,
            },
            "MarketAnalyzerAgent": {
                "capabilities": [AgentCapability.MARKET_ANALYSIS],
                "priority": 2,
                "max_concurrent": 2,
                "success_rate": 0.90,
                "avg_response_time": 4.0,
            },
            "MetaTunerAgent": {
                "capabilities": [AgentCapability.OPTIMIZATION],
                "priority": 2,
                "max_concurrent": 1,
                "success_rate": 0.75,
                "avg_response_time": 15.0,
            },
            "PortfolioManagerAgent": {
                "capabilities": [AgentCapability.PORTFOLIO_MANAGEMENT],
                "priority": 1,
                "max_concurrent": 2,
                "success_rate": 0.88,
                "avg_response_time": 5.0,
            },
            "SystemMonitorAgent": {
                "capabilities": [AgentCapability.SYSTEM_MONITORING],
                "priority": 3,
                "max_concurrent": 5,
                "success_rate": 0.95,
                "avg_response_time": 1.0,
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

    async def process_prompt(
        self, prompt: str, context: Optional[PromptContext] = None
    ) -> ProcessedPrompt:
        """
        Process a user prompt and extract information using enhanced classification.

        Args:
            prompt: User's input prompt
            context: Optional context information

        Returns:
            ProcessedPrompt: Processed prompt information
        """
        if not self.available:
            print("âš ï¸ PromptAgent unavailable due to initialization failure")
            return ProcessedPrompt(
                original_prompt=prompt,
                request_type=RequestType.UNKNOWN,
                confidence=0.0,
                extracted_parameters={},
                context=context or PromptContext(),
                routing_suggestions=[],
                processing_time=0.0,
            )

        start_time = datetime.now()
        trace_id = self.trace_logger.start_trace(prompt)

        if context is None:
            context = PromptContext()

        # Normalize prompt
        normalized_prompt = self._normalize_prompt(prompt)

        # Check memory for similar prompts
        similar_prompts = self.memory.find_similar_prompts(normalized_prompt)
        if similar_prompts:
            logger.info(f"Found {len(similar_prompts)} similar prompts in memory")

        # Parse intent using enhanced methods
        parsed_intent = await self.parse_intent_enhanced(prompt)

        # Map intent to request type
        request_type = self._map_intent_to_request_type(parsed_intent.intent)

        # Extract parameters
        extracted_parameters = self._extract_parameters_enhanced(
            normalized_prompt, parsed_intent
        )

        # Generate routing suggestions
        routing_suggestions = self._generate_routing_suggestions(
            request_type, extracted_parameters
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        # Store interaction in memory
        self.memory.store_interaction(
            prompt=prompt,
            intent=parsed_intent.intent,
            action="parse_intent",
            outcome={
                "success": parsed_intent.error is None,
                "confidence": parsed_intent.confidence,
                "request_type": request_type.value,
                "processing_time": processing_time,
            },
            user_id=context.user_id,
        )

        # Complete trace
        self.trace_logger.complete_trace(
            trace_id,
            {
                "processing_time": processing_time,
                "confidence": parsed_intent.confidence,
                "provider": parsed_intent.provider,
            },
        )

        return ProcessedPrompt(
            original_prompt=prompt,
            request_type=request_type,
            confidence=parsed_intent.confidence,
            extracted_parameters=extracted_parameters,
            context=context,
            routing_suggestions=routing_suggestions,
            processing_time=processing_time,
        )

    async def parse_intent_enhanced(self, prompt: str) -> ParsedIntent:
        """
        Enhanced intent parsing with HuggingFace classification and GPT-4 structured parser.

        Args:
            prompt: User prompt

        Returns:
            ParsedIntent: Structured parsed intent
        """
        start_time = datetime.now()

        # Try HuggingFace classification first
        if self.use_huggingface_first:
            try:
                result = await self._parse_intent_huggingface(prompt)
                if result and result.confidence > 0.7:
                    self.performance_metrics["huggingface_parses"] += 1
                    return result
            except Exception as e:
                logger.warning(f"HuggingFace parsing failed: {e}")

        # Try OpenAI structured parser
        if self.use_openai_fallback:
            try:
                result = await self._parse_intent_openai_structured(prompt)
                if result and result.confidence > 0.6:
                    self.performance_metrics["openai_parses"] += 1
                    return result
            except Exception as e:
                logger.warning(f"OpenAI structured parsing failed: {e}")

        # Fallback to basic parsing
        result = self._parse_intent_fallback(prompt)
        self.performance_metrics["fallback_parses"] += 1

        # Update performance metrics
        parse_time = (datetime.now() - start_time).total_seconds()
        self.performance_metrics["total_requests"] += 1
        self.performance_metrics["successful_parses"] += 1
        self.performance_metrics["avg_parse_time"] = (
            self.performance_metrics["avg_parse_time"]
            * (self.performance_metrics["total_requests"] - 1)
            + parse_time
        ) / self.performance_metrics["total_requests"]

        return result

    async def _parse_intent_huggingface(self, prompt: str) -> Optional[ParsedIntent]:
        """Parse intent using HuggingFace classification.

        Args:
            prompt: User prompt

        Returns:
            ParsedIntent or None if parsing fails
        """
        try:
            # Use zero-shot classification
            result = self.classifier(
                prompt,
                candidate_labels=self.intent_labels,
                hypothesis_template="This text is about {}.",
            )

            # Get best match
            best_label = result["labels"][0]
            confidence = result["scores"][0]

            # Extract parameters using text generation
            param_prompt = f"Extract parameters from: {prompt}\nParameters:"
            param_result = self.generator(
                param_prompt, max_length=100, num_return_sequences=1, temperature=0.3
            )

            # Parse parameters (simplified)
            params_text = param_result[0]["generated_text"]
            extracted_params = self._extract_params_from_text(params_text)

            return ParsedIntent(
                intent=best_label,
                confidence=confidence,
                args=extracted_params,
                provider="huggingface",
                raw_response=result,
                structured_data={
                    "classification_result": result,
                    "parameter_text": params_text,
                },
            )

        except Exception as e:
            logger.error(f"HuggingFace parsing error: {e}")
            return None

    async def _parse_intent_openai_structured(
        self, prompt: str
    ) -> Optional[ParsedIntent]:
        """Parse intent using OpenAI with structured JSON output.

        Args:
            prompt: User prompt

        Returns:
            ParsedIntent or None if parsing fails
        """
        try:
            # Define JSON schema for structured output
            schema = {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": [
                            label.replace(" ", "_") for label in self.intent_labels
                        ],
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string"},
                            "timeframe": {"type": "string"},
                            "strategy_type": {"type": "string"},
                            "risk_level": {"type": "string"},
                            "amount": {"type": "number"},
                        },
                    },
                    "entities": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["intent", "confidence"],
            }

            # Create structured prompt
            structured_prompt = f"""Parse the following user request and return a JSON object with the specified schema:

User request: {prompt}

JSON schema: {json.dumps(schema, indent=2)}

Return only the JSON object, no additional text."""

            # Call OpenAI with structured output
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise intent parser. Return only valid JSON.",
                    },
                    {"role": "user", "content": structured_prompt},
                ],
                temperature=0.1,
                max_tokens=500,
            )

            # Parse response
            response_text = response.choices[0].message.content.strip()

            # Extract JSON from response
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")

            parsed_data = json.loads(json_match.group())

            # Convert intent back to readable format
            intent = parsed_data["intent"].replace("_", " ")

            return ParsedIntent(
                intent=intent,
                confidence=parsed_data.get("confidence", 0.8),
                args=parsed_data.get("parameters", {}),
                provider="openai",
                raw_response=response_text,
                json_spec=parsed_data,
                structured_data=parsed_data,
            )

        except Exception as e:
            logger.error(f"OpenAI structured parsing error: {e}")
            return None

    def _parse_intent_fallback(self, prompt: str) -> ParsedIntent:
        """Fallback intent parsing using basic pattern matching.

        Args:
            prompt: User prompt

        Returns:
            ParsedIntent: Basic parsed intent
        """
        prompt_lower = prompt.lower()

        # Basic pattern matching
        if any(
            word in prompt_lower for word in ["forecast", "predict", "price", "stock"]
        ):
            intent = "forecast stock price"
        elif any(word in prompt_lower for word in ["strategy", "trade", "signal"]):
            intent = "generate trading strategy"
        elif any(word in prompt_lower for word in ["analyze", "market", "trend"]):
            intent = "analyze market"
        elif any(
            word in prompt_lower for word in ["optimize", "portfolio", "allocation"]
        ):
            intent = "optimize portfolio"
        elif any(word in prompt_lower for word in ["position", "manage", "risk"]):
            intent = "manage positions"
        elif any(word in prompt_lower for word in ["system", "status", "health"]):
            intent = "system status"
        elif any(word in prompt_lower for word in ["invest", "buy", "sell", "advice"]):
            intent = "investment advice"
        else:
            intent = "general question"

        # Extract basic parameters
        args = self._extract_basic_parameters(prompt)

        return ParsedIntent(
            intent=intent,
            confidence=0.5,  # Low confidence for fallback
            args=args,
            provider="fallback",
            raw_response=prompt,
        )

    def _extract_parameters_enhanced(
        self, prompt: str, parsed_intent: ParsedIntent
    ) -> Dict[str, Any]:
        """Enhanced parameter extraction.

        Args:
            prompt: User prompt
            parsed_intent: Parsed intent result

        Returns:
            Dictionary of extracted parameters
        """
        params = {}

        # Use structured data if available
        if parsed_intent.structured_data:
            params.update(parsed_intent.structured_data.get("parameters", {}))

        # Extract from parsed intent args
        params.update(parsed_intent.args)

        # Extract additional parameters using regex
        params.update(self._extract_params_regex(prompt))

        return params

    def _extract_params_regex(self, prompt: str) -> Dict[str, Any]:
        """Extract parameters using regex patterns.

        Args:
            prompt: User prompt

        Returns:
            Dictionary of extracted parameters
        """
        params = {}

        # Extract stock symbols
        symbol_pattern = r"\b[A-Z]{1,5}\b"
        symbols = re.findall(symbol_pattern, prompt.upper())
        if symbols:
            params["symbols"] = symbols

        # Extract timeframes
        timeframe_pattern = r"\b(\d+)\s*(day|week|month|year)s?\b"
        timeframes = re.findall(timeframe_pattern, prompt.lower())
        if timeframes:
            params["timeframe"] = f"{timeframes[0][0]} {timeframes[0][1]}"

        # Extract amounts
        amount_pattern = r"\$?(\d+(?:,\d{3})*(?:\.\d{2})?)"
        amounts = re.findall(amount_pattern, prompt)
        if amounts:
            params["amount"] = float(amounts[0].replace(",", ""))

        return params

    def _extract_basic_parameters(self, prompt: str) -> Dict[str, Any]:
        """Extract basic parameters from prompt.

        Args:
            prompt: User prompt

        Returns:
            Dictionary of basic parameters
        """
        return self._extract_params_regex(prompt)

    def _extract_params_from_text(self, text: str) -> Dict[str, Any]:
        """Extract parameters from generated text.

        Args:
            text: Generated parameter text

        Returns:
            Dictionary of parameters
        """
        params = {}

        # Simple extraction from generated text
        lines = text.split("\n")
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                value = value.strip()
                if value:
                    params[key] = value

        return params

    def _map_intent_to_request_type(self, intent: str) -> RequestType:
        """Map intent to request type.

        Args:
            intent: Detected intent

        Returns:
            RequestType: Mapped request type
        """
        return self.label_to_request_type.get(intent, RequestType.UNKNOWN)

    def _normalize_prompt(self, prompt: str) -> str:
        """Normalize prompt text.

        Args:
            prompt: Raw prompt

        Returns:
            Normalized prompt
        """
        return prompt.strip().lower()

    def _generate_routing_suggestions(
        self, request_type: RequestType, parameters: Dict[str, Any]
    ) -> List[str]:
        """Generate routing suggestions based on request type and parameters.

        Args:
            request_type: Type of request
            parameters: Extracted parameters

        Returns:
            List of suggested agents
        """
        suggestions = []

        if request_type == RequestType.FORECAST:
            suggestions.extend(["ModelSelectorAgent", "MarketAnalyzerAgent"])
        elif request_type == RequestType.STRATEGY:
            suggestions.extend(["StrategySelectorAgent", "MetaTunerAgent"])
        elif request_type == RequestType.ANALYSIS:
            suggestions.extend(["MarketAnalyzerAgent", "ModelSelectorAgent"])
        elif request_type == RequestType.OPTIMIZATION:
            suggestions.extend(["MetaTunerAgent", "PortfolioManagerAgent"])
        elif request_type == RequestType.PORTFOLIO:
            suggestions.extend(["PortfolioManagerAgent", "MetaTunerAgent"])
        elif request_type == RequestType.SYSTEM:
            suggestions.extend(["SystemMonitorAgent"])
        else:
            suggestions.extend(["ModelSelectorAgent", "StrategySelectorAgent"])

        return suggestions

    def get_learning_insights(self) -> Dict[str, Any]:
        """Get learning insights from memory.

        Returns:
            Dictionary with learning insights
        """
        return self.memory.get_learning_insights()

    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics.

        Returns:
            Dictionary with performance statistics
        """
        stats = self.performance_metrics.copy()
        stats["success_rate"] = (
            stats["successful_parses"] / stats["total_requests"]
            if stats["total_requests"] > 0
            else 0
        )
        return stats

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

        # Check if strategy was detected
        if not processed.request_type or processed.request_type == RequestType.UNKNOWN:
            # Return fallback for no strategy detected
            return RoutingDecision(
                request_id=f"req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                request_type=RequestType.GENERAL,
                primary_agent="GeneralAgent",
                fallback_agents=[],
                confidence=0.1,
                reasoning="No strategy detected, using general fallback",
                expected_response_time=1.0,
                priority=7,
                timestamp=datetime.now(),
                metadata={
                    "action": "help",
                    "message": "No strategy detected",
                    "processed_prompt": processed,
                    "provider_used": "consolidated",
                },
            )

        # Find suitable agents
        suitable_agents = self._find_suitable_agents(
            processed.request_type, user_request
        )

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

        required_capability = capability_mapping.get(
            request_type, AgentCapability.GENERAL_QUERY
        )

        for agent in self.available_agents.values():
            if (
                required_capability in agent.capabilities
                and agent.is_available
                and agent.current_load < agent.max_concurrent
            ):
                suitable_agents.append(agent)

        return suitable_agents

    def _select_agents(
        self, suitable_agents: List[AgentInfo], context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[str]]:
        """Select primary and fallback agents."""
        if not suitable_agents:
            # Fallback to general agent
            return "GeneralAgent", []

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
            agent.avg_response_time = (agent.avg_response_time * 0.9) + (
                response_time * 0.1
            )

            # Update last used time
            agent.last_used = datetime.now()

    def handle_prompt(
        self, prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for prompt handling with comprehensive trace logging.

        Args:
            prompt: User prompt
            context: Optional context

        Returns:
            Dictionary with routing decision and processing results
        """
        if not self.available:
            print("âš ï¸ PromptAgent unavailable due to initialization failure")
            return {
                "success": False,
                "error": "PromptAgent unavailable due to initialization failure",
                "trace_id": f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                "processing_time": 0.0,
            }

        start_time = datetime.now()

        # Start trace logging
        trace_id = f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        session_id = context.get("session_id") if context else None
        user_id = context.get("user_id") if context else None

        trace = self.trace_logger.start_trace(
            trace_id=trace_id,
            session_id=session_id,
            user_id=user_id,
            original_prompt=prompt,
            context=context,
        )

        try:
            # Process the prompt
            processed = self.process_prompt(prompt)

            # Parse intent and update trace
            parsed_intent = self.parse_intent_enhanced(prompt)
            self.trace_logger.update_intent(
                trace=trace,
                detection_method=parsed_intent.provider,
                intent=parsed_intent.intent,
                confidence=parsed_intent.confidence,
                parameters=parsed_intent.args,
                normalized_prompt=processed.original_prompt,
            )

            # Route the request
            routing_decision = self.route_request(prompt, context)

            # Update trace with action execution
            action_duration = (datetime.now() - start_time).total_seconds()
            self.trace_logger.update_action(
                trace=trace,
                action=f"route_to_{routing_decision.primary_agent}",
                status=(
                    ActionStatus.SUCCESS
                    if routing_decision.confidence > 0.5
                    else ActionStatus.PARTIAL
                ),
                result={
                    "primary_agent": routing_decision.primary_agent,
                    "fallback_agents": routing_decision.fallback_agents,
                    "confidence": routing_decision.confidence,
                    "reasoning": routing_decision.reasoning,
                },
                duration=action_duration,
            )

            # Update provider usage in trace
            self.trace_logger.update_provider_usage(
                trace=trace, provider=parsed_intent.provider, count=1
            )

            # Complete trace logging
            processing_time = (datetime.now() - start_time).total_seconds()
            self.trace_logger.complete_trace(trace, processing_time)

            return {
                "success": True,
                "trace_id": trace_id,
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

            # Update trace with error
            self.trace_logger.add_error(trace, str(e))
            self.trace_logger.update_action(
                trace=trace,
                action="error_handling",
                status=ActionStatus.FAILED,
                result={"error": str(e)},
                duration=(datetime.now() - start_time).total_seconds(),
            )

            # Complete trace logging
            processing_time = (datetime.now() - start_time).total_seconds()
            self.trace_logger.complete_trace(trace, processing_time)

            return {
                "success": False,
                "trace_id": trace_id,
                "error": str(e),
                "processing_time": processing_time,
            }


# Convenience function to create a prompt agent
def create_prompt_agent(
    openai_api_key: Optional[str] = None,
    huggingface_model: str = "microsoft/DialoGPT-medium",
    classification_model: str = "facebook/bart-large-mnli",
    huggingface_api_key: Optional[str] = None,
    enable_debug_mode: bool = False,
    use_huggingface_first: bool = True,
    use_openai_fallback: bool = True,
    redis_url: Optional[str] = None,
) -> PromptAgent:
    """
    Create a configured prompt agent.

    Args:
        openai_api_key: OpenAI API key
        huggingface_model: HuggingFace model for text generation
        classification_model: HuggingFace model for intent classification
        huggingface_api_key: HuggingFace API key
        enable_debug_mode: Enable debug mode
        use_huggingface_first: Use HuggingFace classification first
        use_openai_fallback: Use OpenAI fallback
        redis_url: Redis URL for memory storage

    Returns:
        Configured PromptAgent instance
    """
    return PromptAgent(
        openai_api_key=openai_api_key,
        huggingface_model=huggingface_model,
        classification_model=classification_model,
        huggingface_api_key=huggingface_api_key,
        enable_debug_mode=enable_debug_mode,
        use_huggingface_first=use_huggingface_first,
        use_openai_fallback=use_openai_fallback,
        redis_url=redis_url,
    )
