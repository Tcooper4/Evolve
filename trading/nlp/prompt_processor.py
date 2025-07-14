import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EntityMatch:
    """Data class to hold entity match information."""

    entity_type: str
    value: str
    start: int
    end: int
    confidence: float


@dataclass
class ProcessedPrompt:
    """Data class to hold processed prompt information."""

    original_text: str
    entities: List[EntityMatch]
    intent: str
    confidence: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Intent:
    name: str
    confidence: float
    entities: Dict[str, Any]
    context: Dict[str, Any]


# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add file handler for debug logs
debug_handler = logging.FileHandler("trading/nlp/logs/nlp_debug.log")
debug_handler.setLevel(logging.DEBUG)
debug_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
debug_handler.setFormatter(debug_formatter)
logger.addHandler(debug_handler)

MEMORY_LOG_PATH = os.path.join(os.path.dirname(__file__), "memory_log.jsonl")


class PromptProcessor:
    """Processes and generates prompts for the LLM."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize prompt processor.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.entity_patterns = self._load_entity_patterns()
        logger.info("PromptProcessor initialized with entity patterns")

        # Initialize context
        self.context = {}

    def _load_entity_patterns(self) -> Dict[str, Any]:
        """Load entity patterns from configuration."""
        try:
            config_path = os.path.join(
                os.path.dirname(__file__), "config", "entity_patterns.json"
            )
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load entity patterns: {e}")
            return {}

    def process_prompt(self, text: str) -> ProcessedPrompt:
        """Process a text prompt to extract entities and determine intent.

        Args:
            text: The input text to process

        Returns:
            ProcessedPrompt object with extracted information
        """
        try:
            entities = self._extract_entities(text)
            intent, confidence = self._determine_intent(text, entities)

            return ProcessedPrompt(
                original_text=text,
                entities=entities,
                intent=intent,
                confidence=confidence,
            )
        except Exception as e:
            self.logger.error(f"Error processing prompt: {e}")
            return ProcessedPrompt(
                original_text=text, entities=[], intent="unknown", confidence=0.0
            )

    def _extract_entities(self, text: str) -> List[EntityMatch]:
        """Extract entities from text using regex patterns.

        Args:
            text: The input text to process

        Returns:
            List of EntityMatch objects
        """
        entities = []

        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities.append(
                        EntityMatch(
                            entity_type=entity_type,
                            value=match.group(),
                            start=match.start(),
                            end=match.end(),
                            confidence=1.0,  # Could be improved with ML-based confidence scoring
                        )
                    )

        # Sort entities by start position
        entities.sort(key=lambda x: x.start)

        return entities

    def _determine_intent(
        self, text: str, entities: List[EntityMatch]
    ) -> Tuple[str, float]:
        """Determine the intent of the prompt.

        Args:
            text: The input text to process
            entities: List of extracted entities

        Returns:
            Tuple of (intent, confidence)
        """
        best_intent = "unknown"
        best_confidence = 0.0

        for intent, patterns in self.entity_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    # Calculate confidence based on pattern match and entity presence
                    confidence = 0.5  # Base confidence for pattern match

                    # Boost confidence if relevant entities are present
                    if intent == "forecast" and any(
                        e.entity_type in ["timeframe", "asset"] for e in entities
                    ):
                        confidence += 0.3
                    elif intent == "analyze" and any(
                        e.entity_type in ["asset", "indicator"] for e in entities
                    ):
                        confidence += 0.3
                    elif intent == "recommend" and any(
                        e.entity_type in ["asset", "action"] for e in entities
                    ):
                        confidence += 0.3
                    elif intent == "explain" and any(
                        e.entity_type in ["topic", "concept"] for e in entities
                    ):
                        confidence += 0.3

                    if confidence > best_confidence:
                        best_intent = intent
                        best_confidence = confidence

        return best_intent, best_confidence

    def get_entity_values(self, prompt: ProcessedPrompt, entity_type: str) -> List[str]:
        """Get values for a specific entity type from a processed prompt.

        Args:
            prompt: ProcessedPrompt object
            entity_type: Type of entity to extract

        Returns:
            List of entity values
        """
        return [e.value for e in prompt.entities if e.entity_type == entity_type]

    def get_entity_by_type(
        self, prompt: ProcessedPrompt, entity_type: str
    ) -> Optional[EntityMatch]:
        """Get the first entity of a specific type from a processed prompt.

        Args:
            prompt: ProcessedPrompt object
            entity_type: Type of entity to extract

        Returns:
            EntityMatch object or None if not found
        """
        for entity in prompt.entities:
            if entity.entity_type == entity_type:
                return entity
        return None

    def has_entity(self, prompt: ProcessedPrompt, entity_type: str) -> bool:
        """Check if a processed prompt contains a specific entity type.

        Args:
            prompt: ProcessedPrompt object
            entity_type: Type of entity to check for

        Returns:
            True if entity type is present, False otherwise
        """
        return any(e.entity_type == entity_type for e in prompt.entities)

    def get_required_entities(self, prompt: ProcessedPrompt, intent: str) -> List[str]:
        """Get list of required entities for a specific intent.

        Args:
            prompt: ProcessedPrompt object
            intent: Intent to check requirements for

        Returns:
            List of required entity types
        """
        required_entities = {
            "forecast": ["timeframe", "asset"],
            "analyze": ["asset"],
            "recommend": ["asset", "action"],
            "explain": ["topic"],
            "compare": ["asset"],
            "optimize": ["strategy"],
            "validate": ["test"],
            "monitor": ["asset"],
        }

        return required_entities.get(intent, [])

    def validate_prompt(self, prompt: ProcessedPrompt) -> Tuple[bool, List[str]]:
        """Validate if a processed prompt has all required entities for its intent.

        Args:
            prompt: ProcessedPrompt object to validate

        Returns:
            Tuple of (is_valid, missing_entities)
        """
        required_entities = self.get_required_entities(prompt, prompt.intent)
        missing_entities = []

        for entity_type in required_entities:
            if not self.has_entity(prompt, entity_type):
                missing_entities.append(entity_type)

        return len(missing_entities) == 0, missing_entities

    def _update_context(self, intent: str, entities: Dict[str, List[str]]):
        """Update context with new information."""
        self.context.update(
            {
                "last_intent": intent,
                "last_entities": entities,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def get_context(self) -> Dict[str, Any]:
        """Get current context."""
        return self.context

    def clear_context(self):
        """Clear current context."""
        self.context = {}

    def format_response(self, intent: Intent, response: str) -> str:
        """Format response based on intent and context."""
        try:
            if intent.name == "forecast":
                return self._format_forecast_response(intent, response)
            elif intent.name == "analyze":
                return self._format_analysis_response(intent, response)
            elif intent.name == "recommend":
                return self._format_recommendation_response(intent, response)
            elif intent.name == "explain":
                return self._format_explanation_response(intent, response)
            else:
                return response
        except Exception as e:
            self.logger.error(f"Error formatting response: {str(e)}")
            return response

    def _format_forecast_response(self, intent: Intent, response: str) -> str:
        """Format forecast response."""
        timeframe = intent.entities.get("timeframe", ["unknown timeframe"])[0]
        return f"Forecast for {timeframe}:\n{response}"

    def _format_analysis_response(self, intent: Intent, response: str) -> str:
        """Format analysis response."""
        metric = intent.entities.get("metric", ["current situation"])[0]
        return f"Analysis of {metric}:\n{response}"

    def _format_recommendation_response(self, intent: Intent, response: str) -> str:
        """Format recommendation response."""
        action = intent.entities.get("action", ["action"])[0]
        return f"Recommendation for {action}:\n{response}"

    def _format_explanation_response(self, intent: Intent, response: str) -> str:
        """Format explanation response."""
        return f"Explanation:\n{response}"

    def create_intent_prompt(self, query: str) -> str:
        """Create prompt for intent detection.

        Args:
            query: User query string

        Returns:
            Formatted prompt string
        """
        prompt = f"""Analyze the following query and determine the user's intent:

Query: {query}

Please respond with a JSON object containing:
1. intent: The primary intent (e.g., "forecast", "analyze", "compare", "explain")
2. confidence: A float between 0 and 1 indicating confidence in the intent detection
3. reasoning: Brief explanation of why this intent was chosen

Example response:
{{
    "intent": "forecast",
    "confidence": 0.95,
    "reasoning": "User is asking for future price predictions"
}}
"""
        logger.debug(f"Created intent prompt for query: {query}")
        return prompt

    def create_response_prompt(
        self,
        query: str,
        intent: str,
        entities: Dict[str, Any],
        session_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create prompt for response generation.

        Args:
            query: Original user query
            intent: Detected intent
            entities: Extracted entities
            session_state: Optional session state

        Returns:
            Formatted prompt string
        """
        # Add context from session state
        context = ""
        if session_state:
            if "previous_queries" in session_state:
                context += "\nPrevious queries:\n"
                for prev_query in session_state["previous_queries"][-3:]:
                    context += f"- {prev_query}\n"
            if "user_preferences" in session_state:
                context += "\nUser preferences:\n"
                for pref, value in session_state["user_preferences"].items():
                    context += f"- {pref}: {value}\n"

        prompt = f"""Generate a response to the following query:

Query: {query}
Intent: {intent}
Entities: {json.dumps(entities, indent=2)}
{context}

Please respond with a JSON object containing:
1. response: The main response text
2. visualizations: List of visualizations to include (if any)
3. confidence: A float between 0 and 1 indicating confidence in the response
4. metadata: Additional information about the response

Example response:
{{
    "response": "Based on the analysis, I predict...",
    "visualizations": [
        {{
            "type": "forecast_plot",
            "data": {{...}},
            "narrative": "The forecast shows..."
        }}
    ],
    "confidence": 0.9,
    "metadata": {{
        "timeframe": "1 week",
        "model_used": "LSTM"
    }}
}}
"""
        logger.debug(f"Created response prompt for intent: {intent}")
        return prompt

    def extract_entities(self, query: str, intent: str) -> Dict[str, Any]:
        """Extract entities from query using patterns.

        Args:
            query: User query string
            intent: Detected intent

        Returns:
            Dictionary of extracted entities
        """
        entities = {}

        try:
            # Get relevant patterns for intent
            intent_patterns = self.entity_patterns.get(intent, {})

            # Extract entities using patterns
            for entity_type, pattern in intent_patterns.items():
                if pattern in query.lower():
                    entities[entity_type] = {
                        "value": pattern,
                        "confidence": 0.8,  # Base confidence
                        "source": "pattern_match",
                    }

            # Log extraction results
            logger.debug(f"Extracted entities: {entities}")

            return entities

        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}", exc_info=True)
            return {}

    def expand_entities(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Expand entities with synonyms and clarifications.

        Args:
            entities: Dictionary of extracted entities

        Returns:
            Expanded entities dictionary
        """
        expanded = {}

        try:
            for entity_type, data in entities.items():
                # Get entity patterns
                patterns = self.entity_patterns.get(entity_type, {})

                # Add synonyms
                if "synonyms" in patterns:
                    data["synonyms"] = patterns["synonyms"]

                # Add clarifications
                if "clarifications" in patterns:
                    data["clarifications"] = patterns["clarifications"]

                # Add expanded value if available
                if "expanded" in patterns:
                    data["expanded_value"] = patterns["expanded"]

                expanded[entity_type] = data

            # Log expansion results
            logger.debug(f"Expanded entities: {expanded}")

            return expanded

        except Exception as e:
            logger.error(f"Error expanding entities: {str(e)}", exc_info=True)
            return {}

    def route_to_agent(self, entities: dict, intent: str = None) -> dict:
        """
        Route parsed entities (and optional intent) to the agentic routing system.
        Returns a dict with routed action, reasoning, and confidence.
        Also logs the routing to memory_log.jsonl.
        """
        # Dummy routing logic for demonstration
        # In production, replace with actual agent/ForecastAgent logic
        action = None
        reasoning = ""
        confidence = 0.8
        if intent:
            if intent == "forecast":
                action = "run_forecast"
                reasoning = "Intent classified as forecast."
            elif intent == "backtest":
                action = "run_backtest"
                reasoning = "Intent classified as backtest."
            elif intent == "compare":
                action = "compare_models"
                reasoning = "Intent classified as compare."
            else:
                action = "unknown"
                reasoning = f"Unknown intent: {intent}"
        else:
            # Fallback: infer from entities
            if "ticker" in entities and "timeframe" in entities:
                action = "run_forecast"
                reasoning = "Ticker and timeframe present. Defaulting to forecast."
            else:
                action = "insufficient_info"
                reasoning = "Missing required entities."
                confidence = 0.3
        routed = {"action": action, "reasoning": reasoning, "confidence": confidence}
        # Log to memory
        self.log_memory(
            prompt=getattr(self, "last_prompt", None), entities=entities, routed=routed
        )
        return routed

    def log_memory(self, prompt: str, entities: dict, routed: dict):
        """
        Append a memory log entry: prompt ➜ entities ➜ routed action/reasoning/confidence.
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "prompt": prompt,
            "entities": entities,
            "routed": routed,
        }
        try:
            with open(MEMORY_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write memory log: {e}")

    def process_and_route(self, prompt: str) -> dict:
        """
        Full pipeline: extract entities, classify intent, route to agent, and log memory.
        Returns dict with entities, intent, routed action, reasoning, and confidence.
        """
        self.last_prompt = prompt
        entities = self.extract_entities(prompt)
        intent = (
            self.classify_intent(prompt) if hasattr(self, "classify_intent") else None
        )
        routed = self.route_to_agent(entities, intent)
        return {"entities": entities, "intent": intent, "routed": routed}

    def classify_intent(self, prompt: str) -> str:
        """
        Classify the intent of the prompt (forecast, backtest, compare, interpret, explain, etc).
        Returns a string intent label.
        """
        prompt_lower = prompt.lower()
        if any(
            word in prompt_lower
            for word in ["forecast", "predict", "projection", "outlook"]
        ):
            return "forecast"
        if any(
            word in prompt_lower
            for word in ["backtest", "historical performance", "simulate"]
        ):
            return "backtest"
        if any(
            word in prompt_lower for word in ["compare", "versus", "vs.", "better than"]
        ):
            return "compare"
        if any(word in prompt_lower for word in ["interpret", "explain", "why", "how"]):
            return "interpret"
        if any(word in prompt_lower for word in ["summarize", "summary", "overview"]):
            return "summarize"
        if any(word in prompt_lower for word in ["analyze", "analysis", "insight"]):
            return "analyze"
        return "unknown"
