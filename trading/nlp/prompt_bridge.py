"""
Prompt Bridge - Batch 19
Enhanced prompt parsing with compound intent handling and regex fallbacks
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Types of intents that can be parsed."""

    FORECAST = "forecast"
    STRATEGY = "strategy"
    ANALYSIS = "analysis"
    EXECUTION = "execution"
    OPTIMIZATION = "optimization"
    MONITORING = "monitoring"
    COMPOUND = "compound"


@dataclass
class ParsedIntent:
    """Parsed intent with metadata."""

    intent_type: IntentType
    action: str
    parameters: Dict[str, Any]
    confidence: float
    sub_intents: List["ParsedIntent"] = field(default_factory=list)
    original_text: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PromptParseResult:
    """Result of prompt parsing."""

    intents: List[ParsedIntent]
    success: bool
    error_message: Optional[str] = None
    parse_time: float = 0.0
    compound_detected: bool = False


class PromptBridge:
    """
    Enhanced prompt bridge with compound intent handling.

    Features:
    - Compound prompt splitting
    - Regex fallback for comma-separated and conjunction-joined intents
    - Intent validation and confidence scoring
    - Sub-intent dependency resolution
    """

    def __init__(
        self,
        enable_regex_fallback: bool = True,
        max_compound_intents: int = 5,
        min_confidence_threshold: float = 0.3,
    ):
        """
        Initialize prompt bridge.

        Args:
            enable_regex_fallback: Enable regex-based intent parsing
            max_compound_intents: Maximum number of intents in compound prompt
            min_confidence_threshold: Minimum confidence for intent acceptance
        """
        self.enable_regex_fallback = enable_regex_fallback
        self.max_compound_intents = max_compound_intents
        self.min_confidence_threshold = min_confidence_threshold

        # Intent patterns
        self.intent_patterns = self._initialize_intent_patterns()

        # Conjunction patterns for compound prompts
        self.conjunction_patterns = [
            r"\s+and\s+",
            r"\s+&\s+",
            r"\s+plus\s+",
            r"\s+with\s+",
            r"\s+then\s+",
            r"\s+followed\s+by\s+",
            r"\s+after\s+",
            r"\s+before\s+",
        ]

        # Comma patterns
        self.comma_patterns = [r",\s*", r";\s*", r"\|\s*", r"\.\s+"]

        # Parse history for learning
        self.parse_history: List[PromptParseResult] = []

        logger.info(
            f"PromptBridge initialized with regex fallback: {enable_regex_fallback}"
        )

    def _initialize_intent_patterns(self) -> Dict[str, List[Tuple[str, IntentType]]]:
        """Initialize regex patterns for intent detection."""
        patterns = {
            "forecast": [
                (r"\b(forecast|predict|project)\b", IntentType.FORECAST),
                (
                    r"\b(price|market|trend)\s+(prediction|forecast)\b",
                    IntentType.FORECAST,
                ),
                (r"\b(future|upcoming)\s+(price|value)\b", IntentType.FORECAST),
            ],
            "strategy": [
                (
                    r"\b(apply|use|run|execute)\s+(strategy|strategy)\b",
                    IntentType.STRATEGY,
                ),
                (
                    r"\b(RSI|MACD|SMA|BB|Bollinger|Moving\s+Average)\b",
                    IntentType.STRATEGY,
                ),
                (
                    r"\b(trading|investment)\s+(strategy|approach)\b",
                    IntentType.STRATEGY,
                ),
            ],
            "analysis": [
                (r"\b(analyze|examine|study|review)\b", IntentType.ANALYSIS),
                (r"\b(market|technical|fundamental)\s+analysis\b", IntentType.ANALYSIS),
                (r"\b(performance|returns|risk)\s+analysis\b", IntentType.ANALYSIS),
            ],
            "execution": [
                (r"\b(buy|sell|trade|execute|order)\b", IntentType.EXECUTION),
                (r"\b(place|submit|process)\s+(order|trade)\b", IntentType.EXECUTION),
                (r"\b(entry|exit)\s+(position|trade)\b", IntentType.EXECUTION),
            ],
            "optimization": [
                (r"\b(optimize|tune|adjust|fine-tune)\b", IntentType.OPTIMIZATION),
                (
                    r"\b(parameter|hyperparameter)\s+(optimization|tuning)\b",
                    IntentType.OPTIMIZATION,
                ),
                (
                    r"\b(improve|enhance|boost)\s+(performance|returns)\b",
                    IntentType.OPTIMIZATION,
                ),
            ],
            "monitoring": [
                (r"\b(monitor|watch|track|observe)\b", IntentType.MONITORING),
                (r"\b(alert|notify|report)\b", IntentType.MONITORING),
                (
                    r"\b(dashboard|status|health)\s+(check|monitor)\b",
                    IntentType.MONITORING,
                ),
            ],
        }
        return patterns

    def parse_prompt(self, prompt_text: str) -> PromptParseResult:
        """
        Parse prompt text into structured intents.

        Args:
            prompt_text: Raw prompt text

        Returns:
            PromptParseResult with parsed intents
        """
        start_time = datetime.now()

        try:
            # Check for compound prompts
            if self._is_compound_prompt(prompt_text):
                logger.info("Compound prompt detected, splitting intents")
                intents = self.split_intents(prompt_text)
                compound_detected = True
            else:
                # Single intent parsing
                intents = [self._parse_single_intent(prompt_text)]
                compound_detected = False

            # Validate and filter intents
            valid_intents = []
            for intent in intents:
                if intent.confidence >= self.min_confidence_threshold:
                    valid_intents.append(intent)
                else:
                    logger.warning(
                        f"Low confidence intent filtered: {intent.action} (confidence: {intent.confidence})"
                    )

            # Create result
            parse_time = (datetime.now() - start_time).total_seconds()
            result = PromptParseResult(
                intents=valid_intents,
                success=len(valid_intents) > 0,
                parse_time=parse_time,
                compound_detected=compound_detected,
            )

            # Store in history
            self.parse_history.append(result)

            if result.success:
                logger.info(
                    f"Successfully parsed {len(valid_intents)} intents from prompt"
                )
            else:
                logger.warning("No valid intents found in prompt")
                result.error_message = "No valid intents detected"

            return result

        except Exception as e:
            logger.error(f"Error parsing prompt: {e}")
            parse_time = (datetime.now() - start_time).total_seconds()
            return PromptParseResult(
                intents=[], success=False, error_message=str(e), parse_time=parse_time
            )

    def _is_compound_prompt(self, prompt_text: str) -> bool:
        """Detect if prompt contains multiple intents."""
        # Check for conjunctions
        for pattern in self.conjunction_patterns:
            if re.search(pattern, prompt_text, re.IGNORECASE):
                return True

        # Check for comma separation
        for pattern in self.comma_patterns:
            if re.search(pattern, prompt_text):
                return True

        # Check for multiple action verbs
        action_verbs = [
            "forecast",
            "apply",
            "run",
            "analyze",
            "execute",
            "optimize",
            "monitor",
        ]
        verb_count = sum(
            1 for verb in action_verbs if verb.lower() in prompt_text.lower()
        )

        return verb_count > 1

    def split_intents(self, prompt_text: str) -> List[ParsedIntent]:
        """
        Split compound prompt into individual intents.

        Args:
            prompt_text: Compound prompt text

        Returns:
            List of parsed intents
        """
        intents = []

        # Try conjunction-based splitting first
        split_texts = self._split_by_conjunctions(prompt_text)

        if len(split_texts) == 1:
            # Try comma-based splitting
            split_texts = self._split_by_commas(prompt_text)

        # Parse each split text
        for i, text in enumerate(split_texts):
            if len(intents) >= self.max_compound_intents:
                logger.warning(
                    f"Maximum compound intents ({self.max_compound_intents}) reached"
                )
                break

            text = text.strip()
            if text:
                intent = self._parse_single_intent(text)
                intent.original_text = text
                intents.append(intent)

        # Create compound intent if multiple intents found
        if len(intents) > 1:
            compound_intent = ParsedIntent(
                intent_type=IntentType.COMPOUND,
                action="compound_execution",
                parameters={"sub_intent_count": len(intents)},
                confidence=0.8,
                sub_intents=intents,
                original_text=prompt_text,
            )
            return [compound_intent]

        return intents

    def _split_by_conjunctions(self, prompt_text: str) -> List[str]:
        """Split prompt by conjunction patterns."""
        split_texts = [prompt_text]

        for pattern in self.conjunction_patterns:
            new_splits = []
            for text in split_texts:
                parts = re.split(pattern, text, flags=re.IGNORECASE)
                new_splits.extend(parts)
            split_texts = new_splits

        return [text.strip() for text in split_texts if text.strip()]

    def _split_by_commas(self, prompt_text: str) -> List[str]:
        """Split prompt by comma patterns."""
        split_texts = [prompt_text]

        for pattern in self.comma_patterns:
            new_splits = []
            for text in split_texts:
                parts = re.split(pattern, text)
                new_splits.extend(parts)
            split_texts = new_splits

        return [text.strip() for text in split_texts if text.strip()]

    def _parse_single_intent(self, text: str) -> ParsedIntent:
        """
        Parse single intent from text.

        Args:
            text: Single intent text

        Returns:
            ParsedIntent object
        """
        # Try structured parsing first
        intent = self._structured_parse(text)

        if (
            intent.confidence < self.min_confidence_threshold
            and self.enable_regex_fallback
        ):
            # Fallback to regex parsing
            intent = self._regex_parse(text)

        return intent

    def _structured_parse(self, text: str) -> ParsedIntent:
        """Attempt structured parsing of intent."""
        text_lower = text.lower()

        # Check each intent type
        for intent_category, patterns in self.intent_patterns.items():
            for pattern, intent_type in patterns:
                if re.search(pattern, text_lower):
                    # Extract parameters
                    parameters = self._extract_parameters(text, intent_type)

                    return ParsedIntent(
                        intent_type=intent_type,
                        action=intent_category,
                        parameters=parameters,
                        confidence=0.8,
                        original_text=text,
                    )

        # Default fallback
        return ParsedIntent(
            intent_type=IntentType.ANALYSIS,
            action="general_analysis",
            parameters={"raw_text": text},
            confidence=0.3,
            original_text=text,
        )

    def _regex_parse(self, text: str) -> ParsedIntent:
        """Fallback regex parsing for unknown intents."""
        text_lower = text.lower()

        # Extract key terms
        key_terms = []
        for category, patterns in self.intent_patterns.items():
            for pattern, intent_type in patterns:
                matches = re.findall(pattern, text_lower)
                if matches:
                    key_terms.extend(matches)

        # Determine intent type based on key terms
        if any(term in text_lower for term in ["forecast", "predict", "future"]):
            intent_type = IntentType.FORECAST
            action = "forecast"
        elif any(term in text_lower for term in ["strategy", "rsi", "macd", "sma"]):
            intent_type = IntentType.STRATEGY
            action = "strategy_execution"
        elif any(term in text_lower for term in ["analyze", "examine", "study"]):
            intent_type = IntentType.ANALYSIS
            action = "analysis"
        elif any(term in text_lower for term in ["buy", "sell", "trade"]):
            intent_type = IntentType.EXECUTION
            action = "trade_execution"
        else:
            intent_type = IntentType.ANALYSIS
            action = "general_analysis"

        # Extract parameters using regex
        parameters = self._extract_parameters_regex(text)

        return ParsedIntent(
            intent_type=intent_type,
            action=action,
            parameters=parameters,
            confidence=0.5,  # Lower confidence for regex parsing
            original_text=text,
        )

    def _extract_parameters(self, text: str, intent_type: IntentType) -> Dict[str, Any]:
        """Extract parameters from intent text."""
        parameters = {"raw_text": text}

        if intent_type == IntentType.STRATEGY:
            # Extract strategy names
            strategy_patterns = [
                r"\b(RSI|MACD|SMA|BB|Bollinger)\b",
                r"\b(Moving\s+Average|Exponential\s+Average)\b",
                r"\b(Relative\s+Strength\s+Index)\b",
            ]

            strategies = []
            for pattern in strategy_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                strategies.extend(matches)

            if strategies:
                parameters["strategies"] = strategies

        elif intent_type == IntentType.FORECAST:
            # Extract time horizons
            time_patterns = [
                r"\b(\d+)\s*(day|week|month|year)s?\b",
                r"\b(short|medium|long)\s+term\b",
                r"\b(tomorrow|next\s+week|next\s+month)\b",
            ]

            for pattern in time_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    parameters["time_horizon"] = matches[0]
                    break

        elif intent_type == IntentType.EXECUTION:
            # Extract trade parameters
            if "buy" in text.lower():
                parameters["action"] = "buy"
            elif "sell" in text.lower():
                parameters["action"] = "sell"

            # Extract quantities
            quantity_match = re.search(
                r"\b(\d+)\s*(shares|units|contracts)\b", text, re.IGNORECASE
            )
            if quantity_match:
                parameters["quantity"] = int(quantity_match.group(1))

        return parameters

    def _extract_parameters_regex(self, text: str) -> Dict[str, Any]:
        """Extract parameters using regex patterns."""
        parameters = {"raw_text": text}

        # Extract numbers
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
        if numbers:
            parameters["numbers"] = [float(n) for n in numbers]

        # Extract quoted strings
        quotes = re.findall(r'"([^"]*)"', text)
        if quotes:
            parameters["quoted_strings"] = quotes

        # Extract capitalized terms (potential symbols/names)
        capitalized = re.findall(r"\b[A-Z]{2,}\b", text)
        if capitalized:
            parameters["symbols"] = capitalized

        return parameters

    def resolve_dependencies(self, intents: List[ParsedIntent]) -> List[ParsedIntent]:
        """
        Resolve dependencies between intents.

        Args:
            intents: List of parsed intents

        Returns:
            Ordered list of intents with dependencies resolved
        """
        if not intents:
            return intents

        # Simple dependency resolution
        ordered_intents = []

        # Analysis intents first
        analysis_intents = [i for i in intents if i.intent_type == IntentType.ANALYSIS]
        ordered_intents.extend(analysis_intents)

        # Forecast intents next
        forecast_intents = [i for i in intents if i.intent_type == IntentType.FORECAST]
        ordered_intents.extend(forecast_intents)

        # Strategy intents
        strategy_intents = [i for i in intents if i.intent_type == IntentType.STRATEGY]
        ordered_intents.extend(strategy_intents)

        # Optimization intents
        optimization_intents = [
            i for i in intents if i.intent_type == IntentType.OPTIMIZATION
        ]
        ordered_intents.extend(optimization_intents)

        # Execution intents last
        execution_intents = [
            i for i in intents if i.intent_type == IntentType.EXECUTION
        ]
        ordered_intents.extend(execution_intents)

        # Monitoring intents
        monitoring_intents = [
            i for i in intents if i.intent_type == IntentType.MONITORING
        ]
        ordered_intents.extend(monitoring_intents)

        return ordered_intents

    def get_parse_statistics(self) -> Dict[str, Any]:
        """Get parsing statistics from history."""
        if not self.parse_history:
            return {}

        total_parses = len(self.parse_history)
        successful_parses = sum(1 for result in self.parse_history if result.success)
        compound_parses = sum(
            1 for result in self.parse_history if result.compound_detected
        )

        # Intent type distribution
        intent_counts = {}
        for result in self.parse_history:
            for intent in result.intents:
                intent_type = intent.intent_type.value
                intent_counts[intent_type] = intent_counts.get(intent_type, 0) + 1

        stats = {
            "total_parses": total_parses,
            "success_rate": successful_parses / total_parses,
            "compound_rate": compound_parses / total_parses,
            "avg_parse_time": sum(r.parse_time for r in self.parse_history)
            / total_parses,
            "intent_distribution": intent_counts,
            "regex_fallback_enabled": self.enable_regex_fallback,
        }

        return stats


def create_prompt_bridge(enable_regex_fallback: bool = True) -> PromptBridge:
    """Factory function to create prompt bridge."""
    return PromptBridge(enable_regex_fallback=enable_regex_fallback)
