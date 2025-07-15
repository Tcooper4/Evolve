"""
Fallback LLM with Negation Detection

This module provides a fallback LLM system with advanced prompt parsing
capabilities including negation detection and action skipping.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of actions that can be performed."""
    FORECAST = "forecast"
    ANALYZE = "analyze"
    TRADE = "trade"
    OPTIMIZE = "optimize"
    REPORT = "report"
    MONITOR = "monitor"
    ALERT = "alert"
    SKIP = "skip"  # Action should be skipped


class NegationType(Enum):
    """Types of negation patterns."""
    EXPLICIT = "explicit"  # "don't", "not", "never"
    IMPLICIT = "implicit"  # "avoid", "skip", "ignore"
    CONDITIONAL = "conditional"  # "unless", "except", "but"
    TEMPORAL = "temporal"  # "not now", "later", "wait"


@dataclass
class NegationPattern:
    """Pattern for detecting negation in prompts."""
    pattern: str
    negation_type: NegationType
    strength: float  # 0.0 to 1.0, how strong the negation is
    action_scope: List[str]  # Which actions this negation affects
    description: str


@dataclass
class ParsedAction:
    """Parsed action from prompt."""
    action_type: ActionType
    confidence: float
    parameters: Dict[str, Any]
    is_negated: bool = False
    negation_strength: float = 0.0
    negation_type: Optional[NegationType] = None
    skip_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptParseResult:
    """Result of parsing a prompt."""
    original_prompt: str
    actions: List[ParsedAction]
    skipped_actions: List[ParsedAction]
    confidence: float
    processing_time: float
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class NegationDetector:
    """Detects negation patterns in prompts."""
    
    def __init__(self):
        """Initialize negation detector with patterns."""
        self.negation_patterns = self._load_negation_patterns()
        self.action_keywords = self._load_action_keywords()
    
    def _load_negation_patterns(self) -> List[NegationPattern]:
        """Load negation patterns."""
        patterns = [
            # Explicit negations
            NegationPattern(
                pattern=r"\b(don't|do not|doesn't|does not|didn't|did not)\s+(\w+)",
                negation_type=NegationType.EXPLICIT,
                strength=0.9,
                action_scope=["*"],  # Affects all actions
                description="Explicit verb negation"
            ),
            NegationPattern(
                pattern=r"\b(not|never|no|none|neither|nor)\b",
                negation_type=NegationType.EXPLICIT,
                strength=0.8,
                action_scope=["*"],
                description="Explicit negation words"
            ),
            
            # Implicit negations
            NegationPattern(
                pattern=r"\b(avoid|skip|ignore|exclude|omit|bypass)\b",
                negation_type=NegationType.IMPLICIT,
                strength=0.7,
                action_scope=["*"],
                description="Implicit action avoidance"
            ),
            NegationPattern(
                pattern=r"\b(stop|halt|pause|suspend|cancel)\b",
                negation_type=NegationType.IMPLICIT,
                strength=0.6,
                action_scope=["*"],
                description="Action stopping"
            ),
            
            # Conditional negations
            NegationPattern(
                pattern=r"\b(unless|except|but|however|though|although)\b",
                negation_type=NegationType.CONDITIONAL,
                strength=0.5,
                action_scope=["*"],
                description="Conditional negation"
            ),
            
            # Temporal negations
            NegationPattern(
                pattern=r"\b(not now|later|wait|postpone|delay|hold off)\b",
                negation_type=NegationType.TEMPORAL,
                strength=0.4,
                action_scope=["*"],
                description="Temporal negation"
            ),
            
            # Specific action negations
            NegationPattern(
                pattern=r"\b(don't forecast|no forecast|skip forecast)\b",
                negation_type=NegationType.EXPLICIT,
                strength=0.9,
                action_scope=["forecast"],
                description="Forecast-specific negation"
            ),
            NegationPattern(
                pattern=r"\b(don't trade|no trading|skip trade)\b",
                negation_type=NegationType.EXPLICIT,
                strength=0.9,
                action_scope=["trade"],
                description="Trading-specific negation"
            ),
            NegationPattern(
                pattern=r"\b(don't analyze|no analysis|skip analysis)\b",
                negation_type=NegationType.EXPLICIT,
                strength=0.9,
                action_scope=["analyze"],
                description="Analysis-specific negation"
            ),
        ]
        return patterns
    
    def _load_action_keywords(self) -> Dict[str, List[str]]:
        """Load action keywords for detection."""
        return {
            "forecast": ["forecast", "predict", "future", "next", "upcoming", "tomorrow"],
            "analyze": ["analyze", "analysis", "examine", "study", "review", "assess"],
            "trade": ["trade", "buy", "sell", "position", "order", "execute"],
            "optimize": ["optimize", "tune", "improve", "enhance", "better"],
            "report": ["report", "summary", "overview", "status"],
            "monitor": ["monitor", "watch", "track", "observe", "check"],
            "alert": ["alert", "notify", "warn", "signal"]
        }
    
    def detect_negations(self, prompt: str) -> List[Dict[str, Any]]:
        """
        Detect negation patterns in prompt.
        
        Args:
            prompt: User prompt
            
        Returns:
            List of detected negations
        """
        negations = []
        normalized_prompt = prompt.lower()
        
        for pattern in self.negation_patterns:
            matches = re.finditer(pattern.pattern, normalized_prompt, re.IGNORECASE)
            
            for match in matches:
                negation = {
                    "pattern": pattern.pattern,
                    "match_text": match.group(0),
                    "start_pos": match.start(),
                    "end_pos": match.end(),
                    "negation_type": pattern.negation_type,
                    "strength": pattern.strength,
                    "action_scope": pattern.action_scope,
                    "description": pattern.description
                }
                negations.append(negation)
        
        return negations
    
    def is_action_negated(self, action_type: str, prompt: str) -> Tuple[bool, float, Optional[NegationType]]:
        """
        Check if a specific action is negated.
        
        Args:
            action_type: Type of action to check
            prompt: User prompt
            
        Returns:
            Tuple of (is_negated, strength, negation_type)
        """
        negations = self.detect_negations(prompt)
        
        for negation in negations:
            # Check if this negation affects the action
            if "*" in negation["action_scope"] or action_type in negation["action_scope"]:
                return True, negation["strength"], negation["negation_type"]
        
        return False, 0.0, None


class FallbackLLM:
    """
    Fallback LLM system with advanced prompt parsing and negation detection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize fallback LLM.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.negation_detector = NegationDetector()
        
        # Load configuration
        self.min_confidence = self.config.get("min_confidence", 0.3)
        self.enable_negation_detection = self.config.get("enable_negation_detection", True)
        self.skip_negated_actions = self.config.get("skip_negated_actions", True)
        
        logger.info("FallbackLLM initialized with negation detection")
    
    def parse_prompt(self, prompt: str) -> PromptParseResult:
        """
        Parse prompt and detect actions with negation analysis.
        
        Args:
            prompt: User prompt
            
        Returns:
            PromptParseResult with parsed actions
        """
        start_time = datetime.now()
        
        try:
            # Detect actions in prompt
            actions = self._detect_actions(prompt)
            
            # Apply negation detection
            if self.enable_negation_detection:
                actions = self._apply_negation_detection(actions, prompt)
            
            # Separate skipped and active actions
            active_actions = []
            skipped_actions = []
            
            for action in actions:
                if action.is_negated and self.skip_negated_actions:
                    action.skip_reason = f"Action negated with {action.negation_strength:.2f} strength"
                    skipped_actions.append(action)
                else:
                    active_actions.append(action)
            
            # Calculate overall confidence
            confidence = self._calculate_overall_confidence(actions)
            
            # Generate warnings
            warnings = self._generate_warnings(actions, prompt)
            
            return PromptParseResult(
                original_prompt=prompt,
                actions=active_actions,
                skipped_actions=skipped_actions,
                confidence=confidence,
                processing_time=(datetime.now() - start_time).total_seconds(),
                warnings=warnings,
                metadata={
                    "negation_detection_enabled": self.enable_negation_detection,
                    "skip_negated_actions": self.skip_negated_actions,
                    "total_actions_detected": len(actions),
                    "actions_skipped": len(skipped_actions)
                }
            )
            
        except Exception as e:
            logger.error(f"Error parsing prompt: {e}")
            return PromptParseResult(
                original_prompt=prompt,
                actions=[],
                skipped_actions=[],
                confidence=0.0,
                processing_time=(datetime.now() - start_time).total_seconds(),
                warnings=[f"Error parsing prompt: {str(e)}"],
                metadata={"error": str(e)}
            )
    
    def _detect_actions(self, prompt: str) -> List[ParsedAction]:
        """Detect actions in the prompt."""
        actions = []
        normalized_prompt = prompt.lower()
        
        # Check for each action type
        for action_type_str, keywords in self.negation_detector.action_keywords.items():
            action_type = ActionType(action_type_str)
            
            # Count keyword matches
            matches = 0
            for keyword in keywords:
                if keyword in normalized_prompt:
                    matches += 1
            
            if matches > 0:
                # Calculate confidence based on matches
                confidence = min(1.0, matches / len(keywords) + 0.3)
                
                # Extract parameters
                parameters = self._extract_action_parameters(action_type_str, prompt)
                
                action = ParsedAction(
                    action_type=action_type,
                    confidence=confidence,
                    parameters=parameters
                )
                actions.append(action)
        
        return actions
    
    def _extract_action_parameters(self, action_type: str, prompt: str) -> Dict[str, Any]:
        """Extract parameters for a specific action type."""
        parameters = {}
        normalized_prompt = prompt.lower()
        
        # Extract common parameters
        if "symbol" in normalized_prompt:
            symbol_match = re.search(r'\b([A-Z]{1,5})\b', prompt)
            if symbol_match:
                parameters["symbol"] = symbol_match.group(1)
        
        if "timeframe" in normalized_prompt:
            timeframe_match = re.search(r'\b(1m|5m|15m|30m|1h|4h|1d|1w|1M)\b', normalized_prompt)
            if timeframe_match:
                parameters["timeframe"] = timeframe_match.group(1)
        
        if "days" in normalized_prompt or "period" in normalized_prompt:
            days_match = re.search(r'\b(\d+)\s*(days?|d)\b', normalized_prompt)
            if days_match:
                parameters["days"] = int(days_match.group(1))
        
        # Action-specific parameters
        if action_type == "forecast":
            if "model" in normalized_prompt:
                model_match = re.search(r'\b(lstm|arima|xgboost|prophet|ensemble)\b', normalized_prompt)
                if model_match:
                    parameters["model"] = model_match.group(1)
        
        elif action_type == "trade":
            if "direction" in normalized_prompt:
                if "buy" in normalized_prompt or "long" in normalized_prompt:
                    parameters["direction"] = "buy"
                elif "sell" in normalized_prompt or "short" in normalized_prompt:
                    parameters["direction"] = "sell"
        
        elif action_type == "optimize":
            if "metric" in normalized_prompt:
                metric_match = re.search(r'\b(accuracy|precision|recall|f1|profit|sharpe)\b', normalized_prompt)
                if metric_match:
                    parameters["metric"] = metric_match.group(1)
        
        return parameters
    
    def _apply_negation_detection(self, actions: List[ParsedAction], prompt: str) -> List[ParsedAction]:
        """Apply negation detection to actions."""
        for action in actions:
            is_negated, strength, negation_type = self.negation_detector.is_action_negated(
                action.action_type.value, prompt
            )
            
            if is_negated:
                action.is_negated = True
                action.negation_strength = strength
                action.negation_type = negation_type
                
                # Reduce confidence for negated actions
                action.confidence *= (1.0 - strength)
        
        return actions
    
    def _calculate_overall_confidence(self, actions: List[ParsedAction]) -> float:
        """Calculate overall confidence for the prompt parse."""
        if not actions:
            return 0.0
        
        # Average confidence of all actions
        avg_confidence = np.mean([action.confidence for action in actions])
        
        # Boost confidence if multiple actions detected
        if len(actions) > 1:
            avg_confidence *= 1.1
        
        return min(1.0, avg_confidence)
    
    def _generate_warnings(self, actions: List[ParsedAction], prompt: str) -> List[str]:
        """Generate warnings for the prompt parse."""
        warnings = []
        
        # Check for conflicting actions
        action_types = [action.action_type for action in actions if not action.is_negated]
        if len(set(action_types)) != len(action_types):
            warnings.append("Conflicting actions detected")
        
        # Check for low confidence actions
        low_confidence_actions = [action for action in actions if action.confidence < self.min_confidence]
        if low_confidence_actions:
            warnings.append(f"{len(low_confidence_actions)} actions have low confidence")
        
        # Check for negated actions
        negated_actions = [action for action in actions if action.is_negated]
        if negated_actions:
            warnings.append(f"{len(negated_actions)} actions are negated and will be skipped")
        
        # Check for ambiguous prompts
        if len(actions) > 3:
            warnings.append("Prompt contains many actions - consider simplifying")
        
        return warnings
    
    def should_skip_action(self, action_type: str, prompt: str) -> Tuple[bool, str]:
        """
        Check if an action should be skipped based on negation.
        
        Args:
            action_type: Type of action
            prompt: User prompt
            
        Returns:
            Tuple of (should_skip, reason)
        """
        if not self.enable_negation_detection:
            return False, "Negation detection disabled"
        
        is_negated, strength, negation_type = self.negation_detector.is_action_negated(action_type, prompt)
        
        if is_negated and self.skip_negated_actions:
            reason = f"Action negated with {strength:.2f} strength ({negation_type.value})"
            return True, reason
        
        return False, "Action not negated"
    
    def get_negation_summary(self, prompt: str) -> Dict[str, Any]:
        """
        Get summary of negation detection for a prompt.
        
        Args:
            prompt: User prompt
            
        Returns:
            Negation summary
        """
        negations = self.negation_detector.detect_negations(prompt)
        
        summary = {
            "total_negations": len(negations),
            "negation_types": {},
            "strongest_negation": None,
            "affected_actions": set()
        }
        
        if negations:
            # Count negation types
            for negation in negations:
                neg_type = negation["negation_type"].value
                summary["negation_types"][neg_type] = summary["negation_types"].get(neg_type, 0) + 1
                
                # Track affected actions
                summary["affected_actions"].update(negation["action_scope"])
            
            # Find strongest negation
            strongest = max(negations, key=lambda x: x["strength"])
            summary["strongest_negation"] = {
                "text": strongest["match_text"],
                "strength": strongest["strength"],
                "type": strongest["negation_type"].value
            }
        
        # Convert set to list for JSON serialization
        summary["affected_actions"] = list(summary["affected_actions"])
        
        return summary


def create_fallback_llm(config: Optional[Dict[str, Any]] = None) -> FallbackLLM:
    """
    Create a fallback LLM instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        FallbackLLM instance
    """
    return FallbackLLM(config) 